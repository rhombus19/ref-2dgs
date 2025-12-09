#!/usr/bin/env python3
"""
Lightweight OpenCV viewer for Ref-GS Gaussians.

Controls (window focused):
  - Mouse left-drag: orbit yaw/pitch around the target
  - W/S: zoom in/out
  - A/D: yaw left/right
  - I/K: dolly forward/backward
  - J/L: strafe left/right
  - U/O: move target up/down
  - [/]: narrower/wider FOV
  - R: reset view
  - C: save current frame next to the model
  - Q or ESC: quit
"""

import argparse
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_real
from scene import GaussianModel, Scene
from utils.camera_utils import get_rays
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class OrbitState:
    def __init__(self, center, radius, fov_deg=60.0):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.fov_deg = float(fov_deg)
        self.yaw = 0.0
        self.pitch = 0.0
        self.znear = 0.01
        self.zfar = max(100.0, radius * 4.0)

    def forward_dir(self):
        return np.array(
            [
                math.cos(self.pitch) * math.sin(self.yaw),
                math.sin(self.pitch),
                math.cos(self.pitch) * math.cos(self.yaw),
            ],
            dtype=np.float32,
        )


def build_camera_from_state(state, width, height, device):
    state.pitch = float(np.clip(state.pitch, -1.4, 1.4))
    state.radius = max(0.05, state.radius)
    state.zfar = max(state.zfar, state.radius * 4.0)

    dir_vec = state.forward_dir()
    cam_pos = state.center + dir_vec * state.radius

    backward = cam_pos - state.center
    backward = backward / (np.linalg.norm(backward) + 1e-6)
    right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), backward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(backward, right)
    up = up / (np.linalg.norm(up) + 1e-6)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = backward
    c2w[:3, 3] = cam_pos

    w2c = np.linalg.inv(c2w)
    R_c2w = c2w[:3, :3]
    T_w2c = w2c[:3, 3]

    fovy = math.radians(state.fov_deg)
    fovx = 2 * math.atan(math.tan(fovy * 0.5) * (width / float(height)))

    world_view = (
        torch.tensor(getWorld2View2(R_c2w, T_w2c), device=device, dtype=torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    proj = (
        getProjectionMatrix(state.znear, state.zfar, fovx, fovy)
        .transpose(0, 1)
        .to(device)
        .contiguous()
    )
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    rays_o, rays_d = get_rays(width, height, fovx, fovy, R_c2w, T_w2c)
    rays_o = rays_o.to(device)
    rays_d = F.normalize(rays_d.to(device), dim=-1)

    return SimpleNamespace(
        original_image=torch.zeros(3, height, width, device=device, dtype=torch.float32),
        FoVx=fovx,
        FoVy=fovy,
        image_width=width,
        image_height=height,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
        camera_center=torch.tensor(cam_pos, device=device, dtype=torch.float32),
        rays_o=rays_o,
        rays_d=rays_d,
        zfar=state.zfar,
        znear=state.znear,
    )


def build_initial_state(scene, gaussians, prefer_train=True):
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().cpu().numpy()
    center = xyz.mean(axis=0) if xyz.size else np.zeros(3, dtype=np.float32)
    radius_guess = float(scene.cameras_extent) if scene.cameras_extent else 1.0

    cams = []
    if prefer_train:
        cams = scene.getTrainCameras()
    if not cams:  # fallback to test if no train available
        cams = scene.getTestCameras()
    yaw = 0.0
    pitch = 0.0
    fov_deg = 60.0
    if cams:
        init_cam = cams[0]
        w2c = init_cam.world_view_transform.cpu().numpy().T
        c2w = np.linalg.inv(w2c)
        pos = c2w[:3, 3]
        offset = pos - center
        radius_guess = float(np.linalg.norm(offset))
        yaw = math.atan2(offset[0], offset[2] + 1e-8)
        if radius_guess > 1e-6:
            pitch = math.asin(offset[1] / (radius_guess + 1e-8))
        fov_deg = math.degrees(init_cam.FoVy)

    state = OrbitState(center, max(radius_guess, 0.1), fov_deg=fov_deg)
    state.yaw = yaw
    state.pitch = pitch
    state.zfar = max(100.0, radius_guess * 4.0)
    return state


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for Ref-GS Gaussians")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration (-1 loads the latest)")
    parser.add_argument("--mode", choices=["real", "blender"], default="real", help="Use real renderer or blender-style renderer")
    parser.add_argument("--width", type=int, default=800, help="Viewer width in pixels")
    parser.add_argument("--height", type=int, default=800, help="Viewer height in pixels")
    parser.add_argument("--window", type=str, default="Ref-GS Viewer", help="Window title")
    args = get_combined_args(parser)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    torch.set_grad_enabled(False)

    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, resolution_scales=[1.0])
    gaussians = scene.gaussians

    device = gaussians.get_xyz.device
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )

    env_center = torch.tensor([float(c) for c in dataset.env_scope_center], device=device, dtype=torch.float32)
    env_radius = float(dataset.env_scope_radius)
    xyz_axis = [int(float(c)) for c in dataset.xyz_axis]

    state = build_initial_state(scene, gaussians)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    last_mouse = None
    dirty = True
    frame_bgr = None

    print(
        "Viewer ready. Controls: drag=LMB orbit | W/S zoom | A/D yaw | I/K dolly | "
        "J/L strafe | U/O target up/down | [/]=FOV | R reset | C capture | Q/ESC quit"
    )

    def on_mouse(event, x, y, flags, _param):
        nonlocal last_mouse, dirty
        if event == cv2.EVENT_LBUTTONDOWN:
            last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and last_mouse:
            dx = x - last_mouse[0]
            dy = y - last_mouse[1]
            state.yaw += dx * 0.005
            state.pitch += dy * 0.005
            last_mouse = (x, y)
            dirty = True

    cv2.setMouseCallback(args.window, on_mouse)

    while True:
        if dirty:
            cam = build_camera_from_state(state, args.width, args.height, device)
            if args.mode == "real" and env_radius > 0:
                render_pkg = render_real(
                    cam,
                    gaussians,
                    pipe,
                    bg_color,
                    iteration=1,
                    ENV_CENTER=env_center,
                    ENV_RADIUS=env_radius,
                    XYZ=xyz_axis,
                )
            elif args.mode == "real" and env_radius <= 0:
                print("env_scope_radius is 0; falling back to blender renderer for viewing.")
                render_pkg = render(cam, gaussians, pipe, bg_color, iteration=1)
            else:
                render_pkg = render(cam, gaussians, pipe, bg_color, iteration=1)

            image = render_pkg.get("pbr_rgb", render_pkg.get("render"))
            rgb = torch.clamp(image, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            frame_bgr = (rgb * 255.0).astype(np.uint8)[..., ::-1]
            dirty = False

        if frame_bgr is not None:
            display = frame_bgr.copy()
            info = f"dist {state.radius:.2f} | fov {state.fov_deg:.1f} | yaw {math.degrees(state.yaw)%360:.1f} | pitch {math.degrees(state.pitch):.1f}"
            cv2.putText(display, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(args.window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("w"):
            state.radius *= 0.96
            dirty = True
        elif key == ord("s"):
            state.radius *= 1.04
            dirty = True
        elif key == ord("a"):
            state.yaw -= 0.05
            dirty = True
        elif key == ord("d"):
            state.yaw += 0.05
            dirty = True
        elif key == ord("r"):
            state = build_initial_state(scene, gaussians)
            dirty = True
        elif key == ord("c") and frame_bgr is not None:
            ts = int(time.time())
            out_path = f"{dataset.model_path}/viewer_{ts}.png"
            cv2.imwrite(out_path, frame_bgr)
            print(f"Saved frame to {out_path}")
        elif key == ord("["):
            state.fov_deg = max(20.0, state.fov_deg - 1.0)
            dirty = True
        elif key == ord("]"):
            state.fov_deg = min(120.0, state.fov_deg + 1.0)
            dirty = True
        elif key in (ord("i"), ord("k"), ord("j"), ord("l"), ord("u"), ord("o")):
            forward = -state.forward_dir()
            forward = forward / (np.linalg.norm(forward) + 1e-6)
            right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), forward)
            right = right / (np.linalg.norm(right) + 1e-6)
            step = state.radius * 0.02
            if key == ord("i"):
                state.center += forward * step
            elif key == ord("k"):
                state.center -= forward * step
            elif key == ord("j"):
                state.center -= right * step
            elif key == ord("l"):
                state.center += right * step
            elif key == ord("u"):
                state.center[1] += step
            elif key == ord("o"):
                state.center[1] -= step
            dirty = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
