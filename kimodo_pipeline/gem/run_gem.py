"""Stage 1: GEM-X — Video to Motion.

GEM-X recovers full-body 77-joint SOMA motion from monocular video.
It includes a bundled 2D pose estimator trained for the SOMA skeleton,
making it fully self-contained. Supports dynamic cameras and recovers
global motion trajectories in world space.

Input:  Monocular video (.mp4)
Output: SOMA BVH motion file + optionally retargeted G1 CSV

Repo: https://github.com/NVlabs/GEM-X
Deps: Python 3.10+, PyTorch 2.10+, CUDA 12.6+, SOMA-X (submodule)

Usage:
    python -m kimodo_pipeline.gem.run_gem \\
        --video data/videos/walk.mp4 \\
        --output data/motions_soma/
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_gem_installation() -> bool:
    """Check if GEM-X is installed."""
    try:
        import gem_x  # noqa: F401
        return True
    except ImportError:
        return False


def run_gem_extraction(video_path: str, output_dir: str, device: str = "cuda:0",
                       world_space: bool = True, retarget_g1: bool = False):
    """Run GEM-X to extract SOMA motion from video.

    Args:
        video_path: Path to input video (.mp4)
        output_dir: Directory for output BVH files
        device: CUDA device
        world_space: Recover global trajectory (vs camera-space only)
        retarget_g1: Also run SOMA Retargeter to produce G1 CSV
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not check_gem_installation():
        logger.error(
            "GEM-X is not installed. Install from:\n"
            "  git clone https://github.com/NVlabs/GEM-X.git\n"
            "  cd GEM-X && pip install -e .\n"
            "\nRunning in demo mode with synthetic output."
        )
        _generate_demo_bvh(output_dir / f"{video_path.stem}.bvh")
        return

    # GEM-X extraction pipeline
    try:
        from gem_x import GEMEstimator

        estimator = GEMEstimator(device=device)
        logger.info(f"Running GEM-X on {video_path}")

        result = estimator.process_video(
            str(video_path),
            world_space=world_space,
        )

        # Export as SOMA BVH
        bvh_path = output_dir / f"{video_path.stem}.bvh"
        result.export_bvh(str(bvh_path))
        logger.info(f"SOMA BVH saved: {bvh_path} ({result.num_frames} frames)")

        # Optionally retarget to G1
        if retarget_g1:
            try:
                from soma_retargeter import retarget_bvh_to_g1
                csv_path = output_dir / f"{video_path.stem}_g1.csv"
                retarget_bvh_to_g1(str(bvh_path), str(csv_path))
                logger.info(f"G1 CSV saved: {csv_path}")
            except ImportError:
                logger.warning("soma-retargeter not installed. Skipping G1 retarget.")

    except Exception as e:
        logger.error(f"GEM-X extraction failed: {e}")
        raise


def _generate_demo_bvh(output_path: Path):
    """Generate a minimal demo BVH file for testing the pipeline without GEM-X."""
    import numpy as np

    num_frames = 300
    fps = 30.0
    frame_time = 1.0 / fps

    # Minimal BVH header for SOMA 77-joint skeleton (simplified to root only for demo)
    bvh_content = f"""HIERARCHY
ROOT Hips
{{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftUpLeg
  {{
    OFFSET 0.1 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLeg
    {{
      OFFSET 0.0 -0.4 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {{
        OFFSET 0.0 -0.4 0.0
      }}
    }}
  }}
  JOINT RightUpLeg
  {{
    OFFSET -0.1 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLeg
    {{
      OFFSET 0.0 -0.4 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {{
        OFFSET 0.0 -0.4 0.0
      }}
    }}
  }}
}}
MOTION
Frames: {num_frames}
Frame Time: {frame_time:.6f}
"""
    t = np.linspace(0, 10, num_frames)
    walk_freq = 1.2

    lines = []
    for i in range(num_frames):
        # Root: forward translation + small sway
        x = t[i] * 0.05  # forward
        y = 0.78 + 0.01 * np.sin(2 * np.pi * walk_freq * t[i] * 2)  # height bob
        z = 0.005 * np.sin(2 * np.pi * walk_freq * t[i])  # lateral sway
        rz, rx, ry = 0, 0, 0

        # Left hip/knee oscillation
        l_hip = 20 * np.sin(2 * np.pi * walk_freq * t[i])
        l_knee = 15 * np.abs(np.sin(2 * np.pi * walk_freq * t[i]))

        # Right hip/knee (opposite phase)
        r_hip = 20 * np.sin(2 * np.pi * walk_freq * t[i] + np.pi)
        r_knee = 15 * np.abs(np.sin(2 * np.pi * walk_freq * t[i] + np.pi))

        vals = [x, y, z, rz, rx, ry,
                0, l_hip, 0, 0, l_knee, 0,
                0, r_hip, 0, 0, r_knee, 0]
        lines.append(" ".join(f"{v:.4f}" for v in vals))

    bvh_content += "\n".join(lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bvh_content)
    logger.info(f"Demo BVH generated: {output_path} ({num_frames} frames at {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="GEM-X: Extract motion from video")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--retarget-g1", action="store_true",
                        help="Also retarget to G1 CSV using SOMA Retargeter")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo BVH without running GEM-X")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        output_path = Path(args.output) / f"{Path(args.video).stem}.bvh"
        _generate_demo_bvh(output_path)
    else:
        run_gem_extraction(
            video_path=args.video,
            output_dir=args.output,
            device=args.device,
            retarget_g1=args.retarget_g1,
        )


if __name__ == "__main__":
    main()
