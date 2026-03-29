"""Stage 2: SOMA Retargeter — SOMA BVH to G1 Joint CSV.

Converts SOMA human motion captures into G1 robot joint animations using
GPU-optimized inverse kinematics via Newton physics engine and NVIDIA Warp.

Retargeting pipeline:
  1. Proportional human-to-robot scaling
  2. Multi-objective IK solving with joint limits
  3. Feet stabilization (maintain ground contact)
  4. Per-DOF joint limit clamping

Input:  SOMA BVH motion files (from GEM-X or BONES-SEED dataset)
Output: G1 CSV joint data (playable on robot or usable in ProtoMotions)

Repo: https://github.com/NVIDIA/soma-retargeter
Deps: Python 3.12, Newton physics, NVIDIA Warp, GPU (Maxwell+, driver 545+)

Usage:
    # Single file
    python -m kimodo_pipeline.soma_retarget.run_retarget \\
        --input data/motions_soma/walk.bvh \\
        --output data/motions_g1/

    # Batch conversion
    python -m kimodo_pipeline.soma_retarget.run_retarget \\
        --input data/motions_soma/ \\
        --output data/motions_g1/ \\
        --batch
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def check_soma_retargeter_installation() -> bool:
    """Check if SOMA Retargeter is installed."""
    try:
        import soma_retargeter  # noqa: F401
        return True
    except ImportError:
        return False


def retarget_bvh_to_g1(input_path: str, output_dir: str, viewer: bool = False):
    """Retarget a SOMA BVH file to G1 joint CSV.

    Args:
        input_path: Path to SOMA BVH file
        output_dir: Output directory for G1 CSV
        viewer: Launch OpenGL viewer showing source + retargeted motion
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not check_soma_retargeter_installation():
        logger.warning(
            "SOMA Retargeter not installed. Install from:\n"
            "  git clone https://github.com/NVIDIA/soma-retargeter.git\n"
            "  cd soma-retargeter && pip install -e .\n"
            "\nRunning demo retarget."
        )
        _demo_retarget(input_path, output_dir)
        return

    try:
        from soma_retargeter import SomaToG1Retargeter

        retargeter = SomaToG1Retargeter()
        logger.info(f"Retargeting: {input_path} -> G1 CSV")

        result = retargeter.retarget(
            bvh_path=str(input_path),
            viewer=viewer,
        )

        csv_path = output_dir / f"{input_path.stem}_g1.csv"
        result.export_csv(str(csv_path))
        logger.info(f"G1 CSV saved: {csv_path} ({result.num_frames} frames, "
                     f"{result.num_dof} DOF)")

    except Exception as e:
        logger.error(f"Retargeting failed: {e}")
        raise


def batch_retarget(input_dir: str, output_dir: str):
    """Batch retarget all BVH files in a directory.

    Args:
        input_dir: Directory containing SOMA BVH files
        output_dir: Output directory for G1 CSVs
    """
    input_dir = Path(input_dir)
    bvh_files = sorted(input_dir.glob("*.bvh"))

    if not bvh_files:
        logger.warning(f"No BVH files found in {input_dir}")
        return

    logger.info(f"Batch retargeting {len(bvh_files)} BVH files")

    for i, bvh_path in enumerate(bvh_files):
        logger.info(f"[{i+1}/{len(bvh_files)}] {bvh_path.name}")
        try:
            retarget_bvh_to_g1(str(bvh_path), output_dir, viewer=False)
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    logger.info(f"Batch retarget complete: {len(bvh_files)} files -> {output_dir}")


def _demo_retarget(input_path: Path, output_dir: Path):
    """Demo retargeting: parse BVH and produce a simple G1 CSV."""
    logger.info(f"Demo retarget: {input_path.name}")

    # Parse BVH to get frame count
    num_frames = 300
    fps = 30.0
    try:
        with open(input_path) as f:
            for line in f:
                if line.strip().startswith("Frames:"):
                    num_frames = int(line.split(":")[1].strip())
                elif line.strip().startswith("Frame Time:"):
                    fps = 1.0 / float(line.split(":")[1].strip())
    except Exception:
        pass

    # Generate G1 29-DOF joint trajectory
    # Map simplified BVH walking pattern to G1 joints
    num_dof = 29
    t = np.linspace(0, num_frames / fps, num_frames)
    walk_freq = 1.2

    g1_joints = np.zeros((num_frames, num_dof))

    # G1 joint mapping (index -> joint name):
    # 0: left_hip_yaw, 1: right_hip_yaw, 2: waist_yaw
    # 3: left_hip_roll, 4: right_hip_roll
    # 5: left_shoulder_pitch, 6: right_shoulder_pitch
    # 7: left_hip_pitch, 8: right_hip_pitch
    # 9: left_shoulder_roll, 10: right_shoulder_roll
    # 11: left_knee, 12: right_knee
    # ...

    # Walking pattern
    g1_joints[:, 7] = 0.25 * np.sin(2 * np.pi * walk_freq * t)       # L hip pitch
    g1_joints[:, 8] = 0.25 * np.sin(2 * np.pi * walk_freq * t + np.pi)  # R hip pitch
    g1_joints[:, 11] = 0.35 * np.abs(np.sin(2 * np.pi * walk_freq * t))  # L knee
    g1_joints[:, 12] = 0.35 * np.abs(np.sin(2 * np.pi * walk_freq * t + np.pi))  # R knee
    g1_joints[:, 5] = 0.3 - 0.1 * np.sin(2 * np.pi * walk_freq * t)   # L shoulder
    g1_joints[:, 6] = 0.3 - 0.1 * np.sin(2 * np.pi * walk_freq * t + np.pi)  # R shoulder

    # Export as CSV
    csv_path = output_dir / f"{input_path.stem}_g1.csv"
    header = ",".join(["timestep"] + [f"joint_{i}" for i in range(num_dof)])
    timesteps = t.reshape(-1, 1)
    data = np.hstack([timesteps, g1_joints])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    logger.info(f"Demo G1 CSV: {csv_path} ({num_frames} frames, {num_dof} DOF)")


def main():
    parser = argparse.ArgumentParser(description="SOMA Retargeter: BVH -> G1 CSV")
    parser.add_argument("--input", type=str, required=True,
                        help="Input BVH file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch process all BVH files in input directory")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch OpenGL viewer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.batch:
        batch_retarget(args.input, args.output)
    else:
        retarget_bvh_to_g1(args.input, args.output, viewer=args.viewer)


if __name__ == "__main__":
    main()
