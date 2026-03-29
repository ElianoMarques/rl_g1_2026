"""Stage 1 (alt): Kimodo — Text to Motion.

Kimodo is a kinematic motion diffusion model trained on 700 hours of
commercially-friendly optical mocap data. It generates high-quality 3D motions
from text prompts with extensive kinematic constraints support.

Available models (March 2026):
  - Kimodo-SOMA-RP-v1 / Kimodo-SOMA-SEED-v1 (SOMA skeleton)
  - Kimodo-G1-RP-v1 / Kimodo-G1-SEED-v1 (G1 skeleton, direct robot output)
  - Kimodo-SMPLX-RP-v1 (SMPL-X skeleton)

Input:  Text prompt + optional kinematic constraints
Output: NPZ (posed_joints, rotation matrices, foot contacts) or
        MuJoCo qpos CSV (G1) or AMASS NPZ (SMPL-X)

Repo: https://github.com/nv-tlabs/kimodo
Deps: ~17GB VRAM, PyTorch, CUDA GPU (RTX 3090/4090/A100)

Usage:
    python -m kimodo_pipeline.kimodo_gen.run_kimodo \\
        --prompt "walk forward naturally" \\
        --skeleton g1 \\
        --output data/motions_g1/

    # With constraints
    python -m kimodo_pipeline.kimodo_gen.run_kimodo \\
        --prompt "wave right hand while walking" \\
        --skeleton g1 \\
        --constraints constraints.json \\
        --output data/motions_g1/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Kimodo model variants
KIMODO_MODELS = {
    "soma": {
        "rp": "Kimodo-SOMA-RP-v1",
        "seed": "Kimodo-SOMA-SEED-v1",
    },
    "g1": {
        "rp": "Kimodo-G1-RP-v1",
        "seed": "Kimodo-G1-SEED-v1",
    },
    "smplx": {
        "rp": "Kimodo-SMPLX-RP-v1",
    },
}

# Supported constraint types
CONSTRAINT_TYPES = [
    "full_body_keyframe",    # Full body pose at specific frame
    "2d_root_path",          # 2D root trajectory on ground plane
    "end_effector_pos",      # End-effector position at frame
    "end_effector_rot",      # End-effector rotation at frame
    "waypoint",              # Pass through waypoint at frame
]


def check_kimodo_installation() -> bool:
    """Check if Kimodo is installed."""
    try:
        import kimodo  # noqa: F401
        return True
    except ImportError:
        return False


def run_kimodo_generation(prompt: str, skeleton: str = "g1",
                          model_variant: str = "seed", output_dir: str = ".",
                          device: str = "cuda:0", duration: float = 5.0,
                          constraints_file: str = None):
    """Generate motion from text prompt using Kimodo.

    Args:
        prompt: Text description of desired motion
        skeleton: Target skeleton ("soma", "g1", "smplx")
        model_variant: Model variant ("rp" = Rigplay, "seed" = BONES-SEED)
        output_dir: Output directory
        device: CUDA device
        duration: Motion duration in seconds
        constraints_file: Optional JSON file with kinematic constraints
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if skeleton not in KIMODO_MODELS:
        raise ValueError(f"Unknown skeleton: {skeleton}. Choose from {list(KIMODO_MODELS.keys())}")

    model_name = KIMODO_MODELS[skeleton].get(model_variant)
    if model_name is None:
        raise ValueError(f"No {model_variant} variant for skeleton {skeleton}")

    logger.info(f"Kimodo generation:")
    logger.info(f"  Model:    {model_name}")
    logger.info(f"  Prompt:   '{prompt}'")
    logger.info(f"  Skeleton: {skeleton}")
    logger.info(f"  Duration: {duration}s")

    # Load constraints if provided
    constraints = None
    if constraints_file:
        with open(constraints_file) as f:
            constraints = json.load(f)
        logger.info(f"  Constraints: {len(constraints)} loaded")

    if not check_kimodo_installation():
        logger.warning(
            "Kimodo not installed. Install from:\n"
            "  git clone https://github.com/nv-tlabs/kimodo.git\n"
            "  cd kimodo && pip install -e .\n"
            "\nGenerating demo motion for testing."
        )
        _generate_demo_motion(prompt, skeleton, output_dir, duration)
        return

    # Run Kimodo
    try:
        from kimodo import KimodoGenerator

        gen = KimodoGenerator(model_name=model_name, device=device)

        result = gen.generate(
            prompt=prompt,
            duration=duration,
            constraints=constraints,
        )

        # Export based on skeleton type
        safe_prompt = prompt.replace(" ", "_")[:50]
        if skeleton == "g1":
            # MuJoCo qpos CSV for G1
            csv_path = output_dir / f"{safe_prompt}_g1.csv"
            result.export_mujoco_csv(str(csv_path))
            logger.info(f"G1 CSV saved: {csv_path}")
        elif skeleton == "smplx":
            # AMASS NPZ for SMPL-X
            npz_path = output_dir / f"{safe_prompt}_smplx.npz"
            result.export_amass_npz(str(npz_path))
            logger.info(f"SMPL-X NPZ saved: {npz_path}")
        else:
            # SOMA NPZ
            npz_path = output_dir / f"{safe_prompt}_soma.npz"
            result.export_npz(str(npz_path))
            logger.info(f"SOMA NPZ saved: {npz_path}")

    except Exception as e:
        logger.error(f"Kimodo generation failed: {e}")
        raise


def _generate_demo_motion(prompt: str, skeleton: str, output_dir: Path,
                           duration: float = 5.0):
    """Generate demo motion data for testing without Kimodo installed."""
    fps = 30
    num_frames = int(duration * fps)
    t = np.linspace(0, duration, num_frames)
    walk_freq = 1.2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_prompt = prompt.replace(" ", "_")[:50]

    if skeleton == "g1":
        # G1 29-DOF MuJoCo qpos CSV
        # Columns: timestep, then 29 joint angles
        num_dof = 29
        qpos = np.zeros((num_frames, num_dof))

        # Simulate basic walking pattern on G1 joints
        # Hip pitch oscillation (joints 7, 8 in 23-DOF config)
        qpos[:, 7] = 0.3 * np.sin(2 * np.pi * walk_freq * t)
        qpos[:, 8] = 0.3 * np.sin(2 * np.pi * walk_freq * t + np.pi)
        # Knee flexion (joints 11, 12)
        qpos[:, 11] = 0.4 * np.abs(np.sin(2 * np.pi * walk_freq * t))
        qpos[:, 12] = 0.4 * np.abs(np.sin(2 * np.pi * walk_freq * t + np.pi))
        # Ankle (joints 17, 18)
        qpos[:, 17] = 0.97 + 0.1 * np.sin(2 * np.pi * walk_freq * t)
        qpos[:, 18] = 0.97 + 0.1 * np.sin(2 * np.pi * walk_freq * t + np.pi)
        # Arm swing (joints 5, 6)
        qpos[:, 5] = 0.3 - 0.15 * np.sin(2 * np.pi * walk_freq * t)
        qpos[:, 6] = 0.3 - 0.15 * np.sin(2 * np.pi * walk_freq * t + np.pi)

        csv_path = output_dir / f"{safe_prompt}_g1.csv"
        header = ",".join(["timestep"] + [f"joint_{i}" for i in range(num_dof)])
        timesteps = t.reshape(-1, 1)
        data = np.hstack([timesteps, qpos])
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        logger.info(f"Demo G1 CSV: {csv_path} ({num_frames} frames, {num_dof} DOF)")

    else:
        # SOMA NPZ format
        num_joints = 77
        posed_joints = np.zeros((num_frames, num_joints, 3))
        rotation_matrices = np.tile(np.eye(3), (num_frames, num_joints, 1, 1))
        foot_contacts = np.zeros((num_frames, 2))

        # Basic hip oscillation
        posed_joints[:, 1, 0] = 0.3 * np.sin(2 * np.pi * walk_freq * t)  # L hip
        posed_joints[:, 2, 0] = 0.3 * np.sin(2 * np.pi * walk_freq * t + np.pi)  # R hip

        # Alternating foot contacts
        foot_contacts[:, 0] = (np.sin(2 * np.pi * walk_freq * t) > 0).astype(float)
        foot_contacts[:, 1] = (np.sin(2 * np.pi * walk_freq * t + np.pi) > 0).astype(float)

        npz_path = output_dir / f"{safe_prompt}_soma.npz"
        np.savez(
            npz_path,
            posed_joints=posed_joints,
            rotation_matrices=rotation_matrices,
            foot_contacts=foot_contacts,
            fps=fps,
            prompt=prompt,
        )
        logger.info(f"Demo SOMA NPZ: {npz_path} ({num_frames} frames, {num_joints} joints)")


def main():
    parser = argparse.ArgumentParser(description="Kimodo: Generate motion from text")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--skeleton", type=str, default="g1",
                        choices=["soma", "g1", "smplx"])
    parser.add_argument("--variant", type=str, default="seed",
                        choices=["rp", "seed"])
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--constraints", type=str, default=None,
                        help="Constraints JSON file")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo output without Kimodo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        _generate_demo_motion(args.prompt, args.skeleton, Path(args.output), args.duration)
    else:
        run_kimodo_generation(
            prompt=args.prompt,
            skeleton=args.skeleton,
            model_variant=args.variant,
            output_dir=args.output,
            device=args.device,
            duration=args.duration,
            constraints_file=args.constraints,
        )


if __name__ == "__main__":
    main()
