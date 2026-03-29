"""Stage 3: ProtoMotions — RL Policy Training.

ProtoMotions is NVIDIA's GPU-accelerated framework for training physically
simulated humanoid robot policies via reinforcement learning.

Supported simulation frameworks:
  - Isaac Lab 2.3.0 (recommended)
  - IsaacGym Preview 4
  - Newton (e7a737c)
  - MuJoCo 3.0+

Training pipeline:
  1. Load retargeted G1 motions (CSV from SOMA Retargeter or Kimodo)
  2. Configure motion tracking environment
  3. Train PPO policy with motion imitation + balance rewards
  4. Export ONNX policy for deployment

Repo: https://github.com/NVlabs/ProtoMotions
Scale: 40+ hours AMASS in ~12h on 4x A100; BONES-SEED: 13K motions/GPU on 24x A100

Usage:
    python -m kimodo_pipeline.protomotions.train \\
        --motion-dir data/motions_g1/ \\
        --framework isaac_lab \\
        --num-envs 4096 \\
        --max-iterations 5000
"""

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def check_protomotions_installation() -> bool:
    """Check if ProtoMotions is installed."""
    try:
        import protomotions  # noqa: F401
        return True
    except ImportError:
        return False


def check_isaac_lab_installation() -> bool:
    """Check if Isaac Lab is available."""
    try:
        import isaaclab  # noqa: F401
        return True
    except ImportError:
        return False


def generate_protomotions_config(motion_dir: str, framework: str = "isaac_lab",
                                  num_envs: int = 4096, max_iterations: int = 5000,
                                  output_dir: str = "data/policies/") -> dict:
    """Generate ProtoMotions Hydra config for training.

    Args:
        motion_dir: Directory with G1 motion CSVs
        framework: Simulation backend
        num_envs: Number of parallel environments
        max_iterations: Training iterations
        output_dir: Where to save trained policy

    Returns:
        Config dict compatible with ProtoMotions
    """
    motion_dir = Path(motion_dir)
    motion_files = sorted(motion_dir.glob("*.csv"))

    if not motion_files:
        # Also check for NPZ
        motion_files = sorted(motion_dir.glob("*.npz"))

    logger.info(f"Found {len(motion_files)} motion files in {motion_dir}")

    config = {
        "robot": {
            "name": "unitree_g1",
            "urdf": "data/assets/g1.urdf",
            "num_dof": 29,
        },
        "motion": {
            "dir": str(motion_dir),
            "files": [str(f) for f in motion_files],
            "format": "csv",  # G1 CSV from retargeter
        },
        "simulation": {
            "framework": framework,
            "device": "cuda:0",
            "num_envs": num_envs,
            "dt": 0.005,
            "decimation": 4,
        },
        "training": {
            "algorithm": "PPO",
            "max_iterations": max_iterations,
            "learning_rate": 0.001,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "save_interval": 100,
        },
        "rewards": {
            "motion_tracking": {"weight": 2.0, "std": 0.25},
            "velocity_tracking": {"weight": 1.5, "std": 0.5},
            "balance": {"weight": 1.0},
            "energy": {"weight": -1e-5},
            "action_rate": {"weight": -0.01},
            "termination": {"weight": -200.0},
        },
        "domain_randomization": {
            "friction_range": [0.3, 1.3],
            "mass_perturbation": [-1.0, 3.0],
            "push_velocity": [-0.3, 0.3],
        },
        "export": {
            "format": "onnx",
            "output_dir": output_dir,
        },
    }

    return config


def run_protomotions_training(config: dict):
    """Launch ProtoMotions training.

    Args:
        config: Training configuration dict
    """
    if not check_protomotions_installation():
        logger.error(
            "ProtoMotions not installed. Install from:\n"
            "  git clone https://github.com/NVlabs/ProtoMotions.git\n"
            "  cd ProtoMotions && pip install -e .\n"
            "\nAlso requires Isaac Lab 2.3.0:\n"
            "  https://isaac-sim.github.io/IsaacLab/"
        )
        logger.info("Generating training launch script instead.")
        _generate_launch_script(config)
        return

    framework = config["simulation"]["framework"]
    logger.info(f"Starting ProtoMotions training with {framework}")

    try:
        from protomotions.train import train as pm_train
        pm_train(config)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def _generate_launch_script(config: dict):
    """Generate a shell script to launch ProtoMotions training."""
    framework = config["simulation"]["framework"]
    num_envs = config["simulation"]["num_envs"]
    max_iters = config["training"]["max_iterations"]
    motion_dir = config["motion"]["dir"]
    export_dir = config["export"]["output_dir"]

    script = f"""#!/usr/bin/env bash
# ProtoMotions Training Launch Script
# Generated by kimodo_pipeline
set -e

echo "=========================================="
echo " ProtoMotions: G1 Motion Tracking Training"
echo "=========================================="
echo ""
echo " Framework:   {framework}"
echo " Num envs:    {num_envs}"
echo " Max iters:   {max_iters}"
echo " Motion dir:  {motion_dir}"
echo ""

# Ensure ProtoMotions repo is cloned
if [ ! -d "ProtoMotions" ]; then
    echo "Cloning ProtoMotions..."
    git clone https://github.com/NVlabs/ProtoMotions.git
    cd ProtoMotions && pip install -e . && cd ..
fi

cd ProtoMotions

# Run training with Hydra config
python protomotions/train.py \\
    robot=unitree_g1 \\
    motion.dir={motion_dir} \\
    simulation.framework={framework} \\
    simulation.num_envs={num_envs} \\
    training.max_iterations={max_iters} \\
    training.save_interval=100 \\
    export.format=onnx \\
    export.output_dir={export_dir} \\
    --headless

echo ""
echo "Training complete!"
echo "Policy exported to: {export_dir}"
"""

    script_path = Path("kimodo_pipeline/protomotions/run_training.sh")
    script_path.write_text(script)
    script_path.chmod(0o755)
    logger.info(f"Launch script generated: {script_path}")
    logger.info(f"Run with: bash {script_path}")


def export_onnx(model_path: str, output_path: str, obs_dim: int = 400):
    """Export trained PyTorch policy to ONNX format.

    Args:
        model_path: Path to .pt model checkpoint
        output_path: Output .onnx path
        obs_dim: Observation dimension (80 * history_length=5 = 400)
    """
    try:
        import torch

        logger.info(f"Exporting {model_path} -> {output_path}")
        model = torch.jit.load(model_path)
        model.eval()

        dummy_input = torch.randn(1, obs_dim).to("cuda:0")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model, dummy_input, output_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )
        logger.info(f"ONNX export done: {output_path}")

    except ImportError:
        logger.error("PyTorch not available for ONNX export")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="ProtoMotions: Train G1 policy")
    parser.add_argument("--motion-dir", type=str, required=True,
                        help="Directory with G1 motion CSVs")
    parser.add_argument("--framework", type=str, default="isaac_lab",
                        choices=["isaac_lab", "isaac_gym", "newton", "mujoco"])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/policies/")
    parser.add_argument("--generate-script", action="store_true",
                        help="Only generate launch script")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = generate_protomotions_config(
        motion_dir=args.motion_dir,
        framework=args.framework,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        output_dir=args.output,
    )

    if args.generate_script:
        _generate_launch_script(config)
    else:
        run_protomotions_training(config)


if __name__ == "__main__":
    main()
