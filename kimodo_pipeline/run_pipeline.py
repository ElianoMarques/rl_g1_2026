"""End-to-end Kimodo Pipeline Runner.

Orchestrates the full pipeline from video/text to deployment-ready policy.

Usage:
    # Full pipeline from video
    python -m kimodo_pipeline.run_pipeline --mode video --video input.mp4

    # Full pipeline from text
    python -m kimodo_pipeline.run_pipeline --mode text --prompt "walk forward"

    # Demo mode (no GPU required, generates synthetic data)
    python -m kimodo_pipeline.run_pipeline --mode demo
"""

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_video_pipeline(video_path: str, output_base: str = "data"):
    """Run full pipeline: Video -> GEM -> Retarget -> Train -> Validate -> Deploy."""
    logger.info("="*60)
    logger.info(" KIMODO PIPELINE: Video -> Real G1")
    logger.info("="*60)

    # Stage 1: GEM-X (Video to Motion)
    logger.info("\n[Stage 1] GEM-X: Extracting motion from video...")
    from kimodo_pipeline.gem.run_gem import run_gem_extraction
    run_gem_extraction(
        video_path=video_path,
        output_dir=f"{output_base}/motions_soma/",
        retarget_g1=False,
    )

    # Stage 2: SOMA Retargeter (SOMA BVH -> G1 CSV)
    logger.info("\n[Stage 2] SOMA Retargeter: Retargeting to G1...")
    from kimodo_pipeline.soma_retarget.run_retarget import retarget_bvh_to_g1
    stem = Path(video_path).stem
    retarget_bvh_to_g1(
        input_path=f"{output_base}/motions_soma/{stem}.bvh",
        output_dir=f"{output_base}/motions_g1/",
    )

    # Stage 3: ProtoMotions (RL Training)
    logger.info("\n[Stage 3] ProtoMotions: Training policy...")
    from kimodo_pipeline.protomotions.train import generate_protomotions_config, _generate_launch_script
    config = generate_protomotions_config(
        motion_dir=f"{output_base}/motions_g1/",
        framework="isaac_lab",
    )
    _generate_launch_script(config)
    logger.info("Training launch script generated. Run manually on GPU machine.")

    # Stage 4: MuJoCo Validation (would run after training)
    logger.info("\n[Stage 4] MuJoCo validation ready (run after training)")

    # Stage 5: GEAR-SONIC Deployment (would run after validation)
    logger.info("\n[Stage 5] GEAR-SONIC deployment ready (run on robot)")

    logger.info("\n" + "="*60)
    logger.info(" Pipeline setup complete!")
    logger.info("="*60)


def run_text_pipeline(prompt: str, output_base: str = "data"):
    """Run full pipeline: Text -> Kimodo -> Train -> Validate -> Deploy."""
    logger.info("="*60)
    logger.info(f" KIMODO PIPELINE: Text -> Real G1")
    logger.info(f" Prompt: '{prompt}'")
    logger.info("="*60)

    # Stage 1: Kimodo (Text to G1 Motion - direct, no retargeting needed)
    logger.info("\n[Stage 1] Kimodo: Generating G1 motion from text...")
    from kimodo_pipeline.kimodo_gen.run_kimodo import run_kimodo_generation
    run_kimodo_generation(
        prompt=prompt,
        skeleton="g1",
        model_variant="seed",
        output_dir=f"{output_base}/motions_g1/",
    )

    # Stage 2: ProtoMotions (RL Training)
    logger.info("\n[Stage 2] ProtoMotions: Training policy...")
    from kimodo_pipeline.protomotions.train import generate_protomotions_config, _generate_launch_script
    config = generate_protomotions_config(
        motion_dir=f"{output_base}/motions_g1/",
        framework="isaac_lab",
    )
    _generate_launch_script(config)

    logger.info("\n" + "="*60)
    logger.info(" Pipeline setup complete!")
    logger.info("="*60)


def run_demo_pipeline(output_base: str = "data"):
    """Run full demo pipeline with synthetic data (no GPU required)."""
    logger.info("="*60)
    logger.info(" KIMODO PIPELINE: Demo Mode")
    logger.info(" (synthetic data, no GPU required)")
    logger.info("="*60)

    output_base = Path(output_base)

    # Stage 1: Generate demo BVH (as if from GEM)
    logger.info("\n[Stage 1] GEM-X demo: Generating synthetic SOMA BVH...")
    from kimodo_pipeline.gem.run_gem import _generate_demo_bvh
    bvh_path = output_base / "motions_soma" / "demo_walk.bvh"
    _generate_demo_bvh(bvh_path)

    # Stage 1b: Generate demo G1 motion (as if from Kimodo)
    logger.info("\n[Stage 1b] Kimodo demo: Generating synthetic G1 motion...")
    from kimodo_pipeline.kimodo_gen.run_kimodo import _generate_demo_motion
    _generate_demo_motion(
        prompt="walk forward naturally",
        skeleton="g1",
        output_dir=output_base / "motions_g1",
    )

    # Stage 2: Demo retarget
    logger.info("\n[Stage 2] SOMA Retargeter demo: Retargeting BVH to G1 CSV...")
    from kimodo_pipeline.soma_retarget.run_retarget import _demo_retarget
    _demo_retarget(bvh_path, output_base / "motions_g1")

    # Stage 3: Generate training script
    logger.info("\n[Stage 3] ProtoMotions: Generating training config...")
    from kimodo_pipeline.protomotions.train import generate_protomotions_config, _generate_launch_script
    config = generate_protomotions_config(
        motion_dir=str(output_base / "motions_g1"),
        framework="isaac_lab",
    )
    _generate_launch_script(config)

    # Stage 4: Stub MuJoCo validation
    logger.info("\n[Stage 4] MuJoCo validation (stub)...")
    from kimodo_pipeline.protomotions.validate_mujoco import validate
    result = validate(
        policy_path="stub",
        model_path=None,
        num_episodes=5,
    )
    print(result.summary())

    # Stage 5: GEAR-SONIC dry run
    logger.info("\n[Stage 5] GEAR-SONIC dry run...")
    from kimodo_pipeline.gear_sonic.deploy import GearSonicDeployer
    deployer = GearSonicDeployer(
        policy_path="stub",
        robot_ip="stub",
        dry_run=True,
    )
    deployer.initialize()
    deployer.run(max_steps=100)

    logger.info("\n" + "="*60)
    logger.info(" Demo pipeline complete!")
    logger.info("="*60)

    # List outputs
    logger.info("\nGenerated files:")
    for f in sorted(output_base.rglob("*")):
        if f.is_file():
            logger.info(f"  {f}")


def main():
    parser = argparse.ArgumentParser(description="Kimodo Pipeline Runner")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["video", "text", "demo"])
    parser.add_argument("--video", type=str, help="Input video (for video mode)")
    parser.add_argument("--prompt", type=str, help="Text prompt (for text mode)")
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.mode == "video":
        if not args.video:
            parser.error("--video required for video mode")
        run_video_pipeline(args.video, args.output)
    elif args.mode == "text":
        if not args.prompt:
            parser.error("--prompt required for text mode")
        run_text_pipeline(args.prompt, args.output)
    elif args.mode == "demo":
        run_demo_pipeline(args.output)


if __name__ == "__main__":
    main()
