# Kimodo Pipeline: Video/Text to Real G1 Robot

End-to-end pipeline using NVIDIA's Kimodo ecosystem to go from **video or text**
to a **real Unitree G1 robot** performing learned whole-body motions.

## Ecosystem Tools

| Tool | What it does | Repo |
|------|-------------|------|
| **SOMA-X** | Universal 77-joint parametric body model | [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) |
| **BONES-SEED** | 142K+ production mocap dataset (SOMA + G1 formats) | Dataset release |
| **GEM-X** | Monocular video → 3D motion (SOMA BVH) | [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) |
| **Kimodo** | Text prompt → motion (SOMA/G1 NPZ/CSV) | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) |
| **SOMA Retargeter** | SOMA BVH → G1 joint CSV (Newton GPU IK) | [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter) |
| **ProtoMotions** | RL policy training (Isaac Lab / Newton / MuJoCo) | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) |
| **GEAR-SONIC** | Real G1 deployment (whole-body control) | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) |

## Pipeline Flow

```
                        +-----------+
                        |   Video   |
                        +-----+-----+
                              |
                         GEM-X (video2motion)
                              |
                        +-----v-----+
                        | SOMA BVH  |
                        +-----+-----+
                              |
                     SOMA Retargeter (IK)
                              |            +-----------+
                              |            |   Text    |
                              |            +-----+-----+
                              |                  |
                              |            Kimodo (text2motion)
                              |                  |
                        +-----v-----+      +-----v-----+
                        | G1 CSV    |      | G1 NPZ/CSV|
                        +-----+-----+      +-----+-----+
                              |                  |
                              +--------+---------+
                                       |
                            +----------v----------+
                            |     BONES-SEED      |
                            |    (142K+ motions)   |
                            +----------+----------+
                                       |
                                 ProtoMotions
                           (Isaac Lab / MuJoCo RL)
                                       |
                              +--------v--------+
                              |  ONNX Policy    |
                              +--------+--------+
                                       |
                            MuJoCo sim2sim test
                                       |
                                  GEAR-SONIC
                              (deploy to real G1)
```

## Quick Start

```bash
# Step 1: Extract motion from video using GEM
python -m kimodo_pipeline.gem.run_gem --video data/videos/walk.mp4 --output data/motions_soma/

# Step 2: Retarget SOMA motion to G1 joints
python -m kimodo_pipeline.soma_retarget.run_retarget --input data/motions_soma/walk.bvh --output data/motions_g1/

# Step 2 (alt): Generate motion from text using Kimodo
python -m kimodo_pipeline.kimodo_gen.run_kimodo --prompt "walk forward naturally" --skeleton g1 --output data/motions_g1/

# Step 3: Train policy with ProtoMotions (Isaac Lab)
python -m kimodo_pipeline.protomotions.train --motion-dir data/motions_g1/ --framework isaac_lab

# Step 4: Validate in MuJoCo (sim2sim)
python -m kimodo_pipeline.protomotions.validate_mujoco --policy data/policies/policy.onnx

# Step 5: Deploy to real G1 via GEAR-SONIC
python -m kimodo_pipeline.gear_sonic.deploy --policy data/policies/policy.onnx --robot-ip 192.168.123.161
```

## Prerequisites

```bash
# SOMA-X body model
pip install py-soma-x

# GEM-X (video to motion)
git clone https://github.com/NVlabs/GEM-X.git
cd GEM-X && pip install -e .

# Kimodo (text to motion)
git clone https://github.com/nv-tlabs/kimodo.git
cd kimodo && pip install -e .

# SOMA Retargeter
git clone https://github.com/NVIDIA/soma-retargeter.git
cd soma-retargeter && pip install -e .

# ProtoMotions (training)
git clone https://github.com/NVlabs/ProtoMotions.git
cd ProtoMotions && pip install -e .

# GEAR-SONIC / GR00T Whole-Body Control (deployment)
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git

# Isaac Lab 2.3.0 (for ProtoMotions training)
# See: https://isaac-sim.github.io/IsaacLab/
```

## Hardware Requirements

- NVIDIA GPU with 17GB+ VRAM (RTX 3090/4090/A100)
- CUDA 12.6+
- Unitree G1 robot (for deployment stage)
