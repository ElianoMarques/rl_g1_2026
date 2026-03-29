"""Kimodo Pipeline: Video/Text to Real G1 Robot.

End-to-end pipeline using NVIDIA's Kimodo ecosystem:
  GEM-X (video2motion) -> SOMA Retargeter -> ProtoMotions (RL) -> GEAR-SONIC (deploy)
  Kimodo (text2motion) -> ProtoMotions (RL) -> GEAR-SONIC (deploy)
"""

__version__ = "1.0.0"
