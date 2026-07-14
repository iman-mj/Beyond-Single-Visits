# Beyond Single Visits: Learning Longitudinal Patient Trajectories for Radiology Report Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of the temporally-aware multimodal framework for automated radiology report generation. This model tracks patient disease trajectories over time using continuous time embeddings and a multi-token pathway compressor, effectively bridging the gap between cross-sectional vision-language models and real-world longitudinal clinical workflows.

## ✨ Key Features
- **Longitudinal Memory:** Uses a specialized Temporal Transformer and Trigonometric Time-Gap Encoders to model non-uniform patient visits.
- **Smart Vision Extraction:** Replaces standard pooling with a learnable **Attention Pooling** mechanism on top of Swin Transformers.
- **Multimodal Projector:** Efficiently compresses the patient's entire trajectory into "Soft Prompts" compatible with standard LLMs (BioMistral).
- **Ablation-Ready Architecture:** Built-in runtime flags for comprehensive component analysis and testing.
