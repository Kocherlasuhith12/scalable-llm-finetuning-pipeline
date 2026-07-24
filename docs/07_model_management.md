# 07. Model Management

## Registry & Versioning
- **Semantic Versioning**: Models are registered with version tags (`v1.0.0`, `v1.0.1`, etc.).
- **Lineage Tracking**: Each registered model references its parent `training_job`, `dataset_id`, `base_model`, and `hyperparameters`.
- **Adapter Merging**: Provides single-command adapter-to-base merging (`merge_adapters.py`), creating standalone 16-bit model weights.
- **Quantization**: Converts 16-bit floating point models into INT8 or INT4 formats for low-latency deployment.
