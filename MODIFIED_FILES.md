# Modified and Added Files

This file records the training-related files included here that differ from
Stability AI's upstream `stable-audio-tools`.

## Upstream-Modified Files Kept

- `train.py`: adds demo dataset selection, W&B resume/id handling, filelist overrides, matmul precision handling, validation checkpoint monitoring, and embedding-inversion training conveniences.
- `defaults.ini`: adds defaults used by the local training entry point.
- `pyproject.toml` and `setup.py`: packaging metadata for this curated derivative release.
- `docs/autoencoders.md`, `docs/conditioning.md`, and `docs/datasets.md`: upstream docs with local edits relevant to training/configuration.
- `stable_audio_tools/data/dataset.py`
- `stable_audio_tools/data/utils.py`
- `stable_audio_tools/inference/generation.py`
- `stable_audio_tools/inference/sampling.py`
- `stable_audio_tools/interface/aeiou.py`
- `stable_audio_tools/models/adp.py`
- `stable_audio_tools/models/arc.py`
- `stable_audio_tools/models/autoencoders.py`
- `stable_audio_tools/models/blocks.py`
- `stable_audio_tools/models/bottleneck.py`
- `stable_audio_tools/models/conditioners.py`
- `stable_audio_tools/models/diffusion.py`
- `stable_audio_tools/models/discriminators.py`
- `stable_audio_tools/models/dit.py`
- `stable_audio_tools/models/factory.py`
- `stable_audio_tools/models/fsq.py`
- `stable_audio_tools/models/inpainting.py`
- `stable_audio_tools/models/lm_backbone.py`
- `stable_audio_tools/models/lora/model.py`
- `stable_audio_tools/models/lora/utils.py`
- `stable_audio_tools/models/pqmf.py`
- `stable_audio_tools/models/pretransforms.py`
- `stable_audio_tools/models/transformer.py`
- `stable_audio_tools/models/utils.py`
- `stable_audio_tools/training/arc.py`
- `stable_audio_tools/training/autoencoders.py`
- `stable_audio_tools/training/diffusion.py`
- `stable_audio_tools/training/factory.py`
- `stable_audio_tools/training/lm.py`
- `stable_audio_tools/training/losses/auraloss.py`
- `stable_audio_tools/training/losses/losses.py`
- `stable_audio_tools/training/losses/semantic.py`
- `stable_audio_tools/training/losses/utils.py`
- `stable_audio_tools/training/utils.py`

## Local Additions Kept

- `feature_extraction/`: feature extraction entry point and encoder backends for precomputing conditioning features.
- `stable_audio_tools/configs/model_configs/emb2audio/`: base embedding-inversion model configs for VGGish, CLAP, BEATs, EnCodec, and ConvNeXt.
- `stable_audio_tools/configs/dataset_configs/custom_metadata/`: metadata loaders that align precomputed embedding payloads with audio crops during training.
- `stable_audio_tools/configs/dataset_configs/examples/`: generic dataset configs with placeholder paths.
- `jobs/`: generic sample SLURM jobs without site-specific cluster paths.
- `ATTRIBUTION.md`, `MODIFIED_FILES.md`, and `docs/RELEASE_SCOPE.md`: release curation documentation.
- `feature_extraction/extract_features.py`: in this curated copy, the unused `audiomae` CLI option was removed because no AudioMAE backend is included.

## Excluded Even Though Present in the Working Repository

- `jamendo_demo/`: demo website and assets, not required for training.
- `scoring/` and `JOBS/scoring/`: evaluation/scoring pipelines, not required for training.
- `drafts/`, `analysis_outputs/`, `report_assets/`, and scalability report markdown files: paper/report artifacts.
- `scripts/`: one-off data preparation, scoring, pruning, reconstruction, and local workflow utilities.
- `JOBS/old/`, `JOBS/lr_ablations/`, `JOBS/init_attn_ablations/`, `JOBS/dataset_size_ablations/`, and `JOBS/full_songs_training/`: experiment-specific or ablation job dumps.
- `run_gradio.py`, `stable_audio_tools/interface/gradio.py`, and `stable_audio_tools/interface/interfaces/`: Gradio UI/inference demo surface, not training.
- Hidden files such as `.git`, `.gitignore`, `.DS_Store`, and `.openai/hosting.json`.
- Generated caches such as `__pycache__` and `*.pyc`.
- Separate branch-only packages, configs, tests, and jobs from branch `v2.0`.
