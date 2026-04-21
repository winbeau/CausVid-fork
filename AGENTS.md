# Repository Guidelines

## Project Structure & Module Organization
`causvid/` contains the core training, data, and model code. Key areas are `causvid/models/wan/` for Wan-based causal and bidirectional pipelines, `causvid/models/sdxl/` for SDXL wrappers, `causvid/evaluation/` for evaluation scripts, and `causvid/ode_data/` for LMDB dataset utilities. Runtime configs live in `configs/*.yaml`. Lightweight entry points for checkpointed inference are in `minimal_inference/`. Dataset examples and prompt files are under `sample_dataset/`, while `distillation_data/` holds data download and preprocessing scripts. Tests are grouped by backend in `tests/wan/`, `tests/causal_wan/`, and `tests/sdxl/`.

## Build, Test, and Development Commands
Set up the environment with `uv venv --python 3.10`, `source .venv/bin/activate`, and `uv sync`. Run commands inside the environment directly or prefix them with `uv run`. Example inference: `uv run python minimal_inference/autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --checkpoint_folder <ckpt> --output_folder <out> --prompt_file_path <prompts>`. Long-video inference uses `uv run python minimal_inference/longvideo_autoregressive_inference.py ... --num_rollout <n>`. Training uses `uv run torchrun --nnodes 8 --nproc_per_node=8 causvid/train_distillation.py --config_path configs/wan_causal_dmd.yaml`. The legacy `pip install -r requirements.txt && python setup.py develop` flow remains available for compatibility.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions, variables, files, and YAML configs, and `PascalCase` for classes such as `InferencePipeline` or `CausalWanSelfAttention`. Keep imports grouped logically and prefer small, focused changes over broad refactors. Config names use descriptive model/task prefixes such as `wan_causal_dmd.yaml` or `wan_bidirectional_dmd_from_scratch.yaml`.

## Testing Guidelines
Current tests are mostly GPU-backed smoke or integration scripts rather than isolated unit tests. Keep new coverage close to the affected stack and name files `test_<feature>.py` inside the matching backend directory. Run targeted checks with `python tests/wan/test_text_encoder.py` or `python tests/causal_wan/test_causal_inference_pipeline.py`. If using `pytest`, prefer narrow selection, for example `pytest tests/sdxl -s`, and document any checkpoint or CUDA prerequisites in the PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `fix bidirectional inference typo` and `support various amount of overlapping during sliding window long video generation`. Keep commit titles concise, lowercase, and behavior-focused. PRs should describe the model path or config touched, required weights or datasets, exact validation commands, and sample outputs when generation behavior changes. Do not commit large checkpoints, generated videos, or local dataset artifacts.
