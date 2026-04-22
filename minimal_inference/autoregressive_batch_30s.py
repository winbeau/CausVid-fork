import argparse
import csv
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import List, Sequence

import torch


SEGMENT_LATENT_FRAMES = 21
LATENT_SHAPE = (16, 60, 104)


@dataclass(frozen=True)
class WorkerAssignment:
    gpu_id: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class WorkerConfig:
    assignment: WorkerAssignment
    prompts: Sequence[str]
    config_path: str
    checkpoint_folder: str
    output_dir: str
    target_latent_frames: int
    num_overlap_frames: int
    fps: int
    base_seed: int
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate long autoregressive videos with one Python worker "
            "process per GPU. The default 123 latent frames decode to "
            "approximately 489 RGB frames with the Wan VAE path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        default="configs/wan_causal_dmd.yaml",
        help="Inference config passed to the causal Wan InferencePipeline.",
    )
    parser.add_argument(
        "--checkpoint-folder",
        "--checkpoint_folder",
        dest="checkpoint_folder",
        default="autoregressive_checkpoint",
        help="Folder containing model.pt with the autoregressive generator weights.",
    )
    parser.add_argument(
        "--prompt-file",
        "--prompt_file",
        "--prompt_file_path",
        dest="prompt_file",
        default="prompts/MovieGenVideoBench_num128.txt",
        help="TXT file of prompts. Lines are preserved 1:1, including blank lines.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--output_folder",
        dest="output_dir",
        default="CausVid-30s",
        help="Directory for video_XXX.mp4 outputs and prompts.csv.",
    )
    parser.add_argument(
        "--target-latent-frames",
        "--target_latent_frames",
        dest="target_latent_frames",
        type=int,
        default=123,
        help=(
            "Number of latent frames to assemble before the final decode. "
            "123 latent frames decode to approximately 489 RGB frames."
        ),
    )
    parser.add_argument(
        "--num-overlap-frames",
        "--num_overlap_frames",
        dest="num_overlap_frames",
        type=int,
        default=3,
        help="Latent-frame overlap reused between 21-frame autoregressive segments.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Output MP4 frame rate.",
    )
    parser.add_argument(
        "--base-seed",
        "--base_seed",
        dest="base_seed",
        type=int,
        default=42,
        help="Per-prompt seed base. Prompt i uses base_seed + i.",
    )
    parser.add_argument(
        "--gpu-ids",
        "--gpu_ids",
        dest="gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help="Visible CUDA device ids to use. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing video files instead of skipping them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_latent_frames <= 0:
        raise ValueError("--target-latent-frames must be positive.")
    if args.num_overlap_frames <= 0:
        raise ValueError("--num-overlap-frames must be positive.")
    if args.num_overlap_frames >= SEGMENT_LATENT_FRAMES:
        raise ValueError(
            f"--num-overlap-frames must be smaller than {SEGMENT_LATENT_FRAMES}."
        )
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")


def read_prompts(prompt_file: str) -> List[str]:
    prompts: List[str] = []
    with open(prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            prompts.append(line.rstrip("\n"))
    return prompts


def write_prompts_csv(prompts: Sequence[str], output_dir: str) -> str:
    output_path = os.path.join(output_dir, "prompts.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "prompt"])
        for index, prompt in enumerate(prompts):
            writer.writerow([index, prompt])
    return output_path


def resolve_gpu_ids(gpu_ids: Sequence[int] | None) -> List[int]:
    if gpu_ids is not None:
        if not gpu_ids:
            raise ValueError("--gpu-ids must not be empty when provided.")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--gpu-ids must not contain duplicates.")
        return list(gpu_ids)

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError("No visible CUDA devices found.")
    return list(range(visible_gpu_count))


def partition_prompt_indices(num_prompts: int, gpu_ids: Sequence[int]) -> List[WorkerAssignment]:
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty.")

    common_chunk = num_prompts // len(gpu_ids)
    remainder = num_prompts % len(gpu_ids)

    assignments: List[WorkerAssignment] = []
    start_index = 0
    last_worker = len(gpu_ids) - 1
    for worker_index, gpu_id in enumerate(gpu_ids):
        chunk_size = common_chunk
        if worker_index == last_worker:
            chunk_size += remainder

        end_index = start_index + chunk_size
        assignments.append(
            WorkerAssignment(
                gpu_id=gpu_id,
                start_index=start_index,
                end_index=end_index,
            )
        )
        start_index = end_index

    return assignments


def assemble_latent_segments(
    segment_latents: Sequence[torch.Tensor],
    num_overlap_frames: int,
    target_latent_frames: int,
) -> torch.Tensor:
    if not segment_latents:
        raise ValueError("segment_latents must contain at least one segment.")

    assembled_segments = []
    for segment_index, latents in enumerate(segment_latents):
        if segment_index == 0:
            assembled_segments.append(latents)
        else:
            assembled_segments.append(latents[:, num_overlap_frames:])

    output = torch.cat(assembled_segments, dim=1)
    return output[:, :target_latent_frames]


def encode_video_frames(vae: torch.nn.Module, videos: torch.Tensor) -> torch.Tensor:
    device = videos.device
    dtype = videos.dtype
    scale = [
        vae.mean.to(device=device, dtype=dtype),
        1.0 / vae.std.to(device=device, dtype=dtype),
    ]
    output = [
        vae.model.encode(video.unsqueeze(0), scale).float().squeeze(0)
        for video in videos
    ]
    return torch.stack(output, dim=0)


def build_start_latents(
    pipeline: torch.nn.Module,
    video: torch.Tensor,
    latents: torch.Tensor,
    num_overlap_frames: int,
) -> torch.Tensor:
    if num_overlap_frames == 1:
        start_frame_video = video[:, -1:, :]
        overlap_suffix = latents[:, :0]
    else:
        start_frame_video = video[
            :,
            -4 * (num_overlap_frames - 1) - 1:-4 * (num_overlap_frames - 1),
            :,
        ]
        overlap_suffix = latents[:, -(num_overlap_frames - 1):]

    start_frame = encode_video_frames(
        pipeline.vae,
        (start_frame_video * 2.0 - 1.0).transpose(2, 1).to(torch.bfloat16),
    ).transpose(2, 1).to(torch.bfloat16)

    return torch.cat([start_frame, overlap_suffix], dim=1)


def generate_prompt_latents(
    pipeline: torch.nn.Module,
    prompt: str,
    target_latent_frames: int,
    num_overlap_frames: int,
    prompt_seed: int,
    device: torch.device,
) -> torch.Tensor:
    segment_latents: List[torch.Tensor] = []
    current_length = 0
    rollout_index = 0
    start_latents = None

    while current_length < target_latent_frames:
        rollout_seed = prompt_seed + rollout_index
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(rollout_seed)
            torch.cuda.manual_seed(rollout_seed)
            sampled_noise = torch.randn(
                (1, SEGMENT_LATENT_FRAMES, *LATENT_SHAPE),
                device=device,
                dtype=torch.bfloat16,
            )
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[prompt],
                return_latents=True,
                start_latents=start_latents,
            )

        segment_latents.append(latents)
        current_length += latents.shape[1] if rollout_index == 0 else latents.shape[1] - num_overlap_frames

        if current_length >= target_latent_frames:
            break

        start_latents = build_start_latents(
            pipeline=pipeline,
            video=video,
            latents=latents,
            num_overlap_frames=num_overlap_frames,
        )
        rollout_index += 1

    return assemble_latent_segments(
        segment_latents=segment_latents,
        num_overlap_frames=num_overlap_frames,
        target_latent_frames=target_latent_frames,
    )


def _worker_main(worker_config: WorkerConfig) -> None:
    torch.set_grad_enabled(False)
    assignment = worker_config.assignment

    if assignment.start_index == assignment.end_index:
        print(f"[gpu {assignment.gpu_id}] no prompts assigned")
        return

    import numpy as np
    from diffusers.utils import export_to_video
    from omegaconf import OmegaConf

    from causvid.models.wan.causal_inference import InferencePipeline

    device = torch.device(f"cuda:{assignment.gpu_id}")
    torch.cuda.set_device(device)

    config = OmegaConf.load(worker_config.config_path)
    pipeline = InferencePipeline(config, device=device)
    pipeline.to(device=device, dtype=torch.bfloat16)

    if worker_config.num_overlap_frames % pipeline.num_frame_per_block != 0:
        raise ValueError(
            "num_overlap_frames must be divisible by pipeline.num_frame_per_block"
        )

    checkpoint_path = os.path.join(worker_config.checkpoint_folder, "model.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["generator"]
    pipeline.generator.load_state_dict(state_dict, strict=True)

    print(
        f"[gpu {assignment.gpu_id}] prompt indices "
        f"{assignment.start_index}-{assignment.end_index - 1}"
    )

    for prompt_index in range(assignment.start_index, assignment.end_index):
        output_path = os.path.join(worker_config.output_dir, f"video_{prompt_index:03d}.mp4")
        if os.path.exists(output_path) and not worker_config.overwrite:
            print(f"[gpu {assignment.gpu_id}] skip existing {output_path}")
            continue

        prompt = worker_config.prompts[prompt_index]
        final_latents = generate_prompt_latents(
            pipeline=pipeline,
            prompt=prompt,
            target_latent_frames=worker_config.target_latent_frames,
            num_overlap_frames=worker_config.num_overlap_frames,
            prompt_seed=worker_config.base_seed + prompt_index,
            device=device,
        )

        decoded_video = pipeline.vae.decode_to_pixel(final_latents)
        decoded_video = (decoded_video * 0.5 + 0.5).clamp(0, 1)
        output_video = decoded_video[0].permute(0, 2, 3, 1).cpu().numpy()
        if not isinstance(output_video, np.ndarray):
            raise TypeError("Decoded output must convert to a NumPy array before export.")

        export_to_video(output_video, output_path, fps=worker_config.fps)


def main() -> None:
    args = parse_args()
    validate_args(args)

    prompts = read_prompts(args.prompt_file)
    os.makedirs(args.output_dir, exist_ok=True)
    write_prompts_csv(prompts, args.output_dir)

    gpu_ids = resolve_gpu_ids(args.gpu_ids)
    assignments = partition_prompt_indices(len(prompts), gpu_ids)

    worker_configs = [
        WorkerConfig(
            assignment=assignment,
            prompts=prompts,
            config_path=args.config_path,
            checkpoint_folder=args.checkpoint_folder,
            output_dir=args.output_dir,
            target_latent_frames=args.target_latent_frames,
            num_overlap_frames=args.num_overlap_frames,
            fps=args.fps,
            base_seed=args.base_seed,
            overwrite=args.overwrite,
        )
        for assignment in assignments
    ]

    ctx = mp.get_context("spawn")
    workers = []
    for worker_config in worker_configs:
        process = ctx.Process(target=_worker_main, args=(worker_config,))
        process.start()
        workers.append((worker_config.assignment.gpu_id, process))

    failed_gpu_ids = []
    for gpu_id, process in workers:
        process.join()
        if process.exitcode != 0:
            failed_gpu_ids.append((gpu_id, process.exitcode))

    if failed_gpu_ids:
        failures = ", ".join(
            f"gpu {gpu_id} (exit code {exitcode})"
            for gpu_id, exitcode in failed_gpu_ids
        )
        raise RuntimeError(f"One or more workers failed: {failures}")


if __name__ == "__main__":
    main()
