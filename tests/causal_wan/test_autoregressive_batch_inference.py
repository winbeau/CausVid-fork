import csv

import torch

from minimal_inference.autoregressive_batch_inference import (
    SEGMENT_LATENT_FRAMES,
    assemble_latent_segments,
    partition_prompt_indices,
    read_prompts,
    resolve_output_dir,
    write_prompts_csv,
)


def test_read_prompts_preserves_blank_lines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first line\n\nthird line\n", encoding="utf-8")

    prompts = read_prompts(str(prompt_file))

    assert prompts == ["first line", "", "third line"]


def test_write_prompts_csv_preserves_index_and_order(tmp_path):
    prompts = ["alpha", "", "gamma"]

    output_path = write_prompts_csv(prompts, str(tmp_path))

    with open(output_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ["index", "prompt"],
        ["0", "alpha"],
        ["1", ""],
        ["2", "gamma"],
    ]


def test_resolve_output_dir_uses_target_latent_frames_when_not_provided():
    assert resolve_output_dir(None, 123) == "CausVid-123latents"
    assert resolve_output_dir("custom-dir", 123) == "custom-dir"


def test_partition_prompt_indices_assigns_remainder_to_last_gpu():
    assignments = partition_prompt_indices(num_prompts=5, gpu_ids=[0, 1])

    assert assignments == [
        type(assignments[0])(gpu_id=0, start_index=0, end_index=2),
        type(assignments[1])(gpu_id=1, start_index=2, end_index=5),
    ]


def test_partition_prompt_indices_allows_empty_front_assignments():
    assignments = partition_prompt_indices(num_prompts=2, gpu_ids=[0, 1, 2, 3])

    assert assignments == [
        type(assignments[0])(gpu_id=0, start_index=0, end_index=0),
        type(assignments[1])(gpu_id=1, start_index=0, end_index=0),
        type(assignments[2])(gpu_id=2, start_index=0, end_index=0),
        type(assignments[3])(gpu_id=3, start_index=0, end_index=2),
    ]


def test_assemble_latent_segments_trims_to_exact_target():
    segments = [
        torch.arange(SEGMENT_LATENT_FRAMES, dtype=torch.float32).view(1, SEGMENT_LATENT_FRAMES, 1, 1, 1),
        torch.arange(100, 100 + SEGMENT_LATENT_FRAMES, dtype=torch.float32).view(1, SEGMENT_LATENT_FRAMES, 1, 1, 1),
        torch.arange(200, 200 + SEGMENT_LATENT_FRAMES, dtype=torch.float32).view(1, SEGMENT_LATENT_FRAMES, 1, 1, 1),
    ]

    output = assemble_latent_segments(
        segment_latents=segments,
        num_overlap_frames=3,
        target_latent_frames=40,
    )

    expected = torch.cat(
        [segments[0], segments[1][:, 3:], segments[2][:, 3:]],
        dim=1,
    )[:, :40]

    assert output.shape[1] == 40
    assert torch.equal(output, expected)
