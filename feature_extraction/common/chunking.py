from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torchaudio import transforms as T


def compute_chunk_ranges(
    total_num_samples: int,
    sr: int,
    chunk_sec: float,
    hop_sec: float,
    tail_policy: str = "cover",
) -> List[Tuple[int, int, float, float]]:
    if total_num_samples <= 0:
        return []

    chunk_size = max(1, int(round(chunk_sec * sr)))
    hop_size = max(1, int(round(hop_sec * sr)))

    if total_num_samples <= chunk_size:
        return [(0, total_num_samples, 0.0, total_num_samples / sr)]

    if tail_policy == "pad":
        starts = list(range(0, total_num_samples, hop_size))
    else:
        starts = list(range(0, total_num_samples - chunk_size + 1, hop_size))

    if tail_policy == "cover":
        last_start = total_num_samples - chunk_size
        if starts[-1] != last_start:
            starts.append(last_start)

    ranges: List[Tuple[int, int, float, float]] = []
    for start in starts:
        end = min(start + chunk_size, total_num_samples)
        ranges.append((start, end, start / sr, end / sr))

    return ranges


def load_audio_mono(
    path: Path,
    *,
    target_sr: int,
    clamp: bool = True,
) -> torch.Tensor:
    audio, in_sr = torchaudio.load(str(path))
    if audio.numel() == 0:
        raise ValueError("empty audio tensor after load")
    if in_sr != target_sr:
        audio = T.Resample(in_sr, target_sr)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if clamp:
        audio = audio.clamp(-1.0, 1.0)
    return audio.squeeze(0).contiguous()