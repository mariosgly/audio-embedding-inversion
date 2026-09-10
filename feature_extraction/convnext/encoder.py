from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torchaudio import transforms as T
from audioset_convnext_inf.pytorch.convnext import convnext_tiny


CONVNEXT_SR = 32000
CONVNEXT_EMBED_DIM = 768
CONVNEXT_FREQ_BINS = 7

def compute_chunk_ranges(
    total_num_samples: int,
    sr: int,
    chunk_sec: float,
    hop_sec: float,
    tail_policy: str = "cover",
) -> List[Tuple[int, int]]:
    if total_num_samples <= 0:
        return []

    chunk_size = max(1, int(round(chunk_sec * sr)))
    hop_size = max(1, int(round(hop_sec * sr)))

    if total_num_samples <= chunk_size:
        return [(0, total_num_samples)]

    starts = list(range(0, total_num_samples - chunk_size + 1, hop_size))

    last_full_start = total_num_samples - chunk_size
    if tail_policy == "cover":
        if starts[-1] != last_full_start:
            starts.append(last_full_start)
    elif tail_policy == "pad":
        remainder_start = len(starts) * hop_size
        if remainder_start < total_num_samples:
            starts.append(remainder_start)
    elif tail_policy != "drop":
        raise ValueError(f"Unsupported ConvNeXt tail policy: {tail_policy}")

    ranges: List[Tuple[int, int]] = []
    for start in starts:
        end = min(start + chunk_size, total_num_samples)
        ranges.append((start, end))
    return ranges


def load_audio_mono_32k(path: Path) -> torch.Tensor:
    audio, in_sr = torchaudio.load(str(path))
    if audio.numel() == 0:
        raise ValueError("empty audio tensor after load")
    if in_sr != CONVNEXT_SR:
        audio = T.Resample(in_sr, CONVNEXT_SR)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.clamp(-1.0, 1.0).contiguous()
    return audio.squeeze(0).cpu()


def load_checkpoint(model, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_model as st_load_model

        st_load_model(model, str(checkpoint_path))
        return

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


class ConvNeXtExtractor:
    name = "convnext"

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str,
        chunk_sec: float = 10.0,
        chunk_hop_sec: float = 10.0,
        tail_policy: str = "cover",
        batch_size: int = 8,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.chunk_sec = chunk_sec
        self.chunk_hop_sec = chunk_hop_sec
        self.tail_policy = tail_policy
        self.batch_size = batch_size
        self.model = None

    def __enter__(self) -> "ConvNeXtExtractor":
        model = convnext_tiny(
            pretrained=False,
            strict=False,
            drop_path_rate=0.0,
            after_stem_dim=[252, 56],
            use_speed_perturb=False,
        )
        load_checkpoint(model, self.checkpoint_path)
        model.to(self.device).eval().requires_grad_(False)
        self.model = model
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.model = None

    def extract(self, audio_path: Path) -> Dict[str, torch.Tensor | int | float | str]:
        if self.model is None:
            raise RuntimeError("ConvNeXt extractor not initialized")

        audio = load_audio_mono_32k(audio_path)
        total_num_samples = int(audio.shape[0])
        chunk_size = max(1, int(round(self.chunk_sec * CONVNEXT_SR)))

        chunk_ranges = compute_chunk_ranges(
            total_num_samples=total_num_samples,
            sr=CONVNEXT_SR,
            chunk_sec=self.chunk_sec,
            hop_sec=self.chunk_hop_sec,
            tail_policy=self.tail_policy,
        )
        if not chunk_ranges:
            raise ValueError("no ConvNeXt chunk ranges computed")

        frame_emb_chunks: List[torch.Tensor] = []
        frame_start_chunks: List[torch.Tensor] = []
        frame_end_chunks: List[torch.Tensor] = []
        frame_center_chunks: List[torch.Tensor] = []

        for batch_start in range(0, len(chunk_ranges), self.batch_size):
            batch_ranges = chunk_ranges[batch_start:batch_start + self.batch_size]

            batch_audio: List[torch.Tensor] = []
            actual_lengths_sec: List[float] = []
            chunk_start_sec: List[float] = []

            for start_ix, end_ix in batch_ranges:
                clip = audio[start_ix:end_ix]
                actual_lengths_sec.append(float(clip.shape[0]) / float(CONVNEXT_SR))
                chunk_start_sec.append(float(start_ix) / float(CONVNEXT_SR))
                if clip.shape[0] < chunk_size:
                    padded = torch.zeros(chunk_size, dtype=clip.dtype)
                    padded[:clip.shape[0]] = clip
                    clip = padded
                batch_audio.append(clip)

            batch_waveform = torch.stack(batch_audio, dim=0).to(self.device, non_blocking=True)

            with torch.no_grad():
                frame_map = self.model.forward_frame_embeddings(batch_waveform)

            if frame_map.ndim != 4:
                raise ValueError(f"expected ConvNeXt frame embeddings [B,C,T,F], got {tuple(frame_map.shape)}")
            if frame_map.shape[1] != CONVNEXT_EMBED_DIM or frame_map.shape[3] != CONVNEXT_FREQ_BINS:
                raise ValueError(f"unexpected ConvNeXt frame embedding shape: {tuple(frame_map.shape)}")

            frame_map = frame_map.detach().to(torch.float32).cpu()

            for chunk_ix, raw_map in enumerate(frame_map):
                # [C, T, F] -> [T, C, F]
                raw_tokens = raw_map.permute(1, 0, 2).contiguous()
                num_tokens = raw_tokens.shape[0]

                token_edges = torch.linspace(0.0, float(self.chunk_sec), num_tokens + 1, dtype=torch.float32)
                token_start = token_edges[:-1] + chunk_start_sec[chunk_ix]
                token_end = token_edges[1:] + chunk_start_sec[chunk_ix]
                token_center = 0.5 * (token_start + token_end)

                valid_until = chunk_start_sec[chunk_ix] + actual_lengths_sec[chunk_ix]
                keep = token_center < (valid_until + 1.0e-6)
                if not keep.any():
                    keep = torch.zeros(num_tokens, dtype=torch.bool)
                    keep[0] = True

                frame_emb_chunks.append(raw_tokens[keep].cpu())
                frame_start_chunks.append(token_start[keep].cpu())
                frame_end_chunks.append(token_end[keep].cpu())
                frame_center_chunks.append(token_center[keep].cpu())

        frame_emb = torch.cat(frame_emb_chunks, dim=0)
        frame_start_sec = torch.cat(frame_start_chunks, dim=0)
        frame_end_sec = torch.cat(frame_end_chunks, dim=0)
        frame_center_sec = torch.cat(frame_center_chunks, dim=0)

        duration_sec = float(total_num_samples) / float(CONVNEXT_SR)
        frame_start_sec = frame_start_sec.clamp_(0.0, duration_sec)
        frame_end_sec = frame_end_sec.clamp_(0.0, duration_sec)
        frame_center_sec = frame_center_sec.clamp_(0.0, duration_sec)

        return {
            "payload_type": "frame_sequence_2d",
            "frame_emb": frame_emb,
            "frame_start_sec": frame_start_sec,
            "frame_end_sec": frame_end_sec,
            "frame_center_sec": frame_center_sec,
            "embedding_dim": int(CONVNEXT_EMBED_DIM),
            "freq_bins": int(CONVNEXT_FREQ_BINS),
            "num_frames": int(frame_emb.shape[0]),
            "convnext_target_sample_rate": int(CONVNEXT_SR),
            "convnext_chunk_sec": float(self.chunk_sec),
            "convnext_chunk_hop_sec": float(self.chunk_hop_sec),
            "convnext_tail_policy": str(self.tail_policy),
            "convnext_checkpoint": str(self.checkpoint_path),
        }