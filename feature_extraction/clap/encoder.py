from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T

import laion_clap


CLAP_SR = 48000


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2).clamp_min(eps)


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

    starts: List[int] = list(range(0, total_num_samples - chunk_size + 1, hop_size))

    if tail_policy == "cover":
        last_start = total_num_samples - chunk_size
        if starts[-1] != last_start:
            starts.append(last_start)

    ranges: List[Tuple[int, int, float, float]] = []
    for s in starts:
        e = min(s + chunk_size, total_num_samples)
        ranges.append((s, e, s / sr, e / sr))

    return ranges


def load_audio_mono_48k(path: Path) -> np.ndarray:
    audio, in_sr = torchaudio.load(str(path))
    if audio.numel() == 0:
        raise ValueError("empty audio tensor after load")
    if in_sr != CLAP_SR:
        audio = T.Resample(in_sr, CLAP_SR)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.clamp(-1.0, 1.0).contiguous()
    return audio.squeeze(0).cpu().numpy()


def embed_audio_batch(model: laion_clap.CLAP_Module, audio_batch_np: Sequence[np.ndarray]) -> torch.Tensor:
    with torch.no_grad():
        emb = model.get_audio_embedding_from_data(x=list(audio_batch_np), use_tensor=False)
    emb_t = torch.from_numpy(emb).float()
    if emb_t.ndim != 2:
        raise ValueError(f"expected CLAP output [B, D], got {tuple(emb_t.shape)}")
    if not torch.isfinite(emb_t).all():
        raise ValueError("non-finite values returned by CLAP")
    return emb_t


class CLAPExtractor:
    name = "clap"

    def __init__(
        self,
        *,
        clap_ckpt_path: Path,
        device: str,
        use_fusion: bool,
        mode: str = "global",
        chunk_sec: float = 5.0,
        chunk_hop_sec: float = 1.0,
        chunk_tail_policy: str = "cover",
        batch_size: int = 16,
    ) -> None:
        self.clap_ckpt_path = clap_ckpt_path
        self.device = device
        self.use_fusion = use_fusion
        self.mode = mode
        self.chunk_sec = chunk_sec
        self.chunk_hop_sec = chunk_hop_sec
        self.chunk_tail_policy = chunk_tail_policy
        self.batch_size = batch_size
        self.model = None

    def __enter__(self) -> "CLAPExtractor":
        ckpt_name = self.clap_ckpt_path.name
        use_large_ckpt = (
            "music_audioset" in ckpt_name
            or "music_speech" in ckpt_name
            or "music_speech_audioset" in ckpt_name
        )
        amodel = "HTSAT-base" if use_large_ckpt else "HTSAT-tiny"
        model = laion_clap.CLAP_Module(
            enable_fusion=self.use_fusion,
            amodel=amodel,
            device=self.device,
        )
        model.load_ckpt(str(self.clap_ckpt_path))
        model.eval()
        self.model = model
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.model = None

    def extract(self, audio_path: Path) -> Dict[str, torch.Tensor | int | bool | str]:
        if self.model is None:
            raise RuntimeError("CLAP extractor not initialized")
        audio_np = load_audio_mono_48k(audio_path)
        if self.mode == "global":
            emb_t = embed_audio_batch(self.model, [audio_np])
            if emb_t.shape[0] != 1:
                raise ValueError(f"expected CLAP output [1, D], got {tuple(emb_t.shape)}")
            emb_raw = emb_t[0].cpu()
            emb_l2 = l2_normalize(emb_raw)
            return {
                "payload_type": "global",
                "embedding_raw": emb_raw,
                "embedding_l2": emb_l2,
                "embedding_dim": int(emb_raw.numel()),
                "clap_target_sample_rate": int(CLAP_SR),
                "clap_fusion_enabled": bool(self.use_fusion),
                "clap_checkpoint": str(self.clap_ckpt_path),
            }

        if self.mode != "chunks":
            raise ValueError(f"Unsupported CLAP mode: {self.mode}")

        chunk_ranges = compute_chunk_ranges(
            total_num_samples=audio_np.shape[0],
            sr=CLAP_SR,
            chunk_sec=self.chunk_sec,
            hop_sec=self.chunk_hop_sec,
            tail_policy=self.chunk_tail_policy,
        )
        if not chunk_ranges:
            raise ValueError("no CLAP chunk ranges computed")

        raw_chunks: List[torch.Tensor] = []
        l2_chunks: List[torch.Tensor] = []
        starts_sec: List[float] = []
        ends_sec: List[float] = []

        for batch_start in range(0, len(chunk_ranges), self.batch_size):
            batch_ranges = chunk_ranges[batch_start:batch_start + self.batch_size]
            batch_audio = [audio_np[s:e] for s, e, _, _ in batch_ranges]
            emb_batch = embed_audio_batch(self.model, batch_audio)
            if emb_batch.shape[0] != len(batch_ranges):
                raise ValueError(f"CLAP batch size mismatch: got {emb_batch.shape[0]}, expected {len(batch_ranges)}")

            for (s, e, start_sec, end_sec), emb_raw in zip(batch_ranges, emb_batch):
                emb_raw = emb_raw.cpu()
                raw_chunks.append(emb_raw)
                l2_chunks.append(l2_normalize(emb_raw))
                starts_sec.append(float(start_sec))
                ends_sec.append(float(end_sec))

        embeddings_raw = torch.stack(raw_chunks, dim=0).contiguous()
        embeddings_l2 = torch.stack(l2_chunks, dim=0).contiguous()

        return {
            "payload_type": "chunk_sequence",
            "embedding_raw": embeddings_raw,
            "embedding_l2": embeddings_l2,
            "embedding_dim": int(embeddings_raw.shape[-1]),
            "num_chunks": int(embeddings_raw.shape[0]),
            "chunk_start_sec": torch.tensor(starts_sec, dtype=torch.float32),
            "chunk_end_sec": torch.tensor(ends_sec, dtype=torch.float32),
            "chunk_window_sec": float(self.chunk_sec),
            "chunk_hop_sec": float(self.chunk_hop_sec),
            "chunk_tail_policy": str(self.chunk_tail_policy),
            "clap_target_sample_rate": int(CLAP_SR),
            "clap_fusion_enabled": bool(self.use_fusion),
            "clap_checkpoint": str(self.clap_ckpt_path),
        }
