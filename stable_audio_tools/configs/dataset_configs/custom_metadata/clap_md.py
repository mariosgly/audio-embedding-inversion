from pathlib import Path
import os
import math
import torch

AUDIO_ROOT = Path(os.getenv("CLAP_AUDIO_ROOT"))
FEATURES_ROOT = Path(os.getenv("CLAP_FEATURES_ROOT"))

CLAP_EMB_KEY = os.getenv("CLAP_EMB_KEY", "embedding_raw")  # embedding_raw | embedding_l2
CLAP_NORM = os.getenv("CLAP_NORM", "none")  # none | layer | l2
CLAP_EMB_MODE = os.getenv("CLAP_EMB_MODE", "overlap_all")  # overlap_all | max_overlap | exact_match


def _apply_norm(x: torch.Tensor) -> torch.Tensor:
    if CLAP_NORM == "none":
        return x
    if CLAP_NORM == "l2":
        return torch.nn.functional.normalize(x, p=2, dim=-1)
    if CLAP_NORM == "layer":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    raise ValueError(f"Unknown CLAP_NORM={CLAP_NORM}")


def _feature_path_for_audio(audio_path: str) -> Path:
    audio_path = Path(audio_path)
    rel = audio_path.relative_to(AUDIO_ROOT)
    feat_path = FEATURES_ROOT / rel
    return feat_path.with_suffix(feat_path.suffix + ".clap_chunks.pt")


def _load_clap_for_audio(audio_path: str):
    feat_path = _feature_path_for_audio(audio_path)

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing CLAP file: {feat_path}")

    payload = torch.load(feat_path, map_location="cpu")

    if payload.get("payload_type") != "chunk_sequence":
        raise ValueError(f"Expected chunk_sequence payload in {feat_path}")

    if CLAP_EMB_KEY not in payload:
        raise KeyError(f"Missing key {CLAP_EMB_KEY} in {feat_path}")

    emb = payload[CLAP_EMB_KEY]
    if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
        raise ValueError(f"Expected {CLAP_EMB_KEY} to be [T,D] in {feat_path}")

    emb = emb.to(torch.float32)
    emb = _apply_norm(emb)

    duration_sec = float(payload["duration_sec"])
    chunk_start_sec = payload.get("chunk_start_sec")
    chunk_end_sec = payload.get("chunk_end_sec")

    if not isinstance(chunk_start_sec, torch.Tensor) or not isinstance(chunk_end_sec, torch.Tensor):
        raise ValueError(f"Missing chunk start/end tensors in {feat_path}")
    if chunk_start_sec.ndim != 1 or chunk_end_sec.ndim != 1:
        raise ValueError(f"Expected chunk start/end to be 1D tensors in {feat_path}")
    if chunk_start_sec.shape[0] != emb.shape[0] or chunk_end_sec.shape[0] != emb.shape[0]:
        raise ValueError(f"Chunk metadata length mismatch in {feat_path}")

    chunk_start_sec = chunk_start_sec.to(torch.float32)
    chunk_end_sec = chunk_end_sec.to(torch.float32)
    chunk_center_sec = 0.5 * (chunk_start_sec + chunk_end_sec)

    return emb, duration_sec, chunk_start_sec, chunk_end_sec, chunk_center_sec


def _slice_by_crop(
    emb: torch.Tensor,
    start_times_sec: torch.Tensor,
    end_times_sec: torch.Tensor,
    centers_sec: torch.Tensor,
    dur_sec: float,
    t_start: float,
    t_end: float,
):
    if emb.shape[0] == 0 or dur_sec <= 1e-6:
        return emb
    if emb.shape[0] == 1:
        return emb

    start_sec = max(0.0, min(dur_sec, t_start * dur_sec))
    end_sec = max(0.0, min(dur_sec, t_end * dur_sec))

    if end_sec <= start_sec:
        center = 0.5 * (start_sec + end_sec)
        ix = int(torch.argmin((centers_sec - center).abs()).item())
        return emb[ix:ix + 1]

    # Overlap length between each CLAP chunk [chunk_start, chunk_end)
    # and the crop [start_sec, end_sec).
    overlap = torch.minimum(end_times_sec, torch.tensor(end_sec, dtype=torch.float32)) - torch.maximum(
        start_times_sec, torch.tensor(start_sec, dtype=torch.float32)
    )
    overlap = overlap.clamp_min(0.0)
    crop_duration = end_sec - start_sec
    eps = 1.0e-4

    if CLAP_EMB_MODE == "overlap_all":
        keep = overlap > 0.0
        if keep.any():
            return emb[keep]
    elif CLAP_EMB_MODE == "max_overlap":
        if (overlap > 0.0).any():
            ix = int(torch.argmax(overlap).item())
            return emb[ix:ix + 1]
    elif CLAP_EMB_MODE == "exact_match":
        exact = overlap >= max(0.0, crop_duration - eps)
        if exact.any():
            exact_ixs = torch.nonzero(exact, as_tuple=False).flatten()
            if exact_ixs.numel() == 1:
                ix = int(exact_ixs.item())
            else:
                # If multiple chunks satisfy the overlap threshold due to numerical edge
                # cases, choose the one whose boundaries best match the crop boundaries.
                start_err = (start_times_sec[exact_ixs] - start_sec).abs()
                end_err = (end_times_sec[exact_ixs] - end_sec).abs()
                ix = int(exact_ixs[torch.argmin(start_err + end_err)].item())
            return emb[ix:ix + 1]
        if (overlap > 0.0).any():
            ix = int(torch.argmax(overlap).item())
            return emb[ix:ix + 1]
    else:
        raise ValueError(f"Unknown CLAP_EMB_MODE={CLAP_EMB_MODE}")

    center = 0.5 * (start_sec + end_sec)
    ix = int(torch.argmin((centers_sec - center).abs()).item())
    return emb[ix:ix + 1]


def get_custom_metadata(info, audio):
    audio_path = info["path"]
    t_start, t_end = info["timestamps"]

    try:
        emb, dur_sec, start_times_sec, end_times_sec, centers_sec = _load_clap_for_audio(audio_path)
        emb_seg = _slice_by_crop(
            emb,
            start_times_sec,
            end_times_sec,
            centers_sec,
            dur_sec,
            float(t_start),
            float(t_end),
        )
        return {
            "clap": emb_seg,
            "clap_global": emb_seg,
            "prompt": info.get("relpath", ""),
        }
    except Exception as e:
        print(f"[clap_md] REJECT audio={audio_path} err={repr(e)}")
        return {"__reject__": True}