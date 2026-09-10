from pathlib import Path
import os
import torch


AUDIO_ROOT = Path(os.getenv("CONVNEXT_AUDIO_ROOT"))
FEATURES_ROOT = Path(os.getenv("CONVNEXT_FEATURES_ROOT"))

CONVNEXT_EMB_MODE = os.getenv("CONVNEXT_EMB_MODE", "pooled")  # pooled | flatten
CONVNEXT_NORM = os.getenv("CONVNEXT_NORM", "none")  # none | layer | l2


def _apply_norm(x: torch.Tensor) -> torch.Tensor:
    if CONVNEXT_NORM == "none":
        return x
    if CONVNEXT_NORM == "l2":
        return torch.nn.functional.normalize(x, p=2, dim=-1)
    if CONVNEXT_NORM == "layer":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    raise ValueError(f"Unknown CONVNEXT_NORM={CONVNEXT_NORM}")


def _feature_path_for_audio(audio_path: str) -> Path:
    audio_path = Path(audio_path)
    rel = audio_path.relative_to(AUDIO_ROOT)
    feat_path = FEATURES_ROOT / rel
    return feat_path.with_suffix(feat_path.suffix + ".convnext.pt")


def _project_frame_emb(frame_emb: torch.Tensor) -> torch.Tensor:
    if frame_emb.ndim != 3:
        raise ValueError(f"Expected frame_emb to be [T,C,F], got {tuple(frame_emb.shape)}")

    if CONVNEXT_EMB_MODE == "pooled":
        seq = frame_emb.mean(dim=-1)
    elif CONVNEXT_EMB_MODE == "flatten":
        seq = frame_emb.reshape(frame_emb.shape[0], -1)
    else:
        raise ValueError(f"Unknown CONVNEXT_EMB_MODE={CONVNEXT_EMB_MODE}")

    return _apply_norm(seq.to(torch.float32))


def _load_convnext_for_audio(audio_path: str):
    feat_path = _feature_path_for_audio(audio_path)

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing ConvNeXt file: {feat_path}")

    payload = torch.load(feat_path, map_location="cpu")
    if payload.get("payload_type") != "frame_sequence_2d":
        raise ValueError(f"Expected frame_sequence_2d payload in {feat_path}")

    frame_emb = payload.get("frame_emb")
    frame_start_sec = payload.get("frame_start_sec")
    frame_end_sec = payload.get("frame_end_sec")
    frame_center_sec = payload.get("frame_center_sec")

    if not isinstance(frame_emb, torch.Tensor) or frame_emb.ndim != 3:
        raise ValueError(f"Expected frame_emb to be [T,C,F] in {feat_path}")
    if not isinstance(frame_start_sec, torch.Tensor) or frame_start_sec.ndim != 1:
        raise ValueError(f"Expected frame_start_sec to be [T] in {feat_path}")
    if not isinstance(frame_end_sec, torch.Tensor) or frame_end_sec.ndim != 1:
        raise ValueError(f"Expected frame_end_sec to be [T] in {feat_path}")
    if not isinstance(frame_center_sec, torch.Tensor) or frame_center_sec.ndim != 1:
        raise ValueError(f"Expected frame_center_sec to be [T] in {feat_path}")
    if frame_emb.shape[0] != frame_start_sec.shape[0] or frame_emb.shape[0] != frame_end_sec.shape[0]:
        raise ValueError(f"Frame metadata length mismatch in {feat_path}")
    if frame_emb.shape[0] != frame_center_sec.shape[0]:
        raise ValueError(f"Frame center metadata length mismatch in {feat_path}")

    seq = _project_frame_emb(frame_emb)
    return (
        seq,
        float(payload["duration_sec"]),
        frame_start_sec.to(torch.float32),
        frame_end_sec.to(torch.float32),
        frame_center_sec.to(torch.float32),
    )


def _dedupe_selected_tokens(
    seq: torch.Tensor,
    start_times_sec: torch.Tensor,
    end_times_sec: torch.Tensor,
    crop_start_sec: float,
    crop_end_sec: float,
):
    if seq.shape[0] <= 1:
        return seq

    crop_start = torch.tensor(crop_start_sec, dtype=torch.float32)
    crop_end = torch.tensor(crop_end_sec, dtype=torch.float32)
    clipped_start = torch.maximum(start_times_sec, crop_start)
    clipped_end = torch.minimum(end_times_sec, crop_end)

    selected = []
    covered_until = float(crop_start_sec)
    eps = 1.0e-6
    for ix in range(seq.shape[0]):
        token_start = float(clipped_start[ix].item())
        token_end = float(clipped_end[ix].item())
        if token_end <= token_start + eps:
            continue
        if token_end > covered_until + eps:
            selected.append(torch.tensor(ix, dtype=torch.long))
            covered_until = max(covered_until, token_end)

    if not selected:
        return seq[:1]
    selected_ix = torch.stack(selected).to(torch.long)
    return seq[selected_ix]


def _slice_by_crop(
    seq: torch.Tensor,
    start_times_sec: torch.Tensor,
    end_times_sec: torch.Tensor,
    centers_sec: torch.Tensor,
    dur_sec: float,
    t_start: float,
    t_end: float,
):
    if seq.shape[0] == 0 or dur_sec <= 1e-6:
        return seq
    if seq.shape[0] == 1:
        return seq

    start_sec = max(0.0, min(dur_sec, t_start * dur_sec))
    end_sec = max(0.0, min(dur_sec, t_end * dur_sec))

    if end_sec <= start_sec:
        center = 0.5 * (start_sec + end_sec)
        ix = int(torch.argmin((centers_sec - center).abs()).item())
        return seq[ix:ix + 1]

    overlap = torch.minimum(end_times_sec, torch.tensor(end_sec, dtype=torch.float32)) - torch.maximum(
        start_times_sec, torch.tensor(start_sec, dtype=torch.float32)
    )
    keep = overlap.clamp_min(0.0) > 0.0
    if keep.any():
        return _dedupe_selected_tokens(
            seq[keep],
            start_times_sec[keep],
            end_times_sec[keep],
            start_sec,
            end_sec,
        )

    center = 0.5 * (start_sec + end_sec)
    ix = int(torch.argmin((centers_sec - center).abs()).item())
    return seq[ix:ix + 1]


def get_custom_metadata(info, audio):
    audio_path = info["path"]
    t_start, t_end = info["timestamps"]

    try:
        seq, dur_sec, start_times_sec, end_times_sec, centers_sec = _load_convnext_for_audio(audio_path)
        seq_seg = _slice_by_crop(
            seq,
            start_times_sec,
            end_times_sec,
            centers_sec,
            dur_sec,
            float(t_start),
            float(t_end),
        )
        return {
            "convnext": seq_seg,
            "convnext_global": seq_seg,
            "prompt": info.get("relpath", ""),
        }
    except Exception as e:
        print(f"[convnext_md] REJECT audio={audio_path} err={repr(e)}")
        return {"__reject__": True}