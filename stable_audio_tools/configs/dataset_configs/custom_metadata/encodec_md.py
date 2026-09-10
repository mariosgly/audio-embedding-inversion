from pathlib import Path
import os
import torch


AUDIO_ROOT = Path(os.getenv("ENCODEC_AUDIO_ROOT"))
FEATURES_ROOT = Path(os.getenv("ENCODEC_FEATURES_ROOT"))

ENCODEC_NORM = os.getenv("ENCODEC_NORM", "none")  # none | layer | l2
ENCODEC_SCALE_REDUCTION = os.getenv("ENCODEC_SCALE_REDUCTION", "mean")  # mean | max


def _apply_norm(x: torch.Tensor) -> torch.Tensor:
    if ENCODEC_NORM == "none":
        return x
    if ENCODEC_NORM == "l2":
        return torch.nn.functional.normalize(x, p=2, dim=-1)
    if ENCODEC_NORM == "layer":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    raise ValueError(f"Unknown ENCODEC_NORM={ENCODEC_NORM}")


def _feature_path_for_audio(audio_path: str) -> Path:
    audio_path = Path(audio_path)
    rel = audio_path.relative_to(AUDIO_ROOT)
    feat_path = FEATURES_ROOT / rel
    return feat_path.with_suffix(feat_path.suffix + ".encodec.pt")


def _load_encodec_for_audio(audio_path: str):
    feat_path = _feature_path_for_audio(audio_path)

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing EnCodec file: {feat_path}")

    payload = torch.load(feat_path, map_location="cpu")
    if payload.get("payload_type") != "zq_sequence":
        raise ValueError(f"Expected zq_sequence payload in {feat_path}")

    z_q = payload.get("z_q")
    centers_sec = payload.get("frame_center_sec")
    scales = payload.get("scale")

    if not isinstance(z_q, torch.Tensor) or z_q.ndim != 2:
        raise ValueError(f"Expected z_q to be [T,D] in {feat_path}")
    if not isinstance(centers_sec, torch.Tensor) or centers_sec.ndim != 1:
        raise ValueError(f"Expected frame_center_sec to be [T] in {feat_path}")
    if not isinstance(scales, torch.Tensor) or scales.ndim != 1:
        raise ValueError(f"Expected scale to be [T] in {feat_path}")
    if centers_sec.shape[0] != z_q.shape[0] or scales.shape[0] != z_q.shape[0]:
        raise ValueError(f"Length mismatch in {feat_path}")

    z_q = _apply_norm(z_q.to(torch.float32))
    centers_sec = centers_sec.to(torch.float32)
    scales = scales.to(torch.float32)
    duration_sec = float(payload["duration_sec"])

    return z_q, centers_sec, scales, duration_sec


def _slice_by_crop(z_q: torch.Tensor, centers_sec: torch.Tensor, scales: torch.Tensor, dur_sec: float, t_start: float, t_end: float):
    if z_q.shape[0] == 0 or dur_sec <= 1e-6:
        return z_q, scales[:0]
    if z_q.shape[0] == 1:
        return z_q, scales[:1]

    start_sec = max(0.0, min(dur_sec, t_start * dur_sec))
    end_sec = max(0.0, min(dur_sec, t_end * dur_sec))

    if end_sec <= start_sec:
        center = 0.5 * (start_sec + end_sec)
        ix = int(torch.argmin((centers_sec - center).abs()).item())
        return z_q[ix:ix + 1], scales[ix:ix + 1]

    keep = (centers_sec >= start_sec) & (centers_sec < end_sec)
    if keep.any():
        return z_q[keep], scales[keep]

    center = 0.5 * (start_sec + end_sec)
    ix = int(torch.argmin((centers_sec - center).abs()).item())
    return z_q[ix:ix + 1], scales[ix:ix + 1]


def _reduce_scale(scale_seg: torch.Tensor) -> float:
    if scale_seg.numel() == 0:
        return 1.0
    if ENCODEC_SCALE_REDUCTION == "mean":
        return float(scale_seg.mean().item())
    if ENCODEC_SCALE_REDUCTION == "max":
        return float(scale_seg.max().item())
    raise ValueError(f"Unknown ENCODEC_SCALE_REDUCTION={ENCODEC_SCALE_REDUCTION}")


def get_custom_metadata(info, audio):
    audio_path = info["path"]
    t_start, t_end = info["timestamps"]

    try:
        z_q, centers_sec, scales, dur_sec = _load_encodec_for_audio(audio_path)
        z_q_seg, scale_seg = _slice_by_crop(z_q, centers_sec, scales, dur_sec, float(t_start), float(t_end))
        return {
            "encodec": z_q_seg,
            "encodec_global": z_q_seg,
            "encodec_scale": _reduce_scale(scale_seg),
            "prompt": info.get("relpath", ""),
        }
    except Exception as e:
        print(f"[encodec_md] REJECT audio={audio_path} err={repr(e)}")
        return {"__reject__": True}
