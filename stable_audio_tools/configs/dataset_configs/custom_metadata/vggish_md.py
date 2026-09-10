from pathlib import Path
import os
import torch
import math

AUDIO_ROOT = Path(os.getenv("VGGISH_AUDIO_ROOT"))
FEATURES_ROOT = Path(os.getenv("VGGISH_FEATURES_ROOT"))
VGGISH_HOP_SEC = 0.96

VGGISH_EMB_KEY = os.getenv("VGGISH_EMB_KEY", "emb")
VGGISH_UINT8_SCALE = os.getenv("VGGISH_UINT8_SCALE", "-1_1")  # only for emb_postproc
VGGISH_NORM = os.getenv("VGGISH_NORM", "none")  # none | layer | l2


def _apply_norm(x: torch.Tensor) -> torch.Tensor:
    if VGGISH_NORM == "none":
        return x
    elif VGGISH_NORM == "l2":
        return torch.nn.functional.normalize(x, p=2, dim=-1)
    elif VGGISH_NORM == "layer":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    else:
        raise ValueError(f"Unknown VGGISH_NORM={VGGISH_NORM}")


def _load_vggish_for_audio(audio_path: str):
    audio_path = Path(audio_path)
    rel = audio_path.relative_to(AUDIO_ROOT)

    feat_path = (FEATURES_ROOT / rel)
    feat_path = feat_path.with_suffix(feat_path.suffix + ".vggish.pt")
    
    # print(audio_path)
    # print(feat_path)

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing vggish file: {feat_path}")

    payload = torch.load(feat_path, map_location="cpu")

    if VGGISH_EMB_KEY not in payload:
        raise KeyError(f"Missing key {VGGISH_EMB_KEY} in {feat_path}")

    emb = payload[VGGISH_EMB_KEY]  # [T,128]

    if VGGISH_EMB_KEY == "emb_postproc":
        emb = emb.to(torch.float32)
        if VGGISH_UINT8_SCALE == "none":
            pass
        elif VGGISH_UINT8_SCALE == "0_1":
            emb = emb / 255.0
        elif VGGISH_UINT8_SCALE == "-1_1":
            emb = (emb / 255.0) * 2.0 - 1.0
        else:
            raise ValueError(f"Unknown VGGISH_UINT8_SCALE={VGGISH_UINT8_SCALE}")
    else:
        # optional standardize dtype for raw emb
        if emb.dtype != torch.float32:
            emb = emb.to(torch.float32)

    emb = _apply_norm(emb)
    dur = float(payload["duration_sec"])
    return emb, dur


def _slice_by_crop(emb: torch.Tensor, dur_sec: float, t_start: float, t_end: float):
    T = emb.shape[0]
    if T == 0 or dur_sec <= 1e-6:
        return emb
    if T == 1:
        return emb

    start_sec = max(0.0, min(dur_sec, t_start * dur_sec))
    end_sec   = max(0.0, min(dur_sec, t_end * dur_sec))

    if end_sec <= start_sec:
        return emb[:1]

    start_ix = int(math.floor(start_sec / VGGISH_HOP_SEC))
    end_ix   = int(math.ceil(end_sec / VGGISH_HOP_SEC))

    start_ix = max(0, min(T - 1, start_ix))
    end_ix   = max(start_ix + 1, min(T, end_ix))

    return emb[start_ix:end_ix]


def get_custom_metadata(info, audio):
    audio_path = info["path"]
    t_start, t_end = info["timestamps"]
    # print(f"[vggish_md] LOADED CUSTOM METADATA MODULE v3 PID={os.getpid()}")

    try:
        emb, dur_sec = _load_vggish_for_audio(audio_path)
        emb_seg = _slice_by_crop(emb, dur_sec, float(t_start), float(t_end))
        return {
            "vggish": emb_seg,
            "vggish_global": emb_seg,
            "prompt": info.get("relpath", "")
        }
    except Exception as e:
        print(f"[vggish_md] REJECT audio={audio_path} err={repr(e)}")
        return {"__reject__": True}