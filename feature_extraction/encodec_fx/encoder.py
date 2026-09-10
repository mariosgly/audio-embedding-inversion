from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio


try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
except ImportError as import_err:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    ENCODEC_REPO_ROOT = REPO_ROOT / "encodec"
    if ENCODEC_REPO_ROOT.is_dir() and str(ENCODEC_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(ENCODEC_REPO_ROOT))
    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except ImportError as fallback_err:
        raise ImportError(
            "Could not import EnCodec. Install the 'encodec' Python package in the active environment "
            "or place the EnCodec source repo at '<repo>/encodec'."
        ) from fallback_err


def _model_factory(model_name: str):
    if model_name == "encodec_24khz":
        return EncodecModel.encodec_model_24khz
    if model_name == "encodec_48khz":
        return EncodecModel.encodec_model_48khz
    raise ValueError(f"Unsupported EnCodec model: {model_name}")


class EncodecExtractor:
    name = "encodec"

    def __init__(
        self,
        *,
        model_name: str,
        bandwidth: float,
        device: str,
        checkpoint_repository: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.bandwidth = bandwidth
        self.device = device
        self.checkpoint_repository = Path(checkpoint_repository).resolve() if checkpoint_repository else None
        self.model = None

    def __enter__(self) -> "EncodecExtractor":
        factory = _model_factory(self.model_name)
        self.model = factory(pretrained=True, repository=self.checkpoint_repository)
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.to(self.device).eval().requires_grad_(False)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.model = None

    def extract(self, audio_path: Path) -> Dict[str, torch.Tensor | int | float | str]:
        if self.model is None:
            raise RuntimeError("Encodec extractor not initialized")

        audio, in_sr = torchaudio.load(str(audio_path))
        if audio.numel() == 0:
            raise ValueError("empty audio tensor after load")

        wav = convert_audio(audio, in_sr, self.model.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0).to(self.device, non_blocking=True)

        with torch.no_grad():
            encoded_frames = self.model.encode(wav)

        if not encoded_frames:
            raise ValueError("EnCodec returned no encoded frames")

        if self.model.segment_stride is None:
            stride = wav.shape[-1]
        else:
            stride = int(self.model.segment_stride)

        chunk_zq: List[torch.Tensor] = []
        chunk_centers: List[torch.Tensor] = []
        chunk_scales: List[torch.Tensor] = []
        chunk_frame_lengths: List[int] = []
        chunk_offsets_samples: List[int] = []

        samples_per_frame = float(self.model.sample_rate) / float(self.model.frame_rate)
        offset = 0
        num_quantizers = None

        for codes, scale in encoded_frames:
            if codes.ndim != 3:
                raise ValueError(f"expected EnCodec codes [B,K,T], got {tuple(codes.shape)}")

            codes_for_quantizer = codes.transpose(0, 1).contiguous()
            z_q = self.model.quantizer.decode(codes_for_quantizer)
            if z_q.ndim != 3 or z_q.shape[0] != 1:
                raise ValueError(f"expected EnCodec z_q [1,D,T], got {tuple(z_q.shape)}")

            z_q = z_q[0].transpose(0, 1).contiguous().to(torch.float32).cpu()
            num_frames = z_q.shape[0]

            frame_positions = offset + (torch.arange(num_frames, dtype=torch.float32) + 0.5) * samples_per_frame
            frame_centers = frame_positions / float(self.model.sample_rate)

            if scale is None:
                scale_seq = torch.ones(num_frames, dtype=torch.float32)
            else:
                scale_value = float(scale.view(-1)[0].detach().cpu().item())
                scale_seq = torch.full((num_frames,), scale_value, dtype=torch.float32)

            chunk_zq.append(z_q)
            chunk_centers.append(frame_centers)
            chunk_scales.append(scale_seq)
            chunk_frame_lengths.append(int(num_frames))
            chunk_offsets_samples.append(int(offset))

            num_quantizers = int(codes.shape[1])
            offset += stride

        z_q_full = torch.cat(chunk_zq, dim=0)
        frame_center_sec = torch.cat(chunk_centers, dim=0)
        scale_full = torch.cat(chunk_scales, dim=0)

        duration_sec = float(audio.shape[-1]) / float(in_sr)
        frame_center_sec = frame_center_sec.clamp_(0.0, duration_sec)

        return {
            "payload_type": "zq_sequence",
            "z_q": z_q_full,
            "frame_center_sec": frame_center_sec,
            "scale": scale_full,
            "encodec_chunk_frame_lengths": torch.tensor(chunk_frame_lengths, dtype=torch.int64),
            "encodec_chunk_offsets_samples": torch.tensor(chunk_offsets_samples, dtype=torch.int64),
            "embedding_dim": int(z_q_full.shape[-1]),
            "num_frames": int(z_q_full.shape[0]),
            "frame_rate": int(self.model.frame_rate),
            "encodec_bandwidth": float(self.bandwidth),
            "encodec_model_name": str(self.model_name),
            "encodec_target_sample_rate": int(self.model.sample_rate),
            "encodec_channels": int(self.model.channels),
            "encodec_num_quantizers": int(num_quantizers or 0),
            "encodec_segment_length": int(self.model.segment_length or 0),
            "encodec_segment_stride": int(self.model.segment_stride or 0),
        }
