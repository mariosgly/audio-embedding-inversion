#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Set

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.common.scanning import (
    audio_info,
    iter_audio_paths,
    load_filelist_paths,
    resolve_scan_mode,
)

ENCODER_NAMES = ("vggish", "clap", "beats", "encodec", "convnext")


def feature_suffix(args) -> str:
    if args.encoder == "clap" and args.clap_mode == "chunks":
        return ".clap_chunks.pt"
    return f".{args.encoder}.pt"


def encoder_run_name(args) -> str:
    if args.encoder == "clap" and args.clap_mode == "chunks":
        return "clap_chunks"
    return args.encoder


def feature_out_path(audio_path: Path, audio_root: Path, features_root: Path, args) -> Path:
    rel = audio_path.relative_to(audio_root)
    base = features_root / rel
    return base.with_suffix(base.suffix + feature_suffix(args))


def output_is_valid(path: Path, encoder_name: str, args) -> bool:
    if not path.exists():
        return False
    try:
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            return False
        required = {"duration_sec", "sample_rate", "wav_relpath", "wav_abspath"}
        if not required.issubset(obj.keys()):
            return False
        if encoder_name == "vggish":
            return "emb" in obj and "emb_postproc" in obj
        if encoder_name == "clap":
            if args.clap_mode == "global":
                return (
                    obj.get("payload_type") in (None, "global")
                    and isinstance(obj.get("embedding_raw"), torch.Tensor)
                    and isinstance(obj.get("embedding_l2"), torch.Tensor)
                    and obj["embedding_raw"].ndim == 1
                    and obj["embedding_l2"].ndim == 1
                )
            return (
                obj.get("payload_type") == "chunk_sequence"
                and isinstance(obj.get("embedding_raw"), torch.Tensor)
                and isinstance(obj.get("embedding_l2"), torch.Tensor)
                and obj["embedding_raw"].ndim == 2
                and obj["embedding_l2"].ndim == 2
                and int(obj.get("num_chunks", -1)) == int(obj["embedding_raw"].shape[0])
                and obj["embedding_raw"].shape == obj["embedding_l2"].shape
            )
        if encoder_name == "beats":
            payload_type = obj.get("payload_type")
            if payload_type == "patch_grid":
                return (
                    isinstance(obj.get("embedding_grid"), torch.Tensor)
                    and obj["embedding_grid"].ndim == 3
                    and int(obj.get("num_time_patches", -1)) == int(obj["embedding_grid"].shape[0])
                    and int(obj.get("num_freq_patches", -1)) == int(obj["embedding_grid"].shape[1])
                    and int(obj.get("embedding_dim", -1)) == int(obj["embedding_grid"].shape[2])
                )
            if payload_type == "chunked_patch_grid_sequence":
                return (
                    isinstance(obj.get("embedding_grid"), torch.Tensor)
                    and obj["embedding_grid"].ndim == 4
                    and int(obj.get("num_chunks", -1)) == int(obj["embedding_grid"].shape[0])
                    and int(obj.get("num_time_patches", -1)) == int(obj["embedding_grid"].shape[1])
                    and int(obj.get("num_freq_patches", -1)) == int(obj["embedding_grid"].shape[2])
                    and int(obj.get("embedding_dim", -1)) == int(obj["embedding_grid"].shape[3])
                    and isinstance(obj.get("chunk_start_sec"), torch.Tensor)
                    and isinstance(obj.get("chunk_end_sec"), torch.Tensor)
                    and obj["chunk_start_sec"].ndim == 1
                    and obj["chunk_end_sec"].ndim == 1
                    and obj["chunk_start_sec"].shape[0] == obj["embedding_grid"].shape[0]
                    and obj["chunk_end_sec"].shape[0] == obj["embedding_grid"].shape[0]
                )
            return False
        if encoder_name == "encodec":
            return (
                obj.get("payload_type") == "zq_sequence"
                and isinstance(obj.get("z_q"), torch.Tensor)
                and obj["z_q"].ndim == 2
                and isinstance(obj.get("frame_center_sec"), torch.Tensor)
                and obj["frame_center_sec"].ndim == 1
                and obj["frame_center_sec"].shape[0] == obj["z_q"].shape[0]
                and isinstance(obj.get("scale"), torch.Tensor)
                and obj["scale"].ndim == 1
                and obj["scale"].shape[0] == obj["z_q"].shape[0]
                and (
                    "encodec_chunk_frame_lengths" not in obj
                    or (
                        isinstance(obj["encodec_chunk_frame_lengths"], torch.Tensor)
                        and obj["encodec_chunk_frame_lengths"].ndim == 1
                        and int(obj["encodec_chunk_frame_lengths"].sum().item()) == int(obj["z_q"].shape[0])
                    )
                )
            )
        if encoder_name == "convnext":
            return (
                obj.get("payload_type") == "frame_sequence_2d"
                and isinstance(obj.get("frame_emb"), torch.Tensor)
                and obj["frame_emb"].ndim == 3
                and isinstance(obj.get("frame_start_sec"), torch.Tensor)
                and obj["frame_start_sec"].ndim == 1
                and isinstance(obj.get("frame_end_sec"), torch.Tensor)
                and obj["frame_end_sec"].ndim == 1
                and isinstance(obj.get("frame_center_sec"), torch.Tensor)
                and obj["frame_center_sec"].ndim == 1
                and obj["frame_emb"].shape[0] == obj["frame_start_sec"].shape[0]
                and obj["frame_emb"].shape[0] == obj["frame_end_sec"].shape[0]
                and obj["frame_emb"].shape[0] == obj["frame_center_sec"].shape[0]
            )
        return False
    except Exception:
        return False


def atomic_torch_save(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


def build_encoder(args):
    if args.encoder == "vggish":
        from feature_extraction.vggish.encoder import VGGishExtractor

        checkpoint_path = Path(args.vggish_checkpoint_path).resolve()
        pca_params_path = Path(args.vggish_pca_params_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing VGGish checkpoint: {checkpoint_path}")
        if not pca_params_path.exists():
            raise FileNotFoundError(f"Missing VGGish PCA params: {pca_params_path}")
        return VGGishExtractor(
            checkpoint_path=checkpoint_path,
            pca_params_path=pca_params_path,
            ffmpeg_fallback=args.ffmpeg_fallback,
        )
    if args.encoder == "clap":
        from feature_extraction.clap.encoder import CLAPExtractor

        clap_ckpt_path = Path(args.clap_ckpt_path).resolve()
        if not clap_ckpt_path.exists():
            raise FileNotFoundError(f"Missing CLAP checkpoint: {clap_ckpt_path}")
        return CLAPExtractor(
            clap_ckpt_path=clap_ckpt_path,
            device=args.device,
            use_fusion=args.clap_use_fusion,
            mode=args.clap_mode,
            chunk_sec=args.clap_chunk_sec,
            chunk_hop_sec=args.clap_chunk_hop_sec,
            chunk_tail_policy=args.clap_chunk_tail_policy,
            batch_size=args.clap_batch_size,
        )
    if args.encoder == "beats":
        from feature_extraction.beats.encoder import BEATsExtractor

        beats_checkpoint_path = Path(args.beats_checkpoint_path).resolve()
        if not beats_checkpoint_path.exists():
            raise FileNotFoundError(f"Missing BEATs checkpoint: {beats_checkpoint_path}")
        return BEATsExtractor(
            checkpoint_path=beats_checkpoint_path,
            device=args.device,
            chunk_sec=args.beats_chunk_sec,
            chunk_hop_sec=args.beats_chunk_hop_sec,
            tail_policy=args.beats_tail_policy,
        )
    if args.encoder == "encodec":
        from feature_extraction.encodec_fx.encoder import EncodecExtractor

        checkpoint_repository = args.encodec_checkpoint_repository or None
        return EncodecExtractor(
            model_name=args.encodec_model_name,
            bandwidth=args.encodec_bandwidth,
            device=args.device,
            checkpoint_repository=checkpoint_repository,
        )
    if args.encoder == "convnext":
        from feature_extraction.convnext.encoder import ConvNeXtExtractor

        convnext_ckpt_path = Path(args.convnext_ckpt_path).resolve()
        if not convnext_ckpt_path.exists():
            raise FileNotFoundError(f"Missing ConvNeXt checkpoint: {convnext_ckpt_path}")
        return ConvNeXtExtractor(
            checkpoint_path=convnext_ckpt_path,
            device=args.device,
            chunk_sec=args.convnext_chunk_sec,
            chunk_hop_sec=args.convnext_chunk_hop_sec,
            tail_policy=args.convnext_tail_policy,
            batch_size=args.convnext_batch_size,
        )
    raise ValueError(f"Unsupported encoder: {args.encoder}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified shardable feature extraction entrypoint")
    ap.add_argument("--encoder", type=str, required=True, choices=sorted(ENCODER_NAMES))
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--features_root", type=str, required=True)
    ap.add_argument("--filelist", type=str, default=None)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--exts", type=str, default="wav,mp3")
    ap.add_argument("--min_dur", type=float, default=0.0)
    ap.add_argument("--scan-mode", type=str, default="auto", choices=["auto", "msd_fast", "recursive"])
    ap.add_argument("--msd_fast_scan", action="store_true")
    ap.add_argument("--no_msd_fast_scan", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="For a quick smoke test: process at most N files in this shard")
    ap.add_argument("--log_every", type=int, default=5000)
    ap.add_argument("--progress_every", type=int, default=20000)
    ap.add_argument("--write_filelist", action="store_true")
    ap.add_argument("--filelist_dir", type=str, default=".")
    ap.add_argument("--filelist_prefix", type=str, default="filelist")
    ap.add_argument("--ffmpeg_fallback", action="store_true")
    ap.add_argument("--vggish_checkpoint_path", type=str, default=str((Path(__file__).resolve().parent.parent / "feature_extraction" / "vggish" / "vggish_model.ckpt")))
    ap.add_argument("--vggish_pca_params_path", type=str, default=str((Path(__file__).resolve().parent.parent / "feature_extraction" / "vggish" / "vggish_pca_params.npz")))
    ap.add_argument("--clap_ckpt_path", type=str, default="")
    ap.add_argument("--clap_use_fusion", action="store_true")
    ap.add_argument("--clap_mode", type=str, default="global", choices=["global", "chunks"])
    ap.add_argument("--clap_chunk_sec", type=float, default=5.0)
    ap.add_argument("--clap_chunk_hop_sec", type=float, default=1.0)
    ap.add_argument("--clap_chunk_tail_policy", type=str, default="cover", choices=["cover", "drop"])
    ap.add_argument("--clap_batch_size", type=int, default=16)
    ap.add_argument("--beats_checkpoint_path", type=str, default="")
    ap.add_argument("--beats_chunk_sec", type=float, default=10.0)
    ap.add_argument("--beats_chunk_hop_sec", type=float, default=10.0)
    ap.add_argument("--beats_tail_policy", type=str, default="pad", choices=["cover", "pad", "drop"])
    ap.add_argument("--encodec_model_name", type=str, default="encodec_48khz", choices=["encodec_24khz", "encodec_48khz"])
    ap.add_argument("--encodec_bandwidth", type=float, default=24.0)
    ap.add_argument("--encodec_checkpoint_repository", type=str, default="")
    ap.add_argument("--convnext_ckpt_path", type=str, default="")
    ap.add_argument("--convnext_chunk_sec", type=float, default=10.0)
    ap.add_argument("--convnext_chunk_hop_sec", type=float, default=10.0)
    ap.add_argument("--convnext_tail_policy", type=str, default="cover", choices=["cover", "pad", "drop"])
    ap.add_argument("--convnext_batch_size", type=int, default=8)
    args = ap.parse_args()

    audio_root = Path(args.audio_root).resolve()
    features_root = Path(args.features_root).resolve()
    features_root.mkdir(parents=True, exist_ok=True)

    if not audio_root.exists():
        raise FileNotFoundError(f"--audio_root not found: {audio_root}")
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError(f"--shard must be in [0, {args.num_shards - 1}]")
    if args.encoder == "clap":
        if args.clap_chunk_sec <= 0.0:
            raise ValueError("--clap_chunk_sec must be > 0")
        if args.clap_chunk_hop_sec <= 0.0:
            raise ValueError("--clap_chunk_hop_sec must be > 0")
        if args.clap_batch_size <= 0:
            raise ValueError("--clap_batch_size must be > 0")
    if args.encoder == "convnext":
        if args.convnext_chunk_sec <= 0.0:
            raise ValueError("--convnext_chunk_sec must be > 0")
        if args.convnext_chunk_hop_sec <= 0.0:
            raise ValueError("--convnext_chunk_hop_sec must be > 0")
        if args.convnext_batch_size <= 0:
            raise ValueError("--convnext_batch_size must be > 0")
    if args.encoder == "beats":
        if args.beats_chunk_sec <= 0.0:
            raise ValueError("--beats_chunk_sec must be > 0")
        if args.beats_chunk_hop_sec <= 0.0:
            raise ValueError("--beats_chunk_hop_sec must be > 0")

    exts: Set[str] = {e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}
    if not exts:
        raise ValueError("--exts parsed empty; expected like 'wav,mp3'")

    filelist_paths = None
    if args.filelist:
        filelist_path = Path(args.filelist).resolve()
        if not filelist_path.exists():
            raise FileNotFoundError(f"--filelist not found: {filelist_path}")
        filelist_paths = load_filelist_paths(filelist_path, audio_root=audio_root, exts=exts)
        print(f"[scan] Loaded {len(filelist_paths)} files from filelist {filelist_path}", flush=True)

    scan_mode = resolve_scan_mode(
        audio_root,
        args.scan_mode,
        msd_fast_scan=args.msd_fast_scan,
        no_msd_fast_scan=args.no_msd_fast_scan,
        filelist_paths=filelist_paths,
    )
    print(f"[scan] using scan_mode={scan_mode}", flush=True)

    audio_iter = iter_audio_paths(
        audio_root=audio_root,
        exts=exts,
        scan_mode=scan_mode,
        shard=args.shard,
        num_shards=args.num_shards,
        filelist_paths=filelist_paths,
    )
    if args.max_files > 0:
        audio_iter = (path for _, path in zip(range(args.max_files), audio_iter))

    filelist_path = None
    filelist_fh = None
    run_name = encoder_run_name(args)
    if args.write_filelist:
        filelist_dir = Path(args.filelist_dir).resolve()
        filelist_dir.mkdir(parents=True, exist_ok=True)
        filelist_path = filelist_dir / f"{args.filelist_prefix}.{run_name}.shard_{args.shard:04d}_of_{args.num_shards:04d}.txt"
        filelist_fh = filelist_path.open("w", encoding="utf-8")
        print(f"[filelist] writing shard filelist to {filelist_path}", flush=True)

    progress_path = features_root / f"{run_name}.shard_{args.shard:04d}.progress.txt"
    failed_path = features_root / f"{run_name}.shard_{args.shard:04d}.failed.txt"
    failed_fh = failed_path.open("w", encoding="utf-8")

    processed = 0
    skipped_exists = 0
    skipped_short = 0
    skipped_empty = 0
    failed = 0
    listed = 0

    with build_encoder(args) as extractor:
        bar = tqdm(audio_iter, desc=f"{args.encoder} shard {args.shard}/{args.num_shards}", unit="file", mininterval=2.0)
        for audio_path in bar:
            try:
                rel = audio_path.relative_to(audio_root)
            except Exception as e:
                print(f"[skip] {audio_path} -> {e}", flush=True)
                continue

            out_path = feature_out_path(audio_path, audio_root, features_root, args)
            if output_is_valid(out_path, args.encoder, args):
                skipped_exists += 1
                if filelist_fh is not None:
                    try:
                        duration_sec, sample_rate = audio_info(audio_path)
                        if duration_sec > 0 and (args.min_dur <= 0 or duration_sec >= args.min_dur):
                            filelist_fh.write(f"{audio_path}\t{duration_sec}\t{sample_rate}\n")
                            listed += 1
                    except Exception as e:
                        failed += 1
                        failed_fh.write(f"{audio_path}\tmeta\t{type(e).__name__}\t{e}\n")
                continue

            try:
                duration_sec, sample_rate = audio_info(audio_path)
                if duration_sec <= 0:
                    skipped_empty += 1
                    continue
                if args.min_dur and duration_sec < args.min_dur:
                    skipped_short += 1
                    continue

                enc_payload: Dict[str, torch.Tensor] = extractor.extract(audio_path)
                payload = {
                    **enc_payload,
                    "duration_sec": float(duration_sec),
                    "sample_rate": int(sample_rate),
                    "wav_relpath": str(rel),
                    "wav_abspath": str(audio_path.resolve()),
                }
                atomic_torch_save(payload, out_path)
                processed += 1

                if filelist_fh is not None:
                    filelist_fh.write(f"{audio_path}\t{duration_sec}\t{sample_rate}\n")
                    listed += 1

                if args.progress_every and processed % args.progress_every == 0:
                    progress_path.write_text(
                        f"processed={processed} listed={listed} last={rel}\n",
                        encoding="utf-8",
                    )

                if args.log_every and processed % args.log_every == 0:
                    print(
                        f"[shard {args.shard}] processed={processed} listed={listed} "
                        f"valid_exists={skipped_exists} short={skipped_short} empty={skipped_empty} failed={failed}",
                        flush=True,
                    )

            except Exception as e:
                failed += 1
                print(f"[failed] {audio_path}: {type(e).__name__}: {e}", flush=True)
                failed_fh.write(f"{audio_path}\tproc\t{type(e).__name__}\t{e}\n")

    if filelist_fh is not None:
        filelist_fh.close()
    failed_fh.close()

    progress_path.write_text(
        f"DONE processed={processed} listed={listed} valid_exists={skipped_exists} "
        f"short={skipped_short} empty={skipped_empty} failed={failed}\n",
        encoding="utf-8",
    )
    print(
        f"DONE encoder={args.encoder} shard={args.shard}/{args.num_shards} "
        f"processed={processed} listed={listed} valid_exists={skipped_exists} "
        f"short={skipped_short} empty={skipped_empty} failed={failed}",
        flush=True,
    )
    if filelist_path is not None:
        print(f"[filelist] DONE wrote {filelist_path}", flush=True)
    print(f"[failed] wrote {failed_path}", flush=True)


if __name__ == "__main__":
    main()
