from __future__ import annotations

import hashlib
import string
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set

import soundfile as sf


LETTERS = list(string.ascii_uppercase)


def stable_shard_id(key: str, num_shards: int) -> int:
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little") % num_shards


def audio_info(path: Path) -> tuple[float, int]:
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        return 0.0, 0
    return float(info.frames) / float(info.samplerate), int(info.samplerate)


def iter_msd_bucket_dirs(audio_root: Path) -> Iterator[Path]:
    for a in LETTERS:
        p1 = audio_root / a
        if not p1.is_dir():
            continue
        for b in LETTERS:
            p2 = p1 / b
            if not p2.is_dir():
                continue
            for c in LETTERS:
                p3 = p2 / c
                if p3.is_dir():
                    yield p3


def detect_msd_structure(audio_root: Path) -> bool:
    return (audio_root / "A").is_dir() and (audio_root / "A" / "A" / "A").is_dir()


def parse_filelist_line(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    return stripped.split("\t", 1)[0].split(",", 1)[0].strip()


def normalize_filelist_entry(entry: str, audio_root: Path) -> Path:
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate
    return audio_root / candidate


def load_filelist_paths(filelist_path: Path, audio_root: Path, exts: Set[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    with filelist_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parsed = parse_filelist_line(raw_line)
            if parsed is None:
                continue
            path = normalize_filelist_entry(parsed, audio_root)
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower().lstrip(".") not in exts:
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def iter_audio_paths(
    audio_root: Path,
    exts: Set[str],
    scan_mode: str,
    shard: int,
    num_shards: int,
    filelist_paths: Optional[Sequence[Path]] = None,
) -> Iterator[Path]:
    if filelist_paths is not None:
        for path in filelist_paths:
            try:
                rel = path.relative_to(audio_root)
            except ValueError:
                continue
            if stable_shard_id(str(rel), num_shards) == shard:
                yield path
        return

    if scan_mode == "msd_fast":
        bucket_dirs = [
            d for d in iter_msd_bucket_dirs(audio_root)
            if stable_shard_id(str(d.relative_to(audio_root)), num_shards) == shard
        ]
        for bucket in bucket_dirs:
            for p in bucket.iterdir():
                if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                    yield p
        return

    for p in audio_root.glob("**/*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue
        rel = p.relative_to(audio_root)
        if stable_shard_id(str(rel), num_shards) == shard:
            yield p


def resolve_scan_mode(
    audio_root: Path,
    scan_mode: str,
    *,
    msd_fast_scan: bool,
    no_msd_fast_scan: bool,
    filelist_paths: Optional[Sequence[Path]],
) -> str:
    if filelist_paths is not None:
        return "recursive"
    if scan_mode == "auto":
        auto_msd = detect_msd_structure(audio_root)
        return "msd_fast" if (msd_fast_scan or auto_msd) and (not no_msd_fast_scan) else "recursive"
    return scan_mode
