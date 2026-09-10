from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import tensorflow.compat.v1 as tf1

from . import vggish_input
from . import vggish_params
from . import vggish_postprocess
from . import vggish_slim


def decode_with_ffmpeg_to_float_mono_16k(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(vggish_params.SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode('utf-8', errors='replace')}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return pcm.astype(np.float32) / 32768.0


def vggish_examples_for_path(path: Path, use_ffmpeg_fallback: bool) -> np.ndarray:
    try:
        return vggish_input.wavfile_to_examples(str(path))
    except Exception:
        if not use_ffmpeg_fallback:
            raise
        wave = decode_with_ffmpeg_to_float_mono_16k(path)
        return vggish_input.waveform_to_examples(wave, vggish_params.SAMPLE_RATE)


class VGGishExtractor:
    name = "vggish"

    def __init__(self, *, checkpoint_path: Path, pca_params_path: Path, ffmpeg_fallback: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.pca_params_path = pca_params_path
        self.ffmpeg_fallback = ffmpeg_fallback
        self.pproc: vggish_postprocess.Postprocessor | None = None
        self.sess: tf1.Session | None = None
        self.features_tensor = None
        self.embedding_tensor = None

    def __enter__(self) -> "VGGishExtractor":
        tf1.disable_eager_execution()
        self.pproc = vggish_postprocess.Postprocessor(str(self.pca_params_path))
        self.graph = tf1.Graph()
        with self.graph.as_default():
            self.sess = tf1.Session(graph=self.graph)
            with self.sess.as_default():
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(self.sess, str(self.checkpoint_path))
                self.features_tensor = self.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                self.embedding_tensor = self.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.sess is not None:
            self.sess.close()

    def extract(self, audio_path: Path) -> Dict[str, torch.Tensor]:
        examples = vggish_examples_for_path(audio_path, use_ffmpeg_fallback=self.ffmpeg_fallback)
        if examples.shape[0] == 0:
            raise ValueError("empty VGGish examples")
        assert self.sess is not None
        assert self.features_tensor is not None
        assert self.embedding_tensor is not None
        assert self.pproc is not None
        [emb] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: examples})
        post = self.pproc.postprocess(emb)
        return {
            "emb": torch.from_numpy(emb).cpu(),
            "emb_postproc": torch.from_numpy(post).cpu(),
        }
