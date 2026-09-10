import math
import random
import torch

from torch import nn
from typing import Optional, Tuple

from torchaudio import transforms as T

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        randomize: bool = True,
        hop_seconds: Optional[float] = None,
        crop_within_chunk_seconds: Optional[float] = None,
        crop_within_chunk_hop_seconds: Optional[float] = None,
    ):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.hop_seconds = hop_seconds
        self.hop_samples = None
        if hop_seconds is not None:
            if hop_seconds <= 0:
                raise ValueError("hop_seconds must be > 0 when provided")
            self.hop_samples = max(1, int(round(hop_seconds * sample_rate)))

        self.crop_within_chunk_samples = None
        self.crop_within_chunk_hop_samples = None
        if crop_within_chunk_seconds is not None:
            if crop_within_chunk_seconds <= 0:
                raise ValueError("crop_within_chunk_seconds must be > 0 when provided")
            self.crop_within_chunk_samples = max(1, int(round(crop_within_chunk_seconds * sample_rate)))
            if self.crop_within_chunk_samples < self.n_samples:
                raise ValueError("crop_within_chunk_seconds must be >= crop duration")

            hop_sec = crop_within_chunk_hop_seconds
            if hop_sec is None:
                hop_sec = crop_within_chunk_seconds
            if hop_sec <= 0:
                raise ValueError("crop_within_chunk_hop_seconds must be > 0 when provided")
            self.crop_within_chunk_hop_samples = max(1, int(round(hop_sec * sample_rate)))

    def _sample_window_constrained_offset(self, n_samples: int, upper_bound: int) -> Optional[int]:
        if self.crop_within_chunk_samples is None or self.crop_within_chunk_hop_samples is None:
            return None

        step = self.hop_samples or 1
        intervals = []
        window_start = 0

        while window_start < n_samples:
            min_offset = window_start
            max_offset = min(window_start + self.crop_within_chunk_samples - self.n_samples, upper_bound)
            if max_offset >= min_offset:
                count = ((max_offset - min_offset) // step) + 1
                if count > 0:
                    intervals.append((min_offset, max_offset, count))
            window_start += self.crop_within_chunk_hop_samples

        total_choices = sum(count for _, _, count in intervals)
        if total_choices <= 0:
            return None

        choice_ix = random.randrange(total_choices)
        for min_offset, _, count in intervals:
            if choice_ix < count:
                return min_offset + choice_ix * step
            choice_ix -= count
        return None

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            constrained_offset = self._sample_window_constrained_offset(n_samples, upper_bound)
            if constrained_offset is not None:
                offset = constrained_offset
            elif self.hop_samples is None:
                offset = random.randint(0, upper_bound)
            else:
                max_step = upper_bound // self.hop_samples
                offset = random.randint(0, max_step) * self.hop_samples

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"
    def __init__(self, params=[-16, 2], sample_rate=16000, energy_threshold=1e-6):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal
        
        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0])
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95
        return signal