from __future__ import annotations

import numpy as np
from pedalboard import Bitcrush, Distortion, Gain, HighpassFilter, LowpassFilter, Pedalboard, Reverb

from .config import FXCfg


def build_hl1_chain(cfg: FXCfg) -> Pedalboard:
    """Black Mesa transit PA voice: bandpass + bitcrush + soft drive + slap reverb."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=cfg.highpass_hz),
        LowpassFilter(cutoff_frequency_hz=cfg.lowpass_hz),
        Bitcrush(bit_depth=cfg.bitcrush_bits),
        Distortion(drive_db=cfg.drive_db),
        Reverb(
            room_size=cfg.reverb_room_size,
            damping=cfg.reverb_damping,
            wet_level=cfg.reverb_wet,
            dry_level=cfg.reverb_dry,
            width=1.0,
        ),
        Gain(gain_db=cfg.output_gain_db),
    ])


def apply_fx(chain: Pedalboard, pcm_f32: np.ndarray, sample_rate: int) -> np.ndarray:
    if pcm_f32.size == 0:
        return pcm_f32
    out = chain(pcm_f32, sample_rate)
    # Hard limit to avoid clipping pop on the speaker.
    return np.clip(out, -1.0, 1.0)
