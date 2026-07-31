"""mlx-signal: Metal/MLX-accelerated signal processing for Apple Silicon.

A scipy.signal-compatible API where the heavy lifting (batched FFTs, polyphase
resampling, FFT convolution) runs on the GPU through MLX. Inputs can be NumPy
or MLX arrays; outputs are MLX arrays in unified memory, so ``np.array(result)``
is cheap and feeding an MLX model requires zero copies.
"""

from ._config import (
    Config,
    DowncastWarning,
    FallbackWarning,
    config_context,
    get_config,
    set_config,
)
from ._fft import next_fast_len
from .convolution import convolve, correlate, correlation_lags, fftconvolve, oaconvolve
from .filtering import filtfilt, firwin, firwin2, hilbert, lfilter, sosfilt, sosfiltfilt
from .peaks import find_peaks, peak_prominences, peak_widths
from .resampling import decimate, resample, resample_poly, upfirdn
from .spectral import coherence, csd, istft, periodogram, spectrogram, stft, welch
from .windows import get_window

__version__ = "0.1.0.dev0"

__all__ = [
    "Config",
    "DowncastWarning",
    "FallbackWarning",
    "coherence",
    "convolve",
    "config_context",
    "correlate",
    "decimate",
    "correlation_lags",
    "csd",
    "filtfilt",
    "fftconvolve",
    "firwin",
    "firwin2",
    "find_peaks",
    "get_config",
    "get_window",
    "hilbert",
    "istft",
    "lfilter",
    "next_fast_len",
    "oaconvolve",
    "peak_prominences",
    "peak_widths",
    "periodogram",
    "resample",
    "resample_poly",
    "set_config",
    "sosfilt",
    "sosfiltfilt",
    "spectrogram",
    "stft",
    "upfirdn",
    "welch",
]
