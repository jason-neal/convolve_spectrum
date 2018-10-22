import numpy as np
import pytest
from hypothesis import given, strategies as st

from convolve_spectrum import wav_selector


@given(st.lists(st.floats()), st.floats(allow_nan=False), st.floats(allow_nan=False))
def test_wav_selector(wav, wav_min, wav_max):
    y = np.copy(wav)
    wav2, y2 = wav_selector(wav, y, wav_min, wav_max)

    assert isinstance(wav2, np.ndarray)
    assert isinstance(y2, np.ndarray)
    assert all(wav2 >= wav_min)
    assert all(wav2 <= wav_max)
    assert len(wav2) == len(y2)


@pytest.mark.parametrize(
    "wav, wav_min, wav_max", [([np.nan], 1, 2), ([np.inf], 1, 2), ([-1 * np.inf], 3, 4)]
)
def test_wav_selector_with_nans_and_infs(wav, wav_min, wav_max):
    y = np.copy(wav)
    wav2, y2 = wav_selector(wav, y, wav_min, wav_max)

    assert isinstance(wav2, np.ndarray)
    assert isinstance(y2, np.ndarray)
    assert all(wav2 >= wav_min)
    assert all(wav2 <= wav_max)
    assert len(wav2) == len(y2)
    assert len(wav2) == 0


@pytest.mark.parametrize(
    "wav, wav_min, wav_max", [(np.arange(10), np.nan, 2), (range(10), 1, np.nan)]
)
def test_wav_selector_with_nans_inputs(wav, wav_min, wav_max):
    y = np.copy(wav)
    with pytest.raises(AssertionError):
        wav_selector(wav, y, wav_min, wav_max)
