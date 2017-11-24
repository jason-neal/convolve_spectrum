from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest

from convolve_spectrum.IP_multi_Convolution import wav_selector
from convolve_spectrum.IP_Convolution import (fast_convolve,
                                              ip_convolution, unitary_Gauss)
from convolve_spectrum.IP_multi_Convolution import ip_convolution as ip_multi_Convolution

from convolve_spectrum.IP_Convolution import IPconvolution 
from convolve_spectrum.IP_multi_Convolution import IPconvolution as IPmulticonvolution

@given(st.lists(st.floats()), st.floats(), st.floats())
def test_wav_selector(wav, wav_min, wav_max):
    y = np.copy(wav)
    wav2, y2 = wav_selector(wav, y, wav_min, wav_max)
    
    assert isinstance(wav2, np.ndarray)
    assert isinstance(y2, np.ndarray)
    assert all(wav2 >= wav_min)
    assert all(wav2 <= wav_max)
    assert len(wav2) == len(y2)


def test_fast_convolution():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    c = np.ones_like(b)
    R = 50000
    for a_val in a:
        # print(type(fast_convolve(a_val, R, b, c, 5)))
        # print(isinstance(fast_convolve(a_val, R, b, c, 5), np.float64))
        assert type(fast_convolve(a_val, R, b, c, 5)) == np.float64
        assert fast_convolve(a_val, R, b, c, 5) == 1     # Test a flat input of 1s gives a flat ouput of 1s
        assert fast_convolve(a_val, R, b, 0 * c, 5) == 0     # Test a flat input of 1s gives a flat ouput of 1s

# TODO: A result that is not just ones.
def test_ip_convolution():
    wave = [1, 2, 3, 5, 6, 7, 10, 11]
    flux = [1, 1, 1, 1, 1, 1, 1, 1]
    chip_limits = [2, 9]
    R = 100
    new_wav, new_flux = ip_convolution(wave, flux, chip_limits, R, plot=False)
    assert np.all(new_flux == [1, 1, 1, 1, 1])
    assert np.all(new_wav == [2, 3, 5, 6, 7])


def test_ip_wrapper():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    assert np.allclose(IPconvolution(a, b, [2140, 2165], R=50000, plot=False),
                  ip_convolution(a, b, [2140, 2165], R=50000, plot=False))


def test_IPconolution_depreciation():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    with pytest.deprecated_call():
        IPconvolution(a, b, [2140, 2165], R=50000, plot=False)


def test_IPmulticonvolution_depreciation():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    with pytest.deprecated_call():
        IPmulticonvolution(a, b, [2140, 2165], R=50000, plot=False)
