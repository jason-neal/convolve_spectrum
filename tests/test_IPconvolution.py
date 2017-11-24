import numpy as np
import py.test

from octotribble.Convolution.IP_Convolution import (IPconvolution, fast_convolve,
                                                    ip_convolution, unitary_Gauss,
                                                    wav_selector)
from octotribble.Convolution.IP_multi_Convolution import fast_convolve as fast_convolve_multi
from octotribble.Convolution.IP_multi_Convolution import ip_convolution as ip_multi_Convolution
from octotribble.Convolution.IP_multi_Convolution import wav_selector as wav_selector_multi


def test_wav_selector():
    x = [1, 2, 3, 4, 6]
    y = [1, 2, 1, 1, 2]
    x_np = np.array(x)
    y_np = np.array(y)
    assert isinstance(wav_selector(x, y, 1, 3), list)
    assert isinstance(wav_selector(x_np, y_np, 1, 3), list)
    assert isinstance(wav_selector(x, y, 1, 3)[0], list)
    assert isinstance(wav_selector(x, y, 1, 3)[1], list)
    assert isinstance(wav_selector(x_np, y_np, 1, 3)[0], np.ndarray)
    assert isinstance(wav_selector(x_np, y_np, 1, 3)[1], np.ndarray)
    assert wav_selector(x, y, 0, 3)[0] == [1, 2]
    assert wav_selector(x, y, 0, 3)[0] == wav_selector_multi(x, y, 0, 3)[0]


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
        assert np.allclose(fast_convolve(a_val, R, b, c, 5), fast_convolve_multi(a_val, R, b, c, 5))


def test_IPconvolution():
    wave = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
    flux = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    chip_limits = [2, 9]
    R = 100
    ans = IPconvolution(wave, flux, chip_limits, R, plot=False)
    assert ans == [1, 1]


def test_ip_wrapper():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    assert np.all(IPconvolution(a, b, [2140, 2165], R=50000, plot=False),
                  ip_convolution(a, b, [2140, 2165], R=50000, plot=False))


if __name__ == "__main__":
    test_fast_convolution()
