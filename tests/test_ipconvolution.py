import numpy as np
import pytest

from convolve_spectrum.ip_convolution import ip_convolution, fast_convolve
from convolve_spectrum.ip_convolution import IPconvolution


def test_fast_convolution():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    c = np.ones_like(b)
    resolution = 50000
    for a_val in a:
        # print(type(fast_convolve(a_val, resolution, b, c, 5)))
        # print(isinstance(fast_convolve(a_val, resolution, b, c, 5), np.float64))
        assert isinstance(fast_convolve(a_val, resolution, b, c, 5), np.float64)
        assert (
            fast_convolve(a_val, resolution, b, c, 5) == 1
        )  # Test a flat input of 1s gives a flat output of 1s
        assert (
            fast_convolve(a_val, resolution, b, 0 * c, 5) == 0
        )  # Test a flat input of 1s gives a flat output of 1s


# TODO: A result that is not just ones.
def test_ip_convolution():
    wave = [1, 2, 3, 5, 6, 7, 10, 11]
    flux = [1, 1, 1, 1, 1, 1, 1, 1]
    chip_limits = [2, 9]
    resolution = 100
    new_wav, new_flux = ip_convolution(wave, flux, chip_limits, resolution, plot=False)
    assert np.all(new_flux == [1, 1, 1, 1, 1])
    assert np.all(new_wav == [2, 3, 5, 6, 7])


def test_ip_wrapper():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    assert np.allclose(
        IPconvolution(a, b, [2140, 2165], resolution=50000, plot=False),
        ip_convolution(a, b, [2140, 2165], resolution=50000, plot=False),
    )


def test_IPconvolution_depreciation():
    a = np.linspace(2130, 2170, 1024)
    b = np.linspace(2100, 2200, 1024)
    with pytest.deprecated_call():
        IPconvolution(a, b, [2140, 2165], resolution=50000, plot=False)


if __name__ == "__main__":
    pytest.main(args=[__file__])
