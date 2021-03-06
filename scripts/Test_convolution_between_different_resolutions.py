# Test convolution to different resolutions
# Test the effect of convolution straight to 20000 and convolution first to an intermediate resolution say 80000.

import matplotlib.pyplot as plt
import numpy as np

from convolve_spectrum import unitary_Gauss
from convolve_spectrum.ip_convolution import ip_convolution


def main():
    # fwhm = lambda/resolution
    fwhm = 2046 / 100000
    # Starting spectrum
    wav = np.linspace(2040, 2050, 20000)
    flux = (
        np.ones_like(wav)
        - unitary_Gauss(wav, 2045, fwhm)
        - unitary_Gauss(wav, 2047, fwhm)
    )

    # Range in which to have the convolved values. Be careful of the edges!
    chip_limits = [2042, 2049]

    # Convolution to 80k
    resolution = 80000
    wav_80k, flux_80k = ip_convolution(
        wav, flux, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    # Convolution to 50k
    resolution = 50000
    wav_50k, flux_50k = ip_convolution(
        wav, flux, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    wav_80k_50k, flux_80k_50k = ip_convolution(
        wav_80k, flux_80k, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    # Convolution to 20k
    resolution = 20000
    wav_80k_20k, flux_80k_20k = ip_convolution(
        wav_80k, flux_80k, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    wav_50k_20k, flux_50k_20k = ip_convolution(
        wav_50k, flux_50k, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    wav_80k_50k_20k, flux_80k_50k_20k = ip_convolution(
        wav_80k_50k, flux_80k_50k, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    # Convolution straight to 20000
    wav_20k, flux_20k = ip_convolution(
        wav, flux, chip_limits, resolution, fwhm_lim=5.0, plot=False
    )

    # Plot the results
    plt.figure(1)
    plt.xlabel(r"wavelength [nm])")
    plt.ylabel(r"flux [counts] ")
    plt.plot(
        wav, flux / np.max(flux), color="k", linestyle="-", label="Original spectra"
    )
    plt.plot(
        wav_80k,
        flux_80k / np.max(flux_80k),
        color="r",
        linestyle="-.",
        label="resolution=80k-20k",
    )
    plt.plot(
        wav_50k,
        flux_50k / np.max(flux_50k),
        color="b",
        linestyle="--",
        label="resolution=50k",
    )
    plt.plot(
        wav_80k_20k,
        flux_80k_20k / np.max(flux_80k_20k),
        color="r",
        linestyle="-",
        label="resolution=80k-20k",
    )
    plt.plot(
        wav_50k_20k,
        flux_50k_20k / np.max(flux_50k_20k),
        color="b",
        linestyle="-",
        label="resolution=50k20k",
    )
    plt.plot(
        wav_80k_50k_20k,
        flux_80k_50k_20k / np.max(flux_80k_50k_20k),
        color="m",
        linestyle="-",
        label="resolution=80k-50k-20k",
    )
    plt.plot(
        wav_20k,
        flux_20k / np.max(flux_20k),
        color="c",
        linestyle="-",
        label="resolution=20k",
    )
    plt.legend(loc="best")
    plt.title(r"Convolution by different Instrument Profiles")
    plt.show()


if __name__ == "__main__":
    # The ip_convolution fails if it is not run inside __name__ == "__main__"
    main()
