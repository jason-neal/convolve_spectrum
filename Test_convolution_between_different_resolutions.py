# Test convolving to different resolutions
# Test the effect of convolving straight to 20000 and convolving first to an intermediate resolution say 80000.


import matplotlib.pyplot as plt
import numpy as np

from IP_multi_Convolution import ip_convolution, unitary_Gauss


def main():
    # fwhm = lambda/R
    fwhm = 2046 / 100000
    # Starting spectrum
    wav = np.linspace(2040, 2050, 20000)
    flux = (np.ones_like(wav) - unitary_Gauss(wav, 2045, fwhm) -
            unitary_Gauss(wav, 2047, fwhm))

    # range in which to have the convoled values. Be careful of the edges!
    chip_limits = [2042, 2049]

    # Convolution to 80k
    R = 80000
    wav_80k, flux_80k = ip_convolution(wav, flux, chip_limits, R,
                                       fwhm_lim=5.0, plot=False, verbose=True)

    # Convolution to 50k
    R = 50000
    wav_50k, flux_50k = ip_convolution(wav, flux, chip_limits, R,
                                       fwhm_lim=5.0, plot=False, verbose=True)

    wav_80k_50k, flux_80k_50k = ip_convolution(wav_80k, flux_80k, chip_limits, R,
                                               fwhm_lim=5.0, plot=False, verbose=True)

    # Convolution to 20k
    R = 20000
    wav_80k_20k, flux_80k_20k = ip_convolution(wav_80k, flux_80k, chip_limits, R,
                                               fwhm_lim=5.0, plot=False, verbose=True)

    wav_50k_20k, flux_50k_20k = ip_convolution(wav_50k, flux_50k, chip_limits, R,
                                               fwhm_lim=5.0, plot=False, verbose=True)

    wav_80k_50k_20k, flux_80k_50k_20k = ip_convolution(wav_80k_50k, flux_80k_50k,
                                                       chip_limits, R, fwhm_lim=5.0,
                                                       plot=False, verbose=True)

    # Convolution straight to 20000
    wav_20k, flux_20k = ip_convolution(wav, flux, chip_limits, R, fwhm_lim=5.0,
                                       plot=False, verbose=True)

    # Plot the results

    plt.figure(1)
    plt.xlabel(r"wavelength [nm])")
    plt.ylabel(r"flux [counts] ")
    plt.plot(wav, flux / np.max(flux), color='k',
             linestyle="-", label="Original spectra")
    plt.plot(wav_80k, flux_80k / np.max(flux_80k), color='r', linestyle="-.", label="R=80k-20k")
    plt.plot(wav_50k, flux_50k / np.max(flux_50k), color='b', linestyle="--", label="R=50k")
    plt.plot(wav_80k_20k, flux_80k_20k / np.max(flux_80k_20k), color='r',
             linestyle="-", label="R=80k-20k")
    plt.plot(wav_50k_20k, flux_50k_20k / np.max(flux_50k_20k), color='b',
             linestyle="-", label="R=50k20k")
    plt.plot(wav_80k_50k_20k, flux_80k_50k_20k / np.max(flux_80k_50k_20k), color='m',
             linestyle="-", label="R=80k-50k-20k")
    plt.plot(wav_20k, flux_20k / np.max(flux_20k), color='c', linestyle="-", label="R=20k")
    plt.legend(loc='best')
    plt.title(r"Convolution by different Instrument Profiles")
    plt.show()


if __name__ == "__main__":
    # The IPcovolution fails if it is not run inside __name__ == "__main__"
    main()
