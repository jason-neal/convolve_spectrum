# Convolution Code from
# https://github.com/jason-neal/equanimous-octo-tribble/blob/master/IP_Convolution.py
# Convolution of spectra to a Instrument profile of resolution R.
#
# The spectra does not have to be equidistant in wavelength.

# Multiprocess use and speed timing was contributed by Jorge Martins

from __future__ import division, print_function

from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from convolve_spectrum.IP_Convolution import ip_convolution as single_ip_convolution
from convolve_spectrum.IP_Convolution import unitary_Gauss
from convolve_spectrum.IP_multi_Convolution import (
    ip_convolution as multi_ip_convolution
)

if __name__ == "__main__":
    # Example usage of this convolution
    wav = np.linspace(2040, 2050, 20000)
    flux = (
        np.ones_like(wav) - unitary_Gauss(wav, 2045, .6) - unitary_Gauss(wav, 2047, .9)
    )

    # range in which to have the convolved values. Be careful of the edges!
    chip_limits = [2042, 2049]
    R = 2000

    time_init = dt.now()
    single_convolved_wav, single_convolved_flux = single_ip_convolution(
        wav, flux, chip_limits, R, fwhm_lim=5.0, plot=False, verbose=True
    )
    time_end = dt.now()

    multi_convolved_wav, multi_convolved_flux = multi_ip_convolution(
        wav, flux, chip_limits, R, fwhm_lim=5.0, plot=False, verbose=True
    )
    time_end_multi = dt.now()

    print("Time for normal convolution {}".format(time_end - time_init))
    print("Time from multiprocess convolution {}".format(time_end_multi - time_end))

    plt.figure()
    plt.plot(single_convolved_wav, single_convolved_flux, "ro", label="single")
    plt.plot(multi_convolved_wav, multi_convolved_flux, "bo", label="multi")
    plt.plot(wav, flux, "k-", label="original")
    plt.legend(loc="best")
    plt.title(r"Convolution by an Instrument Profile")

    plt.figure()
    plt.title(r"single/multi fluxes")
    plt.plot(
        single_convolved_wav,
        [
            single / multi
            for single, multi in zip(single_convolved_flux, multi_convolved_flux)
        ],
        "r",
        label="single",
    )

    plt.show()
