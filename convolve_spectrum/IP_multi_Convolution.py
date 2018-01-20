# Convolution of spectra to a Instrument profile of resolution R.
#
# The spectra does not have to be equidistant in wavelength.

# Multiprocessing is used to improve speed of convolution.
# The addition of multiprocess was added by Jorge Martins
# If you do not want to use multiprocessing then see IP_Convolution.py

from __future__ import division, print_function

import logging
import warnings
from datetime import datetime as dt

import matplotlib.pyplot as plt
import multiprocess as mprocess
import numpy as np
from tqdm import tqdm
from spectrum_overload import Spectrum

from convolve_spectrum.IP_Convolution import wav_selector, unitary_Gauss, fast_convolve


def setup_debug(debug_val=False):
    """Set debug level."""
    if debug_val:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    return None


def wrapper_fast_convolve(args):
    """Wrapper for fast_convolve.

    Needed to unpack the arguments for fast_convolve as multiprocess.Pool.map does not accept multiple
    arguments.
    """
    return fast_convolve(*args)


def convolve_spectrum(spec, *args, **kwargs):
    """Convolve a spectrum using ip_convolution.

    ip_convolution(wav, flux, chip_limits, R, fwhm_lim=5.0, plot=True,
                       verbose=False, numProcs=None, progbar=True, debug=False)
    """
    spec = spec.copy()
    conv = ip_convolution(spec.xaxis, spec.flux, *args, **kwargs)
    return Spectrum(xaxis=conv[0], flux=conv[1], header=spec.header)


def ip_convolution(wav, flux, chip_limits, R, fwhm_lim=5.0, plot=True,
                   verbose=False, numProcs=None, progbar=True, debug=False):
    """Spectral convolution which allows non-equidistance step values.

    Parameters
    ----------
    verbose: bool
        Does nothing anymore...
    numProcs: int
        NUmber of processes to use. Defualt=None selects cpu_count - 1.
    progbar: bool
        Enable the tqdm progress bar. Default=True.
    debug: bool
        Enable logging debug information. Default=False.
    """
    if verbose:
        """Verbose was turned on when doesn't do anything."""
        logging.warning("ip_convolution's unused 'verbose' parameter was enabled."
                        " It is unused/depreciated so should be avoided.")

    setup_debug(debug_val=debug)
    # Turn into numpy arrays
    wav = np.asarray(wav, dtype='float64')
    flux = np.asarray(flux, dtype='float64')

    timeInit = dt.now()
    wav_chip, flux_chip = wav_selector(wav, flux, chip_limits[0],
                                       chip_limits[1])
    # We need to calculate the fwhm at this value in order to set the starting
    # point for the convolution
    fwhm_min = wav_chip[0] / R  # fwhm at the extremes of vector
    fwhm_max = wav_chip[-1] / R

    # Wide wavelength bin for the resolution_convolution
    wav_min = wav_chip[0] - fwhm_lim * fwhm_min
    wav_max = wav_chip[-1] + fwhm_lim * fwhm_max
    wav_ext, flux_ext = wav_selector(wav, flux, wav_min, wav_max)

    logging.debug("Starting the Resolution convolution...")

    # multiprocessing part
    if numProcs is None:
        numProcs = mprocess.cpu_count() - 1

    mprocPool = mprocess.Pool(processes=numProcs)

    args_generator = tqdm([[wav, R, wav_ext, flux_ext, fwhm_lim]
                           for wav in wav_chip], disable=(not progbar))

    flux_conv_res = np.array(mprocPool.map(wrapper_fast_convolve,
                                           args_generator))

    mprocPool.close()
    timeEnd = dt.now()
    logging.debug("Multi-Proc convolution has been completed in "
                  "{} using {}/{} cores.\n".format(timeEnd - timeInit,
                                                   numProcs,
                                                   mprocess.cpu_count()))

    if (plot):
        plt.figure(1)
        plt.xlabel(r"wavelength [ nm ])")
        plt.ylabel(r"flux [counts] ")
        plt.plot(wav_chip, flux_chip / np.max(flux_chip), color='k',
                 linestyle="-", label="Original spectra")
        plt.plot(wav_chip, flux_conv_res / np.max(flux_conv_res), color='r',
                 linestyle="-", label="Spectrum observed at R={0}.".format(R))
        plt.legend(loc='best')
        plt.title(r"Convolution by an Instrument Profile ")
        plt.show()
    return wav_chip, flux_conv_res


def IPconvolution(wav, flux, chip_limits, R, FWHM_lim=5.0, plot=True,
                  verbose=False, numProcs=None):
    """Wrapper of ip_convolution for backwards compatibility.
    Lower case of variable name of FWHM.
    """
    warnings.warn("IPconvolution is depreciated, should use ip_convolution instead."
                  "IPconvolution is still available for compatibility.", DeprecationWarning)
    return ip_convolution(wav, flux, chip_limits, R, fwhm_lim=FWHM_lim, plot=plot,
                          verbose=verbose, numProcs=numProcs)


if __name__ == "__main__":
    # Example usage of this convolution
    wav = np.linspace(2040, 2050, 30000)
    flux = (np.ones_like(wav) - unitary_Gauss(wav, 2045, .6) -
            unitary_Gauss(wav, 2047, .9))
    # Range in which to have the convolved values. Be careful of the edges!
    chip_limits = [2042, 2049]

    R = 1000
    convolved_wav, convolved_flux = ip_convolution(wav, flux, chip_limits, R,
                                                   fwhm_lim=5.0, plot=True,
                                                   verbose=True)
