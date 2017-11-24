# Convolution of spectra to a Instrument profile of resolution R.
#
# The spectra does not have to be equidistant in wavelength.

# Multiprocessing is used to improve speed of convolution.
# The addition of multiprocess was added by Jorge Martins
# If you do not want to use multiprocessing then see IP_Convolution.py

from __future__ import division, print_function

import logging
from datetime import datetime as dt

import matplotlib.pyplot as plt
import multiprocess as mprocess
import numpy as np
from tqdm import tqdm

from spectrum_overload import Spectrum

def setup_debug(debug_val=False):
    """Set debug level."""
    if debug_val:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    return None


def wav_selector(wav, flux, wav_min, wav_max):
    """Wavelenght selector.

    If passed lists it will return lists.
    If passed np arrays it will return arrays

    """
    wav = np.asarray(wav)
    flux = np.asarray(flux)
    # Super Fast masking with numpy
    mask = (wav > wav_min) & (wav < wav_max)
    wav_sel = wav[mask]
    flux_sel = flux[mask]
    return [wav_sel, flux_sel]


def unitary_Gauss(x, center, fwhm):
    """Gaussian_function of area=1.

    p[0] = A;
    p[1] = mean;
    p[2] = fwhm;
    """
    sigma = np.abs(fwhm) / (2 * np.sqrt(2 * np.log(2)))
    Amp = 1.0 / (sigma * np.sqrt(2 * np.pi))
    tau = -((x - center) ** 2) / (2 * (sigma ** 2))
    result = Amp * np.exp(tau)

    return result


def fast_convolve(wav_val, R, wav_extended, flux_extended, fwhm_lim):
    """IP convolution multiplication step for a single wavelength value."""
    fwhm = wav_val / R
    # Mask of wavelength range within 5 fwhm of wav
    index_mask = ((wav_extended > (wav_val - fwhm_lim * fwhm)) &
                  (wav_extended < (wav_val + fwhm_lim * fwhm)))

    flux_2convolve = flux_extended[index_mask]
    # Gausian Instrument Profile for given resolution and wavelength
    inst_profile = unitary_Gauss(wav_extended[index_mask], wav_val, fwhm)

    sum_val = np.sum(inst_profile * flux_2convolve)
    # Correct for the effect of convolution with non-equidistant postions
    unitary_val = np.sum(inst_profile * np.ones_like(flux_2convolve))

    return sum_val / unitary_val


def wrapper_fast_convolve(args):
    """Wrapper for fast_convolve.

    Needed to unpack the arguments for fast_convolve as multiprocess.Pool.map does not accept multiple
    arguments.
    """
    return fast_convolve(*args)

def convolve_spectrum(spec, *args, **kwargs):
    """Convovle a spectrum using ip_convolution.

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
    fwhm_min = wav_chip[0] / R    # fwhm at the extremes of vector
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
    return [wav_chip, flux_conv_res]


def IPconvolution(wav, flux, chip_limits, R, FWHM_lim=5.0, plot=True,
                  verbose=False, numProcs=None):
    """Wrapper of ip_convolution for backwards compatibility.
    Lower case of variable name of FWHM.
    """
    logging.warning("IPconvolution is depreciated, should use ip_convolution instead."
                    "IPconvolution is still available for compatibility.")
    return ip_convolution(wav, flux, chip_limits, R, fwhm_lim=FWHM_lim, plot=plot,
                         verbose=verbose, numProcs=numProcs)


if __name__ == "__main__":
    # Example useage of this convolution
    wav = np.linspace(2040, 2050, 30000)
    flux = (np.ones_like(wav) - unitary_Gauss(wav, 2045, .6) -
            unitary_Gauss(wav, 2047, .9))
    # Range in which to have the convoled values. Be careful of the edges!
    chip_limits = [2042, 2049]

    R = 1000
    convolved_wav, convolved_flux = ip_convolution(wav, flux, chip_limits, R,
                                                   fwhm_lim=5.0, plot=True,
                                                   verbose=True)
