# Convolution of spectra to a instrument profile of a given resolution.
#
# Unlike PyAstronomy the spectra do not have to be equidistant in wavelength.

# Pixels computed in parallel using multiprocess.


from __future__ import division, print_function

import warnings

import multiprocess as mprocess
import numpy as np
from spectrum_overload import Spectrum
from tqdm import tqdm

from convolve_spectrum.utils import plot_convolution, wav_selector, unitary_Gauss


def wrapper_fast_convolve(args):
    """Wrapper for fast_convolve.

    Needed to unpack the arguments for fast_convolve as multiprocess.Pool.map does not accept multiple
    arguments.
    """
    return fast_convolve(*args)


def convolve_spectrum(spec, *args, **kwargs):
    """Convolve a spectrum using ip_convolution.

    ip_convolution(wav, flux, chip_limits, resolution, fwhm_lim=5.0, plot=True,
                  numProcs=None, progbar=True)
    """
    spec = spec.copy()
    conv = ip_convolution(spec.xaxis, spec.flux, *args, **kwargs)
    return Spectrum(xaxis=conv[0], flux=conv[1], header=spec.header)


def ip_convolution(
    wav,
    flux,
    chip_limits,
    resolution,
    fwhm_lim=5.0,
    plot=True,
    numProcs=None,
    progbar=True,
):
    """Spectral convolution which allows non-equidistant step values.

    Parameters
    ----------
    wav:
        Wavelength
    flux:
        Flux of spectrum
    chip_limits: List[float, float]
        Wavelength limits of region to return after convolution.
    resolution:
        Resolution to convolve to.
    fwhm_lim:
        Number of FWHM of convolution kernel to use as edge buffer.
    plot: bool
        Display the spectrum, and convolved result.
    numProcs: int
        NUmber of processes to use. Default=None selects cpu_count - 1.
    progbar: bool
        Enable the tqdm progress bar. Default=True.
     """

    # Turn into numpy arrays
    wav = np.asarray(wav, dtype="float64")
    flux = np.asarray(flux, dtype="float64")


    wav_chip, flux_chip = wav_selector(wav, flux, chip_limits[0], chip_limits[1])
    # We need to calculate the fwhm at this value in order to set the starting
    # point for the convolution
    fwhm_min = wav_chip[0] / resolution  # fwhm at the extremes of vector
    fwhm_max = wav_chip[-1] / resolution

    # Wide wavelength bin for the resolution_convolution
    wav_min = wav_chip[0] - fwhm_lim * fwhm_min
    wav_max = wav_chip[-1] + fwhm_lim * fwhm_max
    wav_ext, flux_ext = wav_selector(wav, flux, wav_min, wav_max)


    # Multiprocessing part
    if numProcs is None:
        numProcs = mprocess.cpu_count() - 1

    mprocPool = mprocess.Pool(processes=numProcs)

    args_generator = tqdm(
        [[wav, resolution, wav_ext, flux_ext, fwhm_lim] for wav in wav_chip],
        disable=(not progbar),
    )

    flux_conv_res = np.array(mprocPool.map(wrapper_fast_convolve, args_generator))

    mprocPool.close()

    if plot:
        plot_convolution(wav_chip, flux_chip, flux_conv_res, resolution)

    return wav_chip, flux_conv_res


def IPconvolution(
    wav, flux, chip_limits, resolution, FWHM_lim=5.0, plot=True, numProcs=None, **kwargs):
    """Wrapper of ip_convolution for backwards compatibility.
    Lower case of variable name of FWHM.
    """
    warnings.warn(
        "IPconvolution is depreciated, should use ip_convolution instead."
        "IPconvolution is still available for compatibility.",
        DeprecationWarning,
    )
    return ip_convolution(
        wav,
        flux,
        chip_limits,
        resolution,
        fwhm_lim=FWHM_lim,
        plot=plot,
        numProcs=numProcs, **kwargs,
    )


def fast_convolve(wav_val, resolution, wav_extended, flux_extended, fwhm_lim):
    """IP convolution multiplication step for a single wavelength value."""
    fwhm = wav_val / resolution
    # Mask of wavelength range within 5 fwhm of wav
    index_mask = (wav_extended > (wav_val - fwhm_lim * fwhm)) & (
        wav_extended < (wav_val + fwhm_lim * fwhm)
    )

    flux_2convolve = flux_extended[index_mask]
    # Gaussian Instrument Profile for given resolution and wavelength
    inst_profile = unitary_Gauss(wav_extended[index_mask], wav_val, fwhm)

    sum_val = np.sum(inst_profile * flux_2convolve)
    # Correct for the effect of convolution with non-equidistant positions
    unitary_val = np.sum(inst_profile)

    return sum_val / unitary_val


if __name__ == "__main__":
    # Example usage of this convolution
    wav = np.linspace(2040, 2050, 30000)
    flux = (
        np.ones_like(wav) - unitary_Gauss(wav, 2045, .6) - unitary_Gauss(wav, 2047, .9)
    )
    # Range in which to have the convolved values. Be careful of the edges!
    chip_limits = [2042, 2049]

    resolution = 1000
    convolved_wav, convolved_flux = ip_convolution(
        wav, flux, chip_limits, resolution, fwhm_lim=5.0, plot=True,
    )

