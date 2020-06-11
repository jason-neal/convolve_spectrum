import numpy as np
from matplotlib import pyplot as plt


def wav_selector(wav, flux, wav_min, wav_max):
    """Wavelength selector.

    Slice array to within wav_min and wav_max inclusive.
    """
    assert not (np.isnan(wav_min)), "Lower wavelength band is NaN!"
    assert not (np.isnan(wav_max)), "Upper wavelength band is NaN!"

    wav = np.asarray(wav)
    flux = np.asarray(flux)

    # Remove NaN wavelengths
    nan_mask = np.isnan(wav)
    wav = wav[~nan_mask]
    flux = flux[~nan_mask]
    assert not np.any(np.isnan(wav))
    mask = (wav >= wav_min) & (wav <= wav_max)
    wav_sel = wav[mask]
    flux_sel = flux[mask]
    return wav_sel, flux_sel


def unitary_Gauss(x, center, fwhm):
    """Gaussian_function of area=1.

    p[0] = A;
    p[1] = mean;
    p[2] = full with at half maximum (fwhm);
    """
    sigma = np.abs(fwhm) / (2 * np.sqrt(2 * np.log(2)))
    Amp = 1.0 / (sigma * np.sqrt(2 * np.pi))
    tau = -((x - center) ** 2) / (2 * (sigma ** 2))
    return Amp * np.exp(tau)


def plot_convolution(wav_chip, flux_chip, flux_conv_res, res):
    plt.figure(1)
    plt.xlabel(r"Wavelength [ nm ])")
    plt.ylabel(r"Normalized Flux [counts] ")
    plt.plot(
        wav_chip,
        flux_chip / np.max(flux_chip),
        color="k",
        linestyle="-",
        label="Original",
    )
    plt.plot(
        wav_chip,
        flux_conv_res / np.max(flux_conv_res),
        color="r",
        linestyle="--",
        label="Convolved",
    )
    plt.legend(loc="best")
    plt.title(r"Convolution by an Instrument Profile with resolution={0}".format(res))
    plt.show()
