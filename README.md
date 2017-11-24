# Spectrum Convolution

[![Build Status](https://travis-ci.org/jason-neal/convolve_spectrum.svg?branch=master)](https://travis-ci.org/jason-neal/convolve_spectrum)[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c85dfdb9736f4b978566241354e3050b)](https://www.codacy.com/app/jason-neal/convolve_spectrum?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jason-neal/convolve_spectrum&amp;utm_campaign=Badge_Grade)
Useful scripts for astronomy and the like.

## Convolution
Convole a spectrum by a IP of a given Resolution

There is a mulitprocessing version and a normal (slower) version.
Calculates the IP for every pixel/wavelength value individually.
Does not need a eqidistant wavelength axis.

Assumes a gaussian IP profile


## Installation
```
    git clone https://github.com/jason-neal/convolve_spectrum.git   
    cd convolve_spectrum
    pip install -r requirements/requirements.txt
    python setup.py install
```
Usage from python:
```
    from convolve_spectrum import ipconvolution
    conv_wav, conv_flux = ip_convolution(wav, flux, wav_limits, R, fwhm_lim=5.0) 
```
The wavelenght axis is reduced to *wav_limits* due to edge effecs in the convolution.