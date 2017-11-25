# Spectrum Convolution

[![Build Status](https://travis-ci.org/jason-neal/convolve_spectrum.svg?branch=master)](https://travis-ci.org/jason-neal/convolve_spectrum)[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c85dfdb9736f4b978566241354e3050b)](https://www.codacy.com/app/jason-neal/convolve_spectrum?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jason-neal/convolve_spectrum&amp;utm_campaign=Badge_Grade)[![Coverage Status](https://coveralls.io/repos/github/jason-neal/convolve_spectrum/badge.svg?branch=master)](https://coveralls.io/github/jason-neal/convolve_spectrum?branch=master)

- Convole a spectrum by a IP of a given Resolution. 
- Does not need a eqidistant wavelength axis.
- Assumes a gaussian IP profile


There is a mulitprocessing version and a normal (slower) version.
Calculates the IP for every pixel/wavelength value individually (embarrassingly parrallel).

## Installation
```
    git clone https://github.com/jason-neal/convolve_spectrum.git   
    cd convolve_spectrum
    pip install -r requirements/requirements.txt
    python setup.py install
```
## Usage
```
    from convolve_spectrum import ipconvolution
    convolved_wav, convolved_flux = ip_convolution(wav, flux, wav_limits, R, fwhm_lim=5.0) 
```
The wavelength axis is reduced to *wav_limits* due to edge effecs in the convolution.


## Notes
The original version of this code was used for the IP component of [Figueira et. al. 2016](https://arxiv.org/abs/1511.07468) and explained in detail there (page 3).
