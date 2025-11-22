"""
Here are functions to help with evaluating datasets based on their effect on stargazing conditions
"""
import numpy as np
import xarray as xr
# custom fxs
from scripts.utils.logging_tools import logging_setup

def grade_precip(p):
    """ convert given precipitation probability value to letter grade,
    where -1=NA, 0=A+, 1=A, 2=B, 3=C, 4=D, 5=F """
    if np.isnan(p): return -1
    if p > 0.5: return 5
    elif p > 0.2: return 3
    elif p > 0.05: return 2
    else: return 0

def grade_cloud(c):
    """ convert given sky cover percent value to letter grade,
    where -1=NA, 0=A+, 1=A, 2=B, 3=C, 4=D, 5=F """
    if np.isnan(c): return -1
    if c >= 1.0: return 5
    elif c > 0.75: return 5
    elif c > 0.5: return 4
    elif c > 0.2: return 2
    else: return 0

def grade_lightpollution(lp):
    """ convert given light pollution magnitude to letter grade, 
    where -1=NA, 0=A+, 1=A, 2=B, 3=C, 4=D, 5=F """
    if np.isnan(lp): return -1 # NaN
    elif lp >= 0.90: return 0 # A+: pristine nightsky
    elif  0.75 <= lp < 0.9: return 1 # A: nearly-pristine nightsky
    elif 0.5 <= lp < 0.75: return 2 # B: winter + summer milkyway are visible
    elif 0.3 <= lp < 0.50: return 3 # C: winter milky way gone, summer milkyway visible
    elif 0.1 <= lp < 0.30: return 4 # D: you can see some planets and bright stars
    elif lp < 0.1: return 5 # F: night sky is never darker than twilight

def grade_moon(m):
    """ convert given moon illumination percent value to letter grade, 
    where -1=NA, 0=A+, 1=A, 2=B, 3=C, 4=D, 5=F """
    if np.isnan(m): return -1
    if m > 0.75: return 4
    elif m > 0.5: return 3
    elif m > 0.25: return 2
    else: return 0

def grade_dataset(var_da, data_name):
    """ apply letter grade conversion function to given data array variable """
    if data_name == 'lp':
        grades = xr.apply_ufunc(
        np.vectorize(grade_lightpollution), var_da,
        dask="parallelized", output_dtypes=[np.int8])
        return grades.astype('int8') # force int8 to save mem
    if data_name == 'clouds':
        grades = xr.apply_ufunc(
        np.vectorize(grade_cloud), var_da,
        dask="parallelized", output_dtypes=[np.int8])
        return grades.astype('int8')
    if data_name == 'precip':
        grades = xr.apply_ufunc(
        np.vectorize(grade_precip), var_da,
        dask="parallelized", output_dtypes=[np.int8])
        return grades.astype('int8')
    if data_name == 'moon':
        grades = xr.apply_ufunc(
        np.vectorize(grade_moon), var_da,
        dask="parallelized", output_dtypes=[np.int8])
        return grades.astype('int8')
    else:
        logger = logging_setup()
        logger.error('letter grade conversion application failed')
