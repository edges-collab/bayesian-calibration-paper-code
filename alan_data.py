"""
Import/read data that came directly from Alan.

The functions here are based on the notebook which is encoded in Memo 199.
"""

from __future__ import annotations
from multiprocessing.spawn import get_command_line
from pyexpat import model
import h5py
import matplotlib.pyplot as plt
import numpy as np
from edges_cal import modelling as mdl
from pathlib import Path
from edges_cal import receiver_calibration_func as rcf
from yabf import ParamVec
from edges_analysis.analysis.calibrate import LabCalibration
from edges_estimate.eor_models import AbsorptionProfile
from edges_cal.modelling import LinLog, Polynomial, UnitTransform
from edges_estimate.likelihoods import (
    DataCalibrationLikelihood,
    LinearFG,
    NoiseWaveLikelihood,
)
from edges_cal import CalibrationObservation
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from typing import Dict
from yabf import run_map
from edges_analysis.analysis import averaging
from edges_io import io
from astropy import units as u
import attr
from edges_cal.tools import FrequencyRange

here = Path(__file__).parent
data = here / "alan-data"

S11_FILES = sorted(
    Path("/data5/edges/data/S11_antenna/low_band/20151207").glob(
        "run_0001_*20151208_0314*.s1p"
    )
)

def get_unmodeled_s11s():
    alan_s11s = np.genfromtxt(data / "s11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt")

    return {
        'freq': alan_s11s[:, 0],
        'lna': alan_s11s[:, 1] + 1j*alan_s11s[:, 2],
        'ambient': alan_s11s[:, 3] + 1j*alan_s11s[:, 4],
        'hot_load': alan_s11s[:, 5] + 1j*alan_s11s[:, 6],
        'open': alan_s11s[:, 7] + 1j*alan_s11s[:, 8],
        'short': alan_s11s[:, 9] + 1j*alan_s11s[:, 10],
        'semi_rigid_s11': alan_s11s[:, 11] + 1j*alan_s11s[:, 12],
        'semi_rigid_s12': alan_s11s[:, 13] + 1j*alan_s11s[:, 14],
        'semi_rigid_s22': alan_s11s[:, 15] + 1j*alan_s11s[:, 16],
        'antsim1': alan_s11s[:, 17] + 1j*alan_s11s[:, 18],
        'antsim2': alan_s11s[:, 19] + 1j*alan_s11s[:, 20],
    }

def get_modeled_s11s():
    """Note that 'all_modeled_s11s.txt' is produced by Alan's updated pipeline."""
    modeled_s11s = np.genfromtxt(
        data/ "all_modeled_s11s.txt",
        dtype=[
            ('freq', float),
            ('lna_re', float),
            ('lna_im', float),
            ('ambient_re', float),
            ('ambient_im', float),
            ('hot_load_re', float),
            ('hot_load_im', float),
            ('open_re', float),
            ('open_im', float),
            ('short_re', float),
            ('short_im', float),
            ('sr_s11_re', float),
            ('sr_s11_im', float),
            ('sr_s12_re', float),
            ('sr_s12_im', float),
            ('sr_s22_re', float),
            ('sr_s22_im', float),
        ]
    )
    mask = (modeled_s11s['freq'] >= 50.0) & (modeled_s11s['freq'] <= 100.0)
    modeled_s11s = modeled_s11s[mask]

    return modeled_s11s

def get_sky_data() -> Dict[str, np.ndarray]:
    alan_sky_data = np.genfromtxt(data / "spe0.txt")

    flg = alan_sky_data[:, 12] > 0
    alan_sky_data = {
        "freq": alan_sky_data[:, 1][flg],
        "t_ant": alan_sky_data[:, 3][flg],
    }
    return alan_sky_data


def get_losses(freq):
    loss = np.genfromtxt(data /"loss_file.txt")
    loss_fq = loss[:, 0]
    loss_temp = spline(loss_fq, loss[:, 2])(freq)
#    bmcorr = spline(loss_fq, loss[:, -1])(freq)
    loss = spline(loss_fq, loss[:, 1])(freq)

    # This is the new "averaged" beam correction (over days)
    bmcorr = np.genfromtxt(data / "beamcorr.txt")
    bmcorr = spline(bmcorr[:, 0], bmcorr[:, 1])(freq)
    return loss, loss_temp, bmcorr



def get_alan_cal(f=None):
    # Note that specal_final_case2.txt is equivalent to specal.txt
    alan_cal = np.genfromtxt(data / "specal_final_case2.txt")

    _f = alan_cal[:, 1]

    mask = (_f >= 50.0) & (_f <= 100.0)
    
    _f = _f[mask]
    alan_cal = alan_cal[mask]

    if f is None:
        f = _f

    # To make it easier to compare to our results, we spline it.
    alan_cal = {
        "freq": f,
        "lna_s11": spline(_f, alan_cal[:, 3])(f) + 1j * spline(_f, alan_cal[:, 4])(f),
        "scale": spline(_f, alan_cal[:, 6])(f),
        "offset": spline(_f, alan_cal[:, 8])(f),
        "tunc": spline(_f, alan_cal[:, 10])(f),
        "tcos": spline(_f, alan_cal[:, 12])(f),
        "tsin": spline(_f, alan_cal[:, 14])(f),
        "weight": spline(_f, alan_cal[:, 16])(f),
    }
    return alan_cal


def get_ant_s11(freq=None):
    alan_ant_s11 = np.genfromtxt(data / "antenna_s11_file.txt")
    f = alan_ant_s11[:, 0]
    if freq is None:
        freq = f

    rl = spline(f, alan_ant_s11[:, 1])
    im = spline(f, alan_ant_s11[:, 2])
    return rl(freq) + 1j * im(freq), rl, im


def decalibrate(t_sky, f_sky):
    """Convert tsky into Q."""
    ra, rb = rcf.get_linear_coefficients(
        gamma_ant=ant_s11,
        gamma_rec=calibration["lna_s11"],
        sca=calibration["scale"],
        off=calibration["offset"],
        t_unc=calibration["tunc"],
        t_cos=calibration["tcos"],
        t_sin=calibration["tsin"],
        t_load=300,
    )
    
    ra = spline(cal_freq, ra)(f_sky)
    rb = spline(cal_freq, rb)(f_sky)

    return ((loss * bmcorr) * t_sky + loss_temp - rb - ra * 300) / (1000 * ra)

modeled_s11s = get_modeled_s11s()
unmodeled_s11s = get_unmodeled_s11s()


sky_data = get_sky_data()
loss, loss_temp, bmcorr = get_losses(sky_data["freq"])
calibration = get_alan_cal()
ant_s11, _ant_s11_rl, ant_s11_im = get_ant_s11()

unmodeled_ant_s11_file = data / "S11_blade_low_band_2015_342_03_14.txt.csv"
sky_freq = sky_data['freq']
cal_freq = calibration['freq']
s11_freq = unmodeled_s11s['freq']
sky_temp = sky_data['t_ant']

sky_q = decalibrate(sky_data['t_ant'], sky_data['freq'])


def ant_s11_func(freq):
    return _ant_s11_rl(freq) + 1j * ant_s11_im(freq)
