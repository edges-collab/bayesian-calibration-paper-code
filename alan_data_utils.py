"""
Import/read data from Alan, and create edges-cal objects to be close to Alan's data.

The functions here are based on the notebook which is encoded in Memo 199.
"""

from __future__ import annotations
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
import alan_data
from edges_cal.cal_coefficients import HotLoadCorrection
from edges_analysis.analysis.s11 import AntennaS11
import yaml
from astropy import units as u
import pickle

with open(Path('~/.edges.yml').expanduser(), 'r') as fl:
    CALPATH = Path(yaml.safe_load(fl)['cal']['raw-cals']).expanduser() / 'Receiver01'

CALOBS_PATH = CALPATH / 'Receiver01_25C_2015_09_02_040_to_200MHz'
here = Path(__file__).parent

def make_absorption(freq, fix=tuple()):
    params = {
        "A": {"max": 2, "min": 0.05, "fiducial": 0.5},
        "nu0": {"min": 60, "max": 90, "fiducial": 78.5},
        "tau": {"min": 1, "max": 40, "fiducial": 7},
        "w": {"min": 1, "max": 25, "fiducial": 15},
    }

    fid = {p: params.pop(p)["fiducial"] for p in fix}
    
    if isinstance(fix, dict):
        for p, v in fix.items():
            fid[p] = v

    return AbsorptionProfile(name="absorption", fiducial=fid, params=params, freqs=freq)


def get_labcal(calobs,raw=False, use_spline=False):
    if use_spline:
        return LabCalibration(
            calobs=calobs, 
            antenna_s11_model=AntennaS11(
                freq=calobs.freq, raw_s11=alan_data.ant_s11, use_spline=True, 
                complex_model_type=mdl.ComplexRealImagModel,
                internal_switch=calobs.internal_switch
            )
        )

    if raw:
        s11_files = alan_data.S11_FILES
    else:
        s11_files = str(alan_data.unmodeled_ant_s11_file)
        
    return LabCalibration.from_s11_files(calobs=calobs, s11_files=s11_files)


def get_calio():
    return io.CalibrationObservation(
        CALOBS_PATH, 
        run_num={"receiver_reading":  (1,2,4,5,6,8)},
        repeat_num=1
    )

def get_calobs(cterms: int=6, wterms: int=5, smooth: int=8, use_alan_s11: bool=True, use_spline: bool = False, **kwargs):
    calio = get_calio()

    calobs = CalibrationObservation.from_io(
        calio,
        f_low=50.0*u.MHz,          # Sets the final frequency range 
        f_high=100.0*u.MHz,        # used in calibration
        cterms=cterms,
        wterms=wterms,
        freq_bin_size=smooth,      # Equivalent to Alan's "smooth" parameter
        spectrum_kwargs= {
            "default": {
                "f_low": 40*u.MHz,             # Frequency range for reading/smoothing
                "f_high": 110*u.MHz,           # the spectra, from which a subset is used for calibration
                "t_load": 300,
                "t_load_ns": 1000, 
                "ignore_times_percent": 7200,  # Since it's >100, sets the number of seconds to ignore  
                'frequency_smoothing': 'gauss',# Smooth with gaussian filter like Alan. Prefer 'bin' usually.
                'time_coordinate_swpos': 0,    # Ignore times based on swpos=0 timings.
            },
            'ambient': {'temperature': 296.0},  # Set the assumed temperature of the loads.
            'hot_load': {"temperature": 399.0}, # edges-cal by default uses the thermistor
            'open': {'temperature': 296.0},     # to read the actual temperature, instead of
            'short': {'temperature': 296.0}     # assuming it. This makes a reasonably large difference.
        },
        s11_kwargs={
            "default": {
                'model_type': mdl.Fourier,       # Use Fourier model for all loads.
                'complex_model_type': mdl.ComplexRealImagModel,  # Fit on real/imag instead of abs/phase
                'model_transform': mdl.ZerotooneTransform(range=(50, 100)),  # Alan uses (0, 1) range of freq.
                'model_kwargs': {'period': 1.5}, # Alan uses 2pi/1.5 in his cos/sin terms
                'n_terms': 27,                   # corresponds to nfit2 in alans pipeline
            },
            'hot_load': {'model_delay': 2*np.pi*5e-10*u.s},  # Alan fits for a delay. We just use his output value.
            'ambient': {'model_delay': 2*np.pi*6e-10*u.s},   # this value is printed out in his updated pipeline.
            'open': {'model_delay': 2*np.pi*6.09e-8*u.s},
            'short': {'model_delay': 2*np.pi*6.09e-8*u.s},
        },
        receiver_kwargs = {
            'n_terms': 11,           # Use 11 terms
            'model_type': 'fourier', # Alan used Fourier series on receiver
            'model_transform': mdl.ZerotooneTransform(range=(50,100)),  # See above for what these represent.
            'model_kwargs': {'period': 1.5},
            'complex_model_type': mdl.ComplexRealImagModel,
            'model_delay': -2*np.pi*1e-9*u.s
        },
        hot_load_loss_kwargs = {
            'model': mdl.Fourier(n_terms=27, period=1.5, transform=mdl.ZerotooneTransform(range=(50,100))),
            'complex_model': mdl.ComplexRealImagModel,
        }
    )
    
    if use_alan_s11:
        if use_spline:
            hlc = HotLoadCorrection(
                freq=FrequencyRange(alan_data.modeled_s11s['freq']*u.MHz), 
                raw_s11=alan_data.modeled_s11s['sr_s11_re']+ 1j*alan_data.modeled_s11s['sr_s11_im'],
                raw_s12s21=alan_data.modeled_s11s['sr_s12_re']+ 1j*alan_data.modeled_s11s['sr_s12_im'],
                raw_s22=alan_data.modeled_s11s['sr_s22_re']+ 1j*alan_data.modeled_s11s['sr_s22_im'],
                use_spline=True,
                complex_model=mdl.ComplexRealImagModel
            )

            mfreq=FrequencyRange(alan_data.modeled_s11s['freq']*u.MHz)
            return calobs.clone(
                receiver=attr.evolve(
                    calobs.receiver, 
                    freq = mfreq,
                    raw_s11=alan_data.modeled_s11s['lna_re'] + 1j*alan_data.modeled_s11s['lna_im'],
                    use_spline=True
                ),
                loads={
                    name: attr.evolve(
                        load, 
                        reflections=attr.evolve(
                            load.reflections,
                            use_spline=True,
                            freq=mfreq,
                            raw_s11=alan_data.modeled_s11s[name+"_re"] + 1j*alan_data.modeled_s11s[name+"_im"]
                        ),
                        loss_model=hlc if name=='hot_load' else None
                    )
                    for name, load in calobs.loads.items()
                }
            )
        else:
            hlc = HotLoadCorrection(
                freq=FrequencyRange(alan_data.unmodeled_s11s['freq']*u.MHz), 
                raw_s11=alan_data.unmodeled_s11s['semi_rigid_s11'],
                raw_s12s21=alan_data.unmodeled_s11s['semi_rigid_s12'],
                raw_s22=alan_data.unmodeled_s11s['semi_rigid_s22'],
                model = mdl.Fourier(n_terms=27, period=1.5, transform=mdl.ZerotooneTransform(range=(50,100))),
                complex_model=mdl.ComplexRealImagModel,
                
            )

            afreq = FrequencyRange(alan_data.unmodeled_s11s['freq']*u.MHz)

            return calobs.clone(
                receiver=attr.evolve(
                    calobs.receiver, 
                    raw_s11=alan_data.unmodeled_s11s['lna'], 
                    freq=afreq,
                ),
                loads={
                    name: attr.evolve(
                        load, 
                        reflections=attr.evolve(
                            load.reflections,
                            raw_s11=alan_data.unmodeled_s11s[name], 
                            freq=afreq,
                        ),
                        loss_model=hlc if name=='hot_load' else None
                    )
                    for name, load in calobs.loads.items()
                }
            )


def decalibrate(labcal, f_sky, t_sky):
    """Convert tsky into Q."""
    ra, rb = labcal.get_linear_coefficients()
    ra = spline(labcal.calobs.freq.freq.to_value("MHz"), ra)(f_sky)
    rb = spline(labcal.calobs.freq.freq.to_value("MHz"), rb)(f_sky)

    return ((alan_data.loss * alan_data.bmcorr) * t_sky + alan_data.loss_temp - rb - ra * 300) / (1000 * ra)


def get_cal_lk(calobs, tns_width=500, est_tns=None, ignore_sources=(), as_sim=(), **kwargs):
    return NoiseWaveLikelihood.from_calobs(
        calobs,
        t_ns_params=get_tns_params(calobs, tns_width=tns_width, est_tns=est_tns, cterms=kwargs.get('cterms', None)),
        derived=(
            "logdet_cinv",
            "logdet_sig",
            "rms",
            "tunchat",
            "tcoshat",
            "tsinhat",
            "tloadhat",
            "rms_parts",
        ),
        sources=[src for src in calobs.load_names if src not in ignore_sources],
        as_sim=as_sim,
        **kwargs
    )


def recalibrate(labcal, t_sky, f_sky, cal_lk=None, cal_optx=None, with_same=False, a=None, b=None):
    f = labcal.calobs.freq.freq
    if with_same:
        q = decalibrate(labcal, f_sky, t_sky)
    else:
        q = alan_data.decalibrate(t_sky, f_sky)

    if a is None or b is None:
        if cal_lk is not None:
            if cal_optx is None:
                cal_optx = run_map(cal_lk.partial_linear_model).x
            a, b = cal_lk.get_linear_coefficients(
                freq=f_sky, labcal=labcal, params=cal_optx
            )
        else:
            a, b = labcal.get_linear_coefficients()

            a = spline(f, a)(f_sky)
            b = spline(f, b)(f_sky)

    return (labcal.calobs.t_load_ns * a * q + a * 300 + b - alan_data.loss_temp) / (alan_data.bmcorr * alan_data.loss)


def get_var_q(fsky, qant, n_terms=25) -> np.ndarray:
    fit = Polynomial(
        n_terms=n_terms, transform=UnitTransform(range=(fsky.min(), fsky.max()))
    ).fit(fsky, ydata=qant)
    return np.ones_like(qant) * np.var(fit.residual)  # the value for 25 terms.


def get_tns_params(calobs, tns_width=100, est_tns=None, cterms=None):
    # has to be an actual calobs
    assert isinstance(calobs, CalibrationObservation)
    cterms = cterms or calobs.cterms

    zero = est_tns is not None and np.all(est_tns == 0)

    if est_tns is None or zero:
        est_tns = calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns

    if zero:
        mean = np.zeros(calobs.cterms)
        mean[0] = est_tns[0]
    else:
        mean = est_tns

    if len(est_tns) < cterms:
        est_tns = np.concatenate((est_tns, [0]*(cterms - len(est_tns))))
        mean = np.concatenate((mean, [0]*(cterms - len(mean))))
    elif len(est_tns) > cterms:
        est_tns = est_tns[:cterms]
        mean = mean[:cterms]

    return ParamVec(
        "t_lns",
        length=cterms,
        min=mean - tns_width,
        max=mean + tns_width,
        fiducial=est_tns,
        ref=[stats.norm(v, scale=1.0) for v in est_tns],
    )


def get_tns_params_direct(tns_fid, tns_width):
    if not hasattr(tns_width, "__len__"):
        tns_width = tns_width * np.ones_like(tns_fid)

    return ParamVec(
        "t_lns",
        length=len(tns_fid),
        min=tns_fid - tns_width,
        max=tns_fid + tns_width,
        fiducial=tns_fid,
        ref=[stats.norm(v, scale=w / 10) for v, w in zip(tns_fid, tns_width)],
    )


def get_likelihood(
    labcal,
    calobs,
    fsky,
    fg=LinLog(n_terms=5),
    eor=None,
    var_terms=25,
    cal_noise="data",
    tns_width=100,
    as_sim=(),
    est_tns=None,
    include_antsim=False,
    ignore_sources=(),
    s11_systematic_params=(),
    cterms=None, 
    wterms=None,
    sim_sky=False,
    add_noise=True,
    seed=1234,
    remove_eor:bool=False,
):
    qant_var = get_var_q(fsky, alan_data.sky_q, n_terms=var_terms)

    if eor is None:
        eor = make_absorption(fsky)

    sources = [src for src in calobs.load_names if src not in ignore_sources]
    loads = {name: calobs.loads[name] for name in sources}

    if include_antsim:
        for name in calobs.metadata["io"].s11.simulators:
            loads[name] = calobs.new_load(name, io_obj=calobs.metadata["io"])

    sources = tuple(loads.keys())

    if sim_sky:
        # Do a kinda dumb thing... fit just the FG model to the sky data, and get those
        # parameters as fiducial parameters for the fg model.
        p = fg.fit(xdata=alan_data.sky_freq, ydata=alan_data.sky_temp).model_parameters
        sky_q = decalibrate(labcal, t_sky=fg(x=alan_data.sky_freq, parameters=p) + eor()['eor_spectrum'], f_sky=alan_data.sky_freq)
        fg = fg.with_params(p)
    else:
        sky_q = alan_data.sky_q

    if remove_eor:
        teor = eor()['eor_spectrum']
        qeor = decalibrate(labcal, t_sky=teor, f_sky=alan_data.sky_freq)
        sky_q = sky_q - qeor
        
    return DataCalibrationLikelihood.from_labcal(
        labcal,
        calobs,
        loads=loads,
        field_freq=fsky*u.MHz,
        q_ant=sky_q,
        qvar_ant=qant_var,
        fg_model=fg,
        eor_components=(eor,),
        as_sim=as_sim,
        cal_noise=cal_noise,
        t_ns_params=get_tns_params(calobs, tns_width=tns_width, est_tns=est_tns),
        loss=alan_data.loss,
        loss_temp=alan_data.loss_temp,
        bm_corr=alan_data.bmcorr,
        s11_systematic_params=s11_systematic_params,
        cterms=cterms,
        wterms=wterms,
        add_noise=add_noise,
        seed=seed
    )


def get_isolated_likelihood(
    labcal, calobs, fsky, fg=LinLog(n_terms=5), eor=None, ml_solution=None, mcdef=None,
):
    qant_var = get_var_q(fsky, alan_data.sky_q)

    if eor is None:
        eor = make_absorption(fsky)

    if ml_solution:
        with open(ml_solution, 'rb') as fl:
            optx = pickle.load(fl)['optres'].x

        cal_lk = mcdef.get_likelihood_from_label(Path(ml_solution).parent.name)
    else:
        optx = None

    recal_tsky = recalibrate(labcal, t_sky=alan_data.sky_data["t_ant"], f_sky=fsky, cal_lk=cal_lk, cal_optx=optx)

    return LinearFG(
        freq=fsky,
        t_sky=recal_tsky,
        var=qant_var * (labcal.calobs.C1(fsky*u.MHz) * labcal.calobs.t_load_ns) ** 2,
        fg=fg,
        eor=eor,
    )


def view_results(lk, res_data, calobs, slf=None):
    eorspec = lk.partial_linear_model.get_ctx(params=res_data.x)

    fig, ax = plt.subplots(2, 2, figsize=(15, 7), sharex=True)

    sim_tns = calobs.C1() * calobs.t_load_ns

    nu = calobs.freq.freq

    ax[0, 0].plot(nu, eorspec["tns"], label="Simultaneous")
    ax[0, 0].plot(nu, sim_tns, label="Isolated (Fid.)")
    ax[1, 0].plot(nu, eorspec["tns"] - sim_tns, color="k", label=r"$\Delta T_{\rm NS}$")
    ax[0, 0].set_title(r"$T_{\rm NS}$")
    ax[0, 0].set_ylabel("Temperature [K]")

    nu = lk.nwfg_model.field_freq

    if slf is None:
        fideor = fid_eor()["eor_spectrum"]
    else:
        res = slf()
        fideor = slf.get_eor(res.x) + slf.get_resid(res.x)

    ax[0, 1].plot(nu, eorspec["eor_spectrum"])
    ax[0, 1].plot(nu, fideor, label="")
    ax[0, 1].set_title(r"$T_{21}$")
    ax[1, 1].plot(nu, eorspec["eor_spectrum"] - fideor, color="k")
    ax[1, 0].set_ylabel("Difference [K]")

    ax[1, 0].set_xlabel("Frequency")
    ax[1, 1].set_xlabel("Frequency")

    ax[0, 0].legend()
    ax[1, 0].legend()




def get_percs(arr):
    arr = arr.reshape((-1, arr.shape[-1]))
    return np.percentile(arr, [4, 16, 50, 84, 96], axis=0)


def plot_regions(ax, x, percs, afid=0, rfid=1, color="C0", label=None, fills=True):
    if isinstance(afid, int) and isinstance(rfid, np.ndarray):
        afid = rfid

    if fills:
        ax.fill_between(
            x,
            (percs[0] - afid) / rfid,
            (percs[-1] - afid) / rfid,
            color=color,
            alpha=0.2,
            lw=0,
            label=r"2-$\sigma$" if label is None else None,
        )
        ax.fill_between(
            x,
            (percs[1] - afid) / rfid,
            (percs[-2] - afid) / rfid,
            color=color,
            alpha=0.5,
            lw=0,
            label=r"1-$\sigma$" if label is None else None,
        )
    if percs.ndim == 1:
        # plot single index (ideally ML solution)
        (line,) = ax.plot(x, (percs - afid) / rfid, color=color, label=label or r"ML")
    else:
        # plot the median
        (line,) = ax.plot(
            x, (percs[2] - afid) / rfid, color=color, label=label or r"Median"
        )
    ax.set_xlim(x.min(), x.max())

    return line


def plot_cal_solutions(blobs, calobs, fig=None, ax=None, cidx=0, label=None):

    if fig is None:
        fig, ax = plt.subplots(5, 1, sharex=True)

    for i, name in enumerate(["tns", "tload", "Tunc", "Tcos", "Tsin"]):
        # Hack to ensure calibration likelihoods work.
        if name == "tns" and name.lower() not in blobs:
            name = "t_lns"

        percs = get_percs(blobs[name.lower()])
        if name == "tns":
            fid = calobs.C1(blobs["freq"]) * calobs.t_load_ns
        elif name == "tload":
            fid = calobs.t_load - calobs.C2(blobs["freq"])
        else:
            fid = getattr(calobs, name)(blobs["freq"])

        plot_regions(
            ax[i], blobs["freq"], percs, afid=fid, color=f"C{cidx}", label=label
        )
        ax[i].set_ylabel(r"$\Delta T_{\rm %s}$" % name[1:])

    if label is None:
        ax[0].legend(ncol=3)
    ax[-1].set_xlabel("Frequency [MHz]")

    return fig, ax


def plot_recalibrated_sources(
    src_temps: Dict[str, np.ndarray],
    lk,
    calobs,
    fills=True,
    index: int | None = None,
    fig=None,
    ax=None,
    cidx=0,
    with_rms=True,
    smooth: int = 1,
    std_units: bool=False,
):
    if fig is None:
        fig, ax = plt.subplots(len(src_temps), 1, sharex=True, figsize=(8, 7))

    for i, (name, data) in enumerate(src_temps.items()):
        if name in calobs.load_names:
            load = getattr(calobs, name)
        else:
            load = calobs.new_load(name)

        temps = data['cal_temp']

        # Use only a single recalibrated temperature from the posterior.
        if index is not None or temps.ndim == 1:
            temps = temps[index]

        if smooth > 1:
            freq, temps, _ = averaging.bin_array_unbiased_irregular(
                temps, coords=calobs.freq.freq, bins=smooth
            )
            if freq.ndim > 1:
                freq = freq[0]
        else:
            freq = calobs.freq.freq

        if index is not None or temps.ndim == 1:
            percs = temps
            best = percs
        else:
            percs = get_percs(temps)
            best = percs[2]

        if np.isscalar(load.temp_ave):
            fid = load.temp_ave * np.ones(temps.shape[-1])
        else:
            _, fid, _ = averaging.bin_array_unbiased_irregular(load.temp_ave, coords=calobs.freq.freq, bins=smooth)
            
        vr = data['cal_var'] / smooth
        vr = spline(calobs.freq.freq, vr)(freq)
            
        lbl = ""
        if with_rms:
            rms = np.sqrt(np.mean(np.square(best - fid) / vr))
            lbl += f"{rms:.2f}"

        line = plot_regions(
            ax[i], freq, percs, afid=fid, rfid=np.sqrt(vr) if std_units else 1, color=f"C{cidx}", label=lbl, fills=fills
        )

        ax[i].set_ylabel(name)

    ax[-1].set_xlabel("Frequency [MHz]")

    return fig, ax, line


def plot_multi_recalibrated_sources(
    src_temps: dict[str, dict], lks, calobs, fig=None, ax=None, labeller=None, **kwargs
):

    lines = []
    names = []
    for i, (name, srctmp) in enumerate(src_temps.items()):
        fig, ax, line = plot_recalibrated_sources(
            srctmp, lks[name], calobs, cidx=i, fig=fig, ax=ax, **kwargs
        )
        lines.append(line)
        if labeller is not None:
            name = labeller(name)

        names.append(name)

    for axx in ax:
        axx.legend(ncol=6)

    fig.legend(lines, names, "upper right")
    return fig, ax


def plot_linear_coeffs(blobs, labcal, fig=None, ax=None, label=None, cidx=0):
    if fig is None:
        fig, ax = plt.subplots(3, 1, sharex=True)

    a, b = labcal.get_linear_coefficients(freq=alan_data.sky_data["freq"])

    if "a" not in blobs:
        x = np.linspace(50, 100, 200)
        blobs_a = np.zeros((len(blobs["tunc"]), len(alan_data.sky_freq)))
        blobs_b = np.zeros((len(blobs["tunc"]), len(alan_data.sky_freq)))

        ant_s11 = labcal.antenna_s11_model(freq=alan_data.sky_freq)
        lna_s11 = labcal.calobs.lna.s11_model(alan_data.sky_freq)
        for i, (tns, tload, tunc, tcos, tsin) in enumerate(
            zip(
                blobs["tns"],
                blobs["tload"],
                blobs["tunc"],
                blobs["tcos"],
                blobs["tsin"],
            )
        ):
            blobs_a[i], blobs_b[i] = rcf.get_linear_coefficients(
                gamma_ant=ant_s11,
                gamma_rec=lna_s11,
                sca=spline(x, tns / labcal.calobs.t_load_ns)(alan_data.sky_freq),
                off=spline(x, labcal.calobs.t_load - tload)(alan_data.sky_freq),
                t_unc=spline(x, tunc)(alan_data.sky_freq),
                t_cos=spline(x, tcos)(alan_data.sky_freq),
                t_sin=spline(x, tsin)(alan_data.sky_freq),
            )
        blobs["a"] = blobs_a * labcal.calobs.t_load_ns
        blobs["b"] = blobs_b + blobs_a * labcal.calobs.t_load
        blobs["recal_sky"] = recalibrate(
            labcal,
            t_sky=alan_data.sky_data["t_ant"],
            f_sky=alan_data.sky_data["freq"],
            a=blobs_a,
            b=blobs_b,
        )

    percs = get_percs(blobs["a"])
    plot_regions(
        ax[0],
        alan_data.sky_freq,
        percs,
        afid=a * labcal.calobs.t_load_ns,
        color=f"C{cidx}",
        label=label,
    )
    percs = get_percs(blobs["b"])
    plot_regions(
        ax[1],
        alan_data.sky_freq,
        percs,
        afid=b + a * labcal.calobs.t_load,
        label=label,
        color=f"C{cidx}",
    )
    percs = get_percs(blobs["recal_sky"])
    plot_regions(
        ax[2],
        alan_data.sky_freq,
        percs,
        afid=recalibrate(labcal, t_sky=alan_data.sky_data["t_ant"], f_sky=alan_data.sky_freq),
        color=f"C{cidx}",
        label=label,
    )

    ax[0].set_ylabel(r"$\Delta T_0$ [K]")
    ax[1].set_ylabel(r"$\Delta T_1$ [K]")
    ax[2].set_ylabel(r"$\Delta T_{\rm sky}$ [K]")
    ax[-1].set_xlabel("Frequency [MHz]")
    ax[0].legend(ncol=3)


def plot_sky_models(blobs, fig=None, ax=None, cidx=0, label=None):
    if fig is None:
        fig, ax = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"hspace": 0.05, "wspace": 0.05}
        )

    percs = get_percs(blobs["t21"])
    plot_regions(ax[0], alan_data.sky_freq, percs, color=f"C{cidx}", label=label)

    percs = get_percs(blobs["resids"])
    plot_regions(ax[1], alan_data.sky_freq, percs, color=f"C{cidx}", label=label)
    ax[0].set_ylabel(r"$T_{21}$ [K]")
    ax[1].set_ylabel(r"Resid. Temp [K]")

    ax[0].legend(ncol=1)


def get_evidence(root):
    with open(root + ".stats") as fl:
        for line in fl.readlines():
            if line.startswith("log(Z)"):
                return float(line.split("=")[1].split("+/-")[0].strip())


calobs = get_calobs()
labcal = get_labcal(calobs, use_spline=True)
fid_eor = make_absorption(alan_data.sky_data["freq"])


