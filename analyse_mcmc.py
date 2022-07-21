
import numpy as np
from pathlib import Path
import run_mcmc_utils as run
import logging
from p_tqdm import p_map
from edges_cal import receiver_calibration_func as rcf

log = logging.getLogger(__name__)


def get_all_cal_curves(mcsamples, mcdef: run.MCDef, field_freq=None, nthreads=1, force=False):
    kwargs = mcdef.get_kwargs_from_mcsamples(mcsamples)
    outfile = Path(mcsamples.root + "_blobs.npz")


    if force and outfile.exists():
        log.warning(f"Overwriting {outfile} since force=True")    
    elif outfile.exists():
        return dict(np.load(outfile))
    else:
        log.warning(f"{outfile} doesn't exist, so producing it.")
        
    lk = mcdef.get_likelihood(**kwargs)
    freq = lk.nw_model.freq
    nfreq = len(freq)

    if field_freq is not None:
        freq = np.concatenate((freq, field_freq))

    # Get the EQUAL WEIGHTS samples!
    samples = np.genfromtxt(mcsamples.root + "_equal_weights.txt")[
        :, 2 : (2 + len(lk.partial_linear_model.child_active_params))
    ]

    del lk

    nper_thread = len(samples) // nthreads
    last_n = nper_thread
    if len(samples) % nthreads:
        nper_thread += 1
        last_n = len(samples) - (nthreads - 1) * nper_thread

    def do_stuff(thread):
        cal_lk = mcdef.get_likelihood(**kwargs)

        nn = last_n if thread == nthreads - 1 else nper_thread

        tcal = dict(
            tns=np.zeros((nn, len(freq))),
            tload=np.zeros((nn, len(freq))),
            tunc=np.zeros((nn, len(freq))),
            tcos=np.zeros((nn, len(freq))),
            tsin=np.zeros((nn, len(freq))),
            params=np.zeros((nn, cal_lk.partial_linear_model.linear_model.n_terms)),
        )
        start = thread * nper_thread

        for i, sample in enumerate(samples[start : start + nn]):
            out = cal_lk.get_cal_curves(params=sample, freq=freq, sample=True)
            for name, val in out.items():
                tcal[name][i] = val

        return tcal

    out = p_map(do_stuff, range(nthreads), num_cpus=nthreads)

    out_dict = {"samples": samples, "freq": freq}
    for i, name in enumerate(out[0]):
        out_dict[name] = np.concatenate([o[name] for o in out])

    if field_freq is not None:
        out_dict['tload_field'] = out_dict['tload'][:, nfreq:]
        out_dict['tns_field'] = out_dict['tns'][:, nfreq:]
        out_dict['tunc_field'] = out_dict['tunc'][:, nfreq:]
        out_dict['tcos_field'] = out_dict['tcos'][:, nfreq:]
        out_dict['tsin_field'] = out_dict['tsin'][:, nfreq:]
        
        out_dict['tload'] = out_dict['tload'][:, :nfreq]
        out_dict['tns'] = out_dict['tns'][:, :nfreq]
        out_dict['tunc'] = out_dict['tunc'][:, :nfreq]
        out_dict['tcos'] = out_dict['tcos'][:, :nfreq]
        out_dict['tsin'] = out_dict['tsin'][:, :nfreq]

    np.savez(outfile, **out_dict)

    return out_dict


def get_recalibrated_src_temps(mcdef, blobs, root, calobs, nthreads=1, force=False):
    if Path(root+"_src_temps.npz").exists() and not force:
        return np.load(root+ "_src_temps.npz")

    n = len(blobs["samples"])

    nper_thread = n // nthreads
    last_n = nper_thread
    if n % nthreads:
        nper_thread += 1
        last_n = n - (nthreads - 1) * nper_thread

    freq = calobs.freq.freq
    tload = calobs.t_load
    lna_s11 = calobs.lna_s11
    tload_ns = calobs.t_load_ns
    loads = list(calobs._loads.keys())


    s11corr = calobs.s11_correction_models
    uncal_temps = {
        name: calobs.t_load_ns * load.spectrum.averaged_Q + calobs.t_load
        for name, load in calobs._loads.items()
    }

    for name in calobs.io.s11.simulators:
        load = calobs.new_load(name)

        s11corr[name] = load.s11_model(freq)
        uncal_temps[name] = calobs.t_load_ns * load.spectrum.averaged_Q + calobs.t_load

    def put(thread):
        kwargs = mcdef.get_kwargs(Path(root).name)
        cal_lk = mcdef.get_likelihood(**kwargs)

        nn = last_n if thread == nthreads - 1 else nper_thread
        start = thread * nper_thread
        model = cal_lk.nw_model.linear_model.model

        cal_temps = np.zeros((len(loads), nn, len(freq)))

        for i in range(nn):
            tnsp = blobs["samples"][start + i]
            pset = blobs["params"][start + i]

            tns = cal_lk.t_ns_model.model.model(x=freq, parameters=tnsp)
            off = tload - model.get_model("tload", x=freq, parameters=pset)
            t_unc = model.get_model("tunc", x=freq, parameters=pset)
            t_cos = model.get_model("tcos", x=freq, parameters=pset)
            t_sin = model.get_model("tsin", x=freq, parameters=pset)

            for j, (name, load_s11) in enumerate(s11corr.items()):
                a, b = rcf.get_linear_coefficients(
                    load_s11,
                    lna_s11,
                    sca=tns / tload_ns,
                    off=off,
                    t_unc=t_unc,
                    t_cos=t_cos,
                    t_sin=t_sin,
                    t_load=tload,
                )
                cal_temps[j, i] = a * uncal_temps[name] + b

        return cal_temps

    out = p_map(put, range(nthreads), num_cpus=nthreads)

    out = np.concatenate(out, axis=1)
    out = {name: out[i] for i, name in enumerate(calobs._loads)}

    np.savez(root + '_src_temps.npz', **out)
    return out

def get_recalibrated_src_temp_best(mcdef, mcsamples, calobs, labcal):
    loads = {**calobs.loads}
    for name in calobs.metadata['io'].s11.simulators:
        loads[name] = calobs.new_load(name, io_obj = calobs.metadata['io'])

    uncal_temps = {
        name: calobs.t_load_ns * load.spectrum.averaged_Q + calobs.t_load
        for name, load in loads.items()
    }
    uncal_vars = {
        name: calobs.t_load_ns**2 * load.spectrum.variance_Q / load.spectrum.n_integrations
        for name, load in loads.items()
    }

    kwargs = mcdef.get_kwargs(Path(mcsamples.root).parent.name)
    cal_lk = mcdef.get_likelihood(**kwargs)

    cal_temps = {}
    # NOTE: loglikes in mcsamples is actually -2*loglike :/
    pbest = mcsamples.samples[np.argmax(-mcsamples.loglikes)]
    n = len(cal_lk.partial_linear_model.child_active_params)

    for name, load in loads.items():
        cal_temps[name] = {}

        a, b = cal_lk.get_linear_coefficients(freq=calobs.freq.freq.to_value("MHz"), labcal=labcal, load=load, params=pbest[:n])
        cal_temps[name]['a'] = a
        cal_temps[name]['b'] = b
        cal_temps[name]['cal_temp'] = a*uncal_temps[name] + b
        cal_temps[name]['cal_var'] = a**2 * uncal_vars[name]
        cal_temps[name]['uncal_var'] = uncal_vars[name]
        
    return cal_temps
