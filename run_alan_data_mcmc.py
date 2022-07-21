import click
import alan_data_utils as utils
import alan_data as adata
from edges_cal.modelling import LinLog
from pathlib import Path
import numpy as np
from p_tqdm import p_map
import run_alan_precal_mcmc as precal
import run_alan_cal_mcmc as cal
import run_mcmc_utils as run

main = click.Group()

def get_likelihood(
    nterms_fg, fix_tau, simultaneous, cterms, wterms, smooth, tns_width, 
    fit_cterms, fit_wterms, nscale, ndelay, est_tns=None, ignore_sources=(), as_sim=(), 
    s11_sys=(), antsim=False, sim_sky=False, add_noise=True, seed=1234, remove_eor=False,
):
    calobs = utils.get_calobs(cterms=cterms, wterms=wterms, smooth=smooth)
    labcal = utils.get_labcal(calobs)

    s11_systematic_params = run.define_s11_systematics(s11_sys, ndelay=ndelay, nscale=nscale)
    if simultaneous:
        return utils.get_likelihood(
            labcal, 
            calobs,
            fsky=adata.sky_data['freq'],
            fg=LinLog(n_terms=nterms_fg),
            eor=utils.make_absorption(adata.sky_data['freq'], fix=('tau',) if fix_tau else ()),
            tns_width=tns_width,
            ignore_sources=ignore_sources, 
            as_sim=as_sim, 
            s11_systematic_params=s11_systematic_params, 
            est_tns=est_tns,
            include_antsim=antsim,
            cterms=fit_cterms,
            wterms=fit_wterms,
            sim_sky=sim_sky,
            add_noise=add_noise,
            seed=seed,
            remove_eor=remove_eor,
        )
    else:
        return utils.get_isolated_likelihood(
            labcal, 
            calobs,
            fsky=adata.sky_data['freq'],
            fg=LinLog(n_terms=nterms_fg), 
            eor=utils.make_absorption(adata.sky_data['freq'], fix=('tau',) if fix_tau else ()),
        )

mcdef = run.MCDef(
    label_formats = (
        "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim{antsim}_fg{nterms_fg}_simul{simultaneous}_taufx{fix_tau}_ns{nscale:02d}_nd{ndelay:02d}_sd{seed:d}",
        "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim{antsim}_fg{nterms_fg}_simul{simultaneous}_taufx{fix_tau}_ns{nscale:02d}_nd{ndelay:02d}_sd{seed:d}_rmeor{remove_eor}",
    ),
    folder= 'alan_field_and_cal',
    default_kwargs={
        **{k: v for k,v in precal.mcdef.default_kwargs.items() if k not in ('unweighted', 'cable_noise_factor')},
        **{
            'antsim': False,
            'add_noise': True,
            'seed': 1234,
            'remove_eor': False,
        }
    },
    get_likelihood=get_likelihood,
)



def get_linear_distribution(mcdef, mcsamples,  nthreads=1):
    freq = np.linspace(50, 100, 200)
    kwargs = mcdef.get_kwargs(Path(mcsamples.root).parent.name)

    outfile = Path(mcsamples.root + "_blobs.npz")
    if outfile.exists():
        return dict(np.load(outfile))
    samples = mcsamples.samples

    nper_thread = len(samples) // nthreads
    last_n = nper_thread
    if len(samples) % nthreads:
        nper_thread += 1
        last_n = len(samples) - (nthreads-1)*nper_thread
    
    iso = 'isolated' in mcsamples.root

    def do_stuff(thread):
        lk = mcdef.get_likelihood(**kwargs)

        nn = last_n if thread == nthreads-1 else nper_thread
        model = lk.partial_linear_model.linear_model.model

        out = dict(
            params = np.zeros((nn, model.n_terms)),
            tfg = np.zeros((nn, len(adata.sky_data['freq']))),
            t21 = np.zeros((nn, len(adata.sky_data['freq']))),
            resids = np.zeros((nn, len(adata.sky_data['freq']))),
        )
        
        if not iso:
            out = {
                **out,
                **dict(
                    tns = np.zeros((nn, len(freq))),
                    tload = np.zeros((nn, len(freq))),
                    tunc = np.zeros((nn, len(freq))),
                    tcos = np.zeros((nn, len(freq))),
                    tsin = np.zeros((nn, len(freq))),
                    a = np.zeros((nn, len(adata.sky_data['freq']))),
                    b = np.zeros((nn, len(adata.sky_data['freq']))),
                    recal_sky = np.zeros((nn, len(adata.sky_data['freq']))),
                )
            }
            tns_model = lk.t_ns_model.model.model.at(x=freq)

        start = thread*nper_thread
        
        for i, sample in enumerate(samples[start:start+nn]):
            if len(sample) != len(lk.partial_linear_model.child_active_params):
                p = sample[-model.n_terms:]
                sample = sample[:len(lk.partial_linear_model.child_active_params)]
                ctx = lk.partial_linear_model.get_ctx(params=sample)
            else:
                ctx = lk.partial_linear_model.get_ctx(params=sample)
                fit = lk.partial_linear_model.reduce_model(ctx=ctx, params=sample)[0]
                p = fit.get_sample()[0]

            out['params'][i] = p
            if not iso:
                out['tns'][i] = tns_model(parameters=sample[:tns_model.n_terms])
                out['tunc'][i] = model.get_model('tunc', x=freq, parameters=p)
                out['tcos'][i] = model.get_model('tcos', x=freq, parameters=p)
                out['tsin'][i] = model.get_model('tsin', x=freq, parameters=p)
                out['tload'][i] = model.get_model('tload', x=freq, parameters=p)
                out['a'][i], out['b'][i] = a, b = lk.get_linear_coefficients(ctx=ctx, linear_params=p)
                out['recal_sky'][i] = lk.recalibrated_sky_temp(a=a, b=b)
                
            out['t21'][i] = ctx['eor_spectrum']
            if iso:
                out['tfg'][i] = model(x=adata.sky_data['freq'], parameters=p)
                out['resids'][i] = lk.t_sky - out['t21'][i] - out['tfg'][i]
            else:
                out['tfg'][i] = model.get_model('fg', x=adata.sky_data['freq'], parameters=p)
                out['resids'][i] = out['recal_sky'][i] - out['t21'][i] - out['tfg'][i]
        
        return out

    out = p_map(do_stuff, range(nthreads), num_cpus=nthreads)
    
    out_dict = {'samples': samples, 'freq': freq}
    for name in out[0]:
        out_dict[name] = np.concatenate(tuple(o[name] for o in out))
    
    np.savez(outfile, **out_dict)
        
    return out_dict

@main.command()
@run.all_mc_options
@cal.cterms
@cal.wterms
@cal.fit_cterms
@cal.fit_wterms
@cal.antsim
@precal.smooth
@precal.tns_width
@precal.set_widths
@precal.tns_mean_zero
@precal.ignore_sources
@precal.as_sim
@precal.s11_sys
#@precal.unweighted
#@precal.cable_noise_factor
@precal.ndelay
@precal.nscale
@click.option('--nterms-fg', default=5)
@click.option('--fix-tau/--no-fix-tau', default=True)
@click.option('--simultaneous/--isolated', default=True)
@click.option('--seed', default=1234)
@click.option('--remove-eor/--keep-eor', default=False)
def clirun(**kwargs):
    run.clirun(mcdef, **kwargs)

if __name__ == '__main__':
    clirun()