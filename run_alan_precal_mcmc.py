from __future__ import annotations
import click
import alan_data_utils as utils

import run_mcmc_utils as run 

main = click.Group()


def get_likelihood(
    smooth, tns_width, est_tns=None, ignore_sources=(), as_sim=(), 
    s11_sys=(), nscale=1, ndelay=1, unweighted=False, cable_noise_factor=1,
    fit_cterms=None, fit_wterms=None, cterms=6, wterms=5, seed=1234, variance='data',
    add_noise=True, antsim=False
):
    calobs = utils.get_calobs(cterms=cterms, wterms=wterms, smooth=smooth)
    s11_systematic_params = run.define_s11_systematics(s11_sys, nscale=nscale, ndelay=ndelay)
    return utils.get_cal_lk(
        calobs, tns_width=tns_width, est_tns=est_tns, 
        ignore_sources=ignore_sources, as_sim=as_sim, 
        s11_systematic_params=s11_systematic_params,
        sig_by_sigq=not unweighted, sig_by_tns=not unweighted,
        cable_noise_factor=cable_noise_factor, cterms=fit_cterms, wterms=fit_wterms,
        seed=seed, add_noise=add_noise, include_antsim=antsim
    )

mcdef = run.MCDef(
    label_formats = (
        "smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}",
        "smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_nscale{nscale:02d}_ndelay{ndelay:02d}_unw-{unweighted}_cnf{cable_noise_factor:d}",
    ),
    folder = "alan_precal",
    default_kwargs = {
        'tns_width': 500, 
        'ignore_sources': (), 
        's11_sys': (), 
        "nscale": 1, 
        "ndelay": 1, 
        'unweighted': False, 
        'cable_noise_factor': 1
    },
    get_likelihood=get_likelihood,
)



smooth= click.option("-s", "--smooth", default=8)
tns_width = click.option("-p", "--tns-width", default=500)
set_widths = click.option("--set-widths/--no-set-widths", default=False)
tns_mean_zero = click.option("--tns-mean-zero/--est-tns", default=True)
ignore_sources = click.option('--ignore-sources', multiple=True, type=click.Choice(['short', 'open','hot_load', 'ambient']))
as_sim = click.option('--as-sim', multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient']))
s11_sys = click.option("--s11-sys", multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'rcv']))
unweighted = click.option("--unweighted/--weighted", default=False)
cable_noise_factor = click.option("--cable-noise-factor", default=1, type=int)
ndelay = click.option("--ndelay", default=1, type=int)
nscale = click.option("--nscale", default=1, type=int)

@main.command()
@run.all_mc_options
@smooth
@tns_width
@set_widths
@tns_mean_zero
@ignore_sources
@as_sim
@s11_sys
@unweighted
@cable_noise_factor
@ndelay
@nscale
def clirun(**kwargs):
    run.clirun(mcdef, **kwargs)




if __name__ == "__main__":
    clirun()
