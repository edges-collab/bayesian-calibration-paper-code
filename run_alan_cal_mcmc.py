from __future__ import annotations
import click
import run_alan_precal_mcmc as precal

import run_mcmc_utils as run
import run_alan_precal_mcmc as precal

main = click.Group()

mcdef = run.MCDef(
    label_formats = (
        "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}",
        "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim-{antsim}",
        "c{cterms:02d}_w{wterms:02d}_cf{fit_cterms:02d}_wf{fit_wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim-{antsim}",
    ),
    folder = 'alan_cal',
    default_kwargs = {**precal.mcdef.default_kwargs, **{'antsim': False}},
    get_likelihood=precal.get_likelihood,
)


cterms = click.option("-c", "--cterms", default=6)
wterms = click.option("-w", "--wterms", default=5)
fit_cterms = click.option("--fit-cterms", default=None, type=int)
fit_wterms = click.option("--fit-wterms", default=None, type=int)
antsim = click.option("--antsim/--no-antsim", default=False)


@main.command()
@cterms
@wterms
@fit_cterms
@fit_wterms
@antsim
@run.all_mc_options
@precal.smooth
@precal.tns_width
@precal.set_widths
@precal.tns_mean_zero
@precal.ignore_sources
@precal.as_sim
@precal.s11_sys
@precal.unweighted
@precal.cable_noise_factor
@precal.ndelay
@precal.nscale
def clirun(**kwargs):
    run.clirun(mcdef, **kwargs)


if __name__ == "__main__":
    clirun()
