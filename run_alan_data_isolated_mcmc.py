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
    nterms_fg, fix_tau, cterms, wterms, smooth, cal_map=None
):
    calobs = utils.get_calobs(cterms=cterms, wterms=wterms, smooth=smooth)
    labcal = utils.get_labcal(calobs)

    return utils.get_isolated_likelihood(
        labcal, 
        calobs,
        fsky=adata.sky_data['freq'],
        fg=LinLog(n_terms=nterms_fg), 
        eor=utils.make_absorption(adata.sky_data['freq'], fix=('tau',) if fix_tau else ()),
        ml_solution=cal_map,
        mcdef=cal.mcdef,
    )

def label_transforms(**kw):
    if kw['cal_map'] is not None:
        kw['cal_map'] = True
    return kw

mcdef = run.MCDef(
    label_formats = (
        "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_fg{nterms_fg}_taufx{fix_tau}_calmap{cal_map}",
    ),
    folder = "alan_field_isolated",
    default_kwargs={
        'cterms': 6,
        'wterms': 5, 
        'smooth': 32,
        'nterms_fg': 5,
        'fix_tau': False,
        'cal_map': None,
    },
    get_likelihood = get_likelihood,
    label_transforms=label_transforms,
)


@main.command()
@run.all_mc_options
@cal.cterms
@cal.wterms
@precal.smooth
@click.option('--nterms-fg', default=5)
@click.option('--fix-tau/--no-fix-tau', default=True)
@click.option("--cal-map", default=None, type=click.Path(exists=True))
def clirun(**kwargs):
    run.clirun(mcdef, **kwargs)

if __name__ == '__main__':
    clirun()