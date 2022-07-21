"""
Utilities for setting up mcmc runs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict
from pathlib import Path
import yaml
import parse
import numpy as np
from edges_cal.modelling import Polynomial, UnitTransform
from yabf.core import mpi
import time
import yaml
from datetime import datetime
from scipy.stats import lognorm, norm
import pickle
from edges_cal.modelling import Polynomial, UnitTransform
import git
import edges_io
import edges_cal
import edges_analysis
import edges_estimate
from rich.rule import Rule
from scipy import optimize as opt
from numdifftools import Hessian
from rich.console import Console
import logging
from rich.logging import RichHandler
import sys
from yabf.samplers.polychord import polychord
from yabf import run_map
import attr
import click

cns = Console(width=200)
log = logging.getLogger(__name__)
log.addHandler(RichHandler(rich_tracebacks=True, console=cns))


def make_s11_sys(name, nscale, ndelay, scale_max, delay_max):
    out = dict(
        scale_model = Polynomial(n_terms=nscale, transform=UnitTransform(range=(50, 100))),
        delay_model = Polynomial(n_terms=ndelay, transform=UnitTransform(range=(50, 100)))
    )
    
    scale_params = {f"{name}_logscale_{i}": {'min': np.log10(1 - scale_max), 'max': np.log10(1 + scale_max), 'determines': f'logscale_{i}'} for i in range(nscale)}
    delay_params = {f"{name}_delay_{i}": {'min': -delay_max, 'max': delay_max, 'determines': f'delay_{i}'} for i in range(ndelay)}
    
    return {
        **out,
        **scale_params,
        **delay_params
    }

def define_s11_systematics(s11_sys: tuple[str], nscale: int, ndelay: int, scale_max: float=1e-2, delay_max: float=10):
    s11_systematic_params = {}
    for src in s11_sys:
        s11_systematic_params[src] = make_s11_sys(src, nscale=nscale, ndelay=ndelay, scale_max=scale_max, delay_max=delay_max)

    return s11_systematic_params

@dataclass
class MCDef:
    label_formats: tuple[str]
    folder: str
    default_kwargs: dict[str, Any]
    get_likelihood: Callable
    label_transforms: Callable | None = None

    @property
    def path(self) -> Path:
        return Path('outputs') / self.folder

    def get_path(self, label=None, **kwargs) -> Path:
        if label is None:
            label = self.get_label(**kwargs)

        return self.path / label

    def get_label(self, label_format=None, **kwargs):
        s11_sys = kwargs.pop("s11_sys", ()) or ()
        
        label_format = label_format or self.label_formats[-1]

        if self.label_transforms is not None:
            kwargs = self.label_transforms(**kwargs)

        return label_format.format(s11_sys=s11_sys, **kwargs)

    def get_likelihood_from_label(self, label: str):
        kw = self.get_kwargs(label)
        return self.get_likelihood(**kw)

    def get_kwargs(self, label: str) -> dict[str, Any]:
        yaml_file = self.path / label / 'bayescal.lkargs.yaml'
        if yaml_file.exists():
            with open(yaml_file, 'r') as fl:
                kw = yaml.load(fl, Loader=yaml.UnsafeLoader)
        else:
            for fmt in self.label_formats[::-1]:
                kw = parse.parse(fmt, label)
                if kw:
                    kw = kw.named
                    break
                else:
                    kw = {}

            for k, v in kw.items():
                # Convert booleans
                if v in ("True", "False"):
                    kw[k] = (v == "True")

                # Convert tuples of strings.    
                if isinstance(v, str) and v[0] == '(' and v[-1] == ')':
                    if len(v)>2:
                        kw[k] = tuple(v[1:-1].replace("'", "").replace(' ','').split(','))
                    else:
                        kw[k] = ()

                if k=='variance'  and v != "data":
                    kw["variance"] = float(v)

        return {**self.default_kwargs, **kw}
        
    def get_mcroot(self, label=None, **kwargs) -> str:
        return str(self.get_path(label, **kwargs)/ "bayescal")


    def get_kwargs_from_mcsamples(self, mc):
        return self.get_kwargs(Path(mc.root).parent.name)




class MCMCBoundsError(ValueError):
    pass


def run_lk(
    mcdef: MCDef, 
    resume=False,
    nlive_fac=100,
    clobber=False,
    raise_on_prior=True,
    optimize=True,
    truth=None,
    prior_width=10,
    set_widths: bool=False,
    run_mcmc: bool=True,
    opt_iter: int = 10,
    **lk_kwargs
):
    label = mcdef.get_label(**lk_kwargs)
    print(lk_kwargs)
    lk = mcdef.get_likelihood(**lk_kwargs)

    repo = git.Repo(str(Path(__file__).parent.absolute()))

    root = 'bayescal'
    folder = mcdef.get_path(label)
    mcroot = mcdef.get_mcroot(label)
    out_txt = Path(mcroot +  '.txt')
    out_yaml = Path(mcroot + '.meta.yml')

    if mpi.am_single_or_primary_process:

        if not folder.exists():
            folder.mkdir(parents=True)

        cns.print(f"[bold]Running [blue]{label}")
        cns.print(f"Output will go to '{folder}'")

        cns.print(
            f"[bold]Fiducial Parameters[/]: {[p.fiducial for p in lk.partial_linear_model.child_active_params]}"
        )
        t = time.time()
        lnl, derived = lk.partial_linear_model()
        t1 = time.time()
        cns.print(f"[bold]Fiducial likelihood[/]: {lnl}")
        for nm, d in zip(lk.partial_linear_model.child_derived, derived):
            cns.print(f"\t{nm if isinstance(nm, str) else nm.__name__}: {d}")

        cns.print(f"Took {t1 - t:1.2e} seconds to evaluate likelihood.")

        if not resume and out_txt.exists():
            if not clobber:
                sys.exit(
                    f"Run with label '{label}' already exists. Use --resume to resume it, or delete it."
                )
            else:
                all_files = folder.glob(f"{root}*")
                flstring = "\n\t".join(str(fl.name) for fl in all_files)
                log.warning(f"Removing following files:\n{flstring}")
                for fl in all_files:
                    fl.unlink()

        # Write out the likelihood args
        with open(folder / (root + '.lkargs.yaml'), 'w') as fl:
            yaml.dump(lk_kwargs, fl)

        with open(folder / 'data.pkl', 'wb') as fl:
            pickle.dump(lk.partial_linear_model.data, fl)

    if optimize and (not resume or not out_txt.exists() or not run_mcmc):
        # Only run the optimizatino if we're not just resuming an MCMC
        lk = optimize_lk(lk, truth, prior_width,  folder, root, dual_annealing = optimize == 'dual_annealing', niter=opt_iter, set_widths=set_widths)
        

    if mpi.am_single_or_primary_process:
        if resume and out_txt.exists() and out_yaml.exists():
            with open(out_yaml, 'r') as fl:
                yaml_args = yaml.safe_load(fl)
        else:
            yaml_args = {
                'start_time': time.time(), 
                'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'optimize': optimize,
                'prior_width': prior_width,
                'set_widths': set_widths,
                'githash': repo.head.ref.commit.hexsha + ('.dirty' if repo.is_dirty() else ''),
                'edges-io': edges_io.__version__,
                'edges-cal': edges_cal.__version__,
                'edges-analysis': edges_analysis.__version__,
                'edges-estimate': edges_estimate.__version__,
            }

            with open(out_yaml, 'w') as fl:
                yaml.dump(yaml_args, fl)

    if run_mcmc:
        poly = polychord(
            save_full_config=False,
            likelihood=lk.partial_linear_model,
            output_dir=folder,
            output_prefix=root,
            sampler_kwargs=dict(
                nlive=nlive_fac * len(lk.partial_linear_model.child_active_params),
                read_resume=resume,
                feedback=2,
            ),
        )

        def time_dumper(live, dead, logweights, logZ, logZerr):
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            this_time = time.time()
            cns.print(f"{now}: {(this_time - time_dumper.last_call) / 60} min since last dump.")
            time_dumper.last_call = this_time

        t = time.time()
        time_dumper.last_call = t
        samples = poly.sample(dumper=time_dumper)
        cns.print(f"Sampling took {(time.time() - t)/60/60:1.3} hours.")
        samples.saveAsText(f"{folder}/{root}")

        if mpi.am_single_or_primary_process:
            yaml_args['end_time'] = t
            yaml_args['end_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(out_yaml, 'w') as fl:
                yaml.dump(yaml_args, fl)

        # Do some basic diagnostics
        means = samples.getMeans()
        stds = np.sqrt(samples.getVars())
        err = False
        for i, p in enumerate(lk.partial_linear_model.child_active_params):
            if means[i] - 5 * stds[i] < p.min or means[i] + 5 * stds[i] > p.max:
                err = True
                log.error(
                    f"Parameter '{p.name}' has posterior out of bounds. Posterior: {means[i]} += {stds[i]}, bounds = {p.min} - {p.max}."
                )

        if err and raise_on_prior:
            raise MCMCBoundsError("Parameter posteriors are out of prior bounds!")

        return samples, err

def optimize_lk(lk, truth, prior_width, folder, label, dual_annealing: bool=False, niter: int=10, set_widths=False):
    if mpi.am_single_or_primary_process:
        outfile = folder / (label + '.map')
        if outfile.exists():
            cns.print("Getting previously run optimization.")
            with open(outfile, 'rb') as fl:
                d = pickle.load(fl)
                
            resx = d['optres'].x
            std = np.sqrt(np.diag(d['cov']))
            widths =  std * prior_width

        else:
            cns.print(f"Optimizing with {niter} global iterations using {'dual_annealing' if dual_annealing else 'basinhopping'}...", end='')
            t = time.time()

            minima = []

            def callback(x, f, accept):
                minima.append((x, f))

            if dual_annealing:
                opt_res = run_map(
                    lk.partial_linear_model,
                    dual_annealing_kw={"maxiter": niter, "callback": callback},
                )
                if not opt_res.success:
                    log.warning(f"Optimization unsuccessful with message: {opt_res.message}")
            else:
                opt_res = run_map(
                    lk.partial_linear_model,
                    basinhopping_kw={"niter": niter, "callback": callback},
                )

                if opt_res.minimization_failures > 0:
                    log.warning(
                            f"There were {opt_res.minimization_failures} minimization failures!"
                        )

                if not opt_res.lowest_optimization_result.success:
                    log.warning(
                            f"The lowest optimization was not successful! Message: {opt_res.lowest_optimization_result.message}"
                        )

            def obj(x):
                out = -lk.partial_linear_model.logp(params=x)
                # cns.print(x, out)
                if np.isnan(out) or np.isinf(out):
                    log.warning(f"Hessian got NaN/inf value for parameters: {x}, {out}")
                return out
            
            cns.print("Computing Hessian...")

            hess = Hessian(obj, base_step=0.1, step_ratio=3, num_steps=30)(opt_res.x)
            cns.print("[bold]Hessian: ")
            cns.print(hess)

            cov = np.linalg.inv(hess)
            cns.print("[bold]Covariance: ")
            cns.print(cov)
            std = np.sqrt(np.diag(cov))
            widths = prior_width * std
            resx = opt_res.x

            if minima:
                f = [f for _, f in minima]
                minf = min(f)
                _minima = [m for m in minima if m[1] < (minf + 100)]

                if len(_minima) == 1:
                    log.warning(f"Only one minima was any good, got: {f}")
                else:
                    for i, param in enumerate(lk.partial_linear_model.child_active_params):
                        rxx = resx[i]
                        ww = widths[i]

                        xx  = [x[i] for x, _ in _minima]
                        
                        if any((np.abs(xxx - rxx) > ww and (ff < minf + 100)) for xxx, ff in zip(xx, f)):
                            log.error(
                                f"For '{param.name}', got minima at {xx} "
                                f"when it should have been between {rxx-ww} and {rxx+ww}."
                                f"Corresponding -lks: {[f for _, f in _minima]} (high likelihoods omitted)."
                            )
                        
            if truth is not None and np.any(np.abs(truth - resx) > 3 * std):
                raise RuntimeError(
                        "At least one of the estimated parameters was off the truth by > 3σ"
                    )
            cns.print(f" done in {time.time() - t:.2f} seconds.")

            # Write out the results in a pickle.
            with open(outfile, 'wb') as fl:
                pickle.dump(
                    {
                        'optres': opt_res,
                        'cov': cov,
                        'minima': minima,
                    },
                    fl
                )

        cns.print("[bold]Estimated Parameters: [/]")
        for p, r, s in zip(lk.partial_linear_model.child_active_params, resx, std):
            outside = abs(p.fiducial - r)>3*s
            cns.print(f"\t{p.name:>14}: {r:+1.3e} +- {s:1.3e} | {p.min:+1.3e} < {'[red]' if outside else ''}{p.fiducial: 1.3e}{'[/]' if outside else ''} < {p.max: 1.3e}")

        best = lk.partial_linear_model.logp(params=resx)
        fid = lk.partial_linear_model.logp()

        cns.print(f"Likelihood at MAP vs. Fiducial: {best} vs. {fid}", style='red' if fid>best else 'green')    

    else:
        resx = None
        widths = None

    widths = mpi.mpi_comm.bcast(widths, root=0)
    resx = mpi.mpi_comm.bcast(resx, root=0)

    if set_widths:
        # NOTE: If this is true, then the final evidence one calculates should be modified
        #       like so: log(Z) ~ log(Z_polychord) - n*log(f),
        #       where n is the number of dimensions that have been compressed by some
        #       width and f is the factor by which each dimension is compressed.
        #       More generally, it's \Sum(log(f_i)) where the sum goes over each dimension.
        #       HOWEVER, this only works if the posterior is essentially zero outside
        #       the actual prior range chosen for polychord. 
        #       For a prior_width of 10, this is a very good approximation for dimensions
        #       well beyond 30. 
        new_tns_params = attr.evolve(
                lk.t_ns_params, fiducial=resx, min=resx - widths, max=resx + widths
            )

        lk = attr.evolve(lk, t_ns_params=new_tns_params)

    elif hasattr(lk, 't_ns_params'):
        for i, p in enumerate(lk.t_ns_params.get_params()):
            if resx[i] - widths[i] < p.min or resx[i] + widths[i] > p.max:
                raise ValueError(f"You need to set Tns[{i}] to have greater width. At least {resx[i] + widths[i]}")

    return lk


resume = click.option("--resume/--no-resume", default=False)
nlive_fac = click.option("-n", "--nlive-fac", default=100)
optimize = click.option("-o", "--optimize", type=click.Choice(['none', 'dual_annealing', 'basinhopping'], case_sensitive=False), default='basinhopping')
clobber = click.option("--clobber/--no-clobber", default=False)
log_level = click.option("--log-level", default='info', type=click.Choice(['info', 'debug', 'warn', 'error']))
run_mcmc = click.option("--run-mcmc/--no-mcmc", default=True)
opt_iter = click.option("--opt-iter", default=10)

def all_mc_options(f):
    return resume(nlive_fac(optimize(clobber(log_level(run_mcmc(opt_iter(f)))))))

def clirun(mcdef, **kwargs):
    log_level = kwargs.pop("log_level")
    root_logger = logging.getLogger('yabf')
    root_logger.setLevel(log_level.upper())
    root_logger.addHandler(RichHandler(rich_tracebacks=True, console=cns))
    
    optimize = kwargs.pop('optimize').lower()

    if optimize == 'none':
        optimize = None

    if 'tns_mean_zero' in kwargs:
        kwargs['est_tns'] = np.zeros(6) if kwargs.pop('tns_mean_zero') else None

    if 'fit_cterms' in kwargs and kwargs['fit_cterms'] is None:
        kwargs['fit_cterms'] = kwargs['cterms']
    if 'fit_wterms' in kwargs and kwargs['fit_wterms'] is None:
        kwargs['fit_wterms'] = kwargs['wterms']

    for k, v in kwargs.items():
        if isinstance(v, list):
            kwargs[k] = tuple(v)

    run_lk(
        mcdef,
        optimize=optimize,
        **kwargs
    )
