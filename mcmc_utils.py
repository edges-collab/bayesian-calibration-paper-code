import click
from rich.console import Console
from rich.logging import RichHandler
import logging
import alan_data_utils as ut
from pathlib import Path
import numpy as np
from datetime import datetime
import importlib
from getdist import loadMCSamples

cns = Console(width=200)
log = logging.getLogger(__name__)
log.addHandler(RichHandler(rich_tracebacks=True, console=cns))

main = click.Group()

def get_evidence(root: str) -> float:
    with open(root + ".stats") as fl:
        for line in fl.readlines():
            if line.startswith("log(Z)"):
                return float(line.split('=')[1].split('+/-')[0].strip())
    return np.nan

def get_all_files(direc, running_only: bool, complete_only: bool, glob=(), exc_glob=()) -> tuple[dict[str, Path], set[Path]]:
    direc = Path(direc)
    
    runs = set()

    if not glob:
        glob = ('',)

    for gl in glob:
        gl = f"*{gl}*" if gl else '*'
        runs.update({fl for fl in direc.glob(gl)})

    for gl in exc_glob:
        runs = {r for r in runs if gl not in str(r)}

    completed = {fl for fl in runs if (fl / 'bayescal.paramnames').exists()}

    if running_only:
        runs = {fl for fl in runs if fl not in completed}

    return {fl.name: fl for fl in runs}, completed
    

def get_completed_mcsamples(folder, include_incomplete=False):
    pth = Path('outputs') / folder

    all_runs = [p for p in sorted(pth.glob('*'))]
    deleteme = False
    
    if not include_incomplete:
        completed_runs = []
        for run in all_runs:
            if (run / 'bayescal.paramnames').exists():
                completed_runs.append(run / 'bayescal')
    else:
        completed_runs = []
        for run in all_runs:
            if (run / 'bayescal.txt').exists():
                s = np.genfromtxt(run / 'bayescal.txt')
                nparams = s.shape[1] - 2
                
                if not (run / 'bayescal.paramnames').exists():
                    deleteme = True
                    # Make a temporary paramnames file
                    with open(run / 'bayescal.paramnames', 'w') as fl:
                        lines = [f'param{i}*\tparam{i}\n' for i in range(nparams)]
                        fl.writelines(lines)

                completed_runs.append(run / 'bayescal')

    out = {fl.parent.name: loadMCSamples(str(fl)) for fl in completed_runs}
    
    if deleteme:
        (run/'bayescal.paramnames').unlink()

    return out


@main.command()
@click.argument('direc', type=click.Path(exists=True, file_okay=False))
@click.option("--running-only/--not-running-only", default=False, help="Only show MCMCs that are still running.")
@click.option('-c/-C', "--complete-only/--not-complete-only", default=False, help='Only show completed runs')
@click.option('-e/-E', "--show-evidence/--no-evidence", default=True, help='Print out evidence for run if available')
@click.option("--glob", multiple=True, help='Glob pattern that must be included to print')
@click.option("--exc-glob", multiple=True, help='Glob pattern that must be included to print')
def show(direc, running_only: bool, complete_only: bool, show_evidence: bool, glob: list[str], exc_glob: list[str]):
    direc = Path(direc)

    names, completed = get_all_files(direc, running_only, complete_only, glob, exc_glob)

    evidence = {name: get_evidence(str(fl / 'bayescal')) for name, fl in names.items() if fl in completed}

    nlongest = max(len(name) for name in names)

    cns.print(f"[bold]Runs for {direc}:[/]")

    for name, fl in sorted(names.items()):
        cns.print(f"[blue]{name:>{nlongest}}[/]\t", end="")
        if name in evidence:
            cns.print(f"lnZ={evidence[name]:.1f}", end='\t')
            mod_time = datetime.fromtimestamp((fl/'bayescal.txt').stat().st_mtime)
            cns.print(mod_time.strftime('%Y-%m-%d %H:%M'))

        elif (fl/'bayescal.txt').exists():
            cns.print("[red](Still Running...)[/]", end='\t')

            mod_time = datetime.fromtimestamp((fl/'bayescal.txt').stat().st_mtime)
            cns.print(mod_time.strftime('%Y-%m-%d %H:%M'))
        elif (fl/'bayescal.map').exists():
            mod_time = datetime.fromtimestamp((fl/'bayescal.map').stat().st_mtime)
            cns.print(f"[red](Only optimized...)[/]\t{mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            cns.print(f"[red](Not properly started...)[/]")
            
@main.command()
@click.argument('direc', type=click.Path(exists=True, file_okay=False))
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry/--not-dry", default=False)
def rename(direc, config, dry):
    direc = Path(direc)
    names, _ = get_all_files(direc, running_only=False, complete_only=True)

    endings = [
        '_dead-birth.txt',
        '_dead.txt',
        '_equal_weights.txt',
        '.paramnames',
        '_phys_live-birth.txt',
        '_phys_live.txt',
        '.prior_info',
        '_prior.txt',
        '.ranges',
        '.resume',
        '_src_temps.npz',
        '.stats',
        '.txt',
        '.meta.yml'
    ]
    
    config = Path(config)
    mdl = importlib.import_module(config.with_suffix('').name)
    print(mdl, mdl.__dict__)

    for root in names.values():
    
        kwargs = mdl.get_kwargs(root.name)
        label = mdl.get_label(**kwargs)

        new_folder = root.parent / label

        if not new_folder.exists():
            new_folder.mkdir()

        printed = False
        for ending in endings:
            fl = Path(str(root) + ending)

            new_file = new_folder / ('bayescal' + ending)

            if str(fl) == str(new_file) or not fl.exists():
                continue
            
            if not printed:
                cns.print(f"[bold]Replacing [blue]{str(root)}[/]  ⟶  [green]{str((new_file))}[/]")
                printed=True
                cns.print(f' ⇒ {ending}', end='')

            else:
                cns.print(' | ' + ending, end='')

            if not dry:
                fl.rename(new_file)

        if printed:
            cns.print()


@main.command()
@click.argument('direc', type=click.Path(exists=True, file_okay=False))
@click.option("--dry/--not-dry", default=False)
@click.option("--glob", multiple=True, help='Glob pattern that must be included to remove')
@click.option("--exc-glob", multiple=True, help='Glob pattern that must NOT be included to remove')
def rm(direc, dry, glob, exc_glob):
    direc = Path(direc)
    names, _ = get_all_files(direc, running_only=False, complete_only=False, glob=glob, exc_glob=exc_glob)

    endings = [
        '_dead-birth.txt',
        '_dead.txt',
        '_equal_weights.txt',
        '.paramnames',
        '_phys_live-birth.txt',
        '_phys_live.txt',
        '.prior_info',
        '_prior.txt',
        '.ranges',
        '.resume',
        '_src_temps.npz',
        '.stats',
        '.txt',
        '_blobs.npz'
    ]

    for root in names.values():

        for ending in endings:
            fl = Path(str(root) + ending)

            if fl.exists():
                cns.print(f'[bold]Removing [blue]{fl}[/]')

                if not dry:
                    fl.unlink()

if __name__ == '__main__':
    main()