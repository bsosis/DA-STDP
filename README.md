# DA-STDP

Code to reproduce results in "Distinct dopaminergic spike-timing-dependent plasticity rules are suited to different functional roles"

Dependencies:
- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `einops`

To run the experiments, run `run_experiments.py`:
>        [-h] [-s SAMPLES] [-p PROCS] [-f FOLDER] [figures ...]
>
> positional arguments:
>   figures               Figures to generate. Either strings 'Fig1', 'Fig2',
>                         etc. or figure number.
>
> options:
>   -h, --help            show this help message and exit
>   -s SAMPLES, --samples SAMPLES
>                         Number of samples
>   -p PROCS, --procs PROCS
>                         Number of processes
>   -f FOLDER, --folder FOLDER

`poisson_model.py` contains the main simulation code. `data_collection.py` contains helper functions to run experiments. `plotter.py` and `plottools.py` generate figures.