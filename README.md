# DA-STDP

Code to reproduce results in "Distinct dopaminergic spike-timing-dependent plasticity rules are suited to different functional roles"

Tested on Python 3.10.5.

Dependencies:
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `einops`
- `seaborn`

To run the experiments and plot the results, run `run_experiments.py`:
> [-h] [--samples SAMPLES] [--data DATA] [--plots PLOTS] [--procs PROCS] [--models MODELS [MODELS ...]] [figures ...]
>
> positional arguments:
> 
> figures: Figures to generate. Either strings 'Fig1', 'Fig2', etc. or figure number.
>
> options:
>   -h, --help            show this help message and exit
>   --samples SAMPLES     Number of samples
>   --data DATA           Folder to save data in
>   --plots PLOTS         Folder to save plots in
>   --procs PROCS         Number of processes to use for multiprocessing
>   --models MODELS [MODELS ...]
>                         Models to run


`poisson_model.py` contains the main simulation code. `data_collection.py` and `data_manager.py` contains helper functions to run experiments. `plotter.py`, and `plot_utils.py` generate figures; `phase_planes_action_selection.py`, `phase_planes_value_estimation.py`, and `phase_plane_utils.py` contain helper functions to plot phase planes. To reproduce the results in the paper, use the experiments set up in `run_experiments.py`.