import os
import copy
import argparse

import numpy as np

import data_collection
import plotter

# ===============================================================================
# Helper functions to run experiments and plot the data
# ===============================================================================

def run_action_selection_final_weights_delay(
        num_samples, models, delay_vals, a_sel, sim_kwargs, data_folder=None, plot_folder=None, silent=False):
    """
    Action selection setting; measure long-term weight values while varying T_del
    and whether to use sustained activity (a_sel) or not 
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', 'a_sel', 'T_del'],
        [models, [None, a_sel], delay_vals],
        save_folder=data_folder)

    plotter.plot_action_selection_final_weights_delay(
        weights, delay_vals, models, save_folder=plot_folder, silent=silent)

def run_action_selection_weights_over_time_contingency_switching(
        num_samples, models, sim_kwargs, param_to_vary, param_vals, data_folder=None, plot_folder=None, silent=False):
    """
    Action selection setting with contingency switching; measure weights over time while varying a parameter
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', param_to_vary],
        [models, param_vals],
        save_folder=data_folder)
    plotter.plot_action_selection_weights_over_time_contingency_switching(
        weights, actions, models, sim_kwargs['switch_period'], param_to_vary, param_vals,
        save_folder=plot_folder, silent=silent)

def run_action_selection_weights_over_time_2d(
        num_samples, models, sim_kwargs, param_to_vary, param_vals, data_folder=None, plot_folder=None, silent=False):
    """
    Action selection setting with N=2; measure weights over time while varying a parameter
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', param_to_vary],
        [models, param_vals],
        save_folder=data_folder)
    plotter.plot_action_selection_weights_over_time_2d(
        weights, actions, models,
        param_to_vary, param_vals,
        save_folder=plot_folder, silent=silent)

    
def run_value_estimation_weights_over_time_varying_w_init(
        num_samples, models, sim_kwargs, w_init_vals, data_folder=None, plot_folder=None, silent=False):
    """
    Value estimation setting; measure weights over time for multiple initial weights
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', 'w_init'],
        [models, w_init_vals],
        save_folder=data_folder)

    plotter.plot_value_estimation_density_varying_w_init(
        weights, actions, action_probs, w_init_vals, models, sim_kwargs,
        save_folder=plot_folder, silent=silent)

def run_value_estimation_phase_planes_density(
        num_samples, param_to_vary, param_vals, models, sim_kwargs, no_dynamics=False, 
        data_folder=None, plot_folder=None, silent=False):
    """
    Value estimation setting; measure weights over time while varying a parameter and plot phase planes
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', param_to_vary],
        [models, param_vals],
        save_folder=data_folder)

    plotter.plot_value_estimation_phase_planes_density(
        weights, action_probs, param_to_vary, param_vals, models, sim_kwargs,
        no_dynamics=no_dynamics, save_folder=plot_folder, silent=silent)

def run_value_estimation_weights_over_time_contingency_switching_varying_param(
        num_samples, models, sim_kwargs, param_to_vary, param_vals, data_folder=None, plot_folder=None, silent=False):
    """
    Value estimation setting with contingency switching; measure weights over time while varying a parameter
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', param_to_vary],
        [models, param_vals],
        save_folder=data_folder)
    
    plotter.plot_value_estimation_weights_over_time_contingency_switching(
        weights, actions, action_probs, models, sim_kwargs['switch_period'], param_to_vary, param_vals,
        save_folder=plot_folder, silent=silent)


def run_value_estimation_instantaneous_drift(
        num_samples, models, param1, param1_vals, param2, param2_vals, sim_kwargs, val_for_predictions=None,
        data_folder=None, plot_folder=None, silent=False):
    """
    Value estimation setting; measure instantaneous drift (weight change per time step) while varying two parameters
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
        num_samples, sim_kwargs,
        ['model', param1, param2],
        [models, param1_vals, param2_vals],
        save_folder=data_folder)
    
    plotter.plot_value_estimation_instantaneous_drift(weights, param1, param1_vals,
                                        param2, param2_vals, models,
                                        sim_kwargs=sim_kwargs, val_for_predictions=val_for_predictions,
                                        save_folder=plot_folder, silent=silent)

def run_action_selection_phase_planes_density(
        num_samples, param_to_vary, param_vals, models, sim_kwargs, no_dynamics=False, 
        data_folder=None, plot_folder=None, silent=False):
    """
    Value estimation setting; measure weights over time while varying a parameter and plot phase planes
    """
    weights, actions, DA, action_probs = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model', param_to_vary],
            [models, param_vals],
            save_folder=data_folder)

    plotter.plot_action_selection_phase_planes_density(
            weights, param_to_vary, param_vals, models, sim_kwargs,
            no_dynamics=no_dynamics, save_folder=plot_folder, silent=silent)

# ===============================================================================
# Simulation parameters
# ===============================================================================

simulation_kwargs_action_selection = {
    'N': 1,
    'r': 10,
    'Rstar': [2, 1],
    'task': 'action selection',
    'num_steps': 1000,
    'alpha': 1,
    'tau': 0.02, # From Gutig et al.
    'tau_dop': 1, # From Riley et al. 2024
    'tau_eli': 1, # From Fisher et al. 2017, Yagishita et al. 2014 (but see Shindou et al. 2019)
    'T_del': 10,
    'T_win': 1,
    'lambda_': 0.01,
    'eps': 0.001, # Need exp(-eps/tau) ~= 1
    'w_init': 0.5,
    'r_dop': 1/21,
    'beta': 100000, # Arbitrary large number
    'a_sel': 0.7,
    # These parameters not used
    'lambda_bar': None,
}

simulation_kwargs_value_estimation = {
    'N': 1,
    'r': 10,
    'Rstar': [7.5, 2.5],
    'task': 'value estimation',
    'num_steps': 1000,
    'alpha': 1,
    'tau': 0.02, # From Gutig et al.
    'tau_dop': 1, # From Riley et al. 2024
    'tau_eli': 1, # From Fisher et al. 2017, Yagishita et al. 2014 (but see Shindou et al. 2019)
    'T_del': 3,
    'T_win': 1,
    'lambda_': 0.001,
    'eps': 0.001, # Need exp(-eps/tau) ~= 1
    'w_init': 0.5,
    'r_dop': 1/7,
    'beta': 1,
    'lambda_bar': 0.0025,
    # These parameters not used
    'a_sel': None,
}

simulation_kwargs_action_selection_N_2 = copy.copy(simulation_kwargs_action_selection)
simulation_kwargs_action_selection_N_2['N'] = 2
simulation_kwargs_action_selection_N_2['r'] = [15, 5]

simulation_kwargs_value_estimation_N_2 = copy.copy(simulation_kwargs_value_estimation)
simulation_kwargs_value_estimation_N_2['N'] = 2
simulation_kwargs_value_estimation_N_2['r'] = [10, 10]

contingency_switching_action_selection_kwargs = copy.copy(simulation_kwargs_action_selection)
contingency_switching_action_selection_kwargs['switch_period'] = simulation_kwargs_action_selection['num_steps']
contingency_switching_action_selection_kwargs['num_steps'] = 5*simulation_kwargs_action_selection['num_steps']
contingency_switching_action_selection_kwargs['r'] = [10, 10] # Keep the same r but switch Rstar
contingency_switching_action_selection_kwargs['Rstar'] = [[2,1], [1,2]]

contingency_switching_value_estimation_kwargs = copy.copy(simulation_kwargs_value_estimation)
contingency_switching_value_estimation_kwargs['switch_period'] = simulation_kwargs_value_estimation['num_steps']
contingency_switching_value_estimation_kwargs['num_steps'] = 5*simulation_kwargs_value_estimation['num_steps']
contingency_switching_value_estimation_kwargs['lambda_'] = 0.00015
contingency_switching_value_estimation_kwargs['lambda_bar'] = 0.005
contingency_switching_value_estimation_kwargs['r'] = [10, 10] # Keep the same r but switch Rstar
contingency_switching_value_estimation_kwargs['Rstar'] = [[7.5,2.5], [2.5,7.5]]

if __name__ == '__main__':
    all_figs = [f'Fig{i}' for i in range(1, 11)] + [f'FigD{i}' for i in range(1, 4)]
    parser = argparse.ArgumentParser('Run experiments in the paper "Distinct dopaminergic spike-timing-dependent plasticity rules are suited to different functional roles".')
    parser.add_argument("figures", nargs="*", type=str, default=all_figs,
                        help="Figures to generate. Either strings 'Fig1', 'Fig2', etc. or figure number.")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples")
    parser.add_argument("--data", type=str, default="Data",
                        help="Folder to save data in")
    parser.add_argument("--plots", type=str, default="Plots",
                        help="Folder to save plots in")
    parser.add_argument("--procs", type=int, default=1,
                        help="Number of processes to use for multiprocessing")
    parser.add_argument("--models", nargs="+", type=str, default=['additive', 'symmetric', 'corticostriatal'],
                        help="Models to run")
    args = parser.parse_args()
    num_samples = args.samples
    project_data_folder = args.data
    project_plot_folder = args.plots
    figs = args.figures
    models = args.models
    # Set number of processes to use
    data_collection.NUM_PROCS = args.procs

    if 'Fig3' in figs or '3' in figs:
        # Action selection setting; plot phase planes for different values of alpha
        run_action_selection_phase_planes_density(
            num_samples=num_samples,
            param_to_vary='alpha',
            param_vals=[1, 5, 9],
            models=models,
            sim_kwargs=simulation_kwargs_action_selection,
            data_folder=os.path.join(project_data_folder, 'Fig3'),
            plot_folder=os.path.join(project_plot_folder, 'Fig3'),
            silent=True)

    if 'Fig4' in figs or '4' in figs:
        # Action selection setting with N=2; plot weights over time for different values of alpha
        run_action_selection_weights_over_time_2d(
            num_samples=num_samples,
            models=models,
            sim_kwargs=simulation_kwargs_action_selection_N_2,
            param_to_vary='alpha',
            param_vals=[1, 5],
            data_folder=os.path.join(project_data_folder, 'Fig4'),
            plot_folder=os.path.join(project_plot_folder, 'Fig4'),
            silent=True)

    if 'Fig5' in figs or '5' in figs:
        # Action selection setting with contingency switching; plot weights over time for different values of alpha
        run_action_selection_weights_over_time_contingency_switching(
            num_samples=num_samples,
            models=models,
            sim_kwargs=contingency_switching_action_selection_kwargs,
            param_to_vary='alpha',
            param_vals=[1, 5],
            data_folder=os.path.join(project_data_folder, 'Fig5'),
            plot_folder=os.path.join(project_plot_folder, 'Fig5'),
            silent=True)
    
    if 'Fig6' in figs or '6' in figs:
        # Action selection setting; measure long-term weight values while varying T_del
        # and whether to use sustained activity (a_sel) or not 
        run_action_selection_final_weights_delay(
            num_samples=num_samples,
            models=models,
            delay_vals=[0,0.5,1,1.5,2,2.5,3],
            a_sel=0.7,
            sim_kwargs=simulation_kwargs_action_selection,
            data_folder=os.path.join(project_data_folder, 'Fig6'),
            plot_folder=os.path.join(project_plot_folder, 'Fig6'),
            silent=True)
        
    if 'Fig7' in figs or '7' in figs:
        # Value estimation setting; plot phase planes for different values of alpha
        run_value_estimation_phase_planes_density(
            num_samples=num_samples,
            param_to_vary='alpha',
            param_vals=[1, 4, 7],
            models=models,
            sim_kwargs=simulation_kwargs_value_estimation,
            data_folder=os.path.join(project_data_folder, 'Fig7'),
            plot_folder=os.path.join(project_plot_folder, 'Fig7'),
            silent=True)

    if 'Fig8' in figs or '8' in figs:
        # Value estimation setting; plot weights over time for different initial weights
        run_value_estimation_weights_over_time_varying_w_init(
            num_samples=num_samples,
            models=models,
            sim_kwargs=simulation_kwargs_value_estimation_N_2,
            w_init_vals=[[0.75,.25], [.5,.5], [.25,.75]],
            data_folder=os.path.join(project_data_folder, 'Fig8'),
            plot_folder=os.path.join(project_plot_folder, 'Fig8'),
            silent=True)
    
    if 'Fig9' in figs or '9' in figs:
        # Value estimation setting with contingency switching; plot weights over time for different values of alpha
        run_value_estimation_weights_over_time_contingency_switching_varying_param(
            num_samples=num_samples,
            models=models,
            sim_kwargs=contingency_switching_value_estimation_kwargs,
            param_to_vary='alpha',
            param_vals=[1, 4],
            data_folder=os.path.join(project_data_folder, f'Fig9'),
            plot_folder=os.path.join(project_plot_folder, f'Fig9'),
            silent=True)
    
    if 'Fig10' in figs or '10' in figs:
        # Value estimation setting; measure instantaneous drift (weight change per time step) while varying T_del
        # and w_init
        instantaneous_drift_kwargs = copy.copy(simulation_kwargs_value_estimation)
        instantaneous_drift_kwargs['num_steps'] = 2
        run_value_estimation_instantaneous_drift(
            num_samples=num_samples,
            models=models,
            param1='T_del',
            param1_vals=[0, 3],
            param2='w_init',
            param2_vals=np.linspace(0, 1, 11),
            sim_kwargs=instantaneous_drift_kwargs,
            val_for_predictions=3, # Use T_del=3 when calculating theoretical predictions
            data_folder=os.path.join(project_data_folder, 'Fig10'),
            plot_folder=os.path.join(project_plot_folder, 'Fig10'),
            silent=True)

    if 'Fig11' in figs or '11' in figs:
        # Value estimation setting; measure instantaneous drift (weight change per time step) while varying eps
        # and w_init
        instantaneous_drift_kwargs = copy.copy(simulation_kwargs_value_estimation)
        instantaneous_drift_kwargs['num_steps'] = 2
        run_value_estimation_instantaneous_drift(
            num_samples=num_samples,
            models=models,
            param1='eps',
            param1_vals=[0.001, 0.005],
            param2='w_init',
            param2_vals=np.linspace(0, 1, 11),
            sim_kwargs=instantaneous_drift_kwargs,
            val_for_predictions=0.001, # Use eps=0.001 when calculating theoretical predictions
            data_folder=os.path.join(project_data_folder, 'Fig11'),
            plot_folder=os.path.join(project_plot_folder, 'Fig11'),
            silent=True)
    
    if 'FigD1' in figs or 'D1' in figs:
        # Action selection setting with single trace models; plot phase planes for different values of gamma
        run_action_selection_phase_planes_density(
            num_samples=num_samples,
            param_to_vary='gamma',
            param_vals=[1, 5, 9],
            models=models,
            sim_kwargs={**simulation_kwargs_action_selection, 'single_trace': True},
            no_dynamics=True,
            data_folder=os.path.join(project_data_folder, 'FigD1'),
            plot_folder=os.path.join(project_plot_folder, 'FigD1'),
            silent=True)

    if 'FigD2' in figs or 'D2' in figs:
        # Action selection setting with contingency switching with single trace models; 
        # plot weights over time for different values of gamma
        run_action_selection_weights_over_time_contingency_switching(
            num_samples=num_samples,
            models=models,
            sim_kwargs={**contingency_switching_action_selection_kwargs, 'single_trace': True},
            param_to_vary='gamma',
            param_vals=[1, 5],
            data_folder=os.path.join(project_data_folder, 'FigD2'),
            plot_folder=os.path.join(project_plot_folder, f'FigD2'),
            silent=True)
    
    if 'FigD3' in figs or 'D3' in figs:
        # Value estimation setting with single trace models; plot phase planes for different values of gamma
        run_value_estimation_phase_planes_density(
            num_samples=num_samples,
            param_to_vary='gamma',
            param_vals=[1, 4, 7],
            models=models,
            sim_kwargs={**simulation_kwargs_value_estimation, 'single_trace': True},
            no_dynamics=True,
            data_folder=os.path.join(project_data_folder, 'FigD3'),
            plot_folder=os.path.join(project_plot_folder, f'FigD3'),
            silent=True)
    
    if 'FigD4' in figs or 'D4' in figs:
        # Value estimation setting with contingency switching with single trace models; 
        # plot weights over time for different values of gamma
        run_value_estimation_weights_over_time_contingency_switching_varying_param(
        num_samples=num_samples,
        models=models,
        sim_kwargs={**contingency_switching_value_estimation_kwargs, 'single_trace': True},
        param_to_vary='gamma',
        param_vals=[1, 4],
        data_folder=os.path.join(project_data_folder, f'FigD4'),
        plot_folder=os.path.join(project_plot_folder, f'FigD4'),
        silent=True)