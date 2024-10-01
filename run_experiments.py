import os
import copy

import numpy as np

import data_collection
import plotter

def run_random_DA_weights_over_time(
        num_samples, params_to_vary, param_vals_list, models, sim_kwargs,
        plot_folder=None, silent=False):
    weights_list = data_collection.get_weights_over_time_multiple(
            num_samples, sim_kwargs,
            [(['model', p], [models, v]) for p,v in zip(params_to_vary, param_vals_list)])

    plotter.plot_random_DA_weights_over_time(
            weights_list, params_to_vary, param_vals_list, models, 
            save_folder=plot_folder, silent=silent)

def run_reward_prediction_phase_planes_density(
        num_samples, param_to_vary, param_vals, models, sim_kwargs,
        no_dynamics=False, plot_folder=None, silent=False):
    weights = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model', param_to_vary],
            [models, param_vals])

    plotter.plot_reward_prediction_phase_planes_density(
            weights, param_to_vary, param_vals, models, sim_kwargs,
            no_dynamics=no_dynamics, save_folder=plot_folder, silent=silent)

def run_reward_prediction_phase_planes_task_switching_weights_over_time_slow_fast_density(
        num_samples, Rstar_vals, switch_period_slow, models, sim_kwargs,
        plot_folder=None, silent=False):
    weights = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model', 'Rstar', 'switch_period'],
            [models, Rstar_vals, [switch_period_slow, 1]])

    plotter.plot_reward_prediction_phase_planes_task_switching_density(weights, 
                                        Rstar_vals, models, sim_kwargs,
                                        save_folder=plot_folder, silent=silent)


def run_action_selection_weights_over_time(num_samples, models, sim_kwargs,
            plot_folder=None, silent=False):
    weights, actions, DA = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model'],
            [models])

    plotter.plot_action_selection_weights_over_time(
            weights, actions, models,
            save_folder=plot_folder, silent=silent)


def run_action_selection_weight_limits_delay(num_samples, models, delay_vals, persistent_activity_rate,
            sim_kwargs, plot_folder=None, silent=False):
    weights, actions, DA = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model', 'persistent_activity_rate', 'T_del'],
            [models, [None, persistent_activity_rate], delay_vals])

    plotter.plot_action_selection_weight_limits_delay(weights, delay_vals, models,
                                        save_folder=plot_folder, silent=silent)

def run_action_selection_weights_over_time_contingency_switching(num_samples, models, sim_kwargs,
            plot_folder=None, silent=False):
    weights, actions, DA = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model'],
            [models])
    if len(models) > 1:
        plotter.plot_action_selection_weights_over_time_contingency_switching(
            weights, actions, models, sim_kwargs['switch_period'],
            save_folder=plot_folder, silent=silent)
    else:
        plotter.plot_action_selection_weights_over_time_contingency_switching_one_model(
            weights, actions, sim_kwargs['switch_period'],
            save_folder=plot_folder, silent=silent)

def run_action_selection_task_switching_density(num_samples, models, sim_kwargs,
            plot_folder=None, silent=False):

    weights, actions, DA = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model'],
            [models])
    
    plotter.plot_action_selection_task_switching_density(weights, actions, models, sim_kwargs,
                                        save_folder=plot_folder, silent=silent)

def run_single_model_all_settings(num_samples, model, winit_vals, sim_kwargs_random_DA,
            sim_kwargs_reward_prediction, sim_kwargs_action_selection,
            plot_folder=None, silent=False):
    weights_random_DA_winit = data_collection.get_weights_over_time(
            num_samples, {**sim_kwargs_random_DA, 'model': model},
            ['w_init'],
            [winit_vals])
    weights_reward_prediction = data_collection.get_weights_over_time(
            num_samples, sim_kwargs_reward_prediction,
            ['model'],
            [[model]])
    weights_action_selection, actions, DA = data_collection.get_weights_over_time(
            num_samples, sim_kwargs_action_selection,
            ['model'],
            [[model]])
    
    # Run without actions
    plotter.plot_single_model_all_settings(model, weights_random_DA_winit, winit_vals, 
                             weights_reward_prediction, sim_kwargs_reward_prediction,
                             weights_action_selection, None,
                             save_folder=plot_folder, silent=silent)

def run_reward_prediction_instantaneous_drift(num_samples, models, param1, param1_vals,
            param2, param2_vals, sim_kwargs, val_for_predictions=None, log_x=False, scale_y=False, 
            plot_folder=None, silent=False):

    weights = data_collection.get_weights_over_time(
            num_samples, sim_kwargs,
            ['model', param1, param2],
            [models, param1_vals, param2_vals])
    
    plotter.plot_reward_prediction_instantaneous_drift(weights, param1, param1_vals,
                                        param2, param2_vals, models,
                                        sim_kwargs=sim_kwargs, val_for_predictions=val_for_predictions,
                                        log_x=log_x, scale_y=scale_y,
                                        save_folder=plot_folder, silent=silent)

simulation_kwargs_random_DA = {
    'N': 1,
    'r': 5,
    'task': 'random',
    'num_steps': 100,
    'DA_mean': 0,
    'DA_std': 1,
    'alpha': 1,
    'tau': 0.02, # From Gutig et al. 2003, Bi & Poo 1998 (also see Bi & Poo 2001)
    'tau_dop': 1, # From Riley et al. 2024
    'tau_eli': 1, # From Fisher et al. 2017, Yagishita et al. 2014 (but see Shindou et al. 2019)
    'lambda_': 0.01,
    'eps': 0.001, # Need exp(-eps/tau) ~= 1
    'w_init': 0.5,
    'r_dop': 1/6,
    # These params not used for random DA
    'Rstar': None,
    'T_del': 0,
    'T_win': 0,
    'beta': None,
}

simulation_kwargs_reward_prediction = {
    'N': 2,
    'r': [15, 10],
    'Rstar': 7.5,
    'task': 'reward prediction',
    'num_steps': 100,
    'alpha': 1,
    'tau': 0.02, # From Gutig et al.
    'tau_dop': 1, # From Riley et al. 2024
    'tau_eli': 1, # From Fisher et al. 2017, Yagishita et al. 2014 (but see Shindou et al. 2019)
    'T_del': 3,
    'T_win': 1,
    'lambda_': 0.0033,
    'eps': 0.001, # Need exp(-eps/tau) ~= 1
    'w_init': 0.33,
    'r_dop': 1/7,
    # These params not used for reward prediction
    'DA_mean': None,
    'DA_std': None,
    'beta': None,
}

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
    'T_del': 0, # No delay
    'T_win': 1,
    'lambda_': 0.025,
    'eps': 0.001, # Need exp(-eps/tau) ~= 1
    'w_init': 0.5,
    'r_dop': 1/7,
    'beta': 100000, # Arbitrary large number
    'persistent_activity_rate': None,
    # These params not used for reward prediction
    'DA_mean': None,
    'DA_std': None,
}


if __name__ == '__main__':
    import argparse
    all_figs = [f'Fig{i}' for i in range(1, 19)]
    parser = argparse.ArgumentParser('Run experiments in the paper "Distinct dopaminergic spike-timing-dependent plasticity rules are suited to different functional roles".')
    parser.add_argument("-s", "--samples", type=int, default=1000,
                        help="Number of samples")
    parser.add_argument("-p", "--procs", type=int, default=1,
                        help="Number of processes")
    parser.add_argument("-f", "--folder", type=str, default="Plots",
                        help="Folder to save results in")
    parser.add_argument("figures", nargs="*", type=str, default=all_figs,
                        help="Figures to generate. Either strings 'Fig1', 'Fig2', etc. or figure number.")
    args = parser.parse_args()
    num_samples = args.samples
    project_plot_folder = args.folder
    figs = args.figures
    # Set number of processes to use
    data_collection.NUM_PROCS = args.procs

    if 'Fig7' in figs or '7' in figs:
        instantaneous_drift_kwargs = copy.copy(simulation_kwargs_reward_prediction)
        instantaneous_drift_kwargs['N'] = 1
        instantaneous_drift_kwargs['r'] = 10
        instantaneous_drift_kwargs['Rstar'] = 6
        instantaneous_drift_kwargs['num_steps'] = 2
        instantaneous_drift_kwargs['lambda_'] = 0.0005
        run_reward_prediction_instantaneous_drift(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            param1='eps',
            param1_vals=[0.001, 0.005],
            param2='w_init',
            param2_vals=np.linspace(0, 1, 11),
            sim_kwargs=instantaneous_drift_kwargs,
            val_for_predictions=0.001,
            plot_folder=os.path.join(project_plot_folder, f'Fig7'),
            silent=True)

    if 'Fig6' in figs or '6' in figs:
        instantaneous_drift_kwargs = copy.copy(simulation_kwargs_reward_prediction)
        instantaneous_drift_kwargs['N'] = 1
        instantaneous_drift_kwargs['r'] = 10
        instantaneous_drift_kwargs['Rstar'] = 6
        instantaneous_drift_kwargs['lambda_'] = 0.0005
        instantaneous_drift_kwargs['num_steps'] = 2
        run_reward_prediction_instantaneous_drift(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            param1='T_del',
            param1_vals=[0, 3],
            param2='w_init',
            param2_vals=np.linspace(0, 1, 11),
            sim_kwargs=instantaneous_drift_kwargs,
            val_for_predictions=3,
            plot_folder=os.path.join(project_plot_folder, f'Fig6'),
            silent=True)

    if 'Fig16' in figs or '16' in figs:
        run_reward_prediction_phase_planes_density(
            num_samples=num_samples,
            param_to_vary='gamma',
            param_vals=[1, 2, 3],
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs={**simulation_kwargs_reward_prediction, 'single_trace': True},
            no_dynamics=True,
            plot_folder=os.path.join(project_plot_folder, f'Fig16'),
            silent=True)

    if 'Fig15' in figs or '15' in figs:
        run_random_DA_weights_over_time(
            num_samples=num_samples,
            params_to_vary= ['w_init', 'alpha', 'gamma'],
            param_vals_list=[[0.25, 0.5, 0.75], [1, 2, 3], [1, 2, 3]],
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs={**simulation_kwargs_random_DA, 'single_trace': True},
            plot_folder=os.path.join(project_plot_folder, f'Fig15'),
            silent=True)

    if 'Fig18' in figs or '18' in figs:
        contingency_switching_kwargs = copy.copy(simulation_kwargs_action_selection)
        contingency_switching_kwargs['switch_period'] = simulation_kwargs_action_selection['num_steps']
        contingency_switching_kwargs['num_steps'] = 5*simulation_kwargs_action_selection['num_steps']
        contingency_switching_kwargs['lambda_'] = 0.05
        contingency_switching_kwargs['r'] = [10, 10] # Keep the same r but switch Rstar
        contingency_switching_kwargs['Rstar'] = [[2,1], [1,2]]
        run_action_selection_weights_over_time_contingency_switching(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs={**contingency_switching_kwargs, 'single_trace': True},
            plot_folder=os.path.join(project_plot_folder, f'Fig18'),
            silent=True)

    if 'Fig17' in figs or '17' in figs:
        run_action_selection_weights_over_time(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs={**simulation_kwargs_action_selection, 'single_trace': True},
            plot_folder=os.path.join(project_plot_folder, f'Fig17'),
            silent=True)
    
    if 'Fig14' in figs or '14' in figs:
        contingency_switching_kwargs = copy.copy(simulation_kwargs_action_selection)
        contingency_switching_kwargs['switch_period'] = simulation_kwargs_action_selection['num_steps']
        contingency_switching_kwargs['num_steps'] = 5*simulation_kwargs_action_selection['num_steps']
        contingency_switching_kwargs['lambda_'] = 4*simulation_kwargs_action_selection['lambda_']
        contingency_switching_kwargs['r'] = [10, 10] # Keep the same r but switch Rstar
        contingency_switching_kwargs['Rstar'] = [[2,1], [1,2]]
        run_action_selection_weights_over_time_contingency_switching(
            num_samples=num_samples,
            models=['symmetric'],
            sim_kwargs=contingency_switching_kwargs,
            plot_folder=os.path.join(project_plot_folder, f'Fig14'),
            silent=True)
    
    if 'Fig13' in figs or '13' in figs:
        run_single_model_all_settings(
            num_samples=num_samples,
            model='symmetric',
            winit_vals=[0.25, 0.5, 0.75],
            sim_kwargs_random_DA={**simulation_kwargs_random_DA, 
                                'lambda_': 2*simulation_kwargs_random_DA['lambda_']},
            sim_kwargs_reward_prediction={**simulation_kwargs_reward_prediction, 
                                        'lambda_': 2*simulation_kwargs_reward_prediction['lambda_']},
            sim_kwargs_action_selection={**simulation_kwargs_action_selection, 
                                        'lambda_': 2*simulation_kwargs_action_selection['lambda_']},
            plot_folder=os.path.join(project_plot_folder, f'Fig13'),
            silent=True)
    
    if 'Fig11' in figs or '11' in figs:
        task_switching_kwargs = copy.copy(simulation_kwargs_action_selection)
        task_switching_kwargs['num_steps'] = 5*simulation_kwargs_action_selection['num_steps']
        task_switching_kwargs['N'] = 2
        task_switching_kwargs['r'] = [[15,5], [5,15]]
        task_switching_kwargs['Rstar'] = [[2,1], [1,2]]
        task_switching_kwargs['lambda_'] = 0.005
        run_action_selection_task_switching_density(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs={'switch_period': simulation_kwargs_action_selection['num_steps'], 
                        **task_switching_kwargs},
            plot_folder=os.path.join(project_plot_folder, f'Fig11'),
            silent=True)
    
    if 'Fig12' in figs or '12' in figs:
        task_switching_kwargs = copy.copy(simulation_kwargs_action_selection)
        task_switching_kwargs['num_steps'] = 5*simulation_kwargs_action_selection['num_steps']
        task_switching_kwargs['N'] = 2
        task_switching_kwargs['r'] = [[15,5], [5,15]]
        task_switching_kwargs['Rstar'] = [[2,1], [1,2]]
        task_switching_kwargs['lambda_'] = 0.005
        run_action_selection_task_switching_density(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs={'switch_period': 1, **task_switching_kwargs},
            plot_folder=os.path.join(project_plot_folder, f'Fig12'),
            silent=True)

    if 'Fig10' in figs or '10' in figs:
        contingency_switching_kwargs = copy.copy(simulation_kwargs_action_selection)
        contingency_switching_kwargs['switch_period'] = simulation_kwargs_action_selection['num_steps']
        contingency_switching_kwargs['num_steps'] = 5*simulation_kwargs_action_selection['num_steps']
        contingency_switching_kwargs['lambda_'] = 0.05
        contingency_switching_kwargs['r'] = [10, 10] # Keep the same r but switch Rstar
        contingency_switching_kwargs['Rstar'] = [[2,1], [1,2]]
        run_action_selection_weights_over_time_contingency_switching(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs=contingency_switching_kwargs,
            plot_folder=os.path.join(project_plot_folder, f'Fig10'),
            silent=True)

    if 'Fig9' in figs or '9' in figs:
        run_action_selection_weight_limits_delay(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            delay_vals=[0,0.5,1,1.5,2,2.5,3],
            persistent_activity_rate=0.7,
            sim_kwargs=simulation_kwargs_action_selection,
            plot_folder=os.path.join(project_plot_folder, f'Fig9'),
            silent=True)
    
    if 'Fig8' in figs or '8' in figs:
        run_action_selection_weights_over_time(
            num_samples=num_samples,
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs=simulation_kwargs_action_selection,
            plot_folder=os.path.join(project_plot_folder, f'Fig8'),
            silent=True)
    
    if 'Fig5' in figs or '5' in figs:
        run_reward_prediction_phase_planes_task_switching_weights_over_time_slow_fast_density(
            num_samples=num_samples,
            Rstar_vals=[[6,6], [7,4]],
            switch_period_slow=int(simulation_kwargs_reward_prediction['num_steps']/5),
            models=['additive', 'multiplicative'],
            sim_kwargs={**simulation_kwargs_reward_prediction, 'r': [[15,5], [10,20]], 'w_init': 0.5},
            plot_folder=os.path.join(project_plot_folder, f'Fig5'),
            silent=True)

    if 'Fig4' in figs or '4' in figs:
        run_reward_prediction_phase_planes_density(
            num_samples=num_samples,
            param_to_vary='alpha',
            param_vals=[1, 2, 3],
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs=simulation_kwargs_reward_prediction,
            plot_folder=os.path.join(project_plot_folder, f'Fig4'),
            silent=True)

    if 'Fig3' in figs or '3' in figs:
        run_random_DA_weights_over_time(
            num_samples=num_samples,
            params_to_vary= ['w_init', 'alpha'],
            param_vals_list=[[0.25, 0.5, 0.75], [1, 2, 3]],
            models=['additive', 'multiplicative', 'corticostriatal'],
            sim_kwargs=simulation_kwargs_random_DA,
            plot_folder=os.path.join(project_plot_folder, f'Fig3'),
            silent=True)