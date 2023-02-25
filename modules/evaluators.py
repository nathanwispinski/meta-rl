"""Module for multiprocessing evaluators."""

import sys
import time
import copy
import multiprocessing as mp
from typing import Sequence
from ml_collections import ConfigDict
import numpy as np
import jax
import csv

import modules.envs as envs
import modules.agents as agents

def create_evaluators(
    evaluator_type: str,
    config,
    manager,
    ) -> Sequence[any]:
    """Create multiprocessing evaluators."""
    num_evaluators = config.num_evaluators
    worker_seeds = np.random.randint(0, 42e5, size=(num_evaluators,))

    context = mp.get_context("spawn")
    all_evaluators = []
    for i in range(num_evaluators):
        if evaluator_type == 'train':
            evaluator = TrainingEvaluator
        elif evaluator_type == 'eval':
            evaluator = TestingEvaluator
        else:
            raise ValueError(f'Evaluator {evaluator_type} does not exist.')

        cur_evaluator = context.Process(
                target=evaluator,
                args=('evaluator_'+str(i), worker_seeds[i], config, manager),
                name='evaluator_'+str(i), daemon=True
            )

        all_evaluators.append(cur_evaluator)

    return all_evaluators


class TrainingEvaluator(mp.Process):
    """Evaluator class for each parallel agent-environment worker."""

    def __init__(self, name: str, seed: int, config: ConfigDict, parameter_manager: mp.Process) -> None:
        super(TrainingEvaluator, self).__init__()
        self.name = name
        self.eval(seed, config, parameter_manager)

    def eval(self, seed: int, config: ConfigDict, parameter_manager: mp.Process):
        """Check manager for current step, evaluate copy of agent, repeat."""

        parameter_manager.log_worker_message(self.name, 'Starting')

        jax.config.update('jax_platform_name', 'cpu') # Make sure workers run on CPUs

        np.random.seed(seed) # Each process should run on a different seed

        eval_env_config = config.eval_environment
        agent_config = config.agent
        path = config.path

        # Initialize environment
        env = envs.create_env(env_config=eval_env_config)
        observation = env.reset()

        # Initialize agent
        agent = agents.create_agent(
            observation=observation,
            num_actions=env.num_actions,
            agent_config=agent_config)

        initial_lstm_state = agent.get_initial_lstm_state()

        # Define csv headers
        headers = ['step', 'avg-reward']
        # Create csv file with headers for evaluation results
        with open(path + self.name + '.csv', "w", newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, dialect='excel')
            writer.writerow(headers)

        eval_every_steps =  config.eval_every_steps

        num_eval_episodes = config.num_eval_episodes
        steps_per_episode = env._steps_per_episode

        eval_count = 0

        while not parameter_manager.is_finished():

            # Evaluation
            if (parameter_manager.get_global_counters()[0] / eval_every_steps) >= eval_count:
                agent.params, _ = parameter_manager.get_global_params()
                cur_eval_step = parameter_manager.get_global_counters()[0]

                # Initialize variables for saving data
                all_actions = np.zeros((num_eval_episodes, steps_per_episode))
                all_rewards = np.zeros((num_eval_episodes, steps_per_episode))
                all_arm_probs = np.zeros((num_eval_episodes, env.num_actions))
                all_vector_inputs = np.zeros((num_eval_episodes, steps_per_episode, len(observation['vector_input'])))

                # Evaluate
                total_reward = 0

                for eval_ep in range(num_eval_episodes):
                    step = 0
                    done = False
                    lstm_state = initial_lstm_state
                    observation = env.reset()

                    while not done:
                        action, _, _, lstm_state, _ = agent.get_action(observation, lstm_state)
                        next_observation, reward, done, info = env.step(action)
                        observation = next_observation

                        all_actions[eval_ep, step] = int(action)
                        all_rewards[eval_ep, step] = float(reward)
                        all_arm_probs[eval_ep, :] = env._arm_probs
                        all_vector_inputs[eval_ep, step, :] = observation['vector_input']

                        step += 1
                        total_reward += reward

                # Export results to csv
                avg_total_reward = total_reward / (num_eval_episodes * steps_per_episode)
                summary_results = [cur_eval_step, avg_total_reward]
                with open(path + self.name + '.csv', "a", newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file, dialect='excel')
                    writer.writerow(summary_results)

                eval_count += 1

            time.sleep(10) # Sleep so that evaluator isn't overloading manager.

        # Exit after shutdown flag from manager
        parameter_manager.log_worker_message(self.name, 'Exiting')
        sys.exit()


class TestingEvaluator(mp.Process):
    """Evaluator class for each parallel agent-environment worker."""

    def __init__(self, name, seed, config, parameter_manager):
        super(TestingEvaluator, self).__init__()
        self.name = name
        self.eval(seed, config, parameter_manager)

    def eval(self, seed, config, parameter_manager):
        """Check manager for current episode, evaluate copy of agent, repeat."""

        parameter_manager.log_worker_message(self.name, 'Starting')

        jax.config.update('jax_platform_name', 'cpu') # Make sure workers run on CPUs

        np.random.seed(seed) # Each process should run on a different seed

        path = config.path
        eval_env_config = config.eval_environment
        agent_config = config.agent
        log_dynamics = config.log_dynamics
        steps_per_episode = eval_env_config.steps_per_episode
        num_evaluators = config.num_evaluators
        # If distributing evaluation between multiple evaluators, split evaluation episodes
        num_eval_episodes = int(np.ceil(config.num_eval_episodes / num_evaluators))

        # Initialize environment
        env = envs.create_env(env_config=eval_env_config)
        observation = env.reset()

        # Initialize agent
        agent = agents.create_agent(
            observation=observation,
            num_actions=env.num_actions,
            agent_config=agent_config)

        initial_lstm_state = agent.get_initial_lstm_state()

        # Overwrite agent with saved parameters from training
        agent.params, _ = parameter_manager.get_global_params()

        # Initialize variables for saving data to .pickle
        data = []

        # Initialize data
        episode_data_keys = ['seed']
        if log_dynamics:
            episode_data_keys.extend([
                'action', 'pi_out', 'v_out', 'lstm_state', 'lstm_output',
                'reward', 'arm_probs', 'vector_input',
            ])
        empty_episode_data = {key: [] for key in episode_data_keys}

        # Evaluate
        for eval_ep in range(num_eval_episodes):

            step = 0
            done = False
            lstm_state = initial_lstm_state
            observation = env.reset()

            # Initialize data at start of each episode
            episode_data = copy.deepcopy(empty_episode_data)
            episode_data['seed'] = np.random.get_state()

            while not done:
                action, pi_out, v_out, lstm_state, lstm_output = agent.get_action(observation, lstm_state)
                next_observation, reward, done, info = env.step(action)
                observation = next_observation

                # Save data for analysis.ipynb
                if log_dynamics:
                    episode_data['action'].append(np.array(action))
                    episode_data['pi_out'].append(np.array(pi_out))
                    episode_data['v_out'].append(np.array(v_out))
                    episode_data['lstm_state'].append(np.array(lstm_state.hidden))
                    episode_data['lstm_output'].append(np.array(lstm_output))
                    episode_data['reward'].append(reward)
                    episode_data['arm_probs'].append(env._arm_probs)
                    episode_data['vector_input'].append(observation['vector_input'])

                step += 1

            # Append data at the end of each episode
            data.append(episode_data)

        # Send results to parameter manager
        parameter_manager.append_testing_data(data)

        # Exit after evaluation done
        parameter_manager.log_worker_message(self.name, 'Exiting')
        sys.exit()
