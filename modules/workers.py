"""Module for multiprocessing workers."""

import sys
import multiprocessing as mp
from typing import Sequence
from ml_collections import ConfigDict
import numpy as np
import jax

import modules.envs as envs
import modules.agents as agents

def create_workers(
    worker_type: str,
    config: ConfigDict,
    manager: mp.Process,
    ) -> Sequence[any]:
    """Create multiprocessing workers."""
    num_workers = config.num_workers
    worker_seeds = np.random.randint(0, 42e5, size=(num_workers,))

    context = mp.get_context("spawn")
    all_workers = []
    for i in range(num_workers):
        if worker_type == 'train':
            worker = context.Process(
                    target=TrainingWorker,
                    args=('worker_'+str(i), worker_seeds[i], config, manager),
                    name='worker_'+str(i), daemon=True
                )
        else:
            raise ValueError(f'Worker {worker_type} does not exist.')

        all_workers.append(worker)

    return all_workers


class TrainingWorker(mp.Process):
    """Worker class for each parallel agent-environment worker."""

    def __init__(self, name: str, seed: int, config: ConfigDict, parameter_manager: mp.Process) -> None:
        super(TrainingWorker, self).__init__()
        self.name = name
        self.train(seed, config, parameter_manager)

    def train(self, seed: int, config: ConfigDict, parameter_manager: mp.Process) -> None:
        """Get global params, run for max unroll steps, send gradients to manager, repeat."""

        parameter_manager.log_worker_message(self.name, 'Starting')

        jax.config.update('jax_platform_name', 'cpu') # Make sure workers run on CPUs

        np.random.seed(seed) # Each process should run on a different seed

        env_config = config.environment
        agent_config = config.agent
        total_training_steps = agent_config.total_training_steps

        # Initialize environment
        env = envs.create_env(env_config)
        observation = env.reset()

        # Initialize agent
        agent = agents.create_agent(
            observation=observation,
            num_actions=env.num_actions,
            agent_config=agent_config)

        # Overwrite local agent params with global params
        agent.params, agent.e_loss_coef = parameter_manager.get_global_params()

        initial_lstm_state = agent.get_initial_lstm_state()
        lstm_state = initial_lstm_state

        step, episode, loss = 0, 0, 0
        while step < total_training_steps:

            action, _, v_out, new_lstm_state, _ = agent.get_action(observation, lstm_state)
            next_observation, reward, done, info = env.step(action)

            agent.buffer.append(
                obs=observation,
                action=action,
                reward=reward,
                next_obs=next_observation,
                done=done,
                lstm_state=lstm_state,
            )

            observation = next_observation
            lstm_state = new_lstm_state

            if self.name == 'worker_0':
                parameter_manager.log_step_results(
                    worker_step=step,
                    reward=reward,
                    info=info,
                    loss=loss,
                )

            loss, grads, num_steps = agent.update(done, update_params=False)
            step += 1

            if grads is not None:
                agent.params, agent.e_loss_coef = parameter_manager.update(grads, done, num_steps)
                if parameter_manager.is_finished():
                    parameter_manager.log_worker_message(self.name, 'Exiting')
                    sys.exit()

            if done:
                episode += 1
                done = False
                lstm_state = initial_lstm_state
                observation = env.reset()
