"""Module for multiprocessing managers."""

from typing import Dict, Tuple
import copy
import pickle
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from absl import logging
from ml_collections import ConfigDict
import numpy as np

import modules.envs as envs
import modules.agents as agents
import modules.loggers as loggers

class BaseManagerClass(BaseManager):
    pass

lock = mp.Lock()

def create_manager(manager_type: str, config: ConfigDict):
    """Create a multiprocessing manager."""

    BaseManagerClass.register('ParamManager', DistributedManager, exposed = [
        'is_finished',
        'get_global_counters',
        'get_global_params',
        'update',
        'log_step_results',
        'append_testing_data',
        'log_worker_message'])
    mymanager = BaseManagerClass()
    mymanager.start()

    if manager_type == "train":
        parameter_manager = mymanager.ParamManager(config=config)

    elif manager_type == "eval":
        # Retain settings in training config unless overridden by eval config
        path = config.path
        params_filename = config.params_filename
        with open(path + '/' + params_filename + '.pickle', 'rb') as fp:
            training_results = pickle.load(fp)
        training_config = training_results[0]['config']
        training_params = training_results[0]['params']
        training_config = ConfigDict(training_config)
        old_training_config = copy.deepcopy(training_config)
        training_config.update(config)
        config = copy.deepcopy(training_config)
        parameter_manager = mymanager.ParamManager(
            config=config,
            training_config=old_training_config,
            training_params=training_params)

    else:
        raise ValueError(f'Manager {manager_type} does not exist.')

    return mymanager, parameter_manager, config


class DistributedManager(object):
    """Global state and parameter manager."""

    def __init__(self, config: ConfigDict, training_config=None, training_params=None) -> None:
        self.name = "Manager"
        self.config = config
        self.training_config = training_config
        self.path = config.path
        self.params_filename = config.params_filename
        agent_config = config.agent
        env_config = config.environment
        self.total_training_steps = agent_config.total_training_steps

        self.global_steps = 0
        self.global_episodes = 0

        # Initialize environment
        env = envs.create_env(env_config)
        observation = env.reset()

        # Initialize agent
        self.agent = agents.create_agent(
            observation=observation,
            num_actions=env.num_actions,
            agent_config=agent_config)

        # Overwrite agent with saved parameters from training
        if training_params is not None:
            self.agent.params = training_params

        # Initialize logger
        self.logger = loggers.create_logger(logger_name=env_config.env_name, config=config)
        self._last_worker_loss = 0.

        self.exit_flag = False

        self.num_evaluators = config.num_evaluators
        self.evaluation_results = []
        self.evaluators_done = 0

    def is_finished(self) -> bool:
        """Send exit flag to workers and evaluators if work is done."""
        return self.exit_flag

    def get_global_counters(self) -> Tuple[int, int]:
        tmp_global_steps = self.global_steps
        tmp_global_episodes = self.global_episodes
        return tmp_global_steps, tmp_global_episodes

    def _increment_global_counters(self, done: bool, num_steps: int) -> None:
        self.global_steps += num_steps

        if done:
            self.global_episodes += 1
        if self.global_steps >= self.total_training_steps and not self.exit_flag:
            self.save_model()
            self.exit_flag = True

    def save_model(self) -> None:
        """Save model and config."""
        results = {
            "params": self.agent.params,
            "config": self.config.to_dict(),
        }
        with open(self.path + '/' + self.params_filename + '.pickle', 'wb') as fp:
            pickle.dump([results], fp)
        logging.info("Saved parameters.")

    def get_global_params(self) -> None:
        with lock:
            tmp_params = copy.deepcopy(self.agent.params)
            tmp_e_loss_coef = copy.deepcopy(self.agent.e_loss_coef)
            return tmp_params, tmp_e_loss_coef

    def update(self, grads, done: bool, num_steps: int):
        """Apply gradients, and update params and optimizer state."""
        with lock:
            self.agent.params, self.agent.opt_state = self.agent.update_model(
                grads,
                self.agent.params,
                self.agent.opt_state)
            self.agent.decrement_e_loss()
            self._increment_global_counters(done, num_steps)
            # Return new params
            tmp_params = copy.deepcopy(self.agent.params)
            tmp_e_loss_coef = copy.deepcopy(self.agent.e_loss_coef)
            return tmp_params, tmp_e_loss_coef

    def append_testing_data(self, testing_results):
        with lock:
            self.evaluation_results.extend(testing_results)
            self.evaluators_done += 1
            if self.evaluators_done == self.num_evaluators:

                # Log some simple summary metrics
                total_reward = np.sum([np.sum(trial['reward']) for trial in self.evaluation_results])
                logging.info(f"Total reward: {total_reward}")

                results = {
                    "eval_config": self.config,
                    "training_config": self.training_config,
                    "params": self.agent.params,
                    "data": self.evaluation_results,
                }
                # Evaluation done, save results from distributed testing
                with open(self.path + '/data_' + self.params_filename + '.pickle', 'wb') as fp:
                    pickle.dump([results], fp)
                logging.info("Saved distributed testing results. Manager shutting down.")
                self.exit_flag = True
                self.logger.close_logger()

    def log_step_results(self,
        worker_step: int,
        reward: float,
        info: Dict[str, any],
        loss: float,
    ) -> None:
        if loss is not None:
            self._last_worker_loss = loss
        self.logger.log_step(
            global_step=self.global_steps,
            worker_step=worker_step,
            reward=reward,
            info=info,
            loss=self._last_worker_loss,
            entropy_coef=self.agent.e_loss_coef,
        )

    def log_worker_message(self, worker_name: str, message: str) -> None:
        logging.info(f"{worker_name}\t|"
        f"{message}\t|")
