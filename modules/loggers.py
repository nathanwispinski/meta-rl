"""Loggers that take data, process it, and periodically log metrics to file."""

from typing import Dict
import time
from ml_collections import ConfigDict
from absl import logging


def create_logger(logger_name: str, config=None, log_to_console=False):
    """Create a logger."""
    if logger_name == "bandit":
        logger = BanditLogger(config=config, log_to_console=log_to_console)
    else:
        raise ValueError(f'Logger {logger_name} does not exist.')
    return logger

def initialize_logger(config: ConfigDict) -> None:
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    logging.get_absl_handler().use_absl_log_file('log', config.path)
    logging.get_absl_handler().setFormatter(None)
    logging.info(f"{config.to_json_best_effort()}")
    logging.info(f"Started at: {cur_time}")

class BanditLogger:
    """Logger for bandit env."""

    def __init__(self, config: ConfigDict, log_to_console: bool) -> None:
        self._log_to_console = log_to_console
        self._log_every_steps = config.log_every_steps
        initialize_logger(config)

        self._episode_reward_log = 0
        self._log_step_count = 0
        self._start_time = time.time()

        self._step_reward_log = 0

    def log_step(
        self,
        global_step: int,
        worker_step: int,
        reward: float,
        info: Dict,
        loss: float,
        entropy_coef: float,
    )  -> None:
        """Method to call on every step to log step or episode metrics."""
        # Step loggers
        self._step_reward_log += reward

        if (worker_step / self._log_every_steps) >= self._log_step_count:
            self._log_step_count += 1
            batch_time = time.time() - self._start_time
            self._start_time = time.time()
            if self._log_to_console:
                print(
                    f"Global step:\t{global_step}\t|"
                    f" Worker step:\t{worker_step}\t|"
                    f" T:\t{batch_time:0.2f}\t|"
                    f" Mean Reward:\t{(self._step_reward_log / self._log_every_steps):0.4f}\t|"
                    f" Entropy coef:\t{entropy_coef:0.4f}\t|"
                    f" Loss:\t{loss:0.5f}\t|"
                    )
            else:
                logging.info(
                    f"Global step:\t{global_step}\t|"
                    f" Worker step:\t{worker_step}\t|"
                    f" T:\t{batch_time:0.2f}\t|"
                    f" Mean Reward:\t{(self._step_reward_log / self._log_every_steps):0.4f}\t|"
                    f" Entropy coef:\t{entropy_coef:0.4f}\t|"
                    f" Loss:\t{loss:0.5f}\t|"
                    )
            self._step_reward_log = 0

        # Episode loggers

    def close_logger(self) -> None:
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        logging.info('Shutting down.')
        logging.info(f"Ended at: {cur_time}")
