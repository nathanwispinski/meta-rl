"""Launch distributed training or testing."""

import sys
from absl import app, flags
from ml_collections.config_flags import config_flags
import numpy as np
import jax

import modules.managers as managers
import modules.workers as workers
import modules.evaluators as evaluators

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config') # NOTE: this is prod

def main(_):
    """Launch distributed training or testing.

    Launches one manager, some number of workers, and some number of evaluators.
    Workers take a copy of the agent and run it through their own environment.
    Workers then compute the gradient on their experience and send gradients to the manager.
    Manager updates the global agent parameters with each worker's gradients.
    Evaluators take a periodic copy of the agent and evaluate agent performance.
    """

    jax.config.update('jax_platform_name', 'cpu') # Make sure main() runs on CPU

    config = FLAGS.config
    phase = config.phase

    # Random seeds
    random_seed = config.random_seed
    np.random.seed(random_seed)

    started_manager, manager, config = managers.create_manager(
        manager_type=phase,
        config=config
    )

    all_workers = workers.create_workers(
        worker_type=phase,
        config=config,
        manager=manager,
    )

    all_evaluators = evaluators.create_evaluators(
        evaluator_type=phase,
        config=config,
        manager=manager,
    )

    all_subprocesses = all_workers + all_evaluators
    for subprocess in all_subprocesses:
        subprocess.start()
    for subprocess in all_subprocesses:
        subprocess.join()

    # All workers done, save and shutdown manager
    started_manager.shutdown()
    sys.exit()

if __name__ == "__main__":
    app.run(main)
