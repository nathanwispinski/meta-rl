"""Launch single-core training."""

import pickle
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import modules.envs as envs
import modules.agents as agents
import modules.loggers as loggers

FLAGS = flags.FLAGS

# PROD
# config_flags.DEFINE_config_file('config') # NOTE: this is prod

# DEBUG
from configs.bandit_config_train import get_config
config_flags.DEFINE_config_dict('config', get_config()) # NOTE: this is debug


def main(_):
    """Train model."""

    config = FLAGS.config

    env_config = config.environment
    agent_config = config.agent
    path = config.path
    params_filename = config.params_filename
    total_training_steps = agent_config.total_training_steps

    # Initialize environment
    env = envs.create_env(env_config)
    observation = env.reset()

    # Initialize agent
    agent = agents.create_agent(
        agent_config=agent_config,
        observation=observation,
        num_actions=env.num_actions,
    )
    initial_lstm_state = agent.get_initial_lstm_state()
    lstm_state = initial_lstm_state

    # Initialize logger
    logger = loggers.create_logger(logger_name=env_config.env_name, config=config)

    step, episode, loss = 0, 0, 0

    # Training loop
    while step < total_training_steps:

        action, pi_out, v_out, new_lstm_state, _ = agent.get_action(observation, lstm_state)
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

        logger.log_step(
            global_step=step,
            worker_step=step,
            reward=reward,
            info=info,
            loss=loss,
            entropy_coef=agent.e_loss_coef,
        )

        loss, grads, num_steps = agent.update(done)
        step += 1

        if done:
            episode += 1
            done = False
            lstm_state = initial_lstm_state
            observation = env.reset()

    # Save final trained parameters
    results = {
        "params": agent.params,
        "config": config.to_dict(),
    }
    with open(path + '/' + params_filename + '.pickle', 'wb') as fp:
        pickle.dump([results], fp)
    logging.info("Saved parameters.")
    logging.info("Done training.")

if __name__ == "__main__":
    app.run(main)
