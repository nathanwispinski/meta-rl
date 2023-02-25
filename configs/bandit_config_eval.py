import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.phase = 'eval'

    config.num_workers = 0
    config.num_evaluators = 1

    # Saving
    config.path = './'
    config.params_filename = 'train_test'

    # Evaluation
    config.random_seed = 42
    config.num_eval_episodes = 400
    config.log_dynamics = True

    # Environment
    config.environment = ml_collections.ConfigDict()

    # Evaluation environment
    config.eval_environment = ml_collections.ConfigDict()
    config.eval_environment.steps_per_episode = int(100)
    config.eval_environment.reward_structure = "correlated"

    # Agent
    config.agent = ml_collections.ConfigDict()

    return config