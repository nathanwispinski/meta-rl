import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.phase = 'train'

    # Saving
    config.path = './'
    config.params_filename = 'train_test'

    # Training
    config.random_seed = 42
    config.num_workers = 5
    config.num_evaluators = 1
    config.eval_every_steps = 20_000*10
    config.num_eval_episodes = 400

    # Logging
    config.log_every_steps = int(100*200)

    # Environment
    config.environment = ml_collections.ConfigDict()
    config.environment.env_name = "bandit"
    config.environment.steps_per_episode = int(100)
    config.environment.reward_structure = "correlated"

    # Evaluation environment
    config.eval_environment = ml_collections.ConfigDict()
    config.eval_environment.steps_per_episode = int(100)
    config.eval_environment.reward_structure = "correlated"

    # Agent
    config.agent = ml_collections.ConfigDict()
    config.agent.total_training_steps = int(100*20_000)
    config.agent.random_seed = 42
    config.agent.num_lstm_units = 48
    config.agent.learning_rate_start = 3e-4
    config.agent.learning_rate_end = 0.0
    config.agent.gamma = 0.9
    config.agent.v_loss_coef = 0.05
    config.agent.e_loss_coef_start = 0.0
    config.agent.e_loss_coef_end = 0.0
    config.agent.e_loss_decay_factor = 3
    config.agent.max_unroll_steps = 200
    config.agent.global_norm_grad_clip = 50.0

    return config
