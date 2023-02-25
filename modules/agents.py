"""Agents module."""

from typing import Tuple, Dict
from functools import partial
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from modules.trajectory import Trajectory, Buffer


def create_agent(
    observation: Dict[str, np.ndarray],
    num_actions: int,
    agent_config=None,
    ):
    """Create selected agent."""
    agent = DefaultAgent(
        observation=observation,
        num_actions=num_actions,
        **agent_config)
    return agent


class DefaultAgent:
    """Advantage actor-critic agent that responds with discrete actions."""

    def __init__(self,
            observation: np.ndarray,
            random_seed: int = 42,
            num_lstm_units: int = 64,
            num_actions: int = 2,
            learning_rate_start: float = 0.0001,
            learning_rate_end: float = 0.0001,
            total_training_steps: int = 100000,
            gamma: float = 1.0,
            v_loss_coef: float = 0.5,
            e_loss_coef_start: float = 0.0,
            e_loss_coef_end: float = 0.0,
            e_loss_decay_factor: float = 3,
            max_unroll_steps: int = 300,
            global_norm_grad_clip: float = 50.0,
            ) -> None:

        super().__init__()
        self.name = 'Default'
        self._key = jax.random.PRNGKey(random_seed)
        self._num_lstm_units = num_lstm_units
        self._num_actions = num_actions
        self._learning_rate_start = learning_rate_start
        self._learning_rate_end = learning_rate_end
        self._gamma = gamma
        self._v_loss_coef = v_loss_coef
        self._max_unroll_steps = max_unroll_steps
        self._total_training_steps = total_training_steps

        # Approximate how many learning steps
        approx_learning_steps = self._total_training_steps // self._max_unroll_steps
        self._learning_rate_decay_steps = approx_learning_steps

        # Get schedule for annealing entropy loss term
        self.entropy_schedule = optax.polynomial_schedule(
                                    init_value=e_loss_coef_start,
                                    end_value=e_loss_coef_end,
                                    power=e_loss_decay_factor,
                                    transition_steps=self._learning_rate_decay_steps)
        self.e_loss_count = 0
        self.e_loss_coef = self.entropy_schedule(self.e_loss_count)

        self._global_norm_grad_clip = global_norm_grad_clip

        self.model = None
        self.params = None
        self.opt = None
        self.opt_state = None

        self.buffer = Buffer()

        self._init_model(observation)
        self._init_optimizer()

    def net(self,
            task_input: jnp.ndarray,
            lstm_state: hk.LSTMState,
            ) -> Tuple[jnp.ndarray, jnp.ndarray, hk.LSTMState]:
        """Actor-critic RNN with one LSTM layer.

        Args:
            task_input: Inputs [Time/Batch, Features].
            lstm_state: Previous LSTM hidden and cell state.

        Returns:
            pi_out: Activations for each action [num_actions, Time].
            v_out: Value baseline [1, Time].
            state: New LSTM hidden and cell state.
        """

        rnn_input = task_input
        rnn_input = jnp.expand_dims(rnn_input, 1) # rnn_output is [Time, Batch, Features]
        core = hk.LSTM(self._num_lstm_units, name='LSTM')
        lstm_output, state = hk.dynamic_unroll(core, rnn_input, lstm_state)

        pi_out = hk.BatchApply(hk.Linear(self._num_actions, name='Actions'))(lstm_output)
        pi_out = jnp.squeeze(pi_out)
        v_out = hk.BatchApply(hk.Linear(1, name='Values'))(lstm_output)
        v_out = jnp.squeeze(v_out)
        return pi_out, v_out, state, lstm_output

    def _init_model(self, observation):
        """Initialize model and parameters."""
        self.model = hk.transform(self.net)
        initial_lstm_state = self.get_initial_lstm_state()
        self.params = self.model.init(
            self._key,
            task_input=jnp.expand_dims(observation['vector_input'], 0),
            lstm_state=initial_lstm_state,
            )

    def get_initial_lstm_state(self) -> hk.LSTMState:
        """Create initial LSTM state of zeros."""
        initial_lstm_state = hk.LSTMState(
                hidden=jnp.zeros((1, self._num_lstm_units), dtype=jnp.float32),
                cell=jnp.zeros((1, self._num_lstm_units), dtype=jnp.float32))
        return initial_lstm_state

    def _init_optimizer(self) -> None:
        """Initialize Adam optimizer for training."""
        lr_schedule = optax.polynomial_schedule(
                                            init_value=self._learning_rate_start,
                                            end_value=self._learning_rate_end,
                                            power=1, # 1: Linear decay
                                            transition_steps=self._learning_rate_decay_steps,
                                            )
        self.opt = optax.chain(
            optax.adam(lr_schedule),
            optax.clip_by_global_norm(self._global_norm_grad_clip),
            )
        self.opt_state = self.opt.init(self.params)

    def get_action(self,
        observation: Dict[str, np.ndarray],
        lstm_state: hk.LSTMState
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, hk.LSTMState, np.ndarray]:
        """Wrapper for model_step."""
        new_key, action, pi_out, v_out, lstm_state, lstm_output = self.model_step(
            self._key,
            self.params,
            observation,
            lstm_state)
        self._key = new_key
        return action, pi_out, v_out, lstm_state, lstm_output

    @partial(jax.jit, static_argnums=(0,))
    def model_step(self, key, params, observation, lstm_state):
        """Step the model once and select action via softmax policy."""
        pi_out, v_out, lstm_state, lstm_output = self.model.apply(
            params,
            None,
            task_input=jnp.expand_dims(observation['vector_input'], 0),
            lstm_state=lstm_state,
            )
        new_key, action_key = jax.random.split(key)
        action = jax.random.choice(action_key,
            jnp.arange(self._num_actions),
            p=jnp.array(jax.nn.softmax(pi_out)))
        return new_key, action, pi_out, v_out, lstm_state, lstm_output

    def update(self, done, update_params=True):
        """Update model from an episode trajectory."""
        loss = None
        grads = None
        num_steps = 0
        if self.buffer.t == self._max_unroll_steps or done:
            num_steps = self.buffer.t
            trajectory = self.buffer.drain()
            trajectory = self.process_trajectory_observations(trajectory)
            loss, grads = self.calculate_grads(
                params=self.params,
                trajectory=trajectory,
                e_loss_coef=self.e_loss_coef)
            if update_params:
                self.params, self.opt_state = self.update_model(
                    grads,
                    self.params,
                    self.opt_state)
                self.decrement_e_loss()
        return loss, grads, int(num_steps)

    def process_trajectory_observations(self, trajectory) -> Dict[str, any]:
        """Unpack observations from dictionary into their own np.ndarray."""
        all_vector_input = jnp.stack([obs['vector_input'] for obs in trajectory.observations], axis=0)
        trajectory = {
            "vector_input": all_vector_input,
            "actions": trajectory.actions,
            "rewards": trajectory.rewards,
            "discounts": trajectory.discounts,
            "lstm_state": trajectory.lstm_state
        }
        return trajectory

    @partial(jax.jit, static_argnums=(0,))
    def calculate_grads(self, params, trajectory, e_loss_coef):
        """Calculate loss and gradients."""
        loss, grads = jax.value_and_grad(self.loss_fn)(
                                params,
                                trajectory,
                                e_loss_coef)
        return loss, grads

    @partial(jax.jit, static_argnums=(0,))
    def update_model(self, grads, params, opt_state):
        """Update parameters and optimizer state from gradients."""
        grads, opt_state = self.opt.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state

    def decrement_e_loss(self) -> None:
        self.e_loss_count += 1
        self.e_loss_coef = self.entropy_schedule(self.e_loss_count)

    def loss_fn(self, params: hk.Params, trajectory: Trajectory, e_loss_coef) -> jnp.ndarray:
        """Discrete actor-critic loss."""

        all_vector_input = trajectory['vector_input']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        discounts = trajectory['discounts']

        lstm_state = trajectory['lstm_state']

        # Run experienced trajectory through model
        pi_out, v_out, _, _ = self.model.apply(
            params,
            None,
            task_input=all_vector_input,
            lstm_state=lstm_state,
            )

        # Calculate discounted td errors
        batched_td_errors = jax.vmap(rlax.td_learning)
        td_errors = batched_td_errors(
            v_tm1=jnp.squeeze(v_out[:-1]),
            r_t=jnp.squeeze(rewards),
            discount_t=jnp.squeeze(discounts * self._gamma),
            v_t=jnp.squeeze(v_out[1:]),
        )

        # Critic loss
        critic_loss = jnp.mean(td_errors**2)

        # Actor loss
        actor_loss = rlax.policy_gradient_loss(
            logits_t=pi_out[:-1],
            a_t=actions,
            adv_t=td_errors,
            w_t=jnp.ones_like(td_errors))

        # Entropy loss
        action_probs = jax.nn.softmax(pi_out)
        e_loss = jnp.sum(jnp.sum(action_probs * jnp.log(action_probs + 1e-7), axis=-1))

        # Weighted sum of all loss terms
        all_loss = actor_loss + (self._v_loss_coef * critic_loss) + (e_loss_coef * e_loss)
        return all_loss
