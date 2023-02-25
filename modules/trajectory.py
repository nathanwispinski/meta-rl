"""Module for reinforcement learning trajectories for recurrent agents."""

from typing import NamedTuple, Dict
import numpy as np
import haiku as hk

class Trajectory(NamedTuple):
    """Class to store agent experience."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    lstm_state: hk.LSTMState

class Buffer:
    """Buffer for reinforcement learning trajectories. See deepmind/bsuite."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset buffers."""
        self._observations = []
        self._actions = []
        self._rewards = []
        self._discounts = []
        self._lstm_state = []
        self.t = 0

    def append(
        self,
        obs: Dict,
        action: any = None,
        reward: float = None,
        next_obs: np.ndarray = None,
        lstm_state: hk.LSTMState = None,
        done: bool = None,
    ) -> None:
        """Appends an observation, action, reward, and discount to the buffer."""
        if len(self._observations) == 0:
            self._observations.append(obs)
            self._lstm_state = lstm_state

        self._observations.append(next_obs)
        self._actions.append(action)
        self._rewards.append(float(reward))
        if done:
            self._discounts.append(0.0)
        else:
            self._discounts.append(1.0)
        self.t += 1

    def drain(self) -> Trajectory:
        """Return Trajectory of experience, and then clear Trajectory."""
        trajectory = Trajectory(
            observations = np.array(self._observations),
            actions = np.array(self._actions),
            rewards = np.array(self._rewards),
            discounts = np.array(self._discounts),
            lstm_state = self._lstm_state,
        )
        self.reset()
        return trajectory
