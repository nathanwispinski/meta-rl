"""Environments module."""

from typing import Tuple, Dict, Any
import numpy as np


def create_env(env_config=None):
    """Create a reinforcement learning environment."""
    env = BanditEnv(**env_config)
    return env


class BanditEnv:
    """Environment for an N-armed bandit task.
    
    An agent has access to as many discrete actions as there are unique bandit arms.
    Each bandit arm has a win probability determined at episode start.
    Win probabilities are constant within an episode.
    Episodes consist of several bandit decisions, also known as trials, and continue for steps_per_episode trials.
    The env can have several reward structures:
    - "independent": Win probabilities of each arm are independent [0, 1].
    - "correlated": Win probabilities of all arms sum to 1.
    """

    def __init__(self,
        env_name: str = "bandit",
        steps_per_episode: int = 100,
        num_arms: int = 2,
        include_prev_action: bool = True,
        include_prev_reward: bool = True,
        reward_structure: str = "correlated",
        ) -> None:
        """Initialize Env class for a multi-armed bandit task.

        Args:
            steps_per_episode: Steps/trials per episode.
            num_arms: Number of available bandit arms.
            include_prev_action: Include action from last step in observation.
            include_prev_reward: Include reward from last step in observation.
            reward_structure: "independent" or "correlated" arm win probabilities.

        Returns:
            None
        """
        self.name = env_name
        self._steps_per_episode = steps_per_episode
        self._num_arms = num_arms
        self._include_prev_action = include_prev_action
        self._include_prev_reward = include_prev_reward
        self._reward_structure = reward_structure

        self._arm_probs = np.zeros(self._num_arms)

        self.num_actions = self._num_arms
        self._done = False

        self.reset()

    def _get_new_bandits(self) -> None:
        """Get bandit arm win probabilites for this episode."""

        self._arm_probs = np.zeros(self._num_arms)

        if self._reward_structure == "independent":
            # Win probabilities of each arm are independent
            self._arm_probs = np.random.uniform(size=self._num_arms)
        elif self._reward_structure == "correlated":
            # Win probabilities of all arms sum to 1
            self._arm_probs = np.random.uniform(size=self._num_arms)
            self._arm_probs = self._arm_probs / np.sum(self._arm_probs)
        else:
            raise ValueError(f'Reward structure {self._reward_structure} does not exist.')

    def reset(self, arm_probs = None) -> Dict[str, np.ndarray]:
        """Reset environment at the start of each episode.

        Args:
            arm_probs: Optional numpy array of arm win probabilities.

        Returns:
            observation: Dictionary of numpy arrays for network input.
        """
        if arm_probs is not None:
            # Manual win probabilities for each arm
            if not isinstance(arm_probs, np.ndarray):
                raise ValueError(f'arm_probs must be a numpy array.')
            if np.shape(arm_probs) != np.shape(self._arm_probs):
                raise ValueError(f'arm_probs shape {np.shape(arm_probs)} does not match required shape {np.shape(self._arm_probs)}.')
            if np.any(self._arm_probs < 0.0) or np.any(self._arm_probs > 1.0):
                raise ValueError(f'Arm win probs {self._arm_probs} need to be between 0 and 1.')
            self._arm_probs = arm_probs
        else:
            # Automatic win probabilities for each arm based on reward_structure
            self._get_new_bandits()

        vector_input = np.array([])

        if self._include_prev_action:
            vector_input = np.concatenate((vector_input, np.zeros(self.num_actions)))
        if self._include_prev_reward:
            vector_input = np.concatenate((vector_input, np.zeros(1)))

        self._info = {
            "episode_reward": 0,
        }
        self._cur_step = 0
        self._done = False

        return {
            "vector_input": vector_input,
        }

    def step(self,
        action: np.ndarray
        ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the environment.

        Args:
            action: Discrete action taken by the agent on this step.

        Returns:
            observation: Dictionary of numpy arrays for network input.
            reward: Scalar reward value for this step.
            done: Boolean for terminal episode state.
            info: Dictionary of episode information.
        """
        action = int(action)
        reward = 0.0

        cur_arm_prob = self._arm_probs[action]
        win = np.random.uniform() < cur_arm_prob

        if win:
            reward += 1.

        vector_input = np.array([])

        if self._include_prev_action:
            one_hot_prev_action = np.zeros(self.num_actions)
            one_hot_prev_action[action] = 1
            vector_input = np.concatenate((vector_input, one_hot_prev_action))
        if self._include_prev_reward:
            vector_input = np.append(vector_input, reward)

        if self._cur_step == self._steps_per_episode - 1:
            self._done = True
        self._cur_step += 1

        return (
            {
                "vector_input": vector_input,
            },
            reward,
            self._done,
            self._info
        )
