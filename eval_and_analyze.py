"""Evaluation and analysis of a trained model."""

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import modules.envs as envs
import modules.agents as agents

path = "/home/natha/meta-rl/"
filename = "trained_agent.pickle"

with open(path + filename, 'rb') as fp:
    training_results = pickle.load(fp)

params = training_results[0]['params']
training_config = training_results[0]['config']
agent_config = training_config['agent']

# Initialize environment
env = envs.BanditEnv(
    steps_per_episode=100,
    num_arms=2,
    reward_structure="correlated",
    )
observation = env.reset()

# Initialize agent
agent = agents.create_agent(
    agent_config=agent_config,
    observation=observation,
    num_actions=env.num_actions,
)
initial_lstm_state = agent.get_initial_lstm_state()
lstm_state = initial_lstm_state

# Overwrite agent paramters with saved training parameters
agent.params = params

# Make evaluation episodes
arm1_probs = np.arange(101)/100
arm2_probs = 1. - arm1_probs

eval_episodes = np.arange(len(arm1_probs))

# Initialize data to save
eval_actions = np.zeros((len(arm1_probs), env._steps_per_episode))
eval_actions[:] = np.nan

# Evaluation loop
for (arm1_prob, arm2_prob, ep) in zip(arm1_probs, arm2_probs, eval_episodes):
    
    step = 0
    done = False
    lstm_state = initial_lstm_state
    observation = env.reset(arm_probs=np.array([arm1_prob, arm2_prob]))
    
    while not done:

        action, pi_out, v_out, new_lstm_state, _ = agent.get_action(observation, lstm_state)
        next_observation, reward, done, info = env.step(action)

        # Save data
        eval_actions[ep, step] = action

        observation = next_observation
        lstm_state = new_lstm_state
        step += 1

print("Done evaluation.")

# Analyze data

# Figure params
tick_fontsize = 12
label_fontsize = 16

# Setup y tick labels to convey arm win probabilities
ytick_vals = [0, eval_episodes[int(len(eval_episodes)/2)], eval_episodes[-1]]
ytick_labels = []
for i in ytick_vals:
    print(i)
    cur_probs = str(int(arm1_probs[i]*100)) + ", " + str(int(arm2_probs[i]*100))
    ytick_labels.append(cur_probs)

# Custom legend setup for actions
cmap_name = 'tab20c'
cmap = matplotlib.cm.get_cmap(cmap_name)
zero_color = cmap(0.)
one_color = cmap(1.)
patch1 = mpatches.Patch(color=zero_color, label='Arm 1')
patch2 = mpatches.Patch(color=one_color, label='Arm 2')

# Plot action results
fig = plt.figure(figsize=(6, 4))
ax = plt.subplot()
plt.imshow(eval_actions, cmap='tab20c', origin='lower')
plt.yticks(
    ticks=ytick_vals,
    labels=ytick_labels,
    fontsize=tick_fontsize,
    )
plt.xticks(
    ticks=[0, env._steps_per_episode],
    labels=[0, env._steps_per_episode],
    fontsize=tick_fontsize,
    )
plt.xlabel("Trials", fontsize=label_fontsize)
plt.ylabel("Arm win probabilities \n (Arm 1 %, Arm 2 %)", fontsize=label_fontsize)
ax.spines[['right', 'top']].set_visible(False)
plt.legend(handles=[patch1, patch2],
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           frameon=False,
           fontsize=label_fontsize,
           borderaxespad=0.,
           )
plt.tight_layout()
plt.show()

print("Done!")
