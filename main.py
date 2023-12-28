from mpl_toolkits.axes_grid1 import make_axes_locatable

from irl_maxent import gridworld as W
from irl_maxent import maxent as M
from irl_maxent import plot as P
from irl_maxent import trajectory as T
from irl_maxent import solver as S
from irl_maxent import optimizer as O

import numpy as np
import math
import matplotlib.pyplot as plt


def setup_mdp():
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = W.IcyGridWorld(size=6, p_slip=0.2)

    # set up the reward function
    reward = np.zeros(world.n_states)
    reward[35] = 1.0
    reward[24] = 0.4
    reward[15] = 0.25

    # set up terminal states
    terminal = [35, 24, 15]

    return world, reward, terminal

# def euclidean_distance(world, t1, t2):
#     min_size = min(len(t1.states), len(t2.states))
#
#     xy1 = [world.state_index_to_point(s) for s in t1.states()]
#     x1, y1 = zip(*xy1)
#
#     xy2 = [world.state_index_to_point(s) for s in t1.states()]
#     x2, y2 = zip(*xy2)
#
#     distance = 0
#     #for i in range(min_size):
#         distance += math.sqrt((x2[i] - x1[i]) ** 2) + (y2[i] - y1[i]) ** 2))


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 100
    discount = 0.9
    # down-weight less optimal actions
    weighting = lambda x: x ** 50

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    # last states of each trajectory
    distribution = []

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal, distribution))

    return tjs, policy, distribution


def generate_hyp_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 100
    discount = 0.1
    # down-weight less optimal actions
    weighting = lambda x: x ** 50

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    # last states of each trajectory
    distribution = []

    # generate naive trajectories
    value_naive = S.value_iteration_naive(world.p_transition, reward, discount)
    policy_naive = S.stochastic_policy_from_value(world, value_naive, w=weighting)
    policy_exec_naive = T.stochastic_policy_adapter(policy_naive)
    tjs_naive = list(T.generate_trajectories(n_trajectories, world, policy_exec_naive, initial, terminal, distribution))

    # generate sophisticated trajectories
    value_soph = S.value_iteration_sophisticated(world.p_transition, reward, discount)
    policy_soph = S.stochastic_policy_from_value(world, value_soph, w=weighting)
    policy_exec_soph = T.stochastic_policy_adapter(policy_soph)
    tjs_soph = list(T.generate_trajectories(n_trajectories, world, policy_exec_soph, initial, terminal, distribution))

    return tjs_naive, policy_naive, tjs_soph, policy_soph, distribution


def maxent_causal(world, terminal, trajectories, discount=0.9):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward, yy, e_svf = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    # last states of each trajectory
    distributionent = []

    policy_ent = S.optimal_policy_from_value(world, e_svf)
    policy_adapted_ent = T.policy_adapter(policy_ent)

    # compute the optimal trajectory after maximum entropy
    trajectoryent = list(T.generate_trajectories(1, world, policy_adapted_ent, initial, terminal, distributionent))
    # list_t_states = list(trajectoryent[0].states())
    # print(list_t_states)

    return reward, yy, trajectoryent, policy_adapted_ent, e_svf


def main():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = setup_mdp()

    # show our original reward
    ax = plt.figure(num='Original Reward').add_subplot(111)
    ax.title.set_text('Original Reward')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward, **style)
    ax.text(0, 4, 'M', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3, 2, 'S', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 5, 'B', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.figure().colorbar(p, cax=cax)
    plt.draw()

    # generate "expert" trajectories
    trajectories, expert_policy, distribution = generate_trajectories(world, reward, terminal)

    # show our expert policies
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    ax.title.set_text('Expert Policy and Trajectories')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_stochastic_policy(ax, world, expert_policy, **style)
    plt.figure().colorbar(p, cax=cax)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()

    s1s = distribution.count(24)
    s2s = distribution.count(15)
    bs = distribution.count(35)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.title.set_text('Original Reward Optimal Policy')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward, **style)
    ax.text(0, 4, 'M', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(3, 2, 'S', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 5, 'B', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward, 0.9), color='red')
    fig.colorbar(p, cax=cax)

    # maximum causal entropy reinforcement learning (non-causal)
    reward_maxcausal, yy, trajectory_ent, policy_adapted_ent, e_svf = maxent_causal(world, terminal, trajectories)

    ax = fig.add_subplot(122)
    ax.title.set_text('Recovered Reward MaxEnt')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward_maxcausal, **style)
    ax.text(0, 4, 'M', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(3, 2, 'S', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 5, 'B', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward_maxcausal, 0.9), color='red')
    fig.colorbar(p, cax=cax)

    fig.tight_layout()
    plt.show()

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    ax.title.set_text('Recovered Reward MaxEnt')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward_maxcausal, **style)
    ax.text(0, 4, 'M', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3, 2, 'S', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 5, 'B', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.figure().colorbar(p, cax=cax)
    plt.draw()

    plt.show()

    # s1sen = reward_maxcausal[24]
    # s2sen = reward_maxcausal[15]
    # bsen = reward_maxcausal[35]
    #
    # # Define the variables and their corresponding values
    # variables = ['M', 'S', 'B']
    # values = np.array([s1s, s2s, bs])
    #
    # # Create the figure and axes
    # fig, ax = plt.subplots()
    # # Set the y-axis limits
    # ax.set_ylim(0, 100)
    # # Set the x-axis tick positions and labels
    # x_pos = np.arange(len(variables))
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(variables)
    # # Plot the values as grouped bars
    # width = 0.3  # Width of each bar
    # bar1 = ax.bar(x_pos, values, width, label='Expert')
    # ax.set_ylabel('No. of trajectories')
    # ax.set_xlabel('Rewards')
    # # Add a legend
    # ax.legend()
    # # Show the plot
    # plt.show()
    #
    # values2 = np.array([s1sen, s2sen, bsen])
    #
    # # Create the figure and axes
    # fig, ax = plt.subplots()
    # # Set the y-axis limits
    # ax.set_ylim(0, 1.5)
    # # Set the x-axis tick positions and labels
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(variables)
    # # Plot the values as grouped bars
    # width = 0.3  # Width of each bar
    # bar1 = ax.bar(x_pos, values2, width, label='MaxEnt')
    # ax.set_ylabel('Recovered reward value')
    # ax.set_xlabel('Rewards')
    # # Add a legend
    # ax.legend()
    # # Show the plot
    # plt.show()

    # ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    # ax.title.set_text('1 Trajectory')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # p = P.plot_state_values(ax, world, reward_maxcausal, **style)
    # P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward_maxcausal, 0.9), color='red')
    # plt.figure().colorbar(p, cax=cax)
    #
    # for t in trajectory_ent:
    #     P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.25)
    #
    # plt.show()

    # Naive and sophisticated agents
    tjs_naive, policy_naive, tjs_soph, policy_soph, distribution = generate_hyp_trajectories(world, reward, terminal)

    # Naive
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.title.set_text('Original Reward Optimal Policy')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward, **style)
    ax.text(0, 4, 'M', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(3, 2, 'S', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 5, 'B', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    P.plot_deterministic_policy(ax, world, S.optimal_policy_naive(world, reward, 0.2), color='red')
    fig.colorbar(p, cax=cax)

    # maximum causal entropy reinforcement learning (non-causal)
    reward_max_naive, yy_naive, trajectory_ent_naive, policy_adapted_ent_naive, e_svf_naive = maxent_causal(world, terminal, tjs_naive)

    ax = fig.add_subplot(122)
    ax.title.set_text('Recovered Reward MaxEnt')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward_maxcausal, **style)
    ax.text(0, 4, 'M', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(3, 2, 'S', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 5, 'B', color='black', ha='center', va='center', fontsize=9, fontweight='bold')
    P.plot_deterministic_policy(ax, world, S.optimal_policy_naive(world, reward_max_naive, 0.2), color='red')
    fig.colorbar(p, cax=cax)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
