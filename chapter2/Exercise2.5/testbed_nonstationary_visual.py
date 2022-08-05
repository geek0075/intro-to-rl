import matplotlib.pyplot as plt
import numpy as np
import random

def run():
    print('testbed_nonstationary_visual.run: steps = 10000')
    x = np.arange(1, 10001)
    with open('./data/average_reward_runs2000_steps10000_epsilon01_sample_avg.npy', 'rb') as f:
        average_reward_steps10000_epsilon01_sample_avg = np.load(f)
    with open('./data/average_reward_runs2000_steps10000_epsilon01_constant_alpha.npy', 'rb') as f:
        average_reward_steps10000_epsilon01_constant_alpha = np.load(f)
    print('testbed_nonstationary_visual.run: average_reward_steps10000_epsilon01_sample_avg[-1] = {0}'.format(average_reward_steps10000_epsilon01_sample_avg[-1]))
    print('testbed_nonstationary_visual.run: average_reward_steps10000_epsilon01_constant_alpha[-1] = {0}'.format(average_reward_steps10000_epsilon01_constant_alpha[-1]))
    fig, ax = plt.subplots()
    ax.plot(x, average_reward_steps10000_epsilon01_sample_avg, label='sample avg')
    ax.plot(x, average_reward_steps10000_epsilon01_constant_alpha, label='constant alpha')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title("Ave Reward vs. Time (10000 Steps, Non-Stationary)")
    ax.legend()
    plt.savefig('./plots/average_reward_runs2000_steps10000_non_stationary.png')
    plt.show()
    fig, ax = plt.subplots()
    cum_average_reward_steps10000_epsilon01_sample_avg = np.cumsum(average_reward_steps10000_epsilon01_sample_avg)
    cum_average_reward_steps10000_epsilon01_constant_alpha = np.cumsum(average_reward_steps10000_epsilon01_constant_alpha)
    print('testbed_nonstationary_visual.run: cum_average_reward_steps10000_epsilon01_sample_avg[-1] = {0}'.format(cum_average_reward_steps10000_epsilon01_sample_avg[-1]))
    print('testbed_nonstationary_visual.run: cum_average_reward_steps10000_epsilon01_constant_alpha[-1] = {0}'.format(cum_average_reward_steps10000_epsilon01_constant_alpha[-1]))
    ax.plot(x, cum_average_reward_steps10000_epsilon01_sample_avg, label='sample avg')
    ax.plot(x, cum_average_reward_steps10000_epsilon01_constant_alpha, label='constant alpha')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cum Ave Reward')
    ax.set_title("Cum Ave Reward vs. Time (10000 Steps, Non-Stationary)")
    ax.legend()
    plt.savefig('./plots/cumulative_average_reward_runs2000_steps10000_non_stationary.png')
    plt.show()
    with open('./data/prob_a_star_runs2000_steps10000_epsilon01_sample_avg.npy', 'rb') as f:
        prob_a_star_steps10000_epsilon01_sample_avg = np.load(f)
    with open('./data/prob_a_star_runs2000_steps10000_epsilon01_constant_alpha.npy', 'rb') as f:
        prob_a_star_steps10000_epsilon01_constant_alpha = np.load(f)
    print('testbed_nonstationary_visual.run: prob_a_star_steps10000_epsilon01_sample_avg[-1] = {0}'.format(prob_a_star_steps10000_epsilon01_sample_avg[-1]))
    print('testbed_nonstationary_visual.run: prob_a_star_steps10000_epsilon01_constant_alpha[-1] = {0}'.format(prob_a_star_steps10000_epsilon01_constant_alpha[-1]))
    fig, ax = plt.subplots()
    ax.plot(x, prob_a_star_steps10000_epsilon01_sample_avg, label='sample avg')
    ax.plot(x, prob_a_star_steps10000_epsilon01_constant_alpha, label='constant alpha')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Prob A*')
    ax.set_title("Prob A* vs. Time (10000 Steps, Non-Stationary)")
    ax.legend()
    plt.savefig('./plots/prob_a_star_runs2000_steps10000_non_stationary.png')
    plt.show()

if __name__ == '__main__':
    run()