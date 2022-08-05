import numpy as np
import random

alpha = 0.1
epsilon = 0.1
max_num_tasks = 2000

def setup():
    print('testbed_nonstationary.setup.1')
    global n, Q, n_a, Q_star, randomness, max_num_tasks
    n = 10
    Q = np.zeros(n, dtype=np.float64)
    Q_star = np.zeros((n, max_num_tasks), dtype=np.float64)
    n_a = np.zeros(n, dtype=np.int64)
    print('testbed_nonstationary.setup.2')
    np.random.seed(42)
    Q_star += np.random.normal(0, 1, max_num_tasks)
    print('testbed_nonstationary.setup: Q_star[:, 0].shape = {0}, Q_star[:, 0] = {1}'.format(Q_star[:, 0].shape, Q_star[:, 0]))
    print('testbed_nonstationary.setup: Q_star[:, 1].shape = {0}, Q_star[:, 1] = {1}'.format(Q_star[:, 1].shape, Q_star[:, 1]))
    print('testbed_nonstationary.setup: Q_star[:, 2].shape = {0}, Q_star[:, 2] = {1}'.format(Q_star[:, 2].shape, Q_star[:, 2]))
    print('testbed_nonstationary.setup: Q_star[:, 3].shape = {0}, Q_star[:, 3] = {1}'.format(Q_star[:, 3].shape, Q_star[:, 3]))
    randomness = np.arange(3000, 3000 + max_num_tasks, 1)

def init():
    global n, Q, n_a
    Q = np.zeros(n, dtype=np.float64)
    n_a = np.zeros(n, dtype=np.int64)

def arg_max_random_tiebreak(arr):
    indices = np.flatnonzero(arr == np.amax(arr))
    # index = np.random.randint(indices.shape[0])
    index = random.randrange(0, indices.shape[0])
    arg = indices[index]
    return arg

def learn_sample_avg(a, r):
    global Q, n_a
    n_a[a] += 1
    Q[a] += (1. / n_a[a]) * (r - Q[a])

def learn_constant_alpha(a, r):
    global Q, alpha
    Q[a] += alpha * (r - Q[a])

def reward(a, arr):
    r = np.random.normal(arr[a], 1)
    return r

def epsilon_greedy(epsilon):
    global n, Q
    c = np.random.choice(2, p=[epsilon, 1-epsilon])
    if c == 0:
        # a = np.random.randint(n)
        a = random.randrange(0, n)
    else:
        a = arg_max_random_tiebreak(Q)
    return a

def greedy():
    global Q
    a = arg_max_random_tiebreak(Q)
    return a

def runs(num_runs=1000, num_steps=100, epsilon=0, action_value_method='sample_avg'):
    global n, Q, n_a, Q_star, randomness, max_num_tasks
    assert num_runs <= max_num_tasks
    average_reward = np.zeros(num_steps, dtype=np.float64)
    prob_a_star = np.zeros(num_steps, dtype=np.float64)
    for run_num in range(num_runs):
        print('testbed_nonstationary.runs: run_num = {0}, num_runs = {1}, progress = {2} of {3}'.format(run_num, num_runs, run_num+1, num_runs))
        Q_star_nonstationary = np.copy(Q_star[:, run_num])
        print('testbed_nonstationary.runs: run_num = {0}, Q_star_nonstationary.shape = {1}, Q_star_nonstationary = {2}'.format(run_num, Q_star_nonstationary.shape, Q_star_nonstationary))
        init()
        seed = randomness[run_num]
        np.random.seed(seed)
        random.seed(seed)
        for time_step in range(num_steps):
            Q_star_nonstationary += np.random.normal(0, 0.01, n)
            print('testbed_nonstationary.runs: run_num = {0}, time_step = {1}, Q_star_nonstationary.shape = {2}, Q_star_nonstationary = {3}'.format(run_num, time_step, Q_star_nonstationary.shape, Q_star_nonstationary))
            a_star = 0
            for a in range(1, n):
                if Q_star_nonstationary[a] > Q_star_nonstationary[a_star]:
                    a_star = a
            print('testbed_nonstationary.runs: run_num = {0}, time_step = {1}, a_star = {2}, Q_star_nonstationary = {3}'.format(run_num, time_step, a_star, Q_star_nonstationary[a_star]))
            print('testbed_nonstationary.runs: run_num = {0}, time_step = {1}, num_steps = {2}, progress = {3} of {4}'.format(run_num, time_step, num_steps, time_step+1, num_steps))
            a = epsilon_greedy(epsilon)
            r = reward(a, Q_star_nonstationary)
            print('testbed.runs: run_num = {0}, time_step = {1}, a = {2}, r = {3}, a_star = {4}, (a == a_star) = {5}'.format(run_num, time_step, a, r, a_star, a == a_star))
            if action_value_method == 'sample_avg':
                learn_sample_avg(a, r)
            elif action_value_method == 'constant_alpha':
                learn_constant_alpha(a, r)
            else:
                learn_sample_avg(a, r)
            average_reward[time_step] += r
            if a == a_star:
                prob_a_star[time_step] += 1
    average_reward /= num_runs
    prob_a_star /= num_runs
    return average_reward, prob_a_star

def run():
    setup()

    print('testbed.run: action_value_method="sample_avg"')
    init()
    average_reward, prob_a_star = runs(num_runs=2000, num_steps=10000, epsilon=0.1, action_value_method='sample_avg')
    with open('average_reward_runs2000_steps10000_epsilon01_sample_avg.npy', 'wb') as f:
        np.save(f, average_reward)
    with open('prob_a_star_runs2000_steps10000_epsilon01_sample_avg.npy', 'wb') as f:
        np.save(f, prob_a_star)
    print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))

    print('testbed.run: action_value_method="constant_alpha"')
    init()
    average_reward, prob_a_star = runs(num_runs=2000, num_steps=10000, epsilon=0.1, action_value_method='constant_alpha')
    with open('average_reward_runs2000_steps10000_epsilon01_constant_alpha.npy', 'wb') as f:
        np.save(f, average_reward)
    with open('prob_a_star_runs2000_steps10000_epsilon01_constant_alpha.npy', 'wb') as f:
        np.save(f, prob_a_star)
    print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))

if __name__ == '__main__':
    run()