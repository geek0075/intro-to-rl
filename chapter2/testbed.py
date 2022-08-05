import numpy as np
import random

epsilon = 0.1
max_num_tasks = 2000

def setup():
    print('simple_bandit.setup.1')
    np.random.seed(42)
    global n, Q, n_a, Q_star, randomness, max_num_tasks
    n = 10
    Q = np.zeros(n, dtype=np.float64)
    n_a = np.zeros(n, dtype=np.int64)
    print('simple_bandit.setup.2')
    Q_star = np.random.normal(0, 1, (n, max_num_tasks))
    print('simple_bandit.setup: Q_star.dtype = {0}'.format(Q_star.dtype))
    print('simple_bandit.setup.3')
    randomness = np.arange(3000, 3000 + max_num_tasks, 1)
    print('simple_bandit.setup: randomness.dtype = {0}'.format(randomness.dtype))

def init():
    global n, Q, n_a
    Q = np.zeros(n, dtype=np.float64)
    n_a = np.zeros(n, dtype=np.int64)

def arg_max_random_tiebreak(arr):
    indices = np.flatnonzero(arr == np.amax(arr))
    # print('testbed.arg_max_random_tiebreak: indices.shape = {0}, indices = {1}'.format(indices.shape, indices))
    # index = np.random.randint(indices.shape[0])
    index = random.randrange(0, indices.shape[0])
    arg = indices[index]
    # print('testbed.arg_max_random_tiebreak: index = {0}, arg = {1}'.format(index, arg))
    return arg

def learn(a, r):
    global Q, n_a
    n_a[a] += 1
    Q[a] += (1. / n_a[a]) * (r - Q[a])

def reward(a, task_num):
    global Q_star
    r = np.random.normal(Q_star[a, task_num], 1)
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

def runs(num_runs=1000, num_steps=100, epsilon=0):
    global n, Q, n_a, Q_star, randomness, max_num_tasks
    assert num_runs <= max_num_tasks
    average_reward = np.zeros(num_steps, dtype=np.float64)
    prob_a_star = np.zeros(num_steps, dtype=np.float64)
    for run_num in range(num_runs):
        print('testbed.runs: run_num = {0}, num_runs = {1}, progress = {2} of {3}'.format(run_num, num_runs, run_num+1, num_runs))
        a_star = 0
        for a in range(1, n):
            if Q_star[a, run_num] > Q_star[a_star, run_num]:
                a_star = a
        print('testbed.runs: run_num = {0}, a_star = {1}, Q_star = {2}'.format(run_num, a_star, Q_star[a_star, run_num]))
        init()
        seed = randomness[run_num]
        np.random.seed(seed)
        random.seed(seed)
        for time_step in range(num_steps):
            print('testbed.runs: run_num = {0}, time_step = {1}, num_steps = {2}, progress = {3} of {4}'.format(run_num, time_step, num_steps, time_step+1, num_steps))
            a = epsilon_greedy(epsilon)
            r = reward(a, run_num)
            print('testbed.runs: run_num = {0}, time_step = {1}, a = {2}, r = {3}, a_star = {4}, (a == a_star) = {5}'.format(run_num, time_step, a, r, a_star, a == a_star))
            learn(a, r)
            average_reward[time_step] += r
            if a == a_star:
                prob_a_star[time_step] += 1
    average_reward /= num_runs
    prob_a_star /= num_runs
    return average_reward, prob_a_star

def run():
    setup()
    
    print('testbed.run: epsilon = 0.1')
    init()
    average_reward, prob_a_star = runs(num_runs=2000, num_steps=1000, epsilon=0.1)
    with open('./data/average_reward_runs2000_steps1000_epsilon01.npy', 'wb') as f:
        np.save(f, average_reward)
    with open('./data/prob_a_star_runs2000_steps1000_epsilon01.npy', 'wb') as f:
        np.save(f, prob_a_star)
    print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))
    
    print('testbed.run: epsilon = 0.01')
    init()
    average_reward, prob_a_star = runs(num_runs=2000, num_steps=1000, epsilon=0.01)
    with open('./data/average_reward_runs2000_steps1000_epsilon001.npy', 'wb') as f:
        np.save(f, average_reward)
    with open('./data/prob_a_star_runs2000_steps1000_epsilon001.npy', 'wb') as f:
        np.save(f, prob_a_star)
    print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))

    print('testbed.run: epsilon = 0.0')
    init()
    average_reward, prob_a_star = runs(num_runs=2000, num_steps=1000, epsilon=0.0)
    with open('./data/average_reward_runs2000_steps1000_epsilon00.npy', 'wb') as f:
        np.save(f, average_reward)
    with open('./data/prob_a_star_runs2000_steps1000_epsilon00.npy', 'wb') as f:
        np.save(f, prob_a_star)
    print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))
     
    # print('testbed.run: epsilon = 0.1, num_steps=10000')
    # init()
    # average_reward, prob_a_star = runs(num_runs=2000, num_steps=10000, epsilon=0.1)
    # with open('average_reward_runs2000_steps10000_epsilon01.npy', 'wb') as f:
    #     np.save(f, average_reward)
    # with open('prob_a_star_runs2000_steps10000_epsilon01.npy', 'wb') as f:
    #     np.save(f, prob_a_star)
    # print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    # print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))
    
    # print('testbed.run: epsilon = 0.01, num_steps=10000')
    # init()
    # average_reward, prob_a_star = runs(num_runs=2000, num_steps=10000, epsilon=0.01)
    # with open('average_reward_runs2000_steps10000_epsilon001.npy', 'wb') as f:
    #     np.save(f, average_reward)
    # with open('prob_a_star_runs2000_steps10000_epsilon001.npy', 'wb') as f:
    #     np.save(f, prob_a_star)
    # print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    # print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))

    # print('testbed.run: epsilon = 0.0, num_steps=10000')
    # init()
    # average_reward, prob_a_star = runs(num_runs=2000, num_steps=10000, epsilon=0.0)
    # with open('average_reward_runs2000_steps10000_epsilon00.npy', 'wb') as f:
    #     np.save(f, average_reward)
    # with open('prob_a_star_runs2000_steps10000_epsilon00.npy', 'wb') as f:
    #     np.save(f, prob_a_star)
    # print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    # print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))
     
    # print('testbed.run: epsilon = 0.1, num_steps=20000')
    # init()
    # average_reward, prob_a_star = runs(num_runs=2000, num_steps=20000, epsilon=0.1)
    # with open('average_reward_runs2000_steps20000_epsilon01.npy', 'wb') as f:
    #     np.save(f, average_reward)
    # with open('prob_a_star_runs2000_steps20000_epsilon01.npy', 'wb') as f:
    #     np.save(f, prob_a_star)
    # print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    # print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))
    
    # print('testbed.run: epsilon = 0.01, num_steps=20000')
    # init()
    # average_reward, prob_a_star = runs(num_runs=2000, num_steps=20000, epsilon=0.01)
    # with open('average_reward_runs2000_steps20000_epsilon001.npy', 'wb') as f:
    #     np.save(f, average_reward)
    # with open('prob_a_star_runs2000_steps20000_epsilon001.npy', 'wb') as f:
    #     np.save(f, prob_a_star)
    # print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    # print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))

    # print('testbed.run: epsilon = 0.0, num_steps=20000')
    # init()
    # average_reward, prob_a_star = runs(num_runs=2000, num_steps=20000, epsilon=0.0)
    # with open('average_reward_runs2000_steps20000_epsilon00.npy', 'wb') as f:
    #     np.save(f, average_reward)
    # with open('prob_a_star_runs2000_steps20000_epsilon00.npy', 'wb') as f:
    #     np.save(f, prob_a_star)
    # print('testbed.run: average_reward.shape = {0}, average_reward = {1}'.format(average_reward.shape, average_reward))
    # print('testbed.run: prob_a_star.shape = {0}, prob_a_star = {1}'.format(prob_a_star.shape, prob_a_star))

if __name__ == '__main__':
    run()