# -*-coding:utf-8-*-

"""
    Author: Microstrong
    Desc: Multi-Armed Bandit: epsilon-greedy
    Date: 2020/01/16
"""

import numpy as np

T = 100000  # T个客人
N = 10  # N道菜

true_rewards = np.random.uniform(low=0, high=1, size=N)  # N道菜好吃的概率
estimated_rewards = np.zeros(N)  # 每道菜的观测概率，初始都为0
number_of_trials = np.zeros(N)  # 每道菜当前已经探索的次数，初始都为0
total_reward = 0


def epsilon_greedy(N, epsilon=0.1):
    if np.random.random() < epsilon:
        item = np.random.randint(low=0, high=N)
    else:
        item = np.argmax(estimated_rewards)
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward


for t in range(1, T):  # T个客人依次进入餐馆
    # 从N道菜中推荐一个，reward = 1表示客人接受，reward = 0 表示客人拒绝并离开
    item, reward = epsilon_greedy(N)
    total_reward += reward  # 一共有多少客人接受了推荐

    # 更新菜的平均成功概率
    number_of_trials[item] += 1
    estimated_rewards[item] = ((number_of_trials[item] - 1) * estimated_rewards[item] + reward) / number_of_trials[item]

print("total_reward=" + str(total_reward))

'''
Reference:
[1] Multi-Armed Bandit: epsilon-greedy - 冯伟的文章 - 知乎，地址：https://zhuanlan.zhihu.com/p/32335683 
[2] 推荐系统遇上深度学习(十二)--推荐系统中的EE问题及基本Bandit算法，地址：https://mp.weixin.qq.com/s/UXQF34PadhUsAU3U8XA2nw
'''
