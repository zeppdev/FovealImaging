import argparse
from optparse import OptionParser

import numpy as np
import pacman
import sys, os
import matplotlib.pyplot as plt
import pickle

import time

LOGGING_ON = False
np.random.seed(6)


# the function we want to optimize
def f(agent, w, epoch, **args):
    # here we would normally:
    # ... 1) create a neural network with weights w
    # ... 2) run the neural network on the environment for some time
    # ... 3) sum up and return the total reward

    # Do not initialize the kernel on 1st epoch
    agent.set_weights(w, epoch == 0)
    agent.train()
    return agent.get_score()


def runExperiment(agent, **args):
    start_time = time.time()

    # hyperparameters
    npop = 50  # population size
    sigma = 0.1  # noise standard deviation
    alpha = 0.1
    alpha_decay = 0.95
    alpha_decay_step = 10
    alpha_decay_stop = 0.003

    nr_epochs = 20
    weights_size = agent.get_weights_size()

    w = np.random.randn(weights_size)  # our initial guess is random
    # rewards per epoch
    rewards = []
    weights = []
    for i in range(nr_epochs):
        if i % alpha_decay_step == 0 and alpha >= alpha_decay_stop:
            alpha *= alpha_decay

        rewards.append(f(agent, w, i, **args))
        # print current fitness with the population average
        if i % 1 == 0:
            print('epoch %d. w: %s, reward: %f' % (i, str(w), rewards[i]))

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(npop, weights_size)  # samples from a normal distribution N(0,1)
        R = np.zeros(npop)
        for j in range(npop):
            w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
            R[j] = f(agent, w_try, i, **args)  # evaluate the jittered version
        print('epoch {}. : rewards: {}'.format(i, R))
        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R))
        stdev = np.std(R)
        if stdev != 0:
            A = A / stdev
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha / (npop * sigma) * np.dot(N.T, A)
        if i % 100 == 0:
            weights.append(w)

    res_directory = "results/"
    run_identifier = "pop" + str(npop) + "_" + str(time.time())[-5:]
    pickle.dump(weights, open(res_directory + run_identifier + ".p", "wb"))
    np.savetxt(res_directory + run_identifier + ".txt", rewards, newline='\r\n')

    print("Reward with the population mean for each epoch: " + str(rewards))
    print("--- Experiment ran for %s seconds ---" % (time.time() - start_time))
    plt.plot(rewards)
    plt.show()
