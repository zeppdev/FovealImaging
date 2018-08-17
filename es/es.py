import argparse
from optparse import OptionParser

import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pickle

import time

from datetime import date, datetime

from NN.Model import Model
from image_processing.draw import create_lattice

LOGGING_ON = False
#np.random.seed(6)

# the function we want to optimize
def f(agent, w, epoch, **args):
    # here we would normally:
    # ... 1) create a neural network with weights w
    # ... 2) run the neural network on the environment for some time
    # ... 3) sum up and return the total reward

    # Do not initialize the kernel on 1st epoch
    if epoch == 0 and not args['restart']:
        # initial settings
        lattice = create_lattice(1, 1, 2, agent.k_size)
        # print(lattice)
        kernel_weights = np.empty((agent.k_size ** 2, 3))
        for i in range(len(kernel_weights)):
            kernel_weights[i] = [lattice[i][0], lattice[i][1], agent.kernel_init_std]
        kernel_weights = kernel_weights.T
        w[:kernel_weights.size] = kernel_weights.flatten()

    agent.set_weights(w)
    agent.train(epoch)
    return agent.get_score()


def runExperiment(agent, **args):
    start_time = time.time()

    # hyperparameters
    npop = 30  # population size
    sigma = 0.1  # noise standard deviation
    alpha = 0.02
    alpha_decay = 0.97
    alpha_decay_step = 10
    alpha_decay_stop = 0.003
    # Nr of images processed in each epoch
    batch_size = 20
    nr_epochs = 3000
    a = agent()
    weights_size = a.get_weights_size()


    if args['restart']:
        w = args['weights']
        # We assume alpha having already decayed to minimum during first run
        alpha = alpha_decay_stop
    else:
        w = np.random.randn(weights_size)
    # rewards per epoch
    rewards = []
    weights = []

    res_directory = "results/{}_{}/".format(datetime.today().date().isoformat(), str(time.time())[-5:])

    os.makedirs(os.path.dirname(res_directory), exist_ok=True)

    for i in range(nr_epochs):
        a.set_batch_size(batch_size)
        rewards.append(f(a, w, i, **args))

        if i % 5 == 0:
            a.visualize(i, res_directory=res_directory)

        # print current fitness with the population average
        if i % 1 == 0:
            print('epoch %d. w: %s, reward: %f' % (i, str(w), rewards[i]))

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(npop, weights_size)  # samples from a normal distribution N(0,1)
        R = np.zeros(npop)
        for j in range(npop):
            w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
            R[j] = f(a, w_try, i, **args)  # evaluate the jittered version
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

        if i % alpha_decay_step == 0 and alpha > alpha_decay_stop:
            alpha *= alpha_decay

        if i % 20 == 0:
            weights.append(w)

            #Intermediate Results

            run_identifier = "intermediate_pop" + str(npop) + "_" + 'epoch' + str(i)
            pickle.dump(w, open(res_directory + run_identifier + ".p", "wb"))
            np.savetxt(res_directory + run_identifier + ".txt", rewards, newline='\r\n')


    # Final Results
    run_identifier = "final_pop" + str(npop) + "_" + 'epoch' + str(nr_epochs)
    pickle.dump(weights, open(res_directory + run_identifier + ".p", "wb"))
    np.savetxt(res_directory + run_identifier + ".txt", rewards, newline='\r\n')

    print("Reward with the population mean for each epoch: " + str(rewards))
    print("--- Experiment ran for %s seconds ---" % (time.time() - start_time))
    plt.plot(rewards)
    plt.show()


def load_results(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
        return results


def recreate_images(agent, weights, file, **args):
    agent.set_weights(np.array(weights).T)
    agent.visualize(epoch=None, filename=file)


if __name__ == '__main__':
    weights = load_results('results/2018-08-09/intermediate_pop50_epoch150_90874.p')
    agent = Model()
    recreate_images(agent, weights[0], file='test_results.png')
