from matplotlib import pyplot as plt
import pickle
import math
import numpy as np
from NN.Model import Model
from pylab import polyfit, poly1d

weights_file = 'D:/experiments/foveal imaging/rob\'s run/intermediate_pop30_epoch18680.p'
weights = np.array(pickle.load(open(weights_file, mode='rb')))

# If these are full results, we are only interested in the last one
if len(weights.shape) == 2:
    weights = weights[-1]

agent = Model()
agent.set_weights(weights)

#Load kernel positions
kernel_weights = agent.kernel_weights
xs = kernel_weights[0]
ys = kernel_weights[1]
rs = kernel_weights[2]

distances = [math.sqrt(x ** 2 + y ** 2) for x, y in zip(xs, ys)]
sorted_distances, sorted_radiuses = zip(*sorted(zip(distances, rs)))

# Clip radiuses below 0 to 0
sorted_radiuses = [max(x, 0) for x in sorted_radiuses]

# Plot radius against distance and linear fit
fit = polyfit(sorted_distances, sorted_radiuses, 1)
fit_fn = poly1d(fit)
plt.xlabel("Distance from the center of the lattice")
plt.ylabel("Radius (standard deviation) of the kernel")
plt.plot(sorted_distances, fit_fn(sorted_distances), 'r')
plt.scatter(sorted_distances, sorted_radiuses)
plt.show()