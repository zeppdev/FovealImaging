import argparse

from NN.Model import Model
from es import es

# For loading weights and running again
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args)
    weights = es.load_results('D:/experiments/foveal imaging/2018-08-17_17801/final_pop30_epoch3000.p')
    args['restart'] = True
    # Most recent weights
    args['weights'] = weights[-1]
    es.runExperiment(Model, **args)
    # es.recreate_images(Model, weights)

