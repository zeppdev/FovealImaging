import argparse

from NN.Model import Model
from es import es

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args)

    es.runExperiment(Model, **args)
    # weights = es.load_results('./results/intermediate_pop1_79133.p')
    # es.recreate_images(Model, weights)

