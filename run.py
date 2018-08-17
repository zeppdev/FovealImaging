import argparse

from NN.Model import Model
from es import es

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args)
    args['restart'] = False
    es.runExperiment(Model, **args)
