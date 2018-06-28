import argparse

from NN.Model import Model
from es import es

def init_model():
    return Model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args)
    m = init_model()
    # es.runExperiment(m, **args)
    weights = es.load_results('./results/intermediate_pop1_79133.p')
    es.recreate_images(m, weights)

