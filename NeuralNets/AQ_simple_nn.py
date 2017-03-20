#!/usr/bin/env python

""" AQ Simple NN

Description:
    Testing a *very* primative neural network using the Python library pyneurgen (Don Smiley)

Usage:
    User selects 1 observed variable to conduct ML using the NN

    python AQ_simple_nn.py

Dataset:
    Observed atmospheric variable at Big Bend NP during the end of August/early September 2012

TODO:
    A lot.

"""
from __future__ import division
import matplotlib
matplotlib.use('TkAgg')  # Display plots in GUI on Mac OS X
import random
import matplotlib.pyplot as plt
import pandas as pd
from pyneurgen.neuralnet import NeuralNet


# ~~~~ USER OPTIONS ~~~~ #

# Path to observations CSV
csv_in = "data/OBS_BBE401_subset.csv"

# Pick a variable to run the model on
# 'TEMPERATURE', 'RELATIVE_HUMIDITY', 'SOLAR_RADIATION', 'OZONE',
# 'PRECIPITATION', 'WINDSPEED', 'WIND_DIRECTION', 'WINDSPEED_SCALAR'
var = 'TEMPERATURE'

# ~~~~ END USER OPTIONS ~~~~ #


def parse_obs(obs_file):

    names = ['DATE_TIME', 'TEMPERATURE', 'RELATIVE_HUMIDITY', 'SOLAR_RADIATION',
         'OZONE', 'PRECIPITATION', 'WINDSPEED', 'WIND_DIRECTION', 'WINDSPEED_SCALAR']

    obs_df = pd.read_csv(csv_in, names=names, header=1)

    return obs_df


def parse_df(df, var):
    obs_len = len(df[var])
    factor = 1.0 / float(obs_len)
    obs_arr = [[i, round(df[var][i])] for i in range(obs_len)]

    return factor, obs_arr


def population_gen(obs_arr):
    """
    This function shuffles the values of the population and yields the
    items in a random fashion.
    """

    obs_sort = [item for item in obs_arr]
    random.shuffle(obs_sort)

    for item in obs_sort:
        yield item


def run_nn():
    net = NeuralNet()

    net.init_layers(2, [10], 1)

    net.randomize_network()
    net.set_halt_on_extremes(True)

    net.set_random_constraint(0.6)
    net.set_learnrate(.01)

    net.set_all_inputs(all_inputs)
    net.set_all_targets(all_targets)

    length = len(all_inputs)
    learn_end_point = int(length * .8)

    # Set learn range
    net.set_learn_range(0, learn_end_point)
    net.set_test_range(learn_end_point + 1, length - 1)

    # Hidden layer activation type
    net.layers[1].set_activation_type('sigmoid')

    net.learn(epochs=10, show_epoch_results=True, random_testing=False)

    mse = net.test()
    print "test mse = ", mse

    test_positions = [item[0][1] * 1000.0 for item in net.get_test_data()]

    all_targets1 = [item[0][0] for item in net.test_targets_activations]

    allactuals = [item[1][0] for item in net.test_targets_activations]

    return net, test_positions, all_targets1, allactuals


def plot_results(ozone_obs, net, test_positions, all_targets1, allactuals):
    # Summarize results
    plt.subplot(3, 1, 1)
    plt.plot([i[1] for i in ozone_obs])
    plt.title("Population")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(test_positions, all_targets1, 'bo', label='targets')
    plt.plot(test_positions, allactuals, 'ro', label='actuals')
    plt.grid(True)
    plt.legend(loc='lower left', numpoints=1)
    plt.title("Test Target Points vs Actual Points")

    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(net.accum_mse) + 1, 1), net.accum_mse)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.title("Mean Squared Error by Epoch")

    plt.show()


if __name__ == '__main__':
    obs = parse_obs(csv_in)
    factor, obs_arr = parse_df(obs, var)

    all_inputs = []
    all_targets = []

    #  Create NN inputs
    for position, target in population_gen(obs_arr):
        pos = float(position)
        all_inputs.append([random.random(), pos * factor])
        all_targets.append([target])

    net, test_positions, all_targets1, allactuals = run_nn()
    plot_results(obs_arr, net, test_positions, all_targets1, allactuals)
