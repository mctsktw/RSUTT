from __future__ import division
import os
import sys
import numpy as np
import random
import time
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.optimize import basinhopping
from config import config_census
from config import config_german
from config import config_bank
import copy
from CT import generateCT
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, './fair_classification/')  # the code for fair classification is in this directory

random.seed(time.time())
experiment = sys.argv[1]
dataset = sys.argv[2]
classifier = sys.argv[3]
N = int(sys.argv[4])

print "start time : " + str(datetime.datetime.now())

X = None
Y = None

params = 0
sensitive_param = 0
perturbation_unit = 0
threshold = 0
input_bounds = []
classifier_name = ""
sensitive_list = []


# Setting for each Dataset and Classsifier


def setting():
    global X, Y, params, sensitive_param, perturbation_unit, threshold, input_bounds, classifier_name, sensitive_list

    # df = None
    # if dataset == "CENSUS":
    #     df = pd.read_csv('classifier/datasets/census.csv')

    # elif dataset == "GERMAN":
    #     df = pd.read_csv('classifier/datasets/german.csv')

    # elif dataset == "BANK":
    #     df = pd.read_csv('classifier/datasets/bank.csv')

    # else:
    #     print "The dataset name is wrong."

    # data = df.values

    # X = data[:, :-1]
    # Y = data[:, -1]

    params = eval('config_' + dataset.lower()).params
    sensitive_param = eval('config_' + dataset.lower()).sensitive_param
    input_bounds = eval('config_' + dataset.lower()).input_bounds

    # Only when you want to change Classsifier
    # eval(classifier.lower())()

    classifier_name = 'classifier/' + dataset.lower() + '/' + classifier.upper() + '_' + dataset.upper() + '.pkl'

    if experiment == "A" or experiment == "B":
        sensitive_list = [input_bounds[sensitive_param - 1][0],
                          input_bounds[sensitive_param - 1][1]]

    elif experiment == "C":
        flg = True
        while flg:
            flg = False
            random_first = random.randint(input_bounds[sensitive_param - 1][0], input_bounds[sensitive_param - 1][1])
            random_second = random.randint(input_bounds[sensitive_param - 1][0], input_bounds[sensitive_param - 1][1])
            if random_first > random_second:
                sensitive_list = [random_second,
                                  random_first]
            elif random_first < random_second:
                sensitive_list = [random_first,
                                  random_second]
            else:
                flg = True

    else:
        print "The experiment name is wrong."
        sys.exit()


setting()

disc_inputs = set()
disc_inputs_list = []
tot_inputs = set()
global_iteration_limit = N
local_iteration_limit = N
model = joblib.load(classifier_name)

total_data = None
duplication_data = None
seed_data = None
global_duplication_data = None
disc_data = None
local_duplication_data = None
elapsed_time = None
duplication_data = 0
local_cnt = 0


def evaluate_disc(ed_input):
    global duplication_data
    inp0 = [int(i) for i in ed_input]
    inp1 = [int(i) for i in ed_input]

    if experiment == "A":
        inp0[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
        inp1[sensitive_param - 1] = random.randint(input_bounds[sensitive_param - 1][0] + 1,
                                                   input_bounds[sensitive_param - 1][1])

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)
        tot_inputs.add(tuple(map(tuple, inp0)))

        if abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) in disc_inputs:
            duplication_data += 1
        if abs(out0 - out1) > threshold and \
                (tuple(map(tuple, inp0)) not in disc_inputs and tuple(map(tuple, inp0)) not in disc_inputs):
            if random.randint(0, 1) == 0:
                disc_inputs.add(tuple(map(tuple, inp0)))
                disc_inputs_list.append(inp0.tolist()[0])
            else:
                disc_inputs.add(tuple(map(tuple, inp1)))
                disc_inputs_list.append(inp1.tolist()[0])

    elif experiment == "B":
        for sensitive in range(input_bounds[sensitive_param - 1][0], input_bounds[sensitive_param - 1][1]):
            inp0[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
            inp1[sensitive_param - 1] = sensitive + 1

            inp0 = np.asarray(inp0)
            inp0 = np.reshape(inp0, (1, -1))

            inp1 = np.asarray(inp1)
            inp1 = np.reshape(inp1, (1, -1))

            out0 = model.predict(inp0)
            out1 = model.predict(inp1)
            tot_inputs.add(tuple(map(tuple, inp0)))

            if abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) in disc_inputs:
                duplication_data += 1
            if abs(out0 - out1) > threshold and \
                    (tuple(map(tuple, inp0)) not in disc_inputs and tuple(map(tuple, inp0)) not in disc_inputs):
                if random.randint(0, 1) == 0:
                    disc_inputs.add(tuple(map(tuple, inp0)))
                    disc_inputs_list.append(inp0.tolist()[0])
                else:
                    disc_inputs.add(tuple(map(tuple, inp1)))
                    disc_inputs_list.append(inp1.tolist()[0])
                break

    else:
        inp0[sensitive_param - 1] = sensitive_list[0]
        inp1[sensitive_param - 1] = sensitive_list[1]

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)
        tot_inputs.add(tuple(map(tuple, inp0)))

        if abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) in disc_inputs:
            duplication_data += 1
        if abs(out0 - out1) > threshold and \
                (tuple(map(tuple, inp0)) not in disc_inputs and tuple(map(tuple, inp0)) not in disc_inputs):
            if random.randint(0, 1) == 0:
                disc_inputs.add(tuple(map(tuple, inp0)))
                disc_inputs_list.append(inp0.tolist()[0])
            else:
                disc_inputs.add(tuple(map(tuple, inp1)))
                disc_inputs_list.append(inp1.tolist()[0])

    # Necessary for basinhopping functions to work.
    return abs(out0 - out1)


def my_local_search(mls_input):
    for param in range(params):
        if param == sensitive_param - 1:
            continue
        else:
            for direction in [-1, 1]:
                inp2 = copy.copy(mls_input)
                inp2[param] = inp2[param] + direction
                if inp2[param] < input_bounds[param][0] and direction == -1:
                    continue
                elif inp2[param] > input_bounds[param][1] and direction == 1:
                    continue
                elif tuple(map(tuple, np.reshape(np.asarray(inp2), (1, -1)))) in tot_inputs:
                    global duplication_data
                    duplication_data += 1
                    continue
                evaluate_disc(inp2)
                global local_cnt
                local_cnt += 1


def extract_testcases(filename):
    x = []
    i = 0
    with open(filename, "r") as ins:
        for line in ins:
            if i < 7:
                i = i + 1
                continue
            line = line.strip()
            line1 = line.split(',')
            y = map(int, line1)
            x.append(y)
    return x


def obtain_ct_length(t):
    # Returns TS in good format and length
    # If this test suite does not exits, create
    strength = str(t)
    if os.path.exists("CT/" + dataset.lower() + "/" + dataset.lower() + "TS" + strength + "w.csv"):
        pass
    else:
        generateCT.generateCT(dataset, t)

    tsaux = extract_testcases("CT/" + dataset.lower() + "/" + dataset.lower() + "TS" + strength + "w.csv")
    tslen = len(tsaux)

    return tsaux, tslen


def select_from_ctfile(number_of_inputs):
    global test_suite_base

    # Initialize the test suite base
    test_suite_base, length_base = obtain_ct_length(1)
    test_suite_alpha, length_alpha = obtain_ct_length(2)

    i = 3

    # Case 1: fewer inputs than length base
    if number_of_inputs <= length_base:
        # Obtain all test cases from
        test_suite_ct_extra, _, _, _ = train_test_split(test_suite_base, [0] * len(test_suite_base),
                                                        test_size=len(test_suite_base) - int(number_of_inputs))
        print len(test_suite_ct_extra)
        return test_suite_ct_extra

    # Case 2: combine two test suites
    # Define base and alpha TS
    while length_alpha < number_of_inputs:
        test_suite_base = test_suite_alpha
        length_base = length_alpha
        test_suite_alpha, length_alpha = obtain_ct_length(i)
        i = i + 1

    set_base = set(tuple(a) for a in test_suite_base)
    set_alpha = set(tuple(a) for a in test_suite_alpha)

    set_difference = set_alpha - set_base

    # print "length of the set difference: "+ str(len(set_difference))
    list_difference = list(set_difference)

    difference_array = np.array(list_difference)

    # Number of inputs to be added
    n_alpha = number_of_inputs - len(test_suite_base)

    # Inputs to be added
    test_suite_ct_extra, _, _, _ = train_test_split(difference_array, [0] * len(difference_array),
                                                    test_size=len(difference_array) - int(n_alpha))

    # Add them
    test_suite_ct_selected = test_suite_base
    test_suite_ct_selected = np.append(test_suite_ct_selected, test_suite_ct_extra, axis=0)

    return test_suite_ct_selected


starting_time = time.time()
test_ct_file = select_from_ctfile(global_iteration_limit)

for inp in test_ct_file:
    evaluate_disc(inp)

seed_data = len(disc_inputs_list)
global_duplication_data = duplication_data

local_iteration_limit *= seed_data

for inp in disc_inputs_list:
    if local_cnt < local_iteration_limit:
        my_local_search(inp)
    else:
        break

local_duplication_data = duplication_data
total_data = len(tot_inputs)
disc_data = len(disc_inputs_list)
elapsed_time = time.time() - starting_time

print "Total evaluated data: " + str(total_data)
print "Number of seed data: " + str(seed_data)
print "Number of global duplication data: " + str(global_duplication_data)
print "Number of discriminatory data: " + str(disc_data)
print "Number of local duplication data: " + str(local_duplication_data)
print "Percentage of discriminatory data: " + str(float(disc_data) / float(total_data) * 100)
print "Number of discriminatory data per second: " + str(disc_data / elapsed_time)
print "Execution_time: {0}".format(elapsed_time) + "[sec]"

with open("result/RSUTT/" + experiment + "/" + dataset + "_" + classifier + "_" + str(N) + ".txt",
          "a") as myfile:
    myfile.write(str(total_data) + " "
                 + str(seed_data) + " "
                 + str(global_duplication_data) + " "
                 + str(disc_data) + " "
                 + str(local_duplication_data) + " "
                 + str(float(disc_data) / float(total_data) * 100) + " "
                 + str(disc_data / elapsed_time) + " "
                 + "{0}".format(elapsed_time) + " "
                 + "\n"
                 )

print "end time : " + str(datetime.datetime.now())
