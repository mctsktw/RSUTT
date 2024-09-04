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
algorithm = sys.argv[1]
dataset = sys.argv[2]
classifier = sys.argv[3]
ace = sys.argv[4]
N = int(sys.argv[5])

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

    # csv_setting_path = 'classifier/datasets/' + dataset.lower() + '_retraining.csv'

    # if not os.path.isfile(csv_setting_path):
    #     print "The path name is wrong."
    #     sys.exit()

    # df = pd.read_csv(csv_setting_path)

    # data = df.values

    # X = data[:, :-1]
    # Y = data[:, -1]

    params = eval('config_' + dataset.lower()).params
    sensitive_param = eval('config_' + dataset.lower()).sensitive_param
    input_bounds = eval('config_' + dataset.lower()).input_bounds

    # Only when you want to change Classsifier
    # eval(classifier.lower())()

    classifier_name = 'classifier/' + dataset.lower() + '/' + classifier.upper() + '_' + dataset.upper() + '.pkl'

    if ace == "FASTEST":
        sensitive_list = [input_bounds[sensitive_param - 1][0],
                          input_bounds[sensitive_param - 1][0] + 1]

    elif ace == "MIDDLE":
        sensitive_list = [input_bounds[sensitive_param - 1][0],
                          (input_bounds[sensitive_param - 1][0] + input_bounds[sensitive_param - 1][1]) // 2]

    elif ace == "BEST":
        sensitive_list = [input_bounds[sensitive_param - 1][0],
                          input_bounds[sensitive_param - 1][1]]

    else:
        print "The ace name is wrong."
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


def evaluate_disc(ed_input):
    inp0 = [int(i) for i in ed_input]
    inp1 = [int(i) for i in ed_input]

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
        global duplication_data
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


# Aequitas algorithm

if algorithm == "AEQUITAS":
    duplication_data = 0

    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001


    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i]) / float(probability_sum)


    class LocalPerturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            param_choice = np.random.choice(xrange(params), p=param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[direction_probability[param_choice],
                                                        (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_disc(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(
                    direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(
                    direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size,
                                                      0)
                normalise_probability()

            return x


    class GlobalDiscovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            for i in xrange(params):
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = sensitive_list[0]
            return x


    starting_time = time.time()
    initial_input = []
    if dataset == "CENSUS":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "GERMAN":
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    elif dataset == "BANK":
        initial_input = [0, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 1, 1, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = GlobalDiscovery()
    local_perturbation = LocalPerturbation()

    basinhopping(evaluate_disc, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    seed_data = len(disc_inputs_list)
    global_duplication_data = duplication_data

    local_iteration_limit = (local_iteration_limit * seed_data) // 200

    copy_list = copy.copy(disc_inputs_list)
    for inp in copy_list:
        basinhopping(evaluate_disc, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)

    local_duplication_data = duplication_data
    total_data = len(tot_inputs)
    disc_data = len(disc_inputs_list)
    elapsed_time = time.time() - starting_time

# KOSEI algorithm

elif algorithm == "KOSEI":
    duplication_data = 0
    local_cnt = 0


    class GlobalDiscovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            for i in xrange(params):
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = sensitive_list[0]
            return x


    # Local search algorithm for each dataset


    def my_local_search_census(mlsc_input):
        for param in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]:
            for direction in [-1, 1]:
                inp2 = copy.copy(mlsc_input)
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


    def my_local_search_german(mlsg_input):
        for param in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
            for direction in [-1, 1]:
                inp2 = copy.copy(mlsg_input)
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


    def my_local_search_bank(mlsb_input):
        for param in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            for direction in [-1, 1]:
                inp2 = copy.copy(mlsb_input)
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


    starting_time = time.time()
    initial_input = []
    if dataset == "CENSUS":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "GERMAN":
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    elif dataset == "BANK":
        initial_input = [0, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 1, 1, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = GlobalDiscovery()

    basinhopping(evaluate_disc, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    seed_data = len(disc_inputs_list)
    global_duplication_data = duplication_data

    local_iteration_limit *= seed_data

    for inp in disc_inputs_list:
        if local_cnt < local_iteration_limit:
            eval("my_local_search_" + dataset.lower())(inp)
        else:
            break

    local_duplication_data = duplication_data
    total_data = len(tot_inputs)
    disc_data = len(disc_inputs_list)
    elapsed_time = time.time() - starting_time

# CGFT algorithm

elif algorithm == "CGFT":
    duplication_data = 0

    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001


    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i]) / float(probability_sum)


    class LocalPerturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            param_choice = np.random.choice(xrange(params), p=param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[direction_probability[param_choice],
                                                        (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_disc(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(
                    direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(
                    direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size,
                                                      0)
                normalise_probability()

            return x


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
    minimizer = {"method": "L-BFGS-B"}

    local_perturbation = LocalPerturbation()

    test_ct_file = select_from_ctfile(global_iteration_limit)

    for inp in test_ct_file:
        evaluate_disc(inp)

    seed_data = len(disc_inputs_list)
    global_duplication_data = duplication_data

    local_iteration_limit = (local_iteration_limit * seed_data) // 200

    copy_list = copy.copy(disc_inputs_list)
    for inp in copy_list:
        basinhopping(evaluate_disc, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)

    local_duplication_data = duplication_data
    total_data = len(tot_inputs)
    disc_data = len(disc_inputs_list)
    elapsed_time = time.time() - starting_time

# RSUTT algorithm

elif algorithm == "RSUTT":
    duplication_data = 0
    local_cnt = 0


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

else:
    print "The algorithm name is wrong."
    sys.exit()

print "Total evaluated data: " + str(total_data)
print "Number of seed data: " + str(seed_data)
print "Number of global duplication data: " + str(global_duplication_data)
print "Number of discriminatory data: " + str(disc_data)
print "Number of local duplication data: " + str(local_duplication_data)
print "Percentage of discriminatory data: " + str(float(disc_data) / float(total_data) * 100)
print "Number of discriminatory data per second: " + str(disc_data / elapsed_time)
print "Execution_time: {0}".format(elapsed_time) + "[sec]"

with open("result/" + algorithm + "/" + dataset + "_" + classifier + "_" + ace + "_" + str(N) + ".txt",
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
