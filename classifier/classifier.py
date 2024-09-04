# Create models
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

dataset = sys.argv[1]
num = int(sys.argv[2])

# Reading the dataset
if dataset == "CENSUS":
    df = pd.read_csv('datasets/census.csv')

elif dataset == "GERMAN":
    df = pd.read_csv('datasets/german.csv')

elif dataset == "BANK":
    df = pd.read_csv('datasets/bank.csv')

else:
    print "The dataset name is wrong."

data = df.values

X = data[:, :-1]
Y = data[:, -1]


def dt():
    global X, Y
    m = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0,
                               random_state=None, splitter='best')

    # Fitting the model with the dataset
    m2 = m.fit(X, Y)

    if dataset == "CENSUS":
        pd.to_pickle(m2, 'census/DT_CENSUS.pkl')

    elif dataset == "GERMAN":
        pd.to_pickle(m2, 'german/DT_GERMAN.pkl')

    elif dataset == "BANK":
        pd.to_pickle(m2, 'bank/DT_BANK.pkl')

    else:
        print "The dataset name is wrong."
        sys.exit()

    return m2


def mlp():
    global X, Y
    m = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                      alpha=0.0001, batch_size='auto', learning_rate='constant',
                      learning_rate_init=0.001, power_t=0.5, max_iter=200,
                      shuffle=True, random_state=None, tol=0.0001,
                      verbose=False, warm_start=False, momentum=0.9,
                      nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                      beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    m2 = m.fit(X, Y)

    if dataset == "CENSUS":
        pd.to_pickle(m2, 'census/MLP_CENSUS.pkl')

    elif dataset == "GERMAN":
        pd.to_pickle(m2, 'german/MLP_GERMAN.pkl')

    elif dataset == "BANK":
        pd.to_pickle(m2, 'bank/MLP_BANK.pkl')

    else:
        print "The dataset name is wrong."
        sys.exit()

    return m2


def rf():
    global X, Y
    m = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=5, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                               oob_score=False, random_state=None, verbose=0,
                               warm_start=False)

    # Fitting the model with the dataset
    m2 = m.fit(X, Y)

    if dataset == "CENSUS":
        pd.to_pickle(m2, 'census/RF_CENSUS.pkl')

    elif dataset == "GERMAN":
        pd.to_pickle(m2, 'german/RF_GERMAN.pkl')

    elif dataset == "BANK":
        pd.to_pickle(m2, 'bank/RF_BANK.pkl')

    else:
        print "The dataset name is wrong."
        sys.exit()

    return m2


if num == 1:
    dt()
    mlp()
    rf()

    print "create classifier"

else:
    print "not create classifier"
