# RSUTT

# Paper Information

Code to the paper "RSUTT: Robust Search Using T-way Testing" by Chisato Matsukawa, Shingo Takada in QUATIC 2024.

# Execution Environment

programming language

* Python : 2.7.18

expansion module

* numpy : 1.14.5
* pandas : 0.24.2
* scipy : 1.1.0
* scikit-learn : 0.19.0

The authors used Pycharm CE 2022.2.3 as the development IDE.

# Links of backend tools

* ACTS : https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/downloadable-tools

# Scikit-Learn Classifiers

* Decision Tree (DT)
* Multi-Layer Perceptron (MLP)
* Random Forest (RF)

Each of the following three directories contains the following

It is in the "classifier" directory.

* Adult (Census Income) : Scikit-Learn classifiers in the "census" directory
* Statlog (German Credit)：Scikit-Learn classifiers in the "german" directory
* Bank Marketing : Scikit-Learn classifiers in the "bank" directory

# Config 

It is in the "config" directory.

* Adult (Census Income) : config_census.py
* Statlog(German Credit)：config_german.py
* Bank Marketing : config_bank.py

The config files has the following data (same as the Udeshi's codes) : 

* params : The number of parameters in the data
* sensitive_param : The parameter under test
* input_bounds : The bounds of each parameter
* classifier_name : Pickled scikit-learn classifier under test (only applicable to the sklearn files)

# Demo

First, you can manually create an empty directory.

<pre>
result
├ AEQUITAS
├ CGFT
├ KOSEI
└ RSUTT
  ├ A
  ├ B
  └ C
</pre>

Running reset.py creates a directory, but be careful where you run it.

## RQ1~RQ3

`python RSUTT.py <algorithm> <dataset> <classifier> <ace> <test_case>`

first parameter : algorithm

* AEQUITAS 
* KOSEI 
* CGFT 
* RSUTT
 
second parameter : dataset

* CENSUS
* GERMAN
* BANK

third parameter : classifier

* DT 
* MLP
* RF

fourth parameter : ace

* FASTEST
* MIDDLE
* BEST

fifth parameter : test_case

test_case is global_iteration_limit.

## RQ4

`python RSUTTABC.py <experiment> <dataset> <classifier> <test_case>`

first parameter : experiment

* A
* B
* C

second parameter : dataset

* CENSUS
* GERMAN
* BANK

third parameter : classifier

* DT 
* MLP
* RF

fourth parameter : test_case

test_case is global_iteration_limit.


# iteration

We set the default iteration limits in the following:

global_iteration_limit = test_case

local_iteration_limit = test_case * seed data (However, Aequitas and CGFT are further divided by 200.)

# Notes

As of August 1 2024, RSUTT is still being updated, so the experimental results may differ from those in the paper.
