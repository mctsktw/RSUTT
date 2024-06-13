import sys
import os
import pandas as pd

num = int(sys.argv[1])
dataset = sys.argv[2]
sensitive_param = int(sys.argv[3])

if num != 1:
    print "not genarate"
    sys.exit()

csv_path = '../classifier/datasets/' + dataset.lower() + '.csv'

if not os.path.isfile(csv_path):
    print "The path name is wrong."
    sys.exit()

df = pd.read_csv(csv_path)

df_data_list = df.values.tolist()

params = len(df_data_list[0]) - 1

input_bounds = []
for param in range(len(df_data_list[0])):
    if param != len(df_data_list[0]) - 1:
        input_bounds.append([df_data_list[0][param], df_data_list[0][param]])
    else:
        continue

for data in df_data_list:
    for param in range(len(data)):
        if param != len(data) - 1:
            if input_bounds[param][0] > data[param]:
                input_bounds[param][0] = data[param]
            elif input_bounds[param][1] < data[param]:
                input_bounds[param][1] = data[param]
        else:
            continue

with open("config_" + dataset.lower() + ".py", "w") as myfile:
    myfile.write("params = " + str(params) + "\n"
                 + "\n"
                 + "sensitive_param = " + str(sensitive_param) + " # Starts at 1.\n"
                 + "\n"
                 + "input_bounds = []\n"
                 + "\n"
                 )

for param in range(len(input_bounds)):
    with open("config_" + dataset.lower() + ".py", "a") as myfile:
        myfile.write("input_bounds.append([" + str(input_bounds[param][0]) + ", " + str(input_bounds[param][1]) + "])")
    if (param + 1) == sensitive_param:
        with open("config_" + dataset.lower() + ".py", "a") as myfile:
            myfile.write(" # Discriminatory parameter")
    with open("config_" + dataset.lower() + ".py", "a") as myfile:
        myfile.write("\n")

print "generate"
