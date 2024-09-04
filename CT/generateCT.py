# generateCT.py

# Once the txt files from each dataset are created, execute the shell command
# python generateCT.py <dataset> <strength>
import sys
import subprocess


def generateCT(dataset, t):
    tstr = str(t)
    command = '-Ddoi=' + tstr + ' -Doutput=csv -jar ../ACTS3.2/acts_3.2.jar ' \
              + dataset.lower() + 'TestModel.txt ' + dataset.lower() + '/' + dataset.lower() + 'TS' + tstr \
              + 'w.csv'
    subprocess.call('java ' + command, shell=True)


if __name__ == "__main__":
    generateCT(sys.argv[1], sys.argv[2])
