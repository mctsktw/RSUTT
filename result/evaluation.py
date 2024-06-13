import sys
import numpy as np

def evaluation(algorithm, dataset, classifier, ace, N):
    aData = []
    sData = []
    dData = []
    dDataPercent = []
    dDataPerSecond = []
    time = []
    gdData = []
    ldData = []

    f = open(algorithm + "/" + dataset + "_" + classifier + "_" + ace + "_" + N + ".txt")

    for data in f.readlines():
        a, s, gd, d, ld, dP, dPS, t = data.split()
        aData.append(float(a))
        sData.append(float(s))
        gdData.append(float(gd))
        dData.append(float(d))
        ldData.append(float(ld))
        dDataPercent.append(float(dP))
        dDataPerSecond.append(float(dPS))
        time.append(float(t))

    f.close()

    if __name__ == '__main__':
        print "Total evaluated data: "
        print "average: " + str(float(sum(aData) / len(aData)))
        print "standard deviation: " + str(np.std(aData))
        print "max: " + str(max(aData))
        print "min: " + str(min(aData))
        print ""

        print "Number of seed data: "
        print "average: " + str(float(sum(sData) / len(sData)))
        print "standard deviation: " + str(np.std(sData))
        print "max: " + str(max(sData))
        print "min: " + str(min(sData))
        print ""

        print "Number of global duplication data: "
        print "average: " + str(float(sum(gdData) / len(gdData)))
        print "standard deviation: " + str(np.std(gdData))
        print "max: " + str(max(gdData))
        print "min: " + str(min(gdData))
        print ""

        print "Number of discriminatory data: "
        print "average: " + str(float(sum(dData) / len(dData)))
        print "standard deviation: " + str(np.std(dData))
        print "max: " + str(max(dData))
        print "min: " + str(min(dData))
        print ""

        print "Number of local duplication data: "
        print "average: " + str(float(sum(ldData) / len(ldData)))
        print "standard deviation: " + str(np.std(ldData))
        print "max: " + str(max(ldData))
        print "min: " + str(min(ldData))
        print ""

        print "Percentage of discriminatory data: "
        print "average: " + str(float(sum(dDataPercent) / len(dDataPercent)))
        print "standard deviation: " + str(np.std(dDataPercent))
        print "max: " + str(max(dDataPercent))
        print "min: " + str(min(dDataPercent))
        print ""

        print "Number of discriminatory data per second: "
        print "average: " + str(float(sum(dDataPerSecond) / len(dDataPerSecond)))
        print "standard deviation: " + str(np.std(dDataPerSecond))
        print "max: " + str(max(dDataPerSecond))
        print "min: " + str(min(dDataPerSecond))
        print ""

        print "Execution_time: "
        print "average: " + str(float(sum(time) / len(time)))
        print "standard deviation: " + str(np.std(time))
        print "max: " + str(max(time))
        print "min: " + str(min(time))
        print ""

    with open(algorithm + "/" + dataset + "_" + classifier + "_" + ace + "_" + N + "_" + "evaluation.txt", "w") as myfile:
        myfile.write("Total evaluated data:\n"
                         + str(float(sum(aData) / len(aData))) + " "
                         + str(np.std(aData)) + " "
                         + str(max(aData)) + " "
                         + str(min(aData)) + " "
                         + "\n"
                         + "Number of seed data:\n"
                         + str(float(sum(sData) / len(sData))) + " "
                         + str(np.std(sData)) + " "
                         + str(max(sData)) + " "
                         + str(min(sData)) + " "
                         + "\n"
                         + "Number of global duplication data:\n"
                         + str(float(sum(gdData) / len(gdData))) + " "
                         + str(np.std(gdData)) + " "
                         + str(max(gdData)) + " "
                         + str(min(gdData)) + " "
                         + "\n"
                         + "Number of discriminatory data:\n"
                         + str(float(sum(dData) / len(dData))) + " "
                         + str(np.std(dData)) + " "
                         + str(max(dData)) + " "
                         + str(min(dData)) + " "
                         + "\n"
                         + "Number of local duplication data:\n"
                         + str(float(sum(ldData) / len(ldData))) + " "
                         + str(np.std(ldData)) + " "
                         + str(max(ldData)) + " "
                         + str(min(ldData)) + " "
                         + "\n"
                         + "Percentage of discriminatory data:\n"
                         + str(float(sum(dDataPercent) / len(dDataPercent))) + " "
                         + str(np.std(dDataPercent)) + " "
                         + str(max(dDataPercent)) + " "
                         + str(min(dDataPercent)) + " "
                         + "\n"
                         + "Number of discriminatory data per second:\n"
                         + str(float(sum(dDataPerSecond) / len(dDataPerSecond))) + " "
                         + str(np.std(dDataPerSecond)) + " "
                         + str(max(dDataPerSecond)) + " "
                         + str(min(dDataPerSecond)) + " "
                         + "\n"
                         + "Execution_time:\n"
                         + str(float(sum(time) / len(time))) + " "
                         + str(np.std(time)) + " "
                         + str(max(time)) + " "
                         + str(min(time)) + " "
                         + "\n")


if __name__ == '__main__':
    evaluation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
