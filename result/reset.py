import os
import sys
import shutil

num = int(sys.argv[1])

if num == 1:
    algorithm = ["AEQUITAS", "KOSEI", "CGFT", "RSUTT"]
    # classifier = ["DT", "MLP", "RF"]
    # dataset = ["CENSUS", "GERMAN", "BANK"]
    # ace = ["FASTEST", "MIDDLE", "BEST"]
    # N = ["3000", "7000"]
    experiment = ["A", "B", "C"]

    for a in algorithm:
        if os.path.isdir(a):
            shutil.rmtree(a)
        os.mkdir(a)
        if a == "RSUTT":
            for b in experiment:
                os.mkdir(a + "/" + b)

    print "reset"

else:
    print "not reset"
