import os
import sys
import shutil

num = int(sys.argv[1])

if num == 1:
    algorithm = ["AEQUITAS", "KOSEI", "CGFT", "RSUTT"]
    # classifier = ["DT", "MLP", "RF", "SGD"]
    # dataset = ["CENSUS", "GERMAN", "BANK"]
    # ace = ["FASTEST", "MIDDLE", "BEST"]
    # N = ["3000", "7000"]

    for a in algorithm:
        shutil.rmtree(a)
        os.mkdir(a)

    print "reset"

else:
    print "not reset"
