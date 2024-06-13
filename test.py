import random
import time
import datetime
import  numpy as np
import copy

z = np.asarray([0, 1, 2])
z = np.reshape(z, (1, -1))

x = set()
x.add(tuple(map(tuple, z)))

y = copy.copy(x)

print(x)

print(y)




