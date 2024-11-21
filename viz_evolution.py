import numpy as np
import matplotlib.pyplot as plt
import sys
import os

expname = sys.argv[1]
reps = int(sys.argv[2])

currentpath = os.getcwd()
os.chdir(currentpath+"/"+expname)

for i in range(reps):
    ev = np.load("evol"+str(i)+".npy")
    plt.plot(ev)
    print(i,ev[-1])

plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Evolution")
plt.legend(range(reps))
plt.show()