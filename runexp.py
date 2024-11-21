import os
import sys

expname = sys.argv[1]
reps = int(sys.argv[2])

currentpath = os.getcwd()

os.mkdir(expname)
os.chdir(currentpath+"/"+expname)

# print(expname)
# print(reps)
# print(currentpath)

for i in range(reps):
    os.system("python ../evolve_ffann_pendulum.py "+str(i)+" &")