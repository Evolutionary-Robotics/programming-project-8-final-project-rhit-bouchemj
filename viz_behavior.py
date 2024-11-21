import ea                  
import fnn2 as fnn          
import invpend
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pybullet as p
import pybullet_data
import time

expname = sys.argv[1]
id = sys.argv[2]

# currentpath = os.getcwd()
# os.chdir(currentpath+'/'+expname)

# ANN Params
layers = [3,10,1] #uses same network as execution?

# Task Params
duration = 200
stepsize = 0.02
noisestd = 0.01
randomPushChance = 0.01

# Time
timer = np.arange(0.0,duration,stepsize)
dt = 1/500
maxTorque = 4

# Load genotype
bestind_genotype = np.load(expname+"/gen"+id+".npy")

def angle_normalize(x): # input = radians
    return (((x+np.pi) % (2*np.pi)) - np.pi) #put range of angle to -pi to pi (0 = hanging down)

def evaluate(genotype): # repeat of fitness function but saving theta
    nn = fnn.FNN(layers)
    nn.setParams(genotype)  #Use the specific genotype of best iteration
    body = invpend.InvPendulum()
    physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    boxId = p.loadURDF("../pendulumThing.urdf", useFixedBase=True)
    # time.sleep(20)

    #initialize all variables for the system (cart and pole)
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=0, controlMode=p.POSITION_CONTROL)
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL)
    time.sleep(dt)
    p.stepSimulation()
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
    out_hist = np.zeros((len(timer),1))
    in_hist = np.zeros((len(timer),3))
    f_hist=np.zeros(len(timer))
    k=0
    for t in timer:
        bodyArr = np.zeros(4)
        bodyArr[0] = p.getLinkState(bodyUniqueId=boxId, linkIndex=2)[0][0] #x-axis value
        bodyArr[1] = p.getLinkState(bodyUniqueId=boxId, linkIndex=2)[0][2] - 1 # y-axis value
        bodyArr[2] = p.getLinkState(bodyUniqueId=boxId, linkIndex=2, computeLinkVelocity=1)[7][1] #angular velocity
        bodyArr[3] = np.pi/2 - np.arctan2(bodyArr[1], bodyArr[0]) #angle in terms of radians
        inp = bodyArr[0:3]  #input array into NN
        out = nn.forward(inp)*2-1 + np.random.normal(0.0,noisestd)
        u = out[0][0]*maxTorque
        u = min(maxTorque, max(u, -maxTorque))
        if(np.random.random() < randomPushChance):
            tmpU = np.random.random()*500 - 250
            p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, controlMode=p.TORQUE_CONTROL, force=tmpU)
        else:
            p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, controlMode=p.TORQUE_CONTROL, force=u)
        cost = angle_normalize(bodyArr[3])**2 + .1*bodyArr[2]**2 + .001*(u**2)  #cosine? angle^2 + angleDer^2 + u^2
        f = -cost*stepsize #returns 0 at peak and (-) value at low angle
        # if(np.random.random() < randomPushChance):
            # p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, controlMode=p.TORQUE_CONTROL, force=np.random.random()*6 - 3)
        # body.render()
        in_hist[k] = inp   #record what variables were
        out_hist[k] = u#out
        f_hist[k] = f       #record instantaneous fitness
        time.sleep(dt)
        p.stepSimulation()
        k += 1              #index
    p.disconnect()
    return in_hist, f_hist, out_hist

in_hist1, f_hist1, out_hist1 = evaluate(bestind_genotype)

plt.plot(in_hist1)
plt.plot(f_hist1*50,'k',label="Output")
plt.plot(out_hist1)
plt.xlabel("Time")
plt.ylabel("Inputs & Outputs & Fitness")
plt.legend(["cos_theta","sin_theta","ang. vel", "instant fitness", "Torque"])
plt.title("Agent #"+id)
plt.show()
