import ea
import fnn2 as fnn
# import invpend
import numpy as np
import matplotlib.pyplot as plt
import sys
import pybullet as p
import pybullet_data
# import time

id = sys.argv[1]

# ANN Params
layers = [3,10,1]

# Task Params
duration = 200
stepsize = 0.1
noisestd = 0.01
dt = 1/500

# Time
time = np.arange(0.0,duration,stepsize)

# EA Params
popsize = 50
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
recombProb = 0.5
mutatProb = 0.01
tournaments = 100*popsize

# Fitness initialization ranges
trials_theta = 2
trials_thetadot = 2
total_trials = trials_theta*trials_thetadot

theta_range = np.linspace(np.pi - np.pi/20, np.pi + np.pi/20, num=trials_theta)        # angle of the pendulum
thetadot_range = np.linspace(-0.25, 0.25, num=trials_thetadot)  # angular velocity of pendulum (angle deriv)

# Fitness function
def fitnessFunction(genotype):
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    # vary initial condition
    k = 0
    fit = 0.0
    maxTorque = 5
    for theta in theta_range: # theta starts at -0.05? then at 0.5 (to train with different starting positions)
        for theta_dot in thetadot_range: # diff angular velocity starts
            physicsClient = p.connect(p.DIRECT) # or p.DIRECT for non-graphical version
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0,0,-10)
            planeId = p.loadURDF("plane.urdf")
            boxId = p.loadURDF("../pendulumThing.urdf", useFixedBase=True)
            #initialize all variables for the system (cart and pole)
            p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=theta, controlMode=p.POSITION_CONTROL)
            p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=theta_dot, controlMode=p.VELOCITY_CONTROL)
            # time.sleep(dt)
            p.stepSimulation()
            p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)


            # make sure getting proper number of steps 
            f = stepsize
            t = 0
            while t < duration:   #Until t is longer than the duration of the sym. or the system goes out of bounds (f becomes 0 when OoB)
                bodyArr = np.zeros(4)
                bodyArr[0] = p.getLinkState(bodyUniqueId=boxId, linkIndex=2)[0][0] #x-axis value
                bodyArr[1] = p.getLinkState(bodyUniqueId=boxId, linkIndex=2)[0][2] - 1 # y-axis value
                bodyArr[2] = p.getLinkState(bodyUniqueId=boxId, linkIndex=2, computeLinkVelocity=1)[7][1] #angular velocity
                bodyArr[3] = np.pi/2 - np.arctan2(bodyArr[1], bodyArr[0]) #angle in terms of radians
                inp = bodyArr[0:3]  #input array into NN
                u = nn.forward(inp)*2 - 1 + np.random.normal(0.0,noisestd)    #average the output around 0 (-1 to 1) and add random noise
                u = u[0][0]*maxTorque
                p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, controlMode=p.TORQUE_CONTROL, force=u)
                cost = angle_normalize(bodyArr[3])**2 + .1*bodyArr[2]**2 + .001*(u**2)  #cosine? angle^2 + angleDer^2 + u^2
                f = -cost*stepsize #returns 0 at peak and (-) value at low angle
                fit += f    #fitness determined based off of how long the system lasts (won't work for pendulumn)
                t += stepsize   #update t <-- get closer to stopping sim
                p.stepSimulation()
            k += 1
            p.disconnect()
            # print("disconnected")
    return fit/(duration*total_trials) #fitness function <-- averages fitness,(-) for down angle, 0 for up angle [never gets positive?]

def angle_normalize(x): # input = radians
    return (((x+np.pi) % (2*np.pi)) - np.pi) #put range of angle to -pi to pi (0 = hanging down)

# Evolve and visualize fitness over generations
ga = ea.MGA(fitnessFunction, genesize, popsize, recombProb, mutatProb, tournaments)
print("ga made")
ga.run()
print("ga ran")
np.save("evol"+id+".npy",ga.bestfit)    #Save the best fit line to a file

# Get best evolved network and show its activity
bestind_num = int(ga.bestind[-1])   #Returns the index of the genotype that had the maximum fitness last
print(bestind_num)
bestind_genotype = ga.pop[bestind_num]
np.save("gen"+id+".npy",bestind_genotype)   #saves the best genotype array to a file
