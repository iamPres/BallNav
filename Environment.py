import math
from scipy import constants

class Environment:

    terminated = False
    def __init__(self, weight, bounds, di):
        self.weight = weight
        self.bounds = bounds
        self.degree_interval = di
        self.reset()

    def reset(self):
        self.pos = [0,0]
        self.vel = [0.1,-0.3]
        self.deg = [0,0]
        self.terminated = False
        return [self.pos, self.vel]
    
    def step(self, action):
        for dimIndex in range(len(action)):
            self.deg[dimIndex] += action[dimIndex] * 5
            self.compute_physics(dimIndex)

        return [self.pos, self.vel]
    
    def compute_physics(self, dimIndex):
        resultant = math.sin(math.radians(self.deg[dimIndex]))
        self.vel[dimIndex] += constants.g*self.weight*resultant
        self.pos[dimIndex] += self.vel[dimIndex]
        if self.pos[0] > self.bounds or self.pos[1] > self.bounds or self.pos[0] < -self.bounds or self.pos[1] < -self.bounds:
            self.terminated = True

