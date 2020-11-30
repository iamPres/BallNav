import math
from scipy import constants
import numpy as np

class Environment:

    test_cases = np.array([
            [[0,0],[0,0],[0,0]],
            [[5,0],[0.5,0],[0,0]],
            [[0,5],[0.0,0.5],[0,0]],
            [[-5,0],[-0.5,0],[0,0]],
            [[0,-5],[0,-0.5],[0,0]],
            [[5,5],[0.5,0.5],[0,0]],
            [[-5,-5],[-0.5,-0.5],[0,0]],
            [[0,0],[0.5,-0.5],[0,0]],
            [[5,0],[0,-0.5],[0,0]],
            [[0,5],[-0.5,0.5],[0,0]],
            [[-5,0],[0,0],[0,0]],
            [[0,-5],[0.5,0.5],[0,0]],
            [[5,5],[0,-0.5],[0,0]],
            [[-5,-5],[0.5,0.5],[0,0]],
        ])
    def __init__(self, weight, bounds, di):
        self.terminated = False
        self.weight = weight
        self.bounds = bounds
        self.degree_interval = di
        self.reset(0)

    def reset(self, ci):
        self.pos = Environment.test_cases[ci][0].copy()
        self.vel = Environment.test_cases[ci][1].copy()
        self.deg = Environment.test_cases[ci][2].copy()
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

