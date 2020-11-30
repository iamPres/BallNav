import math
from scipy import constants
import numpy as np

class Environment:

    test_cases = np.array([
            [[5,1],[0.5,0.2],[0,0]],
            [[2,5],[-0.3,0.5],[0,0]],
            [[-5,-4],[-0.5,0.2],[0,0]],
            [[3,-5],[0.3,-0.5],[0,0]],
            [[5,5],[0.5,0.5],[0,0]],
            [[-5,-5],[-0.5,-0.5],[0,0]],
            [[-1,2],[0.5,-0.5],[0,0]],
            [[5,-3],[0.3,-0.5],[0,0]],
            [[0-4,5],[-0.5,0.5],[0,0]],
            [[-5,2],[0.1,-0.2],[0,0]],
            [[-4,-5],[0.5,0.5],[0,0]],
            [[5,5],[0.4,-0.5],[0,0]],
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
        self.acc = [0,0]
        self.terminated = False
        return [self.pos, self.vel, self.acc]

    def step(self, action):
        for dimIndex in range(2):
            self.deg[dimIndex] += action[dimIndex] * 5
            if self.deg[dimIndex] > 30:
                self.deg[dimIndex] = 30
            self.compute_physics(dimIndex)

        return [self.pos, self.vel, self.acc]

    def compute_physics(self, dimIndex):
        resultant = math.sin(math.radians(self.deg[dimIndex]))
        self.acc[dimIndex] = constants.g*self.weight*resultant
        self.vel[dimIndex] += self.acc[dimIndex]
        self.pos[dimIndex] += self.vel[dimIndex]


class Environment2:

    test_cases = np.array([

        [[5, 1], [0.5, 0.2], [0, 0]],
        [[2, 5], [-0.3, 0.5], [0, 0]],
        [[-5, -4], [-0.5, 0.2], [0, 0]],
        [[3, -5], [0.3, -0.5], [0, 0]],
        [[5, 5], [0.5, 0.5], [0, 0]],
        [[-5, -5], [-0.5, -0.5], [0, 0]],
        [[-1, 2], [0.5, -0.5], [0, 0]],
        [[5, -3], [0.3, -0.5], [0, 0]],
        [[0-4, 5], [-0.5, 0.5], [0, 0]],
        [[-5, 2], [0.1, -0.2], [0, 0]],
        [[-4, -5], [0.5, 0.5], [0, 0]],
        [[5, 5], [0.4, -0.5], [0, 0]],
        [[-5, -5], [0.5, 0.5], [0, 0]],
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
        self.acc = np.array([0, 0]).copy()
        self.terminated = False
        return [self.pos, self.vel, self.acc]

    def step(self, action):
        self.deg = np.add(self.deg, np.dot(action, [5, 5]))
        self.compute_physics()
        return [self.pos, self.vel, self.acc]

    def compute_physics(self):
        resultant = np.sin(np.radians(self.deg))
        self.acc = np.dot(constants.g*self.weight, resultant)
        self.vel = np.add(self.acc, self.vel)
        self.pos += np.add(self.vel, self.pos)
        if self.pos[0] > self.bounds or self.pos[1] > self.bounds or self.pos[0] < -self.bounds or self.pos[1] < -self.bounds:
            self.terminated = True
