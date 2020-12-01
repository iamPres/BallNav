from Environment import Environment
from GeneticAlgorithm import GeneticAlgorithm
from Plotter import Plotter
import uuid
import numpy as np

def main():

    # HYPER-PARAMETERS
    epochs = 1000
    graphing_enabled = True
    population = 10
    max_time = 50
    mutation_rate = 0.05
    mutation_magnitude = 0.05
    bots_mutated = 9
    weight = 0.1
    bounds = 10
    degree_interval = 5

    # INIT
    plt = Plotter(max_time)
    env = Environment(weight, bounds, degree_interval)
    ga = GeneticAlgorithm(population, mutation_rate, mutation_magnitude, bots_mutated)
    ga.load('multicase.csv')

    # TRAIN MODEL
    for gen in range(epochs):
        print("\nGEN: "+str(gen))

        # LOOP THROUGH BOTS
        for bot in range(ga.popSize):

            # LOOP THROUGH CASES
            for caseIndex in range(len(Environment.test_cases)):
                plt.reset()
                state = env.reset(caseIndex)

                # SIMULATION LOOP
                for time in range(max_time):
                    action = ga.subjects[bot].forward(state)
                    state = env.step(action)

                    # PLOTTER
                    if graphing_enabled == True:
                        plt.plot(time, env.deg)
                ga.subjects[bot].fitness += np.sum(np.abs(state[1]))+np.sum(np.abs(state[0]))

            # SAVE TRAINED
            print(ga.subjects[bot].fitness)
            if ga.subjects[bot].fitness < 0.1:
                file = open('trained_model_'+str(uuid.uuid1())+'.csv', "w")
                file.write(str(ga.subjects[bot].network))
                file.close()
                return
            

        # RESET
        ga.compute_generation()
        print("FITTEST: "+str(ga.fittestIndex)+": "+str(ga.subjects[ga.fittestIndex].fitness))
        print("NETWORK: "+str(ga.subjects[ga.fittestIndex].network))
        print("-----")
        ga.reset()

        # SAVE TIMEOUT
        if gen == epochs-1:
            file = open('timeout_model_'+str(uuid.uuid1())+'.csv', "w")
            file.write(str(ga.subjects[bot].network))
            file.close()
            return
main()
