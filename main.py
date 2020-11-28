from Environment import Environment
from GeneticAlgorithm import GeneticAlgorithm
from Plotter import Plotter
import uuid

def main():

    epochs = 1000
    graphing_epoch_interval = 10000
    population = 100
    max_time = 500
    mutation_rate = 0.2
    mutation_magnitude = 0.1
    bots_mutated = 98
    weight = 0.1
    bounds = 10
    degree_interval = 5
    starting_position = [0,0]
    starting_velocity = [-0.1, 0.3]
    starting_slope = [0,0]

    plt = Plotter(max_time)
    env = Environment(weight, bounds, degree_interval)
    ga = GeneticAlgorithm(population, mutation_rate, mutation_magnitude, bots_mutated)
    ga.load('trained_model_125d3f78-31b5-11eb-98c3-784f4381b3ea.csv')

    for gen in range(epochs):
        print("\nGEN: "+str(gen))
        for bot in range(ga.popSize):
            state = env.reset() 
            plt.reset()
            for time in range(max_time):
                action = ga.subjects[bot].forward(state)
                state = env.step(action)  
                if gen % graphing_epoch_interval == 0 and gen != 0:
                    plt.plot(time, env.deg)
                if env.terminated:
                    ga.subjects[bot].fitness = time
                    break
                if time == max_time-1:
                    file = open('trained_model_'+str(uuid.uuid1())+'.csv', "w")
                    file.write(str(ga.subjects[bot].network))
                    file.close()
                    return
        ga.compute_generation()
        print(str(ga.fittestIndex)+": "+str(ga.subjects[ga.fittestIndex].fitness))
        ga.reset()
        if gen == epochs-1:
            file = open('timeout_model_'+str(uuid.uuid1())+'.csv', "w")
            file.write(str(ga.subjects[bot].network))
            file.close()
            return
main()
