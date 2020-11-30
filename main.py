from Environment import Environment
from GeneticAlgorithm import GeneticAlgorithm
from Plotter import Plotter
import uuid

def main():

    # HYPER-PARAMETERS
    epochs = 1000
    graphing_epoch_interval = 1
    population = 100
    max_time = 1000
    mutation_rate = 0.1
    mutation_magnitude = 1
    bots_mutated = 98
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
            
            cases_completed = 0
            fitness = 0

            # LOOP THROUGH CASES
            for caseIndex in range(len(Environment.test_cases)):
                plt.reset()
                state = env.reset(caseIndex)
                case_time = 0

                # SIMULATION LOOP
                for time in range(max_time):
                    case_time = time
                    action = ga.subjects[bot].forward(state)
                    state = env.step(action)  
                    if env.terminated:
                        break                
                    if time == max_time-1:
                        cases_completed += 1
                    
                    # PLOTTER
                    if gen % graphing_epoch_interval == 0:
                        plt.plot(time, env.deg)

                # SAVE TRAINED
                if cases_completed == len(Environment.test_cases):
                    file = open('trained_model_'+str(uuid.uuid1())+'.csv', "w")
                    file.write(str(ga.subjects[bot].network))
                    file.close()
                    return
                
                fitness += case_time

            ga.subjects[bot].fitness = cases_completed*fitness

        # RESET
        ga.compute_generation()
        print("FITTEST: "+str(ga.fittestIndex)+": "+str(ga.subjects[ga.fittestIndex].fitness)+"/"+str((len(Environment.test_cases)**2)*max_time))
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
