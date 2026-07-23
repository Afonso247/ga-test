from abc import ABC, abstractmethod

import sys
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    genetic_code = "" # if we handle the genetic code as chromosomes, we would not destroy good weights at random, however we would not introduce new weights during training, except for random mutations
    fitness = sys.float_info.min
    age = 0

    def __init__(self, genetic_code, fitness = sys.float_info.min, age = 0):
        self.genetic_code = genetic_code
        self.fitness = fitness
        self.age = age

class GeneticSearchSettings:
    fitness_fuction = None
    population_size = -1
    individual_genectic_size = -1
    number_of_generations = -1
    mutation_rate = -1
    store_best_overall_individual = False
    perserve_best_individuals = 0
    aging = False

    def __init__(self, fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, store_best_overall_individual, perserve_best_individuals, aging):
        self.fitness_fuction = fitness_fuction
        self.population_size = population_size
        self.individual_genectic_size = individual_genectic_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.store_best_overall_individual = store_best_overall_individual
        self.perserve_best_individuals = perserve_best_individuals
        self.aging = aging

class GeneticSearch:

    fitness_history = []

    def radom_initialization(self, population_size, individual_genectic_size):
        # TODO validate the parameters

        rng = np.random.default_rng()

        # to understand this line aks an AI about list comprehension
        return [Individual(rng.integers(0, 2, individual_genectic_size)) for _ in range(population_size)]

    def compute_fitness_and_sort_individuals(self, population, fitness_function, consider_aging):
        for individual in population:
            individual.fitness = fitness_function(individual)

            if consider_aging:
              #individual.fitness = individual.fitness - individual.age
              individual.fitness = individual.fitness * (0.9 ** individual.age)


        # sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        return population

    def random_selection(self, population):
        intervals = []
        sum = 0

        for individual in population:
            sum = sum + individual.fitness
            intervals.append(sum)

        rng = np.random.default_rng()

        number = rng.uniform(0, sum)

        for i in range(len(population)):
            if (number <= intervals[i]):
                return population[i]

        # this line should never be executed
        print("ERROR: random selection had no return", sum)

    def reproduce(self, parent1, parent2):
       rng = np.random.default_rng()

       splitting_index = rng.integers(0, len(parent1.genetic_code))

       return Individual(np.concatenate((parent1.genetic_code[:splitting_index], parent2.genetic_code[splitting_index:])))

    def mutation(self, individual, mutation_rate):
        rng = np.random.default_rng()

        number = rng.random()

        if (number < mutation_rate):
            genetic_code = individual.genetic_code
            mutation_index = rng.integers(len(genetic_code))
            genetic_code[mutation_index] = (genetic_code[mutation_index] + 1) % 2

    def geneticSearch(self, settings):
        self.fitness_history = []

        population = self.radom_initialization(settings.population_size, settings.individual_genectic_size)
        population = self.compute_fitness_and_sort_individuals(population, settings.fitness_fuction, settings.aging)
        best_individual = population[0]
        best_individuals = []

        if (settings.perserve_best_individuals > 0):
            best_individuals = population[:settings.perserve_best_individuals]

        for generation in range(1, settings.number_of_generations):
            next_population = []

            for individual in best_individuals:
              individual.age = individual.age + 1
              next_population.append(individual)

            #print(len(next_population))

            for i in range(len(next_population), len(population)):
                parent1 = self.random_selection(population)
                parent2 = self.random_selection(population)
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                next_population.append(child)

            population = next_population

            next_population = self.compute_fitness_and_sort_individuals(population, settings.fitness_fuction, settings.aging)

            if (settings.store_best_overall_individual):
              if (best_individual.fitness < next_population[0].fitness):
                #print(generation, "change", best_individual.fitness, next_population[0].fitness)
                best_individual = next_population[0]
            else:
              best_individual = next_population[0]

            self.fitness_history.append(best_individual.fitness)

        return best_individual


def fitness_ones(individual):
    # 11111111111111111111
    return sum([x == 1 for x in individual.genetic_code])

# function generated by an AI
def plot_chart(data, labels):
    # Create a figure and a set of subplots.
    # fig: The entire figure (window)
    # ax: The axes (the actual plot area where the data is drawn)
    fig, ax = plt.subplots(figsize=(10, 6)) # figsize sets the width and height of the plot in inches

    #ax.set_ylim(0, 100) # This forces the Y-axis to start at 0 and end at 100

    for i in range(len(data)):
      # Plot the line chart
      # ax.plot(x_data, y_data) will plot y_data against x_data.
      # If you only provide one array, it assumes it's y_data and uses 0, 1, 2... for x.
      ax.plot(np.arange(1, len(data[i]) + 1), data[i],
              marker='o',          # Add markers at each data point ('o' for circle)
              linestyle='-',       # Connect points with a solid line ('-' for solid)
              #color='skyblue',     # Set line color
              label= labels[i],     # Label for the legend
              linewidth=2          # Set line width
              )

    # --- 3. Add labels, title, and legend for clarity ---

    ax.set_title('Fitness Evolution Durring Training', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add a legend if you have labels for your plots
    ax.legend(loc='upper left') # Position the legend

    # Customize ticks (optional)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # --- 4. Display the plot ---
    plt.tight_layout() # Adjusts plot parameters for a tight layout
    plt.show()         # Displays the plot


# function generated by an AI
def plot_chart_with_error(averages, error_bars, labels):
  plt.figure(figsize=(5, 5), dpi=150)
  plt.errorbar(labels, averages, yerr=error_bars, fmt='o', capsize=5)
  plt.xlabel('Settings')
  plt.ylabel('Fitness')
  plt.title('Averages with Error Bars')
  plt.grid(True)
  plt.xticks(rotation='vertical')  # Rotaciona os rótulos do eixo X
  plt.tight_layout()
  plt.show()


# training settings
fitness_fuction = fitness_ones
population_size = 100
individual_genectic_size = 100
number_of_generations = 50
mutation_rate = 0.1

test_settings = [ GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, False, 0, False),

                  # elistm
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 0, False),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 5, False),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 10, False),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 15, False),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 20, False),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 25, False),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 30, False),

                  # aging + elitism
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 5, True),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 10, True),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 15, True),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 20, True),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 25, True),
                  GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 30, True)

                  #GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, False, 0, False), #debug
                  #GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 0, False), #debug
                  #GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, True, 25, False), #debug
                  #GeneticSearchSettings(fitness_fuction, population_size, individual_genectic_size, number_of_generations, mutation_rate, False, 25, False) #debug
                  ]

number_of_executions = 100

# statistics and chart data
averages = []
errors = []
fitness_history = []
labels = ["LG", "O1", "OE5", "OE10", "OE15", "OE20", "OE25", "OE30", "AOE5", "AOE10", "AOE15", "AOE20", "AOE25", "AOE30"]

# for each setting
for settings in test_settings:

    best_individuals_of_each_test = []
    chart_data = []
    chart_fitness = []

    # for each run
    for i in range(number_of_executions):
      gs = GeneticSearch()
      individual = gs.geneticSearch(settings)
      print(">", end="")
      best_individuals_of_each_test.append(individual)
      chart_data.append(individual.fitness)

      if (chart_fitness != []):
        chart_fitness = [x + y for x, y in zip(chart_fitness, gs.fitness_history)]
      else:
        chart_fitness = gs.fitness_history
      #print(chart_fitness)

    #print(chart_data)
    average = np.average(chart_data)
    error = np.std(chart_data) / np.sqrt(len(chart_data))

    averages.append(average)
    errors.append(error)

    #print(chart_fitness)
    chart_fitness = [x / number_of_executions for x in chart_fitness]
    #print(chart_fitness)
    fitness_history.append(chart_fitness)

    print(" Average:", average, ", error:", error)

#print(fitness_history)
plot_chart(fitness_history, labels)

#print(averages, errors)
plot_chart_with_error(averages, errors, labels)