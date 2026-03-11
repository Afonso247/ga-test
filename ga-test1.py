from abc import ABC, abstractmethod

import sys
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    genetic_code = "" # if we handle the genetic code as chromosomes, we would not destroy good weights at random, however we would not introduce new weights during training, except for random mutations
    fitness = sys.float_info.min

    def __init__(self, genetic_code, fitness = sys.float_info.min):
        self.genetic_code = genetic_code
        self.fitness_history = []
        self.fitness = fitness

class GeneticSearchSettings:
    fitness_function = None
    population_size = -1
    individual_genectic_size = -1
    number_of_generations = -1
    mutation_rate = -1
    store_best_overall_individual = False
    elite_size = None  # None → 1 individual | float (0.0–1.0) → fraction of population

    def __init__(self, fitness_function, population_size, individual_genectic_size,
                 number_of_generations, mutation_rate, store_best_overall_individual,
                 elite_size=None):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.individual_genectic_size = individual_genectic_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.store_best_overall_individual = store_best_overall_individual
        self.elite_size = elite_size

class GeneticSearch:

    fitness_history = []

    def random_initialization(self, population_size, individual_genectic_size):
        # TODO validate the parameters

        rng = np.random.default_rng()

        # to understand this line aks an AI about list comprehension
        return [Individual(rng.integers(0, 2, individual_genectic_size)) for _ in range(population_size)]

    def compute_fitness_and_find_best_individual(self, population, fitness_function):
        best_individual = Individual([])

        for individual in population:
            individual.fitness = fitness_function(individual)

            if (best_individual.fitness < individual.fitness):
                best_individual = individual

        return best_individual

    def random_selection(self, population):
        intervals = []
        sum = 0

        for individual in population:
            sum = sum + max(individual.fitness, 1e-10)
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

    def _resolve_elite_size(self, elite_size, population_size):
        """
        Converts the elite_size setting into a concrete number of individuals.

        Rules:
            None  → 1 individual (only the best)
            0.25  → 25% of the population
            0.50  → 50% of the population
            0.75  → 75% of the population
            (any float in 0.0–1.0 is accepted)
        """
        if elite_size is None:
            return 1

        if isinstance(elite_size, float) and 0.0 <= elite_size <= 1.0:
            # Ensure at least 1 individual is always kept
            return max(1, int(population_size * elite_size))

        raise ValueError(
            f"elite_size must be None or a float between 0.0 and 1.0, got: {elite_size!r}"
        )

    def geneticSearch(self, settings):
        # Resolve the concrete elite count from the setting
        n_elite = self._resolve_elite_size(settings.elite_size, settings.population_size)

        # Initialize population randomly
        population = self.random_initialization(settings.population_size, settings.individual_genectic_size)

        # Evaluate fitness for the initial population and find the best individual
        best_individual = self.compute_fitness_and_find_best_individual(population, settings.fitness_function)

        for generation in range(1, settings.number_of_generations):

            # Sort population by fitness in descending order so the elite can be sliced easily
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Carry the best n_elite individuals directly to the next generation (elitism)
            next_population = population[:n_elite]

            # Fill the rest of the next generation with children
            while len(next_population) < settings.population_size:
                parent1 = self.random_selection(population)
                parent2 = self.random_selection(population)
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                next_population.append(child)

            population = next_population

            # Evaluate fitness for the new generation and find its best individual
            generation_best_individual = self.compute_fitness_and_find_best_individual(population, settings.fitness_function)

            # Update the best overall individual depending on the chosen strategy
            if settings.store_best_overall_individual:
                if best_individual.fitness < generation_best_individual.fitness:
                    best_individual = generation_best_individual
            else:
                best_individual = generation_best_individual

            self.fitness_history.append(best_individual.fitness)

        return best_individual


def fitness_ones(individual):
    # 11111111111111111111
    return sum([x == 1 for x in individual.genetic_code])

def fitness_zeros(individual):
    # 00000000000000000000
    return sum([x == 0 for x in individual.genetic_code])

def fitness_center_block(individual):
    # 00000011111100000000
    code = individual.genetic_code
    n = len(code)
    target = [1 if n//4 <= i < 3*n//4 else 0 for i in range(n)]
    return sum(g == t for g, t in zip(code, target))

def fitness_royal_road(individual, block_size=5):
    code = individual.genetic_code
    score = 0
    for i in range(0, len(code), block_size):
        block = code[i:i+block_size]
        if all(b == 1 for b in block):
            score += block_size
    return score

def fitness_parity(individual):
    ones = sum(individual.genetic_code)
    penalty = 0 if ones % 2 == 0 else 1
    return ones - penalty * len(individual.genetic_code)

# function generated by an AI
def plot_chart(data):
    y_data = data
    x_data = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_data, y_data,
            marker='o',
            linestyle='-',
            color='skyblue',
            linewidth=2,
            label='Fitness')

    ax.set_title('Fitness Evolution During Training', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()


# function generated by an AI
def plot_chart_with_error(averages, error_bars, labels):
  plt.figure(figsize=(5, 5), dpi=150)
  plt.errorbar(labels, averages, yerr=error_bars, fmt='o', capsize=5)
  plt.xlabel('Settings')
  plt.ylabel('Fitness')
  plt.title('Averages with Error Bars')
  plt.grid(True)
  plt.xticks(rotation='vertical')
  plt.tight_layout()
  plt.show()


# ── Training settings ──────────────────────────────────────────────────────────
fit_ones         = fitness_ones
fit_zeros        = fitness_zeros
fit_center_block = fitness_center_block
fit_royal_road   = fitness_royal_road
fit_parity       = fitness_parity
population_size          = 100
individual_genectic_size = 200
number_of_generations    = 50
mutation_rate            = 0.1

#   elite_size options:
#   None  → apenas 1 indivíduo (o melhor da geração)
#   0.25  → 25 % da população
#   0.50  → 50 % da população
#   0.75  → 75 % da população

test_settings = [
    GeneticSearchSettings(fit_ones, population_size, individual_genectic_size,
                          number_of_generations, mutation_rate,
                          store_best_overall_individual=False, elite_size=0.20),

    GeneticSearchSettings(fit_ones, population_size, individual_genectic_size,
                          number_of_generations, mutation_rate,
                          store_best_overall_individual=False, elite_size=0.25),

    GeneticSearchSettings(fit_ones, population_size, individual_genectic_size,
                          number_of_generations, mutation_rate,
                          store_best_overall_individual=False, elite_size=0.30),
]

labels = ["Elite: 20%", "Elite: 25%", "Elite: 30%"]

number_of_executions = 50

# ── Statistics and chart data ──────────────────────────────────────────────────
averages = []
errors   = []

for settings, label in zip(test_settings, labels):

    best_individuals_of_each_test = []
    chart_data = []

    for i in range(number_of_executions):
        gs         = GeneticSearch()
        individual = gs.geneticSearch(settings)
        print(">", end="", flush=True)
        best_individuals_of_each_test.append(individual)
        chart_data.append(individual.fitness)

    average = np.average(chart_data)
    error   = np.std(chart_data) / np.sqrt(len(chart_data))

    averages.append(average)
    errors.append(error)

    print(f"  [{label}]  Average: {average:.4f},  error: {error:.4f}")

plot_chart_with_error(averages, errors, labels)
