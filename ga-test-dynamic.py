from abc import ABC, abstractmethod

import sys
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    genetic_code = ""
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
    elite_size = None

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
        rng = np.random.default_rng()
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
        if elite_size is None:
            return 1

        if isinstance(elite_size, float) and 0.0 <= elite_size <= 1.0:
            return max(1, int(population_size * elite_size))

        raise ValueError(
            f"elite_size must be None or a float between 0.0 and 1.0, got: {elite_size!r}"
        )

    def geneticSearch(self, settings):
        n_elite = self._resolve_elite_size(settings.elite_size, settings.population_size)

        population = self.random_initialization(settings.population_size, settings.individual_genectic_size)
        best_individual = self.compute_fitness_and_find_best_individual(population, settings.fitness_function)

        for generation in range(1, settings.number_of_generations):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            next_population = population[:n_elite]

            while len(next_population) < settings.population_size:
                parent1 = self.random_selection(population)
                parent2 = self.random_selection(population)
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                next_population.append(child)

            population = next_population
            generation_best_individual = self.compute_fitness_and_find_best_individual(population, settings.fitness_function)

            if settings.store_best_overall_individual:
                if best_individual.fitness < generation_best_individual.fitness:
                    best_individual = generation_best_individual
            else:
                best_individual = generation_best_individual

            self.fitness_history.append(best_individual.fitness)

        return best_individual


# ── Nova fitness function: mede similaridade com um alvo aleatório ─────────────
def make_fitness_target(target):
    """
    Retorna uma fitness function que conta quantos genes
    do indivíduo coincidem com o 'target' (indivíduo perfeito).
    O score máximo possível é len(target).
    """
    def fitness_target(individual):
        return sum(g == t for g, t in zip(individual.genetic_code, target))
    return fitness_target


# function generated by an AI
def plot_chart(data):
    y_data = data
    x_data = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data, marker='o', linestyle='-', color='skyblue',
            linewidth=2, label='Fitness')
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


# ── Parâmetros base ────────────────────────────────────────────────────────────
population_size          = 100
individual_genectic_size = 200
number_of_generations    = 50
mutation_rate            = 0.1

elite_configs = [0.20, 0.25, 0.30]
labels        = ["Elite: 20%", "Elite: 25%", "Elite: 30%"]

number_of_executions = 50

# ── Statistics and chart data ──────────────────────────────────────────────────
# all_results[i] → lista de fitness dos melhores indivíduos para o settings i
all_results = [[] for _ in elite_configs]

rng = np.random.default_rng()

for i in range(number_of_executions):

    # ── Gera um indivíduo-alvo ALEATÓRIO para esta execução ────────────────────
    target = rng.integers(0, 2, individual_genectic_size)
    fitness_fn = make_fitness_target(target)

    # ── Todos os settings usam o MESMO alvo nesta execução ────────────────────
    for j, elite_size in enumerate(elite_configs):
        settings = GeneticSearchSettings(
            fitness_fn,
            population_size,
            individual_genectic_size,
            number_of_generations,
            mutation_rate,
            store_best_overall_individual=False,
            elite_size=elite_size,
        )

        gs         = GeneticSearch()
        individual = gs.geneticSearch(settings)
        all_results[j].append(individual.fitness)
        print(">", end="", flush=True)

    print(f"  [Execução {i+1:02d}]  Alvo gerado: {target[:10]}...")  # mostra os 10 primeiros genes do alvo

# ── Calcula médias e erros por configuração ────────────────────────────────────
averages = []
errors   = []

for j, label in enumerate(labels):
    data    = all_results[j]
    average = np.average(data)
    error   = np.std(data) / np.sqrt(len(data))

    averages.append(average)
    errors.append(error)

    print(f"[{label}]  Average: {average:.4f},  Error: {error:.4f}")

plot_chart_with_error(averages, errors, labels)