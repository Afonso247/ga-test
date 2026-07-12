import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Individual ────────────────────────────────────────────────────────────────

class Individual:
    def __init__(self, genetic_code, fitness=sys.float_info.min, age=0):
        self.genetic_code = genetic_code
        self.fitness_history = []
        self.fitness = fitness
        self.age = age  # the elite aging (Enhanced GA)

    def copy(self):
        clone = Individual(
            np.array(self.genetic_code, copy=True),
            self.fitness,
            self.age
        )
        clone.fitness_history = self.fitness_history.copy()
        return clone

# ── Settings ──────────────────────────────────────────────────────────────────

class GeneticSearchSettings:
    def __init__(self, fitness_function, population_size, individual_genectic_size,
                 number_of_generations, mutation_rate, store_best_overall_individual,
                 elite_size=None, age_decay=0.9):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.individual_genectic_size = individual_genectic_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.store_best_overall_individual = store_best_overall_individual
        self.elite_size = elite_size

        # Enhanced GA only: decay factor for aging.
        # Must be a float in (0.0, 1.0]:
        #   0.9  → minor penalty   (~10 % per generation)
        #   0.7  → medium penalty  (~30 % per generation)
        #   0.5  → strong penalty  (~50 % per generation)
        #   1.0  → no aging
        self.age_decay = age_decay


# ── Core search class ─────────────────────────────────────────────────────────

class GeneticSearch:

    def __init__(self):
        self.fitness_history = []
        self.rng = np.random.default_rng()

    # ── Shared helpers ────────────────────────────────────────────────────────

    def random_initialization(self, population_size, individual_genectic_size):
        return [
            Individual(self.rng.integers(0, 2, individual_genectic_size))
            for _ in range(population_size)
        ]

    def compute_fitness_and_find_best_individual(self, population, fitness_function):
        """Evaluates everyone's fitness and returns the best result. Used by conventional GA."""
        best_individual = Individual([])
        for individual in population:
            individual.fitness = fitness_function(individual)
            if best_individual.fitness < individual.fitness:
                best_individual = individual
        return best_individual

    def compute_fitness_and_sort_population(self, population, fitness_function):
        """Evaluates everyone's fitness, sorts by description, and returns the best. Used by Enhanced GA."""
        for individual in population:
            individual.fitness = fitness_function(individual)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        return population[0]

    def reproduce(self, parent1, parent2):
        """1-point crossover at random position (conventional GA)."""
        splitting_index = self.rng.integers(0, len(parent1.genetic_code))
        return Individual(
            np.concatenate((
                parent1.genetic_code[:splitting_index],
                parent2.genetic_code[splitting_index:]
            ))
        )

    def mutation(self, individual, mutation_rate):
        if self.rng.random() < mutation_rate:
            mutation_index = self.rng.integers(len(individual.genetic_code))
            individual.genetic_code[mutation_index] = (
                individual.genetic_code[mutation_index] + 1
            ) % 2

    def _resolve_elite_size(self, elite_size, population_size):
        if elite_size is None:
            return 1
        if isinstance(elite_size, int):
            if elite_size < 0:
                raise ValueError(f"elite_size int must be >= 0, got: {elite_size!r}")
            return min(population_size, elite_size)
        if isinstance(elite_size, float) and 0.0 <= elite_size <= 1.0:
            return max(1, int(population_size * elite_size))
        raise ValueError(
            f"elite_size must be None, int >= 0, or float between 0.0 and 1.0, got: {elite_size!r}"
        )

    def _update_best(self, best_individual, generation_best, store_best_overall):
        if store_best_overall:
            if best_individual.fitness < generation_best.fitness:
                return generation_best
            return best_individual
        return generation_best

    # ── Seleção por fitness proporcional (GA convencional) ────────────────────

    def random_selection(self, population):
        """Roleta proporcional ao fitness. Usada pelo GA convencional."""
        total = 0
        intervals = []
        for individual in population:
            total += max(individual.fitness, 1e-10)
            intervals.append(total)

        number = self.rng.uniform(0, total)
        for i, threshold in enumerate(intervals):
            if number <= threshold:
                return population[i]

        print("ERROR: random_selection sem retorno", total)

    # ── Seleção por ranking com aging (Enhanced GA) ───────────────────────────

    def random_selection_rank_based_with_aging(self, sorted_population, age_decay):
        """
        Roleta por ranking com penalidade de envelhecimento.

        O peso de cada indivíduo é:
            peso = rank_weight × (age_decay ^ age)

        Exemplo com age_decay=0.9:
            age=0  → fator 1.00  (sem penalidade — indivíduo novo)
            age=1  → fator 0.90
            age=5  → fator 0.59
            age=10 → fator 0.35
        """
        n = len(sorted_population)
        rank_weights = np.arange(n, 0, -1, dtype=float)
        age_penalties = np.array([age_decay ** ind.age for ind in sorted_population])

        weights = rank_weights * age_penalties
        total = weights.sum()

        if total == 0:
            # Fallback: distribuição uniforme se todos os pesos zerarem
            weights = np.ones(n, dtype=float) / n
        else:
            weights /= total

        idx = self.rng.choice(n, p=weights)
        return sorted_population[idx]

    # ── GA Convencional ───────────────────────────────────────────────────────

    def geneticSearch(self, settings):
        """
        GA convencional com elitismo e seleção proporcional ao fitness.
        Sem aging — comportamento original preservado.
        """
        self.fitness_history = []
        n_elite = self._resolve_elite_size(settings.elite_size, settings.population_size)

        population = self.random_initialization(
            settings.population_size, settings.individual_genectic_size
        )

        best_individual = self.compute_fitness_and_find_best_individual(
            population, settings.fitness_function
        )
        self.fitness_history.append(best_individual.fitness)

        for _ in range(settings.number_of_generations):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            next_population = population[:n_elite]

            while len(next_population) < settings.population_size:
                parent1 = self.random_selection(population)
                parent2 = self.random_selection(population)
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                next_population.append(child)

            population = next_population
            generation_best = self.compute_fitness_and_find_best_individual(
                population, settings.fitness_function
            )
            best_individual = self._update_best(
                best_individual, generation_best, settings.store_best_overall_individual
            )
            self.fitness_history.append(best_individual.fitness)

        return best_individual

    # ── Enhanced GA (com aging) ────────────────────────

    def enhancedGeneticSearch(self, settings):
        """
        Enhanced GA com sistema de envelhecimento.

        Diferenças em relação ao GA convencional:
        - Seleção por ranking em vez de proporcional ao fitness.
        - Elites que sobrevivem recebem age += 1 a cada geração.
        - A penalidade (age_decay ^ age) reduz gradualmente a chance de seleção
          de elites que dominam a população há muitas gerações consecutivas.
        - Novos filhos sempre nascem com age = 0.
        """
        self.fitness_history = []
        n_elite = self._resolve_elite_size(settings.elite_size, settings.population_size)

        population = self.random_initialization(
            settings.population_size, settings.individual_genectic_size
        )

        best_individual = self.compute_fitness_and_sort_population(
            population, settings.fitness_function
        )
        self.fitness_history.append(best_individual.fitness)

        for _ in range(settings.number_of_generations):
            ranked_population = sorted(
                population, key=lambda ind: ind.fitness, reverse=True
            )

            # ── Elitismo com aging ─────────────────────────────────────────────
            # Copia os melhores para a próxima geração e incrementa a idade deles.
            elites = [ind.copy() for ind in ranked_population[:n_elite]]
            for elite in elites:
                elite.age += 1

            next_population = elites

            # ── Preenchimento via seleção com penalidade de idade ──────────────
            # Indivíduos mais velhos têm menor chance de ser escolhidos como pais,
            # favorecendo material genético mais recente.
            while len(next_population) < settings.population_size:
                parent1 = self.random_selection_rank_based_with_aging(
                    ranked_population, settings.age_decay
                )
                parent2 = self.random_selection_rank_based_with_aging(
                    ranked_population, settings.age_decay
                )
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                child.age = 0  # filhos nascem sem histórico de sobrevivência
                next_population.append(child)

            population = next_population
            generation_best = self.compute_fitness_and_sort_population(
                population, settings.fitness_function
            )
            best_individual = self._update_best(
                best_individual, generation_best, settings.store_best_overall_individual
            )
            self.fitness_history.append(best_individual.fitness)

        return best_individual


# ── Funções de fitness ─────────────────────────────────────────────────────────

def fitness_ones(individual):
    return sum([x == 1 for x in individual.genetic_code])

def fitness_zeros(individual):
    return sum([x == 0 for x in individual.genetic_code])

def fitness_center_block(individual):
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

def fitness_random(genetic_size, seed=None):
    """
    Gera um alvo binário aleatório UMA ÚNICA VEZ.
    Retorna uma função fitness que avalia a semelhança com esse alvo fixo.
    O alvo é compartilhado por todas as execuções e variações de elite.
    """
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, genetic_size)
    print(f"[Alvo gerado] {target[:20]}...  (primeiros 20 bits)")

    def fitness_random_target(individual):
        return sum(g == t for g, t in zip(individual.genetic_code, target))

    return fitness_random_target


# ── Visualizações ──────────────────────────────────────────────────────────────

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


def plot_comparison_chart(standard_averages, standard_errors,
                          enhanced_averages, enhanced_errors, labels):
    x = np.arange(len(labels))

    enh_mask = [i for i, v in enumerate(enhanced_averages) if v is not None]
    x_enh    = x[enh_mask]
    enh_avg  = [enhanced_averages[i] for i in enh_mask]
    enh_err  = [enhanced_errors[i]   for i in enh_mask]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.errorbar(x, standard_averages, yerr=standard_errors,
                fmt='o', capsize=5, label='Conventional GA')
    ax.errorbar(x_enh, enh_avg, yerr=enh_err,
                fmt='s', capsize=5, label='Enhanced GA')     # ← só plota onde há dados
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('Elitism')
    ax.set_ylabel('Fitness')
    ax.set_title('Conventional GA vs Enhanced GA')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Configurações ──────────────────────────────────────────────────────────────

population_size          = 100
individual_genectic_size = 200
number_of_generations    = 200
mutation_rate            = 0.1

fit_random = fitness_random(individual_genectic_size)

#   elite_size options:
#   None  → apenas 1 indivíduo (o melhor da geração)
#   0.05  → 5 % da população
#   0.10  → 10 % da população
#   e assim em diante

elite_levels = [0, None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
labels = ["Elite: 0", "Elite: 1", "Elite: 5%", "Elite: 10%", "Elite: 15%", "Elite: 20%", "Elite: 25%", "Elite: 30%"]

standard_settings = [
    GeneticSearchSettings(
        fit_random, population_size, individual_genectic_size,
        number_of_generations, mutation_rate,
        store_best_overall_individual=False,
        elite_size=e
    )
    for e in elite_levels
]

enhanced_settings = [
    GeneticSearchSettings(
        fit_random, population_size, individual_genectic_size,
        number_of_generations, mutation_rate,
        store_best_overall_individual=False,
        elite_size=e,
        age_decay=0.9          # fator de decaimento por geração sobrevivida
    )
    for e in elite_levels
]

number_of_executions = 100

# ── Coleta de estatísticas ─────────────────────────────────────────────────────

standard_averages = []
standard_errors   = []
enhanced_averages = []
enhanced_errors   = []

for std_settings, enh_settings, label in zip(standard_settings, enhanced_settings, labels):

    std_chart_data = []
    enh_chart_data = []

    for _ in range(number_of_executions):
        gs_std = GeneticSearch()
        best_std = gs_std.geneticSearch(std_settings)
        std_chart_data.append(best_std.fitness)

        if std_settings.elite_size != 0:
            gs_enh = GeneticSearch()
            best_enh = gs_enh.enhancedGeneticSearch(enh_settings)
            enh_chart_data.append(best_enh.fitness)

        print(">", end="", flush=True)

    std_average = np.average(std_chart_data)
    std_error   = np.std(std_chart_data) / np.sqrt(len(std_chart_data))

    standard_averages.append(std_average)
    standard_errors.append(std_error)

    if std_settings.elite_size != 0:
        enh_average = np.average(enh_chart_data)
        enh_error   = np.std(enh_chart_data) / np.sqrt(len(enh_chart_data))
        enhanced_averages.append(enh_average)
        enhanced_errors.append(enh_error)
    else:
        enhanced_averages.append(None)
        enhanced_errors.append(0)

    print(f"\n[{label}]")
    print(f"  Conventional GA -> Average: {std_average:.4f}, error: {std_error:.4f}")
    if std_settings.elite_size != 0:
        print(f"  Enhanced GA     -> Average: {enh_average:.4f}, error: {enh_error:.4f}")
    else:
        print(f"  Enhanced GA     -> N/A (no elitism)")

plot_comparison_chart(
    standard_averages, standard_errors,
    enhanced_averages, enhanced_errors,
    labels
)