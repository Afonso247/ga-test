import numpy as np
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, genetic_code, fitness=-np.inf, age=0):
        self.genetic_code = np.array(genetic_code, dtype=int, copy=True)
        self.fitness_history = []
        self.fitness = fitness
        self.age = age  # Número de gerações consecutivas que este indivíduo sobreviveu como elite

    def copy(self):
        clone = Individual(self.genetic_code.copy(), self.fitness, self.age)
        clone.fitness_history = self.fitness_history.copy()
        return clone


class GeneticSearchSettings:
    def __init__(self, fitness_function, population_size, individual_genectic_size,
                 number_of_generations, mutation_rate, store_best_overall_individual,
                 elite_size=None, enhancement_top_k=20, age_decay=0.9):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.individual_genectic_size = individual_genectic_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.store_best_overall_individual = store_best_overall_individual
        self.elite_size = elite_size
        self.enhancement_top_k = enhancement_top_k

        # Fator de decaimento por geração de envelhecimento.
        # Deve estar em (0.0, 1.0]:
        #   0.9  → penalidade leve   (peso cai ~10 % por geração sobrevivida)
        #   0.5  → penalidade forte  (peso cai 50 % por geração sobrevivida)
        #   1.0  → sem envelhecimento (padrão)
        self.age_decay = age_decay


class GeneticSearch:
    def __init__(self):
        self.fitness_history = []
        self.rng = np.random.default_rng()

    def random_initialization(self, population_size, individual_genectic_size):
        return [
            Individual(self.rng.integers(0, 2, individual_genectic_size))
            for _ in range(population_size)
        ]

    def compute_fitness_and_sort_population(self, population, fitness_function):
        for individual in population:
            individual.fitness = fitness_function(individual)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        return population[0]

    def random_selection_rank_based(self, sorted_population):
        """
        Seleção por roleta baseada em ranking (sem aging).
        Usada pelo GA convencional.
        """
        n = len(sorted_population)
        weights = np.arange(n, 0, -1, dtype=float)
        weights /= weights.sum()
        idx = self.rng.choice(n, p=weights)
        return sorted_population[idx]

    def random_selection_rank_based_with_aging(self, sorted_population, age_decay):
        """
        Seleção por roleta baseada em ranking com penalidade de envelhecimento.
        O peso de cada indivíduo é multiplicado por (age_decay ^ age), reduzindo
        gradualmente a probabilidade de seleção de indivíduos que já sobreviveram
        muitas gerações consecutivas como elite.

        Exemplo com age_decay=0.9:
          age=0  → fator 1.00  (indivíduo novo, sem penalidade)
          age=1  → fator 0.90
          age=5  → fator 0.59
          age=10 → fator 0.35
        """
        n = len(sorted_population)
        rank_weights = np.arange(n, 0, -1, dtype=float)
        age_penalties = np.array([age_decay ** ind.age for ind in sorted_population])

        weights = rank_weights * age_penalties
        total = weights.sum()

        # Fallback para caso todos os pesos zerarem
        if total == 0:
            weights = np.ones(n, dtype=float) / n
        else:
            weights /= total

        idx = self.rng.choice(n, p=weights)
        return sorted_population[idx]

    def standard_crossover(self, parent1, parent2):
        if len(parent1.genetic_code) <= 1:
            return Individual(parent1.genetic_code.copy())

        splitting_index = self.rng.integers(1, len(parent1.genetic_code))
        child_code = np.concatenate((
            parent1.genetic_code[:splitting_index],
            parent2.genetic_code[splitting_index:]
        ))
        return Individual(child_code)

    def enhancement_crossover(self, parent1, parent2):
        """
        Crossover de Enhancement Phase.
        corte fixo na metade do cromossomo.
        """
        n = len(parent1.genetic_code)
        cut = n // 2

        if cut == 0 or cut == n:
            return self.standard_crossover(parent1, parent2)

        child_code = np.concatenate((
            parent1.genetic_code[:cut],
            parent2.genetic_code[cut:]
        ))
        return Individual(child_code)

    def mutation(self, individual, mutation_rate):
        if self.rng.random() < mutation_rate:
            mutation_index = self.rng.integers(len(individual.genetic_code))
            individual.genetic_code[mutation_index] = 1 - individual.genetic_code[mutation_index]

    def _resolve_elite_size(self, elite_size, population_size):
        if elite_size is None:
            return 1

        if isinstance(elite_size, int):
            if elite_size < 1:
                raise ValueError(f"elite_size inteiro deve ser >= 1, got: {elite_size!r}")
            return min(population_size, elite_size)

        if isinstance(elite_size, float) and 0.0 <= elite_size <= 1.0:
            return max(1, int(round(population_size * elite_size)))

        raise ValueError(
            f"elite_size must be None, int >= 1, or float between 0.0 and 1.0, got: {elite_size!r}"
        )

    def _update_best(self, best_individual, generation_best_individual, store_best_overall_individual):
        if store_best_overall_individual:
            if best_individual.fitness < generation_best_individual.fitness:
                return generation_best_individual.copy()
            return best_individual
        return generation_best_individual.copy()

    def _standard_next_generation(self, population, settings, elite_count):
        next_population = [ind.copy() for ind in population[:elite_count]]

        while len(next_population) < settings.population_size:
            parent1 = self.random_selection_rank_based(population)
            parent2 = self.random_selection_rank_based(population)
            child = self.standard_crossover(parent1, parent2)
            self.mutation(child, settings.mutation_rate)
            next_population.append(child)

        return next_population

    def _enhancement_phase(self, ranked_population, settings):
        """
        Fase de aprimoramento: seleciona os top_k melhores e gera filhos entre eles.
        A seleção interna também considera o envelhecimento, de modo que pais
        antigos dentro do grupo top_k têm menor chance de ser escolhidos.
        Os filhos gerados aqui nascem com age=0.
        """
        top_k = min(settings.enhancement_top_k, len(ranked_population))
        top_group = ranked_population[:top_k]

        enhanced_population = []
        while len(enhanced_population) < top_k:
            parent1 = self.random_selection_rank_based_with_aging(top_group, settings.age_decay)
            parent2 = self.random_selection_rank_based_with_aging(top_group, settings.age_decay)
            child = self.enhancement_crossover(parent1, parent2)
            self.mutation(child, settings.mutation_rate)
            child.age = 0  # filhos nascem sem histórico de sobrevivência
            enhanced_population.append(child)

        self.compute_fitness_and_sort_population(enhanced_population, settings.fitness_function)
        return enhanced_population

    def _modified_elitism(self, normal_population, enhanced_population, elite_count):
        """
        Os melhores indivíduos da população normal e da fase aprimorada competem.
        Apenas os top elite_count sobrevivem.
        A idade é preservada nos indivíduos copiados; o incremento ocorre
        no loop principal de enhancedGeneticSearch após esta etapa.
        """
        candidates = [ind.copy() for ind in normal_population[:elite_count]]
        candidates.extend(ind.copy() for ind in enhanced_population[:elite_count])

        candidates.sort(key=lambda ind: ind.fitness, reverse=True)
        return [ind.copy() for ind in candidates[:elite_count]]

    def geneticSearch(self, settings):
        """
        AG Convencional.
        """
        self.fitness_history = []
        elite_count = self._resolve_elite_size(settings.elite_size, settings.population_size)

        population = self.random_initialization(
            settings.population_size,
            settings.individual_genectic_size
        )

        best_individual = self.compute_fitness_and_sort_population(population, settings.fitness_function)
        self.fitness_history.append(best_individual.fitness)

        for _ in range(settings.number_of_generations):
            population = self._standard_next_generation(population, settings, elite_count)
            generation_best_individual = self.compute_fitness_and_sort_population(population, settings.fitness_function)
            best_individual = self._update_best(
                best_individual,
                generation_best_individual,
                settings.store_best_overall_individual
            )
            self.fitness_history.append(best_individual.fitness)

        return best_individual

    def enhancedGeneticSearch(self, settings):
        """
        Enhanced GA com fase de aprimoramento + elitismo modificado + envelhecimento.

        Sistema de envelhecimento:
        - Todo indivíduo nasce com age=0.
        - A cada geração em que um elite sobrevive, seu age é incrementado em 1.
        - Durante a seleção de pais, o peso de cada indivíduo é multiplicado por
          (age_decay ^ age), reduzindo gradualmente suas chances de ser selecionado.
        - Isso evita que elites antigos dominem a reprodução, forçando renovação
          genética mesmo entre os melhores indivíduos.
        """
        self.fitness_history = []
        elite_count = self._resolve_elite_size(settings.elite_size, settings.population_size)

        population = self.random_initialization(
            settings.population_size,
            settings.individual_genectic_size
        )

        best_individual = self.compute_fitness_and_sort_population(population, settings.fitness_function)
        self.fitness_history.append(best_individual.fitness)

        for _ in range(settings.number_of_generations):
            ranked_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

            # Enhancement Phase
            enhanced_population = self._enhancement_phase(ranked_population, settings)

            # Modified Elitism
            elites = self._modified_elitism(ranked_population, enhanced_population, elite_count)

            # Elite age increment
            # The age of elites is preserved in the next generation
            for elite in elites:
                elite.age += 1

            next_population = list(elites)

            # Filling Next Generation as Standard
            # A seleção de pais considera o envelhecimento: indivíduos mais velhos
            # têm menor probabilidade de ser escolhidos como pais.
            while len(next_population) < settings.population_size:
                parent1 = self.random_selection_rank_based_with_aging(ranked_population, settings.age_decay)
                parent2 = self.random_selection_rank_based_with_aging(ranked_population, settings.age_decay)
                child = self.standard_crossover(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                child.age = 0
                next_population.append(child)

            population = next_population
            generation_best_individual = self.compute_fitness_and_sort_population(
                population, settings.fitness_function
            )
            best_individual = self._update_best(
                best_individual,
                generation_best_individual,
                settings.store_best_overall_individual
            )
            self.fitness_history.append(best_individual.fitness)

        return best_individual


# ── Fitness functions ─────────────────────────────────────────────────────────

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
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, genetic_size)
    print(f"[Alvo gerado] {target[:20]}...  (primeiros 20 bits)")

    def fitness_random_target(individual):
        return sum(g == t for g, t in zip(individual.genetic_code, target))

    return fitness_random_target

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
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.errorbar(x - width/2, standard_averages, yerr=standard_errors,
                fmt='o', capsize=5, label='Conventional GA')
    ax.errorbar(x + width/2, enhanced_averages, yerr=enhanced_errors,
                fmt='o', capsize=5, label='Enhanced GA')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('Elitism')
    ax.set_ylabel('Fitness')
    ax.set_title('Conventional GA vs Enhanced GA')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Configs ──────────────────────────────────────────────────────────────

population_size          = 50
individual_genectic_size = 200
number_of_generations    = 100
mutation_rate            = 0.1

fit_random = fitness_random(individual_genectic_size)

elite_levels = [4, 8, 12]
labels = [f"Elite: {e}" for e in elite_levels]

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
        enhancement_top_k=20,
        age_decay=0.7         # Decay factor per generation
    )
    for e in elite_levels
]

number_of_executions = 50

# ── Display ─────────────────────────────────────────────────────

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

        gs_enh = GeneticSearch()
        best_enh = gs_enh.enhancedGeneticSearch(enh_settings)
        enh_chart_data.append(best_enh.fitness)

        print(">", end="", flush=True)

    std_average = np.average(std_chart_data)
    std_error   = np.std(std_chart_data) / np.sqrt(len(std_chart_data))

    enh_average = np.average(enh_chart_data)
    enh_error   = np.std(enh_chart_data) / np.sqrt(len(enh_chart_data))

    standard_averages.append(std_average)
    standard_errors.append(std_error)
    enhanced_averages.append(enh_average)
    enhanced_errors.append(enh_error)

    print(f"\n[{label}]")
    print(f"  Conventional GA -> Average: {std_average:.4f}, error: {std_error:.4f}")
    print(f"  Enhanced GA     -> Average: {enh_average:.4f}, error: {enh_error:.4f}")

plot_comparison_chart(
    standard_averages, standard_errors,
    enhanced_averages, enhanced_errors,
    labels
)
