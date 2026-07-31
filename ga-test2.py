import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Individual ────────────────────────────────────────────────────────────────

class Individual:
    def __init__(self, genetic_code, fitness=-np.inf, age=0):
        self.genetic_code = genetic_code
        self.fitness_history = []
        self.fitness = fitness
        self.effective_fitness = fitness
        self.age = age  # the elite aging (Enhanced GA)

    def copy(self):
        clone = Individual(
            np.array(self.genetic_code, copy=True),
            self.fitness,
            self.age
        )
        clone.effective_fitness = self.effective_fitness
        clone.fitness_history = self.fitness_history.copy()
        return clone

# ── Settings ──────────────────────────────────────────────────────────────────

class GeneticSearchSettings:
    def __init__(self, fitness_function, population_size, individual_genectic_size,
                 number_of_generations, mutation_rate, store_best_overall_individual,
                 elite_size=None, age_decay=0.9,
                 selection="roulette", tournament_size=3):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.individual_genectic_size = individual_genectic_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.store_best_overall_individual = store_best_overall_individual
        self.elite_size = elite_size

        # Método de seleção de pais:
        #   "roulette"   → proporcional ao fitness (comportamento original)
        #   "tournament" → torneio de `tournament_size` indivíduos; invariante à
        #                  escala do fitness
        if selection not in ("roulette", "tournament"):
            raise ValueError(
                f"selection must be 'roulette' or 'tournament', got: {selection!r}"
            )
        self.selection = selection

        if not isinstance(tournament_size, int) or tournament_size < 1:
            raise ValueError(
                f"tournament_size must be an int >= 1, got: {tournament_size!r}"
            )
        self.tournament_size = tournament_size

        # Enhanced GA only: decay factor for aging.
        # Must be a float in (0.0, 1.0]:
        #   0.9  → minor penalty   (~10 % per generation)
        #   0.7  → medium penalty  (~30 % per generation)
        #   0.5  → strong penalty  (~50 % per generation)
        #   1.0  → no aging
        if not isinstance(age_decay, (int, float)) or not (0.0 < age_decay <= 1.0):
            raise ValueError(
                f"age_decay must be a float in (0.0, 1.0], got: {age_decay!r}"
            )
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
        best_individual = None
        for individual in population:
            individual.fitness = fitness_function(individual)
            if best_individual is None or best_individual.fitness < individual.fitness:
                best_individual = individual
        return best_individual

    def compute_fitness_and_sort_population(self, population, fitness_function, age_decay=1.0):
        """
        Used by Enhanced GA.

        Evaluates RAW fitness for everyone (fitness_function is never told about age),
        then derives an `effective_fitness = raw_fitness * age_decay ** age` used only
        to rank/sort the population for elitism and selection purposes.

        Returns the true best individual of the generation by RAW fitness — this is
        what guarantees an aging elite can never "hide" the fact that it stopped being
        the actual best, and it's also what `_update_best` should compare against, so
        the historical best individual can never be lost just because it grew old.
        """
        raw_best = None
        for individual in population:
            individual.fitness = fitness_function(individual)
            individual.effective_fitness = individual.fitness * (age_decay ** individual.age)
            if raw_best is None or raw_best.fitness < individual.fitness:
                raw_best = individual
        population.sort(key=lambda ind: ind.effective_fitness, reverse=True)
        return raw_best

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
        """
        Bit-flip: cada gene tem probabilidade independente
        `mutation_rate` de ser invertido (0↔1).
        """
        genetic_code = individual.genetic_code
        flip_mask = self.rng.random(len(genetic_code)) < mutation_rate
        genetic_code[flip_mask] = 1 - genetic_code[flip_mask]

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

    # ── Seleção proporcional ao fitness JÁ penalizado por idade (Enhanced GA) ─

    def random_selection_by_effective_fitness(self, population):
        """
        Roleta proporcional ao `effective_fitness` (fitness bruto já multiplicado
        pela penalidade de idade). Como a penalidade já está embutida no fitness
        usado aqui — e também no sort feito em `compute_fitness_and_sort_population`
        — um indivíduo antigo perde peso tanto na seleção quanto no ranking de
        elitismo, deixando de ocupar uma vaga de elite indefinidamente.
        """
        total = 0
        intervals = []
        for individual in population:
            total += max(individual.effective_fitness, 1e-10)
            intervals.append(total)

        number = self.rng.uniform(0, total)
        for i, threshold in enumerate(intervals):
            if number <= threshold:
                return population[i]

        print("ERROR: random_selection_by_effective_fitness sem retorno", total)

    # ── Seleção por torneio ───────────────────────────────────────────────────

    def tournament_selection(self, population, tournament_size, use_effective=False):
        """
        Torneio: sorteia `tournament_size` indivíduos ao acaso e retorna o de maior
        fitness. Invariante à escala do fitness — funciona bem com valores negativos
        (ex.: fitness_parity) e mantém pressão seletiva mesmo quando a população
        converge, dois pontos onde a roleta falha.

        Com use_effective=True, compara pelo `effective_fitness` (penalizado por
        idade), para ser usado pelo Enhanced GA.
        """
        indices = self.rng.integers(0, len(population), tournament_size)
        best = population[indices[0]]
        for i in indices[1:]:
            challenger = population[i]
            if use_effective:
                if challenger.effective_fitness > best.effective_fitness:
                    best = challenger
            elif challenger.fitness > best.fitness:
                best = challenger
        return best

    # ── Despachantes: escolhem o método conforme settings.selection ───────────

    def _select_parent(self, population, settings):
        """Seleção de pai do GA convencional (por fitness bruto)."""
        if settings.selection == "tournament":
            return self.tournament_selection(population, settings.tournament_size)
        return self.random_selection(population)

    def _select_parent_effective(self, population, settings):
        """Seleção de pai do Enhanced GA (por effective_fitness, penalizado por idade)."""
        if settings.selection == "tournament":
            return self.tournament_selection(
                population, settings.tournament_size, use_effective=True
            )
        return self.random_selection_by_effective_fitness(population)

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
            next_population = [ind.copy() for ind in population[:n_elite]]

            while len(next_population) < settings.population_size:
                parent1 = self._select_parent(population, settings)
                parent2 = self._select_parent(population, settings)
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
        - A penalidade de idade (age_decay ^ age) é aplicada sobre o fitness bruto,
          gerando um `effective_fitness` usado tanto para ordenar a população
          (elitismo) quanto para a seleção (`random_selection_by_effective_fitness`).
          Assim, um elite que envelhece perde peso nos dois lugares, e não apenas na
          reprodução — o que evita que ele fique "congelado" no topo do ranking.
        - Elites que sobrevivem recebem age += 1 a cada geração.
        - Novos filhos sempre nascem com age = 0.
        - O melhor indivíduo geral (`store_best_overall_individual`) é sempre
          comparado pelo fitness BRUTO (`.fitness`), nunca pelo `effective_fitness`,
          então ele nunca é descartado só por ter envelhecido.
        """
        self.fitness_history = []
        n_elite = self._resolve_elite_size(settings.elite_size, settings.population_size)

        population = self.random_initialization(
            settings.population_size, settings.individual_genectic_size
        )

        # população inicial: todos com age=0, então effective_fitness == fitness aqui
        best_individual = self.compute_fitness_and_sort_population(
            population, settings.fitness_function, settings.age_decay
        )
        self.fitness_history.append(best_individual.fitness)

        for _ in range(settings.number_of_generations):
            elites = [ind.copy() for ind in population[:n_elite]]
            for elite in elites:
                elite.age += 1

            next_population = elites

            while len(next_population) < settings.population_size:
                parent1 = self._select_parent_effective(population, settings)
                parent2 = self._select_parent_effective(population, settings)
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                child.age = 0
                next_population.append(child)

            population = next_population
            generation_best = self.compute_fitness_and_sort_population(
                population, settings.fitness_function, settings.age_decay
            )
            best_individual = self._update_best(
                best_individual, generation_best, settings.store_best_overall_individual
            )
            self.fitness_history.append(best_individual.fitness)

        return best_individual


# ── Funções de fitness (vetorizadas com NumPy) ─────────────────────────────────

def fitness_ones(individual):
    return int(np.sum(individual.genetic_code == 1))

def fitness_zeros(individual):
    return int(np.sum(individual.genetic_code == 0))

def fitness_center_block(individual):
    code = individual.genetic_code
    n = len(code)
    target = np.array([1 if n // 4 <= i < 3 * n // 4 else 0 for i in range(n)])
    return int(np.sum(code == target))

def fitness_royal_road(individual, block_size=5):
    code = individual.genetic_code
    n = len(code)
    usable_len = (n // block_size) * block_size
    blocks = code[:usable_len].reshape(-1, block_size)
    full_blocks = np.all(blocks == 1, axis=1)
    return int(np.sum(full_blocks) * block_size)

def fitness_plateau_ones(individual, step=10):
    """
    OneMax "em degraus": conta os 1s, mas arredonda para baixo em múltiplos de
    `step`. Cria platôs de largura `step` — dentro do platô, ganhar mais 1s não
    muda o fitness (sem gradiente), então o convencional estagna. Só um salto de
    `step` 1s de uma vez sobe o degrau; o aging sobe os degraus com mais frequência.
    """
    ones = int(np.sum(individual.genetic_code))
    return int((ones // step) * step)

def fitness_parity(individual):
    ones = int(np.sum(individual.genetic_code))
    penalty = 0 if ones % 2 == 0 else 1
    return ones - penalty * len(individual.genetic_code)

# ── Visualizações ──────────────────────────────────────────────────────────────

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


def plot_fitness_evolution(standard_histories, enhanced_histories, labels):

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, label in enumerate(labels):
        color = colors[i % len(colors)]

        std_curve = standard_histories[i]
        ax.plot(np.arange(len(std_curve)), std_curve,
                linestyle='-', color=color, linewidth=2,
                label=f"Conventional {label}")

        enh_curve = enhanced_histories[i]
        if enh_curve is not None:                       # Elite: 0 não tem Enhanced
            ax.plot(np.arange(len(enh_curve)), enh_curve,
                    linestyle='--', color=color, linewidth=2,
                    label=f"Enhanced {label}")

    ax.set_title('Fitness Evolution During Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.show()


# ── Configurações ──────────────────────────────────────────────────────────────

population_size          = 50
individual_genectic_size = 100
number_of_generations    = 100
mutation_rate            = 0.02

#   elite_size options:
#   None  → apenas 1 indivíduo (o melhor da geração)
#   0.05  → 5 % da população
#   0.10  → 10 % da população
#   e assim em diante

elite_levels = [0, None, 0.10, 0.20]
labels = ["Elite: 0", "Elite: 1", "Elite: 10%", "Elite: 20%"]

#   selection_method options:
#   "roulette"   → seleção por roleta (comportamento original)
#   "tournament" → seleção por torneio
selection_method = "tournament"
tournament_size  = 3          # nº de competidores por torneio (usado só em tournament)

standard_settings = [
    GeneticSearchSettings(
        fitness_plateau_ones, population_size, individual_genectic_size,
        number_of_generations, mutation_rate,
        store_best_overall_individual=False,
        elite_size=e,
        selection=selection_method,
        tournament_size=tournament_size
    )
    for e in elite_levels
]

enhanced_settings = [
    GeneticSearchSettings(
        fitness_plateau_ones, population_size, individual_genectic_size,
        number_of_generations, mutation_rate,
        store_best_overall_individual=True,   # garante que o melhor indivíduo
                                                # (por fitness bruto) nunca seja
                                                # perdido por conta do envelhecimento
        elite_size=e,
        age_decay=0.9,
        selection=selection_method,
        tournament_size=tournament_size
    )
    for e in elite_levels
]

number_of_executions = 100

# ── Coleta de estatísticas ─────────────────────────────────────────────────────

standard_averages = []
standard_errors   = []
enhanced_averages = []
enhanced_errors   = []
standard_histories = []
enhanced_histories = []

for std_settings, enh_settings, label in zip(standard_settings, enhanced_settings, labels):

    std_chart_data = []
    enh_chart_data = []

    std_history_sum = None
    enh_history_sum = None

    for _ in range(number_of_executions):
        gs_std = GeneticSearch()
        best_std = gs_std.geneticSearch(std_settings)
        std_chart_data.append(best_std.fitness)

        if std_history_sum is None:
            std_history_sum = list(gs_std.fitness_history)
        else:
            std_history_sum = [a + b for a, b in zip(std_history_sum, gs_std.fitness_history)]

        if std_settings.elite_size != 0:
            gs_enh = GeneticSearch()
            best_enh = gs_enh.enhancedGeneticSearch(enh_settings)
            enh_chart_data.append(best_enh.fitness)

            if enh_history_sum is None:
                enh_history_sum = list(gs_enh.fitness_history)
            else:
                enh_history_sum = [a + b for a, b in zip(enh_history_sum, gs_enh.fitness_history)]

        print(">", end="", flush=True)

    std_average = np.average(std_chart_data)
    std_error   = np.std(std_chart_data) / np.sqrt(len(std_chart_data))

    standard_averages.append(std_average)
    standard_errors.append(std_error)
    standard_histories.append([v / number_of_executions for v in std_history_sum])

    if std_settings.elite_size != 0:
        enh_average = np.average(enh_chart_data)
        enh_error   = np.std(enh_chart_data) / np.sqrt(len(enh_chart_data))
        enhanced_averages.append(enh_average)
        enhanced_errors.append(enh_error)
        enhanced_histories.append([v / number_of_executions for v in enh_history_sum])
    else:
        enhanced_averages.append(None)
        enhanced_errors.append(0)
        enhanced_histories.append(None)

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

plot_fitness_evolution(standard_histories, enhanced_histories, labels)