import sys
import numpy as np
import matplotlib.pyplot as plt


# ── Classes base (sem alterações) ─────────────────────────────────────────────

class Individual:
    genetic_code = ""
    fitness = sys.float_info.min

    def __init__(self, genetic_code, fitness=sys.float_info.min):
        self.genetic_code = genetic_code
        self.fitness_history = []
        self.fitness = fitness


class GeneticSearchSettings:
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
            if best_individual.fitness < individual.fitness:
                best_individual = individual
        return best_individual

    def random_selection(self, population):
        intervals = []
        total = 0
        for individual in population:
            total += max(individual.fitness, 1e-10)
            intervals.append(total)

        rng = np.random.default_rng()
        number = rng.uniform(0, total)

        for i in range(len(population)):
            if number <= intervals[i]:
                return population[i]

        print("ERROR: random selection had no return", total)

    def reproduce(self, parent1, parent2):
        rng = np.random.default_rng()
        splitting_index = rng.integers(0, len(parent1.genetic_code))
        return Individual(np.concatenate((
            parent1.genetic_code[:splitting_index],
            parent2.genetic_code[splitting_index:]
        )))

    def mutation(self, individual, mutation_rate):
        rng = np.random.default_rng()
        if rng.random() < mutation_rate:
            genetic_code = individual.genetic_code
            mutation_index = rng.integers(len(genetic_code))
            genetic_code[mutation_index] = (genetic_code[mutation_index] + 1) % 2

    def _resolve_elite_size(self, elite_size, population_size):
        if elite_size is None:
            return 1
        if isinstance(elite_size, float) and 0.0 <= elite_size <= 1.0:
            return max(1, int(population_size * elite_size))
        raise ValueError(f"elite_size must be None or a float in 0.0–1.0, got: {elite_size!r}")

    def geneticSearch(self, settings):
        n_elite = self._resolve_elite_size(settings.elite_size, settings.population_size)
        population = self.random_initialization(settings.population_size, settings.individual_genectic_size)
        best_individual = self.compute_fitness_and_find_best_individual(population, settings.fitness_function)

        for _ in range(1, settings.number_of_generations):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            next_population = population[:n_elite]

            while len(next_population) < settings.population_size:
                parent1 = self.random_selection(population)
                parent2 = self.random_selection(population)
                child = self.reproduce(parent1, parent2)
                self.mutation(child, settings.mutation_rate)
                next_population.append(child)

            population = next_population
            generation_best = self.compute_fitness_and_find_best_individual(population, settings.fitness_function)

            if settings.store_best_overall_individual:
                if best_individual.fitness < generation_best.fitness:
                    best_individual = generation_best
            else:
                best_individual = generation_best

            self.fitness_history.append(best_individual.fitness)

        return best_individual


# ── Fitness dinâmica: similaridade com o alvo fixo ────────────────────────────

def make_fitness_target(target):
    """Conta quantos genes coincidem com o alvo. Máximo = len(target)."""
    def fitness_target(individual):
        return int(sum(g == t for g, t in zip(individual.genetic_code, target)))
    return fitness_target


# ── Parâmetros base ────────────────────────────────────────────────────────────

population_size          = 100
individual_genectic_size = 100
number_of_generations    = 50
mutation_rate            = 0.1
max_executions           = 200          # limite de tentativas por variação

elite_configs = [None, 0.10, 0.20, 0.30, 0.40, 0.50]
labels        = ["Elite 1", "Elite 10%", "Elite 20%", "Elite 30%", "Elite 40%", "Elite 50%"]

# ── Gera o indivíduo perfeito UMA única vez ────────────────────────────────────

rng    = np.random.default_rng()
target = rng.integers(0, 2, individual_genectic_size)
print(f"Alvo fixo gerado: {target[:20]}...  (primeiros 20 genes)\n")

fitness_fn    = make_fitness_target(target)
perfect_score = individual_genectic_size          # fitness máximo possível

# ── Estado de cada variação ────────────────────────────────────────────────────

found_at      = [None] * len(elite_configs)       # execução em que cada variação achou o alvo
best_fitness  = [0]    * len(elite_configs)       # melhor fitness acumulado por variação

# ── Loop principal ─────────────────────────────────────────────────────────────

for execution in range(1, max_executions + 1):

    all_found = all(f is not None for f in found_at)
    if all_found:
        break

    for j, elite_size in enumerate(elite_configs):

        # Variação já encontrou o alvo — aguarda as demais
        if found_at[j] is not None:
            continue

        settings = GeneticSearchSettings(
            fitness_fn,
            population_size,
            individual_genectic_size,
            number_of_generations,
            mutation_rate,
            store_best_overall_individual=True,
            elite_size=elite_size,
        )

        gs         = GeneticSearch()
        individual = gs.geneticSearch(settings)

        if individual.fitness > best_fitness[j]:
            best_fitness[j] = individual.fitness

        # Verifica se o indivíduo perfeito foi encontrado
        if individual.fitness >= perfect_score:
            found_at[j] = execution
            print(f"  ✔ {labels[j]} encontrou o indivíduo perfeito na execução {execution}!")

    # Progresso resumido a cada 10 execuções
    if execution % 10 == 0:
        status = " | ".join(
            f"{labels[j]}: {'✔' if found_at[j] else f'melhor={best_fitness[j]}'}"
            for j in range(len(elite_configs))
        )
        print(f"[Execução {execution:03d}]  {status}")

print("\n── Resultado final ───────────────────────────────────────────────────────")
for j, label in enumerate(labels):
    result = f"encontrou na execução {found_at[j]}" if found_at[j] else f"NÃO encontrou (melhor fitness: {best_fitness[j]}/{perfect_score})"
    print(f"  {label}: {result}")


# ── Gráfico: execuções necessárias para encontrar o indivíduo perfeito ─────────

def plot_executions_to_find(found_at, best_fitness, labels, max_executions, perfect_score):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = []
    heights = []

    for j in range(len(labels)):
        if found_at[j] is not None:
            heights.append(found_at[j])
            colors.append("#4CAF50")   # verde → encontrou
        else:
            heights.append(max_executions)
            colors.append("#F44336")   # vermelho → não encontrou

    bars = ax.bar(labels, heights, color=colors, edgecolor="black", width=0.5)

    # Linha de limite máximo
    ax.axhline(max_executions, color="gray", linestyle="--", linewidth=1.2, label=f"Limite máximo ({max_executions})")

    # Anotações em cima de cada barra
    for bar, j in zip(bars, range(len(labels))):
        if found_at[j] is not None:
            label_text = f"{found_at[j]} exec."
        else:
            label_text = f"Não encontrou\n(melhor: {best_fitness[j]}/{perfect_score})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_executions * 0.01,
            label_text,
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title("Execuções necessárias para encontrar o indivíduo perfeito", fontsize=13, fontweight="bold")
    ax.set_xlabel("Configuração de Elite", fontsize=11)
    ax.set_ylabel("Número de execuções", fontsize=11)
    ax.set_ylim(0, max_executions * 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


plot_executions_to_find(found_at, best_fitness, labels, max_executions, perfect_score)