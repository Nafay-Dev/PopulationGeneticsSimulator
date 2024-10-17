import numpy as np
import matplotlib.pyplot as plt

def genetic_drift_with_selection_migration_and_mutation(
    pop1_size, pop2_size, freq_A1, freq_A2, generations, migration_rate, mutation_rate, fitness
):
    """Simulates genetic drift with migration, mutation, and natural selection."""
    pop1_history, pop2_history = [], []  # Store allele frequencies

    # Initial frequencies for both populations
    current_freq1, current_freq2 = freq_A1, freq_A2

    for gen in range(generations):
        # Apply natural selection by adjusting the allele counts based on fitness values
        current_freq1 = apply_selection(current_freq1, fitness)
        current_freq2 = apply_selection(current_freq2, fitness)

        # Simulate allele frequency using binomial distribution
        count_A1 = np.random.binomial(pop1_size, current_freq1)
        count_A2 = np.random.binomial(pop2_size, current_freq2)

        # Update frequencies
        current_freq1 = count_A1 / pop1_size
        current_freq2 = count_A2 / pop2_size

        # Apply migration between populations
        migrated_A1 = int(migration_rate * pop1_size * current_freq1)
        migrated_A2 = int(migration_rate * pop2_size * current_freq2)

        # Adjust frequencies after migration
        count_A1 = count_A1 - migrated_A1 + migrated_A2
        count_A2 = count_A2 - migrated_A2 + migrated_A1

        # Apply mutation
        count_A1 = apply_mutation(count_A1, pop1_size, mutation_rate)
        count_A2 = apply_mutation(count_A2, pop2_size, mutation_rate)

        current_freq1 = count_A1 / pop1_size
        current_freq2 = count_A2 / pop2_size

        # Store frequencies for plotting
        pop1_history.append(current_freq1)
        pop2_history.append(current_freq2)

    return pop1_history, pop2_history

def apply_selection(freq, fitness):
    """Applies natural selection by weighting allele frequencies."""
    AA, Aa, aa = fitness  # Fitness values for genotypes
    p = freq  # Frequency of allele A
    q = 1 - p  # Frequency of allele a

    # Weighted contribution of each genotype to the next generation
    w_bar = p**2 * AA + 2 * p * q * Aa + q**2 * aa  # Average fitness
    p_new = (p**2 * AA + p * q * Aa) / w_bar  # New frequency of allele A

    return p_new

def apply_mutation(count_A, pop_size, mutation_rate):
    """Applies random mutation to the population."""
    mutated_A_to_a = np.random.binomial(count_A, mutation_rate)
    mutated_a_to_A = np.random.binomial(pop_size - count_A, mutation_rate)
    return count_A - mutated_A_to_a + mutated_a_to_A

def plot_results(pop1_history, pop2_history):
    """Plots the allele frequencies for two populations over generations."""
    plt.plot(pop1_history, label='Population 1 Allele A Frequency', color='blue')
    plt.plot(pop2_history, label='Population 2 Allele A Frequency', color='green')
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.title('Genetic Drift with Selection, Migration, and Mutation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulation parameters
pop1_size = 100  # Size of population 1
pop2_size = 100  # Size of population 2
freq_A1 = 0.5    # Initial frequency of allele A in population 1
freq_A2 = 0.8    # Initial frequency of allele A in population 2
generations = 50  # Number of generations
migration_rate = 0.05  # Percentage of population migrating each generation
mutation_rate = 0.01  # Probability of mutation per allele
fitness = (1.0, 0.9, 0.8)  # Fitness values for genotypes AA, Aa, and aa

# Run the simulation
pop1_history, pop2_history = genetic_drift_with_selection_migration_and_mutation(
    pop1_size, pop2_size, freq_A1, freq_A2, generations, migration_rate, mutation_rate, fitness
)
plot_results(pop1_history, pop2_history)
