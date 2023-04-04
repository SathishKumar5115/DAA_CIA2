import numpy as np
import matplotlib.pyplot as plt


# Define the problem parameters
desired_frequency = 5
desired_amplitude = 1

# Define the chromosome as a dictionary of parameters
chromosome = {
    "frequency": None,
    "amplitude": None,
    "phase": None,
    "offset": None
}

# Define the fitness function
def fitness(chromosome):
    t = np.linspace(0, 1, 100)
    generated_signal = chromosome["amplitude"] * np.sin(2 * np.pi * chromosome["frequency"] * t + chromosome["phase"]) + chromosome["offset"]
    desired_signal = desired_amplitude * np.sin(2 * np.pi * desired_frequency * t)
    error = np.abs(generated_signal - desired_signal)
    return np.mean(error)

# Define the genetic algorithm functions
def initialize_population(population_size):
    population = []
    for i in range(population_size):
        chromosome["frequency"] = np.random.uniform(0.1, 10)
        chromosome["amplitude"] = np.random.uniform(0.1, 2)
        chromosome["phase"] = np.random.uniform(0, 2*np.pi)
        chromosome["offset"] = np.random.uniform(-1, 1)
        population.append(chromosome.copy())
    return population

def selection(population, num_parents):
    fitness_scores = [fitness(chromosome) for chromosome in population]
    selected_indices = np.argsort(fitness_scores)[:num_parents]
    selected_parents = [population[i] for i in selected_indices]
    return selected_parents

def crossover(parents, num_offspring):
    offspring = []
    for i in range(num_offspring):
        parent1, parent2 = np.random.choice(parents, 2, replace=False)
        child = chromosome.copy()
        for key in chromosome:
            if np.random.uniform() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        offspring.append(child)
    return offspring

def mutation(offspring, mutation_rate):
    for child in offspring:
        for key in chromosome:
            if np.random.uniform() < mutation_rate:
                if key == "frequency":
                    child[key] = np.random.uniform(0.1, 10)
                elif key == "amplitude":
                    child[key] = np.random.uniform(0.1, 2)
                elif key == "phase":
                    child[key] = np.random.uniform(0, 2*np.pi)
                else:
                    child[key] = np.random.uniform(-1, 1)
    return offspring

# Define the main genetic algorithm loop
population_size = 50
num_parents = 10
num_offspring = population_size - num_parents
mutation_rate = 0.1
num_generations = 50

population = initialize_population(population_size)

for i in range(num_generations):
    parents = selection(population, num_parents)
    offspring = crossover(parents, num_offspring)
    offspring = mutation(offspring, mutation_rate)
    population = parents + offspring
    
    best_chromosome = min(population, key=fitness)
    print("Generation ", i+1, ": Fitness = ", fitness(best_chromosome))

# Generate the final signal
t = np.linspace(0, 1, 100)
best_signal = best_chromosome["amplitude"] * np.sin(2 * np.pi * best_chromosome["frequency"] * t + best_chromosome["phase"]) + best_chromosome["offset"]

# Plot the actual and predicted sine waves
t = np.linspace(0, 1, 100)
actual_signal = desired_amplitude * np.sin(2 * np.pi * desired_frequency * t)

plt.plot(t, actual_signal, label="Actual Signal")
plt.plot(t, best_signal, label="Predicted Signal")
plt.legend()
plt.show()
