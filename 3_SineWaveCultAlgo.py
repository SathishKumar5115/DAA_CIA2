import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
desired_frequency = 5
desired_amplitude = 1

# Define the individual as a list [frequency, amplitude, phase, bias]
def create_individual():
    return [random.uniform(0, 10), random.uniform(0, 2), random.uniform(0, 2 * math.pi), random.uniform(-1, 1)]

# Define the fitness function as the mean squared error between the actual and predicted sine waves
def fitness(individual, t):
    frequency, amplitude, phase, bias = individual
    predicted_signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) + bias
    mse = np.mean(np.square(actual_signal - predicted_signal))
    return mse

# Define the cultural algorithm
population_size = 100
max_generations = 100
learning_rate = 0.1

population = [create_individual() for i in range(population_size)]
best_individual = None

# Generate the time vector
t = np.linspace(0, 1, 100)

# Generate the actual sine wave
actual_signal = desired_amplitude * np.sin(2 * np.pi * desired_frequency * t)

for generation in range(max_generations):
    # Evaluate the fitness of each individual
    fitness_values = [fitness(individual, t) for individual in population]
    
    # Find the best individual in the population
    best_index = np.argmin(fitness_values)
    if best_individual is None or fitness(population[best_index], t) < fitness(best_individual, t):
        best_individual = population[best_index]
    
    # Update the population using the cultural algorithm
    for i in range(population_size):
        individual = population[i]
        
        # Perform mutation using learning
        new_individual = [0, 0, 0, 0]
        for j in range(4):
            if random.random() < learning_rate:
                new_individual[j] = random.uniform(0, 10)
            else:
                new_individual[j] = individual[j] + random.uniform(-0.1, 0.1)
        
        # Evaluate the fitness of the new individual
        new_fitness = fitness(new_individual, t)
        
        # Update the individual if it is better than the original
        if new_fitness < fitness_values[i]:
            population[i] = new_individual
            
    # Reduce the learning rate over time
    learning_rate *= 0.99

# Plot the actual and predicted sine waves
best_signal = best_individual[1] * np.sin(2 * np.pi * best_individual[0] * t + best_individual[2]) + best_individual[3]

plt.plot(t, actual_signal, label="Actual Signal")
plt.plot(t, best_signal, label="Predicted Signal")

plt.legend()
plt.show()
