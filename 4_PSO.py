import random
import math
import numpy as np
import matplotlib.pyplot as plt

def obj_function(x):
    return x * math.sin(10 * math.pi * x) + 2

def fitness(pos):
    return -abs(obj_function(pos))

def pso(num_particles, num_iterations):
    
    particles = []
    velocities = []
    for i in range(num_particles):
        pos = random.uniform(-1, 1)
        particles.append(pos)
        vel = random.uniform(-1, 1)
        velocities.append(vel)
    
    own_best_poss = particles.copy()
    own_best_fitnesses = [fitness(pos) for pos in particles]
    
    gl_best_index = own_best_fitnesses.index(max(own_best_fitnesses))
    gl_best_pos = own_best_poss[gl_best_index]
    gl_best_fitness = own_best_fitnesses[gl_best_index]
    
    for i in range(num_iterations):
        for j in range(num_particles):
            
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            vel = velocities[j] + r1 * (own_best_poss[j] - particles[j]) + r2 * (gl_best_pos - particles[j])
            vel = max(vel, -1)
            vel = min(vel, 1)
            velocities[j] = vel
            
            pos = particles[j] + vel
            pos = max(pos, -1)
            pos = min(pos, 1)
            particles[j] = pos
            
            fitness_val = fitness(pos)
            if fitness_val > own_best_fitnesses[j]:
                own_best_poss[j] = pos
                own_best_fitnesses[j] = fitness_val
            
            if fitness_val > gl_best_fitness:
                gl_best_pos = pos
                gl_best_fitness = fitness_val
                
        print("Iteration:", i + 1, "Best pos:", gl_best_pos)
    
    # Plot the function
    x = np.linspace(-1, 1, 500)
    y = [obj_function(xi) for xi in x]
    plt.plot(x, y)
    
    # Plot the best pos found by PSO
    plt.plot(gl_best_pos, obj_function(gl_best_pos), 'ro')
    
    plt.show()
    
    return gl_best_pos, gl_best_fitness

pso(20, 30)
