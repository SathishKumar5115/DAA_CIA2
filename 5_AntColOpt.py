import numpy as np
import random

def ant_colony_optimization(num_ants, num_iterations, eva_rate, alpha, beta, Q, dists):
    num_cities = dists.shape[0]
    phermone = np.ones((num_cities, num_cities)) / num_cities
    best_path = []
    best_dist = np.inf
    
    for iteration in range(num_iterations):
        paths = []
        dists_travelled = []
        
        for ant in range(num_ants):
            cc = random.randint(0, num_cities-1)
            path = [cc]
            dist_travelled = 0
            
            for step in range(num_cities-1):
                possible_next_cities = [city for city in range(num_cities) if city not in path]
                prob = np.zeros(num_cities)
                
                for nc in possible_next_cities:
                    prob[nc] = (phermone[cc][nc] ** alpha) * ((1/dists[cc][nc]) ** beta)
                    
                prob = prob / np.sum(prob)
                nc = np.random.choice(range(num_cities), p=prob)
                
                dist_travelled += dists[cc][nc]
                cc = nc
                path.append(cc)
                
            paths.append(path)
            dists_travelled.append(dist_travelled)
            
            if dist_travelled < best_dist:
                best_path = path
                best_dist = dist_travelled
                
        phermone *= eva_rate
        
        for i, path in enumerate(paths):
            for j in range(num_cities-1):
                city_a, city_b = path[j], path[j+1]
                phermone[city_a][city_b] += Q/dists_travelled[i]
                phermone[city_b][city_a] += Q/dists_travelled[i]
                
    return best_path, best_dist

# Example problem
dists = np.array([[0, 1, 9, 8, 6],
                      [2, 10, 6, 2, 8],
                      [9, 5, 7, 3, 9],
                      [1, 4, 3, 6, 4],
                      [10, 2, 9, 6, 10]])

# Parameters
num_ants = 10
num_iterations = 100
eva_rate = 0.5
alpha = 1
beta = 2
Q = 100

# Run the algorithm
best_path, best_dist = ant_colony_optimization(num_ants, num_iterations, eva_rate, alpha, beta, Q, dists)

# Print the results
print("Best path:", best_path)
print("Best dist:", best_dist)
