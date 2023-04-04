import numpy as np
import matplotlib.pyplot as plt


def fit(sol, dists):
    dist_t = 0
    for i in range(len(sol)-1):
        dist_t += dists[sol[i], sol[i+1]]
    return 1 / dist_t


def crossover(parent1, parent2):

    p1 = np.random.randint(0, len(parent1))
    p2 = np.random.randint(p1, len(parent1))


    ch = [-1] * len(parent1)
    ch[p1:p2+1] = parent1[p1:p2+1]
    rem = [gene for gene in parent2 if gene not in ch]
    ch[:p1] = rem[:p1]
    ch[p2+1:] = rem[p1:]

    return ch

def mutate(sol):
    c1, c2 = np.random.choice(len(sol), 2, replace=False)
    sol[c1], sol[c2] = sol[c2], sol[c1]

    return sol

def genetic_algorithm(dists, pop_size=100, gens=1000):
    num_cities = dists.shape[0]
    pop = [list(np.random.permutation(num_cities)) for _ in range(pop_size)]

    best_fits = []
    best_sols = []

    for gen in range(gens):
        fit_val = [fit(sol, dists) for sol in pop]

        parent_indices = np.random.choice(pop_size, size=pop_size, replace=True, p=fit_val/np.sum(fit_val))
        parents = [pop[i] for i in parent_indices]

        new_pop = []
        for i in range(0, pop_size, 2):
            ch1 = crossover(parents[i], parents[i+1])
            ch2 = crossover(parents[i+1], parents[i])
            new_pop.append(mutate(ch1))
            new_pop.append(mutate(ch2))

        pop = new_pop

        best_sol = max(pop, key=lambda sol: fit(sol, dists))
        best_fit = fit(best_sol, dists)
        best_fits.append(best_fit)
        best_sols.append(best_sol)

    best_sol = max(pop, key=lambda sol: fit(sol, dists))
    best_fit = fit(best_sol, dists)

    return best_sol, best_fit, best_sols, best_fit

num_cities = 10
cities = np.random.rand(num_cities, 2)
dists = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i, num_cities):
        dist = np.linalg.norm(cities[i] - cities[j])
        dists[i, j] = dist
        dists[j, i] = dist

best_sol, best_fit, best_sols, best_fit = genetic_algorithm(dists, pop_size=100, gens=1000)

plt.subplot(1, 2, 1)
plt.plot(cities[:, 0], cities[:, 1], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cities')

plt.subplot(1, 2, 2)
best_route = np.array([cities[i] for i in best_sol])
plt.plot(best_route[:, 0], best_route[:, 1], 'o-')
plt.plot([best_route[-1, 0], best_route[0, 0]], [best_route[-1, 1], best_route[0, 1]], 'o-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best Route')

plt.tight_layout()
plt.show()