import numpy as np
import random
import matplotlib.pyplot as plt

# Function to generate random coordinates for cities
def generate_cities(num_cities, width=100, height=100):
    return np.random.rand(num_cities, 2) * [width, height]

# Function to calculate the distance matrix
def calculate_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(cities[i] - cities[j])
    return distance_matrix

# Define the fitness function for TSP
def fitness(route, distance_matrix):
    return 1 / np.sum([distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)] + [distance_matrix[route[-1], route[0]]])

# Crossover using MWVOSX operator
def crossover_mwvosx(parent1, parent2, distance_matrix):
    size = len(parent1)
    offspring1, offspring2 = [-1]*size, [-1]*size
    
    # Create a sub-path from parent1
    start, end = sorted(random.sample(range(size), 2))
    offspring1[start:end] = parent1[start:end]
    
    # Fill the remaining from parent2
    current_position = end
    for gene in parent2:
        if gene not in offspring1:
            if current_position >= size:
                current_position = 0
            offspring1[current_position] = gene
            current_position += 1
            
    offspring2 = offspring1[::-1]  # reverse the offspring
    
    # Calculate fitness and select the best offspring
    if fitness(offspring1, distance_matrix) > fitness(offspring2, distance_matrix):
        return offspring1
    else:
        return offspring2

# Mutation using local optimization
def mutate(route, distance_matrix):
    size = len(route)
    a, b = sorted(random.sample(range(size), 2))
    new_route = route[:a] + route[a:b][::-1] + route[b:]
    return new_route if fitness(new_route, distance_matrix) > fitness(route, distance_matrix) else route

# Selection method
def select(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

# Genetic Algorithm
def genetic_algorithm(distance_matrix, population_size=100, generations=500):
    num_cities = len(distance_matrix)
    
    # Initial population
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]
    
    for _ in range(generations):
        fitnesses = [fitness(route, distance_matrix) for route in population]
        
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select(population, fitnesses)
            child1 = crossover_mwvosx(parent1, parent2, distance_matrix)
            child2 = crossover_mwvosx(parent2, parent1, distance_matrix)
            new_population.extend([mutate(child1, distance_matrix), mutate(child2, distance_matrix)])
        
        population = new_population
    
    best_route = max(population, key=lambda x: fitness(x, distance_matrix))
    return best_route, 1 / fitness(best_route, distance_matrix)

# Plotting function
def plot_cities_and_route(cities, route):
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='red')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha='right')
    route_cities = cities[route + [route[0]]]  # append the start city to the end to close the loop
    plt.plot(route_cities[:, 0], route_cities[:, 1], 'b-')
    plt.show()

# Example usage
if __name__ == "__main__":
    num_cities = 20
    cities = generate_cities(num_cities)
    distance_matrix = calculate_distance_matrix(cities)
    
    best_route, best_distance = genetic_algorithm(distance_matrix)
    print("Best route:", best_route)
    print("Best distance:", best_distance)
    
    plot_cities_and_route(cities, best_route)
