import numpy as np
import random

# Function to calculate the distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Function to calculate the total distance of a TSP route
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i - 1]][route[i]]
    return total_distance

# Swap mutation operator
def swap_mutation(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return route

# Best Combination Operator BC2,1O
def best_combination_operator(route, distance_matrix):
    n = len(route)
    best_distance = calculate_total_distance(route, distance_matrix)
    best_route = route.copy()

    for length in range(2, n // 2 + 1):
        for start in range(n):
            end = (start + length - 1) % n
            new_route = route.copy()
            new_route[start:end+1] = reversed(new_route[start:end+1])
            new_distance = calculate_total_distance(new_route, distance_matrix)

            if new_distance < best_distance:
                best_distance = new_distance
                best_route = new_route

    return best_route

# Genetic Algorithm with BC2,1O
def genetic_algorithm_tsp(distance_matrix, population_size=100, generations=100):
    # Generate initial population
    num_nodes = len(distance_matrix)
    population = [random.sample(range(num_nodes), num_nodes) for _ in range(population_size)]

    for generation in range(generations):
        # Calculate fitness
        fitness_scores = [(route, calculate_total_distance(route, distance_matrix)) for route in population]
        fitness_scores.sort(key=lambda x: x[1])
        population = [route for route, _ in fitness_scores[:population_size]]

        # Apply mutation and best combination operator
        new_population = []
        for route in population:
            mutated_route = swap_mutation(route.copy())
            improved_route = best_combination_operator(mutated_route, distance_matrix)
            new_population.append(improved_route)

        # Apply elitism
        population.extend(new_population)
        population = list(set(tuple(p) for p in population))  # Remove duplicates
        population.sort(key=lambda x: calculate_total_distance(x, distance_matrix))
        population = [list(p) for p in population[:population_size]]

    best_route = population[0]
    best_distance = calculate_total_distance(best_route, distance_matrix)
    return best_route, best_distance

# Example usage
if __name__ == "__main__":
    # Example TSP instance (coordinates of nodes)
    nodes = [
        (0, 10), (10, 10), (20, 10),
        (20, 0), (10, 0), (0, 0)
    ]

    # Create distance matrix
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i][j] = euclidean_distance(nodes[i], nodes[j])

    # Run the genetic algorithm
    best_route, best_distance = genetic_algorithm_tsp(distance_matrix)
    print(f"Best route: {best_route}")
    print(f"Total distance: {best_distance}")
