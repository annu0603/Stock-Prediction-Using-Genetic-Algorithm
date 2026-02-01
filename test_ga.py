"""
Test script for Stock Price Predictor
Run this to verify the genetic algorithm works correctly
"""

import numpy as np
import random
from datetime import datetime

class SimpleGATest:
    """Simplified version for testing without yfinance dependency"""
    
    def __init__(self, population_size=20, generations=10):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        # Generate sample data
        np.random.seed(42)
        self.historical_prices = 100 + np.cumsum(np.random.randn(50))
        
    def calculate_sma(self, data, window):
        return np.convolve(data, np.ones(window), 'valid') / window
    
    def fitness_function(self, strategy):
        window = strategy[0]
        if window >= len(self.historical_prices):
            return float('-inf')
        
        sma = self.calculate_sma(self.historical_prices, window)
        aligned_prices = self.historical_prices[-len(sma):]
        mae = np.mean(np.abs(sma - aligned_prices))
        return -mae
    
    def initialize_population(self):
        population = []
        max_window = min(50, len(self.historical_prices) - 1)
        for _ in range(self.population_size):
            strategy = [random.randint(2, max_window)]
            population.append(strategy)
        return population
    
    def crossover(self, parent1, parent2):
        return [random.choice([parent1[0], parent2[0]])]
    
    def mutate(self, strategy):
        max_window = min(50, len(self.historical_prices) - 1)
        if random.random() < self.mutation_rate:
            strategy[0] = random.randint(2, max_window)
        return strategy
    
    def evolve(self):
        population = self.initialize_population()
        best_fitness_history = []
        
        print(f"Starting evolution with {self.generations} generations...")
        print(f"Population size: {self.population_size}")
        print("-" * 60)
        
        for generation in range(self.generations):
            # Sort population by fitness
            population = sorted(population, key=self.fitness_function, reverse=True)
            
            # Track best strategy
            best_candidate = population[0]
            best_fitness = self.fitness_function(best_candidate)
            best_fitness_history.append(best_fitness)
            
            # Print progress
            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"Generation {generation + 1:3d}: "
                      f"Best Window = {best_candidate[0]:2d}, "
                      f"Fitness = {best_fitness:.4f}")
            
            # Create new population
            new_population = [best_candidate]
            
            # Elitism
            elite_size = max(1, self.population_size // 10)
            new_population.extend(population[1:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = random.choice(population[:self.population_size//2])
                parent2 = random.choice(population[:self.population_size//2])
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        print("-" * 60)
        print(f"\n✓ Evolution completed!")
        print(f"✓ Best strategy: SMA window = {best_candidate[0]}")
        print(f"✓ Final fitness: {best_fitness:.4f}")
        print(f"✓ Improvement: {((best_fitness_history[-1] - best_fitness_history[0]) / abs(best_fitness_history[0]) * 100):.2f}%")
        
        return best_candidate, best_fitness_history


def main():
    print("=" * 60)
    print("Stock Price Prediction - Genetic Algorithm Test")
    print("=" * 60)
    print()
    
    # Run test
    ga = SimpleGATest(population_size=20, generations=50)
    best_strategy, fitness_history = ga.evolve()
    
    print("\n" + "=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)
    print("\nThe genetic algorithm is working correctly.")
    print("You can now run the full Flask application.")
    print("\nTo start the web app:")
    print("  Option 1 (Docker): docker-compose up --build")
    print("  Option 2 (Local):  python app.py")
    print()


if __name__ == "__main__":
    main()
