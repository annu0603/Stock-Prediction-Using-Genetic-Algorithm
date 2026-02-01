from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import yfinance as yf
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

app = Flask(__name__)

class GeneticStockPredictor:
    def __init__(self, ticker, period='3mo', population_size=20, generations=50, mutation_rate=0.1):
        self.ticker = ticker
        self.period = period
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.historical_prices = None
        self.best_strategy_history = []
        self.fitness_history = []
        
    def fetch_stock_data(self):
        """Fetch stock data using yfinance"""
    try:
        # Add user agent to avoid blocking
        import yfinance as yf
        yf.pdr_override()
        
        stock = yf.Ticker(self.ticker)
        data = stock.history(period=self.period)
        
        # Debugging
        print(f"Attempting to fetch {self.ticker} for period {self.period}")
        print(f"Data shape: {data.shape if not data.empty else 'EMPTY'}")
        
        if data.empty:
            # Try alternative download method
            import yfinance as yf
            data = yf.download(self.ticker, period=self.period, progress=False)
            
        if data.empty or len(data) < 10:
            raise ValueError(f"No data found for ticker {self.ticker}. Please check the ticker symbol.")
            
        self.historical_prices = data['Close'].values
        self.dates = data.index
        return self.historical_prices
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise ValueError(f"Failed to fetch data for {self.ticker}: {str(e)}")
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return np.convolve(data, np.ones(window), 'valid') / window
    
    def fitness_function(self, strategy):
        """Fitness function to evaluate trading strategy"""
        window = strategy[0]
        if window >= len(self.historical_prices):
            return float('-inf')
        
        sma = self.calculate_sma(self.historical_prices, window)
        aligned_prices = self.historical_prices[-len(sma):]
        
        # Fitness based on prediction accuracy
        mae = np.mean(np.abs(sma - aligned_prices))
        return -mae
    
    def initialize_population(self):
        """Initialize population with random strategies"""
        population = []
        max_window = min(50, len(self.historical_prices) - 1)
        for _ in range(self.population_size):
            strategy = [random.randint(2, max_window)]
            population.append(strategy)
        return population
    
    def crossover(self, parent1, parent2):
        """Crossover two parent strategies"""
        if len(parent1) > 1:
            crossover_point = random.randint(0, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
        else:
            child = [random.choice([parent1[0], parent2[0]])]
        return child
    
    def mutate(self, strategy):
        """Mutate a strategy"""
        max_window = min(50, len(self.historical_prices) - 1)
        for i in range(len(strategy)):
            if random.random() < self.mutation_rate:
                strategy[i] = random.randint(2, max_window)
        return strategy
    
    def evolve(self):
        """Run the genetic algorithm"""
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Sort population by fitness
            population = sorted(population, key=self.fitness_function, reverse=True)
            
            # Track best strategy
            best_candidate = population[0]
            best_fitness = self.fitness_function(best_candidate)
            
            self.best_strategy_history.append(best_candidate[0])
            self.fitness_history.append(best_fitness)
            
            # Create new population
            new_population = [best_candidate]
            
            # Elitism: keep top 10%
            elite_size = max(1, self.population_size // 10)
            new_population.extend(population[1:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = min(random.sample(population[:self.population_size//2], 3), 
                            key=self.fitness_function, reverse=True)
                parent2 = min(random.sample(population[:self.population_size//2], 3), 
                            key=self.fitness_function, reverse=True)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best strategy
        population = sorted(population, key=self.fitness_function, reverse=True)
        return population[0]
    
    def generate_prediction_plot(self):
        """Generate prediction visualization"""
        if self.historical_prices is None:
            return None
        
        best_window = self.best_strategy_history[-1]
        sma = self.calculate_sma(self.historical_prices, best_window)
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical prices
        plt.plot(range(len(self.historical_prices)), self.historical_prices, 
                label='Actual Prices', marker='o', markersize=3, linewidth=2)
        
        # Plot SMA prediction
        sma_dates = range(len(self.historical_prices) - len(sma), len(self.historical_prices))
        plt.plot(sma_dates, sma, label=f'SMA Prediction (Window={best_window})', 
                linewidth=2, linestyle='--')
        
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title(f'{self.ticker} Stock Price Prediction using Genetic Algorithm', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def generate_fitness_plot(self):
        """Generate fitness evolution plot"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, 
                marker='o', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Negative MAE)', fontsize=12)
        plt.title('Fitness Evolution over Generations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL').upper()
        period = data.get('period', '3mo')
        generations = int(data.get('generations', 50))
        
        # Create predictor
        predictor = GeneticStockPredictor(
            ticker=ticker,
            period=period,
            population_size=20,
            generations=generations,
            mutation_rate=0.1
        )
        
        # Fetch data
        predictor.fetch_stock_data()
        
        # Run genetic algorithm
        best_strategy = predictor.evolve()
        
        # Generate plots
        prediction_plot = predictor.generate_prediction_plot()
        fitness_plot = predictor.generate_fitness_plot()
        
        # Calculate metrics
        best_window = best_strategy[0]
        sma = predictor.calculate_sma(predictor.historical_prices, best_window)
        aligned_prices = predictor.historical_prices[-len(sma):]
        mae = np.mean(np.abs(sma - aligned_prices))
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'best_window': int(best_window),
            'mae': float(mae),
            'final_fitness': float(predictor.fitness_history[-1]),
            'prediction_plot': prediction_plot,
            'fitness_plot': fitness_plot,
            'data_points': len(predictor.historical_prices)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
