# Stock Price Prediction using Genetic Algorithm

A Dockerized Python web application that leverages genetic algorithms to predict stock prices with interactive visualization. The system evolves trading strategies over multiple generations to improve prediction accuracy.

## 🚀 Features

- **Real-time Stock Data**: Fetches live stock data using yfinance API
- **Genetic Algorithm**: Evolves trading strategies over 50 generations
- **Interactive Web Interface**: User-friendly Flask web application
- **Docker Support**: Fully containerized for easy deployment
- **Visual Analytics**: Interactive charts showing predictions and fitness evolution
- **Multiple Timeframes**: Support for various historical periods (1mo, 3mo, 6mo, 1y, 2y)
- **Performance Metrics**: Real-time display of prediction accuracy and strategy performance

## 🛠️ Technologies Used

- **Python 3.11**: Core programming language
- **Flask**: Web framework for the application
- **yfinance**: Stock market data retrieval
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Docker**: Containerization
- **Genetic Algorithm**: Optimization technique for strategy evolution

## 📋 Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.11+ (for local development)

## 🔧 Installation & Setup

### Option 1: Using Docker (Recommended)

1. **Clone or download the project files**

2. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

3. **Access the application:**
Open your browser and navigate to `http://localhost:5000`

### Option 2: Local Development

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

4. **Access the application:**
Open your browser and navigate to `http://localhost:5000`

## 📊 How It Works

### Genetic Algorithm Overview

1. **Initialization**: Creates a population of random trading strategies (SMA window sizes)
2. **Fitness Evaluation**: Each strategy is evaluated based on prediction accuracy
3. **Selection**: Best-performing strategies are selected for reproduction
4. **Crossover**: Parent strategies combine to create offspring
5. **Mutation**: Random changes introduce diversity
6. **Evolution**: Process repeats for specified generations

### Trading Strategy

The algorithm optimizes the Simple Moving Average (SMA) window size to minimize prediction error:
- **Input**: Historical stock prices
- **Strategy**: SMA window size (2-50 days)
- **Fitness**: Negative Mean Absolute Error (MAE)
- **Output**: Best SMA window for price prediction

## 🎯 Usage

1. **Enter Stock Ticker**: Input any valid stock symbol (e.g., AAPL, GOOGL, TSLA)
2. **Select Time Period**: Choose historical data range
3. **Set Generations**: Define number of evolution cycles (10-200)
4. **Run Prediction**: Click the button to start the genetic algorithm
5. **View Results**: Analyze predictions and fitness evolution charts

## 📁 Project Structure

```
.
├── app.py                  # Main Flask application
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .dockerignore        # Docker ignore patterns
└── README.md            # This file
```

## 🐳 Docker Commands

```bash
# Build the image
docker-compose build

# Start the container
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop the container
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up --build
```

## 📈 Example Stocks to Try

- **Tech**: AAPL, GOOGL, MSFT, TSLA, NVDA
- **Finance**: JPM, BAC, GS, V, MA
- **E-commerce**: AMZN, SHOP, EBAY
- **Entertainment**: DIS, NFLX, SPOT

## 🔬 Algorithm Parameters

- **Population Size**: 20 strategies per generation
- **Mutation Rate**: 10% chance of random changes
- **Elitism**: Top 10% strategies preserved each generation
- **Selection**: Tournament selection for parent choosing
- **Window Range**: 2-50 days for SMA calculation

## 📊 Output Metrics

- **Best Window**: Optimal SMA window size found
- **Mean Absolute Error (MAE)**: Average prediction error in dollars
- **Fitness Score**: Negative MAE (higher is better)
- **Data Points**: Number of historical price points used

## ⚠️ Important Notes

- Stock market predictions are inherently uncertain
- This is a educational/demonstration project
- Past performance doesn't guarantee future results
- Use for learning purposes, not financial advice
- Requires internet connection for stock data

## 🐛 Troubleshooting

**Issue**: Container fails to start
- **Solution**: Ensure Docker is running and ports are available

**Issue**: Stock data not loading
- **Solution**: Check internet connection and ticker symbol validity

**Issue**: Slow performance
- **Solution**: Reduce generations or use shorter time periods

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as a portfolio project demonstrating:
- Genetic algorithm implementation
- Real-time data integration with yfinance
- Full-stack web development with Flask
- Docker containerization
- Data visualization with Matplotlib

## 🔮 Future Enhancements

- Multiple algorithm strategies (LSTM, ARIMA)
- Real-time predictions
- Portfolio optimization
- Backtesting framework
- REST API endpoints
- User authentication
- Historical comparison
- Advanced technical indicators

---

**Note**: This application is for educational purposes only. Always do your own research before making investment decisions.
