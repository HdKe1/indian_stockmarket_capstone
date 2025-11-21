from cust_model import  MultiSourceScorer, ReportGenerator, NIFTYHybridPredictor, IndianMarketDataCollector, IndianWeatherCollector
import os
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime
import numpy as np
# Load environment variables
load_dotenv()

# Configuration
config = {
    'news_api_key': os.getenv('NEWS_API_KEY'),
    'weather_api_key': os.getenv('WEATHER_API_KEY'),
    'output_dir': 'nifty_predictions'
}


class NIFTYPredictionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.market_collector = IndianMarketDataCollector(config)
        self.weather_collector = IndianWeatherCollector(
            config.get('weather_api_key', '')
        )
        self.scorer = MultiSourceScorer()
        self.report_gen = ReportGenerator(
            output_dir=config.get('output_dir', 'nifty_reports')
        )
    
    def run_prediction(self, symbol: str, company_name: str, 
                       sector: str, headquarters: str = 'Mumbai'):
        print(f"\n{'='*70}")
        print(f"Running prediction for {symbol} ({company_name})")
        print(f"{'='*70}\n")
        
        # 1. Fetch NSE stock data
        print("📈 Fetching NSE stock data...")
        nse_data = self.market_collector.fetch_nse_data(symbol)
        price_data = nse_data['history'][['Close', 'High', 'Low']].values
        print(f"   Current Price: ₹{nse_data['current_price']:.2f}")
        
        # 2. Fetch news
        print("\n📰 Fetching news data...")
        news_articles = self.market_collector.fetch_indian_news(company_name, symbol)
        news_features = self._process_news(news_articles)
        print(f"   Found {news_features.get('news_volume', 0)} articles")
        
        # 6. Weather
        print("\n🌤️  Fetching weather data...")
        weather_data = self.weather_collector.fetch_weather_multiple_cities()
        weather_impact = self.weather_collector.calculate_weather_impact_india(
            weather_data, sector, headquarters
        )
        print(f"   Weather Impact: {weather_impact.get('impact_score', 0):+.2f}")
        
        # 7. Indian economic indicators
        print("\n📉 Fetching Indian economic indicators...")
        indian_economic = self.market_collector.get_indian_economic_indicators()
        if 'nifty_50' in indian_economic:
            print(f"   NIFTY 50: {indian_economic['nifty_50']['change_pct']:+.2f}%")
        
        # 8. NSE indices
        print("\n📊 Fetching NSE indices data...")
        nse_indices = self.market_collector.fetch_nse_indices()
        
        # 9. Load model and predict
        print("\n🤖 Generating prediction...")
        timeseries_model = self._load_timeseries_model(symbol, price_data)
        
        predictor = NIFTYHybridPredictor(
            timeseries_model, self.scorer, self.report_gen
        )
        
        result = predictor.predict(
            price_data=price_data,
            news_features=news_features,
           
            weather_impact=weather_impact,
            indian_economic_data=indian_economic,
            nse_sentiment_data=nse_indices,
            symbol=symbol,
            generate_report=True
        )
        
        self._print_results(result, symbol, nse_data['current_price'])
        return result
    
    def _process_news(self, articles):
        if not articles:
            return {'news_volume': 0, 'sentiment_mean': 0, 'sentiment_std': 0}
        
        try:
            from transformers import pipeline
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
            
            sentiments = []
            for article in articles[:20]:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    result = sentiment_analyzer(text[:512])
                    score = result[0]['score'] if result[0]['label'] == 'positive' else -result[0]['score']
                    sentiments.append(score)
            
            return {
                'news_volume': len(sentiments),
                'sentiment_mean': np.mean(sentiments) if sentiments else 0,
                'sentiment_std': np.std(sentiments) if sentiments else 0
            }
        except Exception as e:
            print(f"   Warning: Sentiment analysis failed: {e}")
            return {'news_volume': len(articles), 'sentiment_mean': 0, 'sentiment_std': 0}
    
    def _load_timeseries_model(self, symbol, price_data):
        """
        Load your trained time series model here
        For now, uses simple prediction based on recent average
        """
        class SimplePredictor:
            def predict(self, data):
                if len(data) > 5:
                    # Simple moving average of last 5 days
                    return np.mean(data[-5:, 0])
                elif len(data) > 0:
                    return data[-1][0]
                return 0
        
        return SimplePredictor()
    
    def _print_results(self, result, symbol, current_price):
        print(f"\n{'='*70}")
        print(f"पूर्वानुमान परिणाम / PREDICTION RESULTS - {symbol}")
        print(f"{'='*70}")
        print(f"Current Price:          ₹{current_price:.2f}")
        print(f"Predicted Price:        ₹{result['final_prediction']:.2f}")
        change_pct = ((result['final_prediction']/current_price - 1) * 100) if current_price > 0 else 0
        print(f"Expected Change:        {change_pct:+.2f}%")
        print(f"Confidence Range:       ₹{result['lower_bound']:.2f} - ₹{result['upper_bound']:.2f}")
        print(f"Sentiment Adjustment:   {result['sentiment_adjustment_pct']:+.2f}%")
        print(f"Overall Confidence:     {result['confidence']*100:.1f}%")
        print(f"Market Outlook:         {result['interpretation']}")
        print(f"{'='*70}\n")
# Initialize pipeline
pipeline = NIFTYPredictionPipeline(config)

# Predict for a single stock
# Choose any NIFTY 50 stock:
result = pipeline.run_prediction(
    symbol='RELIANCE',              # Stock symbol (without .NS)
    company_name='Reliance Industries',
    sector='Energy',                 # Energy, IT, Banking, FMCG, etc.
    headquarters='Mumbai'            # City where HQ is located
)

print("\n✅ Prediction Complete!")
print(f"Check reports in: {config['output_dir']}/")