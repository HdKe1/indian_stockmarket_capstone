import sys
import os
# Add the cust_data_model directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cust_data_model'))

from cust_model import MultiSourceScorer, ReportGenerator, NIFTYHybridPredictor, IndianMarketDataCollector, IndianWeatherCollector
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import your Keras model components from app.py
from app import predict_the_future, OHLC, START_DATE, END_DATE, create_dataset, MAX_TIME_STEPS, MAX_RANGE

# Load environment variables
load_dotenv()

# Configuration
config = {
    'news_api_key': os.getenv('NEWS_API_KEY'),
    'weather_api_key': os.getenv('WEATHER_API_KEY'),
    'output_dir': 'nifty_predictions'
}


class KerasTimeSeriesModel:
    """
    Wrapper for your existing Keras LSTM model
    Integrates your app.py prediction function
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
    
    def predict(self, price_data):
        """
        Returns the predicted close price for next day using your Keras model
        
        Args:
            price_data: numpy array of shape (n, 3) with [Close, High, Low]
        
        Returns:
            float: Predicted close price
        """
        try:
            # Get predictions from your Keras model
            pred_open, pred_high, pred_low, pred_close, _ = predict_the_future(self.symbol)
            
            print(f"   Keras Model Predictions:")
            print(f"     Open:  ₹{pred_open:.2f}")
            print(f"     High:  ₹{pred_high:.2f}")
            print(f"     Low:   ₹{pred_low:.2f}")
            print(f"     Close: ₹{pred_close:.2f}")
            
            # Return close price as the base prediction
            return float(pred_close)
            
        except Exception as e:
            print(f"   ⚠️  Error in Keras prediction: {e}")
            print(f"   Falling back to simple moving average...")
            
            # Fallback to simple moving average
            if len(price_data) > 5:
                return float(np.mean(price_data[-5:, 0]))
            return float(price_data[-1][0]) if len(price_data) > 0 else 0.0
    
    def get_all_predictions(self):
        """
        Returns all OHLC predictions if needed for detailed analysis
        
        Returns:
            dict: Contains 'open', 'high', 'low', 'close', 'figure'
        """
        try:
            pred_open, pred_high, pred_low, pred_close, fig = predict_the_future(self.symbol)
            return {
                'open': float(pred_open),
                'high': float(pred_high),
                'low': float(pred_low),
                'close': float(pred_close),
                'figure': fig
            }
        except Exception as e:
            print(f"Error getting all predictions: {e}")
            return None


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
        current_price = nse_data['current_price']
        print(f"   Current Price: ₹{current_price:.2f}")
        print(f"   Historical data: {len(price_data)} days")
        
        # 2. Fetch news
        print("\n📰 Fetching news data...")
        news_articles = self.market_collector.fetch_indian_news(company_name, symbol)
        news_features = self._process_news(news_articles)
        print(f"   Found {news_features.get('news_volume', 0)} articles")
        if news_features.get('news_volume', 0) > 0:
            print(f"   Sentiment: {news_features.get('sentiment_mean', 0):+.3f}")
        
        # 3. Weather
        print("\n🌤️  Fetching weather data...")
        weather_data = self.weather_collector.fetch_weather_multiple_cities()
        weather_impact = self.weather_collector.calculate_weather_impact_india(
            weather_data, sector, headquarters
        )
       

        if weather_impact.get('is_monsoon_season'):
            print(f"   🌧️  Monsoon Season Active")
        
        # 4. Indian economic indicators
        print("\n📉 Fetching Indian economic indicators...")
        indian_economic = self.market_collector.get_indian_economic_indicators()
        if 'nifty_50' in indian_economic:
            print(f"   NIFTY 50: {indian_economic['nifty_50']['change_pct']:+.2f}%")
        if 'india_vix' in indian_economic:
            print(f"   India VIX: {indian_economic['india_vix']['current']:.2f}")
        
        # 5. NSE indices
        print("\n📊 Fetching NSE indices data...")
        nse_indices = self.market_collector.fetch_nse_indices()
        for idx, data in nse_indices.items():
            idx_name = idx.replace('^', '')
            print(f"   {idx_name}: {data.get('change_pct', 0):+.2f}%")
        
        # 6. Load Keras model and predict
        print("\n🤖 Generating Keras LSTM prediction...")
        timeseries_model = self._load_timeseries_model(symbol, price_data)
        
        # Create hybrid predictor
        predictor = NIFTYHybridPredictor(
            timeseries_model, self.scorer, self.report_gen
        )
        
        # Generate final prediction
        print("\n🔮 Calculating hybrid prediction with sentiment adjustment...")
        result = predictor.predict(
            price_data=price_data,
            news_features=news_features,
            weather_impact=weather_impact,
            indian_economic_data=indian_economic,
            nse_sentiment_data=nse_indices,
            symbol=symbol,
            generate_report=True
        )
        
        # Print results
        self._print_results(result, symbol, current_price)
        
        # Get detailed OHLC predictions from Keras model
        print("\n📊 Detailed Keras Model Predictions:")
        all_predictions = timeseries_model.get_all_predictions()
        if all_predictions:
            print(f"   Open:  ₹{all_predictions['open']:.2f}")
            print(f"   High:  ₹{all_predictions['high']:.2f}")
            print(f"   Low:   ₹{all_predictions['low']:.2f}")
            print(f"   Close: ₹{all_predictions['close']:.2f}")
            result['keras_ohlc'] = {
                'open': all_predictions['open'],
                'high': all_predictions['high'],
                'low': all_predictions['low'],
                'close': all_predictions['close']
            }
        
        return result
    
    def _process_news(self, articles):
        """Process news articles and extract sentiment"""
        if not articles:
            return {'news_volume': 0, 'sentiment_mean': 0, 'sentiment_std': 0}
        
        try:
            from transformers import pipeline
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
            
            sentiments = []
            for article in articles[:20]:  # Process up to 20 articles
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    result = sentiment_analyzer(text[:512])
                 
                    label = result[0]['label'].lower()
                    score = result[0]['score']
                    
                    if label == 'positive':
                        sentiments.append(score)
                    elif label == 'negative':
                        sentiments.append(-score)
                    else:  # neutral
                        sentiments.append(0)
            
            if sentiments:
                return {
                    'news_volume': len(sentiments),
                    'sentiment_mean': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments)
                }
            else:
                return {'news_volume': len(articles), 'sentiment_mean': 0, 'sentiment_std': 0}
                
        except Exception as e:
            print(f"   ⚠️  Sentiment analysis failed: {e}")
            return {'news_volume': len(articles), 'sentiment_mean': 0, 'sentiment_std': 0}
    
    def _load_timeseries_model(self, symbol, price_data):
        """
        Load your trained Keras time series model
        This integrates with your existing app.py model
        """
        return KerasTimeSeriesModel(symbol)
    
    def _print_results(self, result, symbol, current_price):
        """Print formatted prediction results"""
        print(f"\n{'='*70}")
        print(f"PREDICTION RESULTS - {symbol}")
        print(f"{'='*70}")
        print(f"Current Price:          ₹{current_price:.2f}")
        print(f"Base TS Prediction:     ₹{result['timeseries_prediction']:.2f}")
        
        # Calculate change from current price
        ts_change = ((result['timeseries_prediction']/current_price - 1) * 100) if current_price > 0 else 0
        print(f"TS Change:              {ts_change:+.2f}%")
        
        print(f"\nAfter Sentiment Adjustment:")
        print(f"Final Predicted Price:  ₹{result['final_prediction']:.2f}")
        
        final_change = ((result['final_prediction']/current_price - 1) * 100) if current_price > 0 else 0
        print(f"Expected Change:        {final_change:+.2f}%")
        
        print(f"Confidence Range:       ₹{result['lower_bound']:.2f} - ₹{result['upper_bound']:.2f}")
        print(f"Sentiment Adjustment:   {result['sentiment_adjustment_pct']:+.2f}%")
        print(f"Overall Confidence:     {result['confidence']*100:.1f}%")
        print(f"Market Outlook:         {result['interpretation']}")
        
        # Show individual source contributions
        print(f"\n{'─'*70}")
        print("Individual Source Scores:")
        for source, score in result['breakdown'].items():
            source_name = source.replace('_', ' ').title()
            sentiment = "📈 Bullish" if score > 0 else "📉 Bearish" if score < 0 else "➡️  Neutral"
            print(f"  {source_name:20s}: {score:+.4f}  {sentiment}")
        
        print(f"{'='*70}\n")


# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = NIFTYPredictionPipeline(config)
    
    # Example: Predict for Reliance Industries
    # You can change this to any stock from your SYMBOLS list in app.py
    result = pipeline.run_prediction(
        symbol='BPCL',              # Must match symbols in your app.py SYMBOLS list
        company_name='Bharat Petroleum',
        sector='Energy',                # Energy, IT, Banking, FMCG, etc.
        headquarters='Mumbai'           # City where HQ is located
    )
    
    print("\n✅ Prediction Complete!")
    print(f"📁 Check reports in: {config['output_dir']}/")
    
    if 'report_files' in result:
        print("\n📄 Generated Reports:")
        for report_type, filepath in result['report_files'].items():
            print(f"   {report_type.upper()}: {filepath}")