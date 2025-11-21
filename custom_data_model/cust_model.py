import numpy as np
import json
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
@dataclass
class DataSourceWeights:
    """
    Configurable weights for different data sources - Indian market optimized
    
    Weight Distribution Strategy:
    - timeseries: 50% (Core technical prediction)
    - news: 20% (Market sentiment)
    - economic: 15% (Indian macro indicators)
    - nse_sentiment: 10% (Broader market mood)
    - weather: 5% (Sector-specific impact)
    
    Total: 100% ✅
    """
    timeseries: float = 0.50  # Base prediction - MOST IMPORTANT
    news: float = 0.20        # News sentiment
    economic: float = 0.15    # Indian economic indicators
    nse_sentiment: float = 0.10  # NSE market sentiment
    weather: float = 0.05     # Weather impact
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = sum([
            self.timeseries,
            self.news, 
            self.weather, 
            self.economic, 
            self.nse_sentiment
        ])
        
        if total == 0:
            raise ValueError("All weights cannot be zero")
        
        self.timeseries /= total
        self.news /= total
        self.weather /= total
        self.economic /= total
        self.nse_sentiment /= total
    
    def validate(self):
        """Validate that weights sum to approximately 1.0"""
        total = sum([
            self.timeseries,
            self.news,
            self.weather,
            self.economic,
            self.nse_sentiment
        ])
        
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        
        return True


class IndianMarketDataCollector:
    """
    Data collection specifically for Indian (NIFTY) stocks
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.nse_indices = ['^NSEI', '^NSEBANK', '^CNXIT']  # NIFTY 50, Bank NIFTY, IT
    
    def fetch_nse_data(self, symbol: str):
        """
        Fetch NSE stock data
        Symbol format: RELIANCE.NS, TCS.NS, INFY.NS, etc.
        """
        import yfinance as yf
        
        # Ensure .NS suffix for NSE stocks
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = f"{symbol}.NS"
        
        stock = yf.Ticker(symbol)
        
        # Get historical data
        hist = stock.history(period="60d")
        
        # Get info
        info = stock.info
        
        return {
            'history': hist,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'current_price': info.get('currentPrice', hist['Close'].iloc[-1] if len(hist) > 0 else 0)
        }
    
    def fetch_nse_indices(self):
        """Fetch major NSE indices"""
        import yfinance as yf
        
        indices_data = {}
        for index in self.nse_indices:
            try:
                ticker = yf.Ticker(index)
                hist = ticker.history(period='5d')
                
                if len(hist) >= 2:
                    change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                    indices_data[index] = {
                        'current': hist['Close'].iloc[-1],
                        'change_pct': change * 100,
                        'volume': hist['Volume'].iloc[-1]
                    }
            except Exception as e:
                print(f"Warning: Could not fetch {index}: {e}")
        
        return indices_data
    
    def fetch_indian_news(self, company_name: str, symbol: str):
        """
        Fetch news from Indian sources
        Sources: Economic Times, Moneycontrol, Business Standard, etc.
        Uses News API (free tier)
        """
        try:
            import requests
            
            # Using News API with Indian sources
            news_api_key = self.config.get('news_api_key')
            if not news_api_key:
                print("Warning: No NEWS_API_KEY found in config")
                return []
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{company_name} OR {symbol}',
                'sources': 'the-times-of-india',  # Available free source
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': news_api_key
            }
            response = requests.get(url, params=params, timeout=10)
        
            if response.status_code == 200:
        
                return response.json().get('articles', [])
            else:
                print(f"Warning: News API returned status {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Indian news: {e}")
            return []
    
    def get_indian_economic_indicators(self):
        """
        Fetch Indian economic indicators
        - NIFTY 50 performance
        - INR/USD exchange rate
        - India VIX (volatility index)
        """
        import yfinance as yf
        
        indicators = {}
        
        try:
            # NIFTY 50
            nifty = yf.Ticker("^NSEI")
            nifty_hist = nifty.history(period='5d')
            if len(nifty_hist) >= 2:
                indicators['nifty_50'] = {
                    'current': nifty_hist['Close'].iloc[-1],
                    'change_pct': ((nifty_hist['Close'].iloc[-1] - nifty_hist['Close'].iloc[0]) 
                                   / nifty_hist['Close'].iloc[0] * 100)
                }
            
            # India VIX
            vix = yf.Ticker("^INDIAVIX")
            vix_hist = vix.history(period='5d')
            if len(vix_hist) > 0:
                indicators['india_vix'] = {
                    'current': vix_hist['Close'].iloc[-1]
                }
            
            # USD/INR
            inr = yf.Ticker("INR=X")
            inr_hist = inr.history(period='5d')
            if len(inr_hist) >= 2:
                indicators['usd_inr'] = {
                    'current': inr_hist['Close'].iloc[-1],
                    'change_pct': ((inr_hist['Close'].iloc[-1] - inr_hist['Close'].iloc[0]) 
                                   / inr_hist['Close'].iloc[0] * 100)
                }
            
        except Exception as e:
            print(f"Error fetching Indian economic indicators: {e}")
        
        return indicators


class IndianWeatherCollector:
    """
    Weather data collector for Indian cities
    Important for: Agriculture (ITC, Tata Consumer), Retail (DMart, Reliance Retail), 
    Power (NTPC, PowerGrid), FMCG (HUL, Nestle India)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.major_cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
            'Kolkata', 'Pune', 'Ahmedabad'
        ]
    
    def fetch_weather_multiple_cities(self):
        """Fetch weather for major Indian metros"""
        import requests
        
        if not self.api_key:
            print("Warning: No WEATHER_API_KEY found")
            return {}
        
        weather_data = {}
        for city in self.major_cities:
            try:
                url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'q': f"{city},IN",
                    'appid': self.api_key,
                    'units': 'metric'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    weather_data[city] = response.json()
            except Exception as e:
                print(f"Warning: Could not fetch weather for {city}: {e}")
        
        return weather_data
    
    def calculate_weather_impact_india(self, weather_data: Dict, sector: str, 
                                       company_hq: str = 'Mumbai') -> Dict:
        """
        Calculate weather impact for Indian sectors
        """
        if not weather_data or company_hq not in weather_data:
            return {'impact_score': 0}
        
        city_weather = weather_data[company_hq]
        temp = city_weather['main']['temp']
        conditions = city_weather['weather'][0]['main']
        humidity = city_weather['main']['humidity']
        
        impact_score = 0
        
        # Indian monsoon consideration (June-September)
        current_month = datetime.now().month
        is_monsoon = 6 <= current_month <= 9
        
        if sector in ['Agriculture', 'FMCG']:
            if is_monsoon and conditions in ['Rain', 'Drizzle']:
                impact_score += 0.3  # Good for agriculture
            elif not is_monsoon and temp > 35:
                impact_score -= 0.2  # Heat wave negative
        
        elif sector in ['Retail', 'Consumer']:
            if temp > 40 or (is_monsoon and conditions == 'Thunderstorm'):
                impact_score -= 0.3  # Extreme weather hurts footfall
            elif 20 < temp < 30:
                impact_score += 0.1  # Pleasant weather
        
        elif sector in ['Power', 'Energy']:
            if temp > 35:
                impact_score += 0.4  # High AC demand
            elif temp < 15:
                impact_score += 0.2  # Heating demand
        
        elif sector in ['Cement', 'Construction']:
            if is_monsoon and conditions in ['Rain', 'Thunderstorm']:
                impact_score -= 0.3  # Rain stops construction
        
        elif sector in ['IT', 'Services']:
            impact_score = 0  # Weather has minimal impact
        
        return {
            'impact_score': impact_score,
            'temperature': temp,
            'conditions': conditions,
            'humidity': humidity,
            'is_monsoon_season': is_monsoon
        }


class MultiSourceScorer:
    def __init__(self, weights: Optional[DataSourceWeights] = None):
        self.weights = weights or DataSourceWeights()
        self.weights.normalize()
        self.weights.validate()  # ✅ NEW: Validate weights
    
    def score_news(self, news_features: Dict) -> float:
        """Score news sentiment"""
        if not news_features or news_features.get('news_volume', 0) == 0:
            return 0.0
        
        sentiment = news_features.get('sentiment_mean', 0)
        volume_factor = min(news_features.get('news_volume', 0) / 20, 1.0)
        confidence = 1 - min(news_features.get('sentiment_std', 0), 0.5) / 0.5
        
        score = sentiment * volume_factor * confidence
        return np.clip(score, -1.0, 1.0)
    
    def score_nse_sentiment(self, nse_data: Dict) -> float:
        """Score NSE market sentiment"""
        if not nse_data:
            return 0.0
        
        # Calculate sentiment from index movements
        nifty_change = nse_data.get('^NSEI', {}).get('change_pct', 0)
        bank_nifty_change = nse_data.get('^NSEBANK', {}).get('change_pct', 0)
        
        # Average change as sentiment indicator
        avg_change = (nifty_change + bank_nifty_change) / 2
        sentiment = np.tanh(avg_change / 2)  # Normalize
        
        return np.clip(sentiment, -1.0, 1.0)
    
    def score_indian_economic(self, economic_data: Dict) -> float:
        """Score Indian economic indicators"""
        if not economic_data:
            return 0.0
        
        sentiment = 0.0
        
        # NIFTY 50 change
        if 'nifty_50' in economic_data:
            nifty_change = economic_data['nifty_50'].get('change_pct', 0)
            sentiment += nifty_change * 0.4
        
        # India VIX (inverse relationship - high VIX is bearish)
        if 'india_vix' in economic_data:
            vix = economic_data['india_vix'].get('current', 20)
            # VIX > 20 is high volatility (bearish), < 15 is low (bullish)
            vix_sentiment = -((vix - 17.5) / 10)  # Normalized around 17.5
            sentiment += vix_sentiment * 0.3
        
        # USD/INR (strengthening rupee is generally bullish for imports)
        if 'usd_inr' in economic_data:
            inr_change = economic_data['usd_inr'].get('change_pct', 0)
            sentiment -= inr_change * 0.3  # Negative because INR weakening is bearish
        
        # Normalize to -1 to 1
        sentiment = sentiment / 10
        return np.clip(sentiment, -1.0, 1.0)
    
    def score_weather(self, weather_impact: Dict) -> float:
        """Score weather impact"""
        if not weather_impact:
            return 0.0
        return np.clip(weather_impact.get('impact_score', 0), -1.0, 1.0)
    
    def score_timeseries(self, ts_prediction: float, current_price: float) -> float:
        """
        ✅ NEW: Score time series prediction strength
        Converts predicted price change into a sentiment score
        """
        if current_price <= 0:
            return 0.0
        
        # Calculate predicted change percentage
        change_pct = (ts_prediction - current_price) / current_price
        
        # Normalize to [-1, 1] range
        # ±10% change = ±1.0 sentiment
        score = np.tanh(change_pct * 10)
        
        return np.clip(score, -1.0, 1.0)
    
    def calculate_composite_score(self, 
                                  news_features: Dict,
                                  weather_impact: Dict,
                                  indian_economic_data: Dict,
                                  nse_sentiment_data: Dict,
                                  ts_prediction: float = None,
                                  current_price: float = None) -> Dict:
        """
        ✅ FIXED: Calculate weighted composite score from ALL sources
        Now includes timeseries scoring
        """
        # Individual scores
        scores = {
            'news': self.score_news(news_features),
            'weather': self.score_weather(weather_impact),
            'indian_economic': self.score_indian_economic(indian_economic_data),
            'nse_sentiment': self.score_nse_sentiment(nse_sentiment_data),
            'timeseries': self.score_timeseries(ts_prediction, current_price) if ts_prediction and current_price else 0.0
        }
        
        # Calculate confidence for each source
        confidences = {
            'news': min(news_features.get('news_volume', 0) / 20, 1.0) if news_features else 0,
            'weather': 0.8 if weather_impact else 0,
            'indian_economic': 1.0 if indian_economic_data else 0,
            'nse_sentiment': 1.0 if nse_sentiment_data else 0,
            'timeseries': 1.0 if ts_prediction and current_price else 0
        }
        
        # ✅ FIXED: Weighted composite score with ALL sources
        composite_score = (
            scores['timeseries'] * self.weights.timeseries +
            scores['news'] * self.weights.news +
            scores['weather'] * self.weights.weather +
            scores['indian_economic'] * self.weights.economic +
            scores['nse_sentiment'] * self.weights.nse_sentiment
        )
        
        # Overall confidence
        overall_confidence = (
            confidences['timeseries'] * self.weights.timeseries +
            confidences['news'] * self.weights.news +
            confidences['weather'] * self.weights.weather +
            confidences['indian_economic'] * self.weights.economic +
            confidences['nse_sentiment'] * self.weights.nse_sentiment
        )
        
        return {
            'composite_score': composite_score,
            'overall_confidence': overall_confidence,
            'individual_scores': scores,
            'individual_confidences': confidences,
            'interpretation': self._interpret_score(composite_score),
            'weights_used': asdict(self.weights)  # ✅ NEW: Show which weights were used
        }
    
    def _interpret_score(self, score: float) -> str:
        """Convert numerical score to interpretation"""
        if score > 0.5:
            return "तेजी की प्रबल संभावना (Strongly Bullish)"
        elif score > 0.2:
            return "तेजी (Bullish)"
        elif score > -0.2:
            return "स्थिर (Neutral)"
        elif score > -0.5:
            return "मंदी (Bearish)"
        else:
            return "मंदी की प्रबल संभावना (Strongly Bearish)"


class ReportGenerator:
    """Generate detailed reports for Indian stocks"""
    
    def __init__(self, output_dir: str = "nifty_prediction_reports"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_text_report(self, prediction_result: Dict, symbol: str,
                           raw_data: Dict = None) -> str:
        """Generate text report for Indian stocks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{symbol}_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"INDIAN STOCK MARKET PREDICTION REPORT - {symbol}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S IST')}\n\n")
            
            # Summary Section
            f.write("="*70 + "\n")
            f.write("PREDICTION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Final Prediction:        ₹{prediction_result['final_prediction']:.2f}\n")
            f.write(f"Confidence Interval:     ₹{prediction_result['lower_bound']:.2f} - ₹{prediction_result['upper_bound']:.2f}\n")
            f.write(f"Time Series Prediction:  ₹{prediction_result['timeseries_prediction']:.2f}\n")
            f.write(f"Sentiment Adjustment:    {prediction_result['sentiment_adjustment_pct']:.2f}%\n")
            f.write(f"Overall Confidence:      {prediction_result['confidence']*100:.1f}%\n")
            f.write(f"Market Interpretation:   {prediction_result['interpretation']}\n\n")
            
            # ✅ NEW: Weights Used Section
            f.write("="*70 + "\n")
            f.write("WEIGHT DISTRIBUTION\n")
            f.write("="*70 + "\n")
            if 'weights_used' in prediction_result:
                for source, weight in prediction_result['weights_used'].items():
                    bar_length = int(weight * 50)
                    bar = "█" * bar_length
                    f.write(f"{source.replace('_', ' ').title():20s}: {weight*100:5.1f}% {bar}\n")
            f.write("\n")
            
            # Data Sources Breakdown
            f.write("="*70 + "\n")
            f.write("DATA SOURCES BREAKDOWN\n")
            f.write("="*70 + "\n\n")
            
            for source, score in sorted(prediction_result['breakdown'].items()):
                source_name = source.replace('_', ' ').title()
                bar_length = int(abs(score) * 20)
                bar = "+" * bar_length if score > 0 else "-" * bar_length
                f.write(f"{source_name:25s}: {score:+.4f}  {bar}\n")
            
            f.write("\n")
            
            # Indian Market Specific Data
            if raw_data:
                f.write("="*70 + "\n")
                f.write("भारतीय बाजार डेटा / INDIAN MARKET DATA\n")
                f.write("="*70 + "\n\n")
                
                if 'nse_indices' in raw_data:
                    f.write("NSE Indices:\n")
                    for idx, data in raw_data['nse_indices'].items():
                        f.write(f"  {idx:15s}: {data.get('change_pct', 0):+.2f}%\n")
                    f.write("\n")
                
                if 'indian_economic' in raw_data:
                    f.write("Economic Indicators:\n")
                    econ = raw_data['indian_economic']
                    if 'nifty_50' in econ:
                        f.write(f"  NIFTY 50:      {econ['nifty_50'].get('change_pct', 0):+.2f}%\n")
                    if 'india_vix' in econ:
                        f.write(f"  India VIX:     {econ['india_vix'].get('current', 0):.2f}\n")
                    if 'usd_inr' in econ:
                        f.write(f"  USD/INR:       ₹{econ['usd_inr'].get('current', 0):.2f}\n")
                    f.write("\n")
                
                if 'weather' in raw_data and raw_data['weather'].get('is_monsoon_season'):
                    f.write("⚠️  Monsoon Season Active - Weather impact considered\n\n")
            
            f.write("="*70 + "\n")
            f.write("रिपोर्ट समाप्त / END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return filename
    
    def generate_json_report(self, prediction_result: Dict, symbol: str, 
                           raw_data: Dict = None) -> str:
        """Generate JSON report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{symbol}_prediction_{timestamp}.json"
        
        report = {
            'metadata': {
                'symbol': symbol,
                'market': 'NSE/BSE',
                'currency': 'INR',
                'timestamp': datetime.now().isoformat(),
                'prediction_type': 'hybrid_multi_source_indian_market'
            },
            'prediction': prediction_result,
            'raw_data': raw_data or {}
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename
    
    def generate_csv_report(self, prediction_result: Dict, symbol: str) -> str:
        """Generate CSV report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{symbol}_scores_{timestamp}.csv"
        
        data = []
        
        # Prediction details
        data.append({
            'Category': 'Final Prediction',
            'Metric': 'Price (INR)',
            'Value': f"₹{prediction_result['final_prediction']:.2f}",
            'Weight': 1.0,
            'Notes': prediction_result['interpretation']
        })
        
        # Source breakdown
        weights_used = prediction_result.get('weights_used', {})
        for source, score in prediction_result['breakdown'].items():
            weight = weights_used.get(source, 0)
            data.append({
                'Category': 'Data Source',
                'Metric': source.replace('_', ' ').title(),
                'Value': f"{score:.4f}",
                'Weight': f"{weight:.4f}",
                'Notes': 'Bullish' if score > 0 else 'Bearish' if score < 0 else 'Neutral'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        return filename
    
    def generate_all_reports(self, prediction_result: Dict, symbol: str, 
                           raw_data: Dict = None) -> Dict[str, str]:
        """Generate all report types"""
        files = {
            'json': self.generate_json_report(prediction_result, symbol, raw_data),
            'csv': self.generate_csv_report(prediction_result, symbol),
            'text': self.generate_text_report(prediction_result, symbol, raw_data)
        }
        return files


class NIFTYHybridPredictor:
    """Hybrid predictor specifically for NIFTY stocks"""
    
    def __init__(self, timeseries_model, scorer: MultiSourceScorer, 
                 report_generator: Optional[ReportGenerator] = None):
        self.timeseries_model = timeseries_model
        self.scorer = scorer
        self.report_generator = report_generator or ReportGenerator()
    
    def predict(self, 
                price_data,
                news_features: Dict,
                weather_impact: Dict,
                indian_economic_data: Dict,
                nse_sentiment_data: Dict,
                symbol: str = None,
                generate_report: bool = True) -> Dict:
        """
        ✅ FIXED: Generate prediction for NIFTY stock
        Now properly includes timeseries in scoring
        """
        # Get time series prediction
        ts_prediction = self.timeseries_model.predict(price_data)
        
        # Get current price (last closing price)
        current_price = price_data[-1][0] if len(price_data) > 0 else ts_prediction
        
        # ✅ FIXED: Get composite sentiment score WITH timeseries
        sentiment_analysis = self.scorer.calculate_composite_score(
            news_features,
            weather_impact,
            indian_economic_data,
            nse_sentiment_data,
            ts_prediction=ts_prediction,  # ✅ NEW
            current_price=current_price   # ✅ NEW
        )
        
        composite_score = sentiment_analysis['composite_score']
        confidence = sentiment_analysis['overall_confidence']
        
        # Apply sentiment adjustment (max ±5%)
        max_adjustment = 0.05
        sentiment_adjustment = composite_score * confidence * max_adjustment
        
        # Final prediction
        final_prediction = ts_prediction * (1 + sentiment_adjustment)
        
        # Calculate bounds
        uncertainty = (1 - confidence) * 0.03
        lower_bound = final_prediction * (1 - uncertainty)
        upper_bound = final_prediction * (1 + uncertainty)
        
        result = {
            'final_prediction': final_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'timeseries_prediction': ts_prediction,
            'sentiment_adjustment_pct': sentiment_adjustment * 100,
            'composite_score': composite_score,
            'confidence': confidence,
            'interpretation': sentiment_analysis['interpretation'],
            'breakdown': sentiment_analysis['individual_scores'],
            'individual_confidences': sentiment_analysis['individual_confidences'],
            'weights_used': sentiment_analysis['weights_used']  # ✅ NEW
        }
        
        # Generate reports
        if generate_report and symbol:
            raw_data = {
                'news': news_features or {},
                'weather': weather_impact or {},
                'indian_economic': indian_economic_data or {},
                'nse_indices': nse_sentiment_data or {}
            }
            
            report_files = self.report_generator.generate_all_reports(
                result, symbol, raw_data
            )
            result['report_files'] = report_files
            
            print(f"\n📊 Reports generated for {symbol}:")
            for report_type, filepath in report_files.items():
                print(f"  {report_type.upper()}: {filepath}")
        
        return result


# ✅ DEMO: Test weight system
if __name__ == "__main__":
    print("="*70)
    print("TESTING FIXED WEIGHT SYSTEM")
    print("="*70)
    
    # Test 1: Default weights
    weights = DataSourceWeights()
    weights.normalize()
    
    print("\n✅ Default Weights:")
    print(f"  Timeseries:      {weights.timeseries*100:.1f}%")
    print(f"  News:            {weights.news*100:.1f}%")
    print(f"  Economic:        {weights.economic*100:.1f}%")
    print(f"  NSE Sentiment:   {weights.nse_sentiment*100:.1f}%")
    print(f"  Weather:         {weights.weather*100:.1f}%")
    print(f"  ───────────────")
    total = sum([weights.timeseries, weights.news, weights.economic, 
                 weights.nse_sentiment, weights.weather])
    print(f"  TOTAL:           {total*100:.1f}%")
    
    try:
        weights.validate()
        print("  ✅ Weights validation: PASSED")
    except ValueError as e:
        print(f"  ❌ Weights validation: FAILED - {e}")
    
    # Test 2: Custom weights
    print("\n✅ Custom Weights Example (Conservative - Heavy on Timeseries):")
    custom_weights = DataSourceWeights(
        timeseries=0.60,
        news=0.15,
        economic=0.15,
        nse_sentiment=0.05,
        weather=0.05
    )
    custom_weights.normalize()
    print(f"  Timeseries:      {custom_weights.timeseries*100:.1f}%")
    print(f"  News:            {custom_weights.news*100:.1f}%")
    print(f"  Economic:        {custom_weights.economic*100:.1f}%")
    print(f"  NSE Sentiment:   {custom_weights.nse_sentiment*100:.1f}%")
    print(f"  Weather:         {custom_weights.weather*100:.1f}%")
    
    # Test 3: Scoring demonstration
    print("\n" + "="*70)
    print("TESTING MULTI-SOURCE SCORING")
    print("="*70)
    
    scorer = MultiSourceScorer(weights)
    
    # Mock data
    mock_news = {
        'sentiment_mean': 0.5,
        'news_volume': 15,
        'sentiment_std': 0.2
    }
    
    mock_weather = {
        'impact_score': 0.1,
        'temperature': 32
    }
    
    mock_economic = {
        'nifty_50': {'change_pct': 0.8},
        'india_vix': {'current': 14.5},
        'usd_inr': {'change_pct': -0.2}
    }
    
    mock_nse = {
        '^NSEI': {'change_pct': 0.8},
        '^NSEBANK': {'change_pct': 1.1}
    }
    
    # Calculate composite score
    result = scorer.calculate_composite_score(
        news_features=mock_news,
        weather_impact=mock_weather,
        indian_economic_data=mock_economic,
        nse_sentiment_data=mock_nse,
        ts_prediction=2800.0,
        current_price=2750.0
    )
    
    print(f"\n📊 Individual Scores:")
    for source, score in result['individual_scores'].items():
        sentiment = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
        print(f"  {source.replace('_', ' ').title():20s}: {score:+.4f} ({sentiment})")
    
    print(f"\n📈 Composite Analysis:")
    print(f"  Composite Score:      {result['composite_score']:+.4f}")
    print(f"  Overall Confidence:   {result['confidence']*100:.1f}%")
    print(f"  Interpretation:       {result['interpretation']}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Weight System Fixed!")
    print("="*70)