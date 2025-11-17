import numpy as np
import json
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class DataSourceWeights:
    """Configurable weights for different data sources - Indian market optimized"""
    news: float = 0.20
    twitter: float = 0.12
    reddit: float = 0.05  # Less relevant for Indian stocks
    moneycontrol: float = 0.15  # Indian financial portal
    economic_times: float = 0.15  # Major Indian business news
    weather: float = 0.03
    economic: float = 0.15  # Indian economic indicators
    nse_sentiment: float = 0.10  # NSE market sentiment
    timeseries: float = 0.05
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = sum([self.news, self.twitter, self.reddit, self.moneycontrol,
                     self.economic_times, self.weather, self.economic, 
                     self.nse_sentiment, self.timeseries])
        self.news /= total
        self.twitter /= total
        self.reddit /= total
        self.moneycontrol /= total
        self.economic_times /= total
        self.weather /= total
        self.economic /= total
        self.nse_sentiment /= total
        self.timeseries /= total


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
            ticker = yf.Ticker(index)
            hist = ticker.history(period='5d')
            
            if len(hist) >= 2:
                change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                indices_data[index] = {
                    'current': hist['Close'].iloc[-1],
                    'change_pct': change * 100,
                    'volume': hist['Volume'].iloc[-1]
                }
        
        return indices_data
    
    def fetch_indian_news(self, company_name: str, symbol: str):
        """
        Fetch news from Indian sources
        Sources: Economic Times, Moneycontrol, Business Standard, etc.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Using News API with Indian sources
            news_api_key = self.config.get('news_api_key')
            if news_api_key:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'{company_name} OR {symbol}',
                    'sources': 'the-times-of-india',  # Available free source
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 50,
                    'apiKey': news_api_key
                }
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json().get('articles', [])
        except Exception as e:
            print(f"Error fetching Indian news: {e}")
        
        return []
    
    def fetch_moneycontrol_sentiment(self, symbol: str):
        """
        Scrape sentiment from Moneycontrol (for demonstration)
        Note: Actual implementation would need proper scraping with respect to robots.txt
        """
        # Placeholder for Moneycontrol sentiment
        # In production, you'd implement proper web scraping or use their API if available
        return {
            'sentiment_score': 0.0,
            'analyst_rating': 'Hold',
            'target_price': 0.0
        }
    
    def fetch_economic_times_news(self, company_name: str):
        """
        Fetch news from Economic Times
        """
        # Placeholder - would implement RSS feed parsing or web scraping
        return []
    
    def get_indian_economic_indicators(self):
        """
        Fetch Indian economic indicators
        - Sensex/NIFTY performance
        - INR/USD exchange rate
        - India VIX (volatility index)
        - FII/DII data (if available)
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
    
    def fetch_twitter_india(self, symbol: str, company_name: str):
        """
        Fetch Twitter data with focus on Indian finance Twitter
        Include hashtags: #NSE #NIFTY50 #IndianStocks
        """
        # Similar to previous Twitter implementation but with Indian hashtags
        hashtags = f"#{symbol} OR #{company_name} OR #NSE OR #NIFTY50"
        return hashtags


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
        
        weather_data = {}
        for city in self.major_cities:
            try:
                url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'q': f"{city},IN",
                    'appid': self.api_key,
                    'units': 'metric'
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    weather_data[city] = response.json()
            except Exception as e:
                print(f"Error fetching weather for {city}: {e}")
        
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
    
    def score_news(self, news_features: Dict) -> float:
        """Score news sentiment"""
        if not news_features or news_features.get('news_volume', 0) == 0:
            return 0.0
        
        sentiment = news_features.get('sentiment_mean', 0)
        volume_factor = min(news_features.get('news_volume', 0) / 20, 1.0)
        confidence = 1 - min(news_features.get('sentiment_std', 0), 0.5) / 0.5
        
        score = sentiment * volume_factor * confidence
        return np.clip(score, -1.0, 1.0)
    
    def score_moneycontrol(self, mc_data: Dict) -> float:
        """Score Moneycontrol data"""
        if not mc_data:
            return 0.0
        
        sentiment = mc_data.get('sentiment_score', 0)
        rating = mc_data.get('analyst_rating', 'Hold')
        
        rating_score = {
            'Strong Buy': 1.0,
            'Buy': 0.6,
            'Hold': 0.0,
            'Sell': -0.6,
            'Strong Sell': -1.0
        }.get(rating, 0.0)
        
        combined = (sentiment * 0.6 + rating_score * 0.4)
        return np.clip(combined, -1.0, 1.0)
    
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
    
    def score_twitter(self, twitter_metrics: Dict) -> float:
        """Score Twitter sentiment"""
        if not twitter_metrics or twitter_metrics.get('volume', 0) == 0:
            return 0.0
        
        sentiment = twitter_metrics.get('sentiment_mean', 0)
        volume = twitter_metrics.get('volume', 0)
        
        volume_factor = min(volume / 50, 1.0)
        score = sentiment * volume_factor
        return np.clip(score, -1.0, 1.0)
    
    def calculate_composite_score(self, 
                                  news_features: Dict,
                                  moneycontrol_data: Dict,
                                  economic_times_data: Dict,
                                  twitter_metrics: Dict,
                                  weather_impact: Dict,
                                  indian_economic_data: Dict,
                                  nse_sentiment_data: Dict) -> Dict:
        """
        Calculate weighted composite score from all sources
        """
        # Individual scores
        scores = {
            'news': self.score_news(news_features),
            'moneycontrol': self.score_moneycontrol(moneycontrol_data),
            'economic_times': self.score_news(economic_times_data),  # Similar to news
            'twitter': self.score_twitter(twitter_metrics),
            'weather': self.score_weather(weather_impact),
            'indian_economic': self.score_indian_economic(indian_economic_data),
            'nse_sentiment': self.score_nse_sentiment(nse_sentiment_data)
        }
        
        # Calculate confidence for each source
        confidences = {
            'news': min(news_features.get('news_volume', 0) / 20, 1.0) if news_features else 0,
            'moneycontrol': 0.9 if moneycontrol_data else 0,
            'economic_times': min(economic_times_data.get('news_volume', 0) / 15, 1.0) if economic_times_data else 0,
            'twitter': min(twitter_metrics.get('volume', 0) / 50, 1.0) if twitter_metrics else 0,
            'weather': 0.8 if weather_impact else 0,
            'indian_economic': 1.0 if indian_economic_data else 0,
            'nse_sentiment': 1.0 if nse_sentiment_data else 0
        }
        
        # Weighted composite score
        composite_score = (
            scores['news'] * self.weights.news +
            scores['moneycontrol'] * self.weights.moneycontrol +
            scores['economic_times'] * self.weights.economic_times +
            scores['twitter'] * self.weights.twitter +
            scores['weather'] * self.weights.weather +
            scores['indian_economic'] * self.weights.economic +
            scores['nse_sentiment'] * self.weights.nse_sentiment
        )
        
        # Overall confidence
        overall_confidence = (
            confidences['news'] * self.weights.news +
            confidences['moneycontrol'] * self.weights.moneycontrol +
            confidences['economic_times'] * self.weights.economic_times +
            confidences['twitter'] * self.weights.twitter +
            confidences['weather'] * self.weights.weather +
            confidences['indian_economic'] * self.weights.economic +
            confidences['nse_sentiment'] * self.weights.nse_sentiment
        )
        
        return {
            'composite_score': composite_score,
            'overall_confidence': overall_confidence,
            'individual_scores': scores,
            'individual_confidences': confidences,
            'interpretation': self._interpret_score(composite_score)
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
            f.write(f"भारतीय शेयर बाजार पूर्वानुमान रिपोर्ट\n")
            f.write(f"INDIAN STOCK MARKET PREDICTION REPORT - {symbol}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S IST')}\n\n")
            
            # Summary Section
            f.write("="*70 + "\n")
            f.write("पूर्वानुमान सारांश / PREDICTION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Final Prediction:        ₹{prediction_result['final_prediction']:.2f}\n")
            f.write(f"Confidence Interval:     ₹{prediction_result['lower_bound']:.2f} - ₹{prediction_result['upper_bound']:.2f}\n")
            f.write(f"Time Series Prediction:  ₹{prediction_result['timeseries_prediction']:.2f}\n")
            f.write(f"Sentiment Adjustment:    {prediction_result['sentiment_adjustment_pct']:.2f}%\n")
            f.write(f"Overall Confidence:      {prediction_result['confidence']*100:.1f}%\n")
            f.write(f"Market Interpretation:   {prediction_result['interpretation']}\n\n")
            
            # Data Sources Breakdown
            f.write("="*70 + "\n")
            f.write("डेटा स्रोत विश्लेषण / DATA SOURCES BREAKDOWN\n")
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
        for source, score in prediction_result['breakdown'].items():
            weight = prediction_result.get('source_weights', {}).get(source, 0)
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
                moneycontrol_data: Dict,
                economic_times_data: Dict,
                twitter_metrics: Dict,
                weather_impact: Dict,
                indian_economic_data: Dict,
                nse_sentiment_data: Dict,
                symbol: str = None,
                generate_report: bool = True) -> Dict:
        """
        Generate prediction for NIFTY stock
        """
        # Get time series prediction
        ts_prediction = self.timeseries_model.predict(price_data)
        
        # Get composite sentiment score
        sentiment_analysis = self.scorer.calculate_composite_score(
            news_features, moneycontrol_data, economic_times_data,
            twitter_metrics, weather_impact, indian_economic_data,
            nse_sentiment_data
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
            'source_weights': asdict(self.scorer.weights)
        }
        
        # Generate reports
        if generate_report and symbol:
            raw_data = {
                'news': news_features or {},
                'moneycontrol': moneycontrol_data or {},
                'economic_times': economic_times_data or {},
                'twitter': twitter_metrics or {},
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


# Example usage for NIFTY stocks
if __name__ == "__main__":
    # Example: Reliance Industries (RELIANCE.NS)
    
    # Mock data
    news_features = {
        'sentiment_mean': 0.5,
        'news_volume': 12,
        'sentiment_std': 0.25
    }
    
    moneycontrol_data = {
        'sentiment_score': 0.6,
        'analyst_rating': 'Buy',
        'target_price': 2850
    }
    
    economic_times_data = {
        'sentiment_mean': 0.45,
        'news_volume': 8
    }
    
    twitter_metrics = {
        'sentiment_mean': 0.3,
        'volume': 35
    }
    
    weather_impact = {
        'impact_score': 0.1,
        'temperature': 32,
        'conditions': 'Clear',
        'humidity': 65,
        'is_monsoon_season': False
    }
    
    indian_economic_data = {
        'nifty_50': {'current': 19800, 'change_pct': 0.8},
        'india_vix': {'current': 14.5},
        'usd_inr': {'current': 83.25, 'change_pct': -0.2}
    }
    
    nse_sentiment_data = {
        '^NSEI': {'current': 19800, 'change_pct': 0.8, 'volume': 250000000},
        '^NSEBANK': {'current': 44500, 'change_pct': 1.1, 'volume': 180000000}
    }
    
    # Mock time series model
    class MockTimeSeriesModel:
        def predict(self, price_data):
            return 2780.50  # Mock prediction for Reliance
    
    # Initialize
    scorer = MultiSourceScorer()
    report_gen = ReportGenerator(output_dir="nifty_reports")
    predictor = NIFTYHybridPredictor(MockTimeSeriesModel(), scorer, report_gen)
    
    # Generate prediction
    result = predictor.predict(
        price_data=None,
        news_features=news_features,
        moneycontrol_data=moneycontrol_data,
        economic_times_data=economic_times_data,
        twitter_metrics=twitter_metrics,
        weather_impact=weather_impact,
        indian_economic_data=indian_economic_data,
        nse_sentiment_data=nse_sentiment_data,
        symbol="RELIANCE",
        generate_report=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print("भारतीय शेयर बाजार पूर्वानुमान / INDIAN STOCK PREDICTION")
    print("="*70)
    print(f"Symbol:                 RELIANCE.NS")
    print(f"Final Prediction:       ₹{result['final_prediction']:.2f}")
    print(f"Confidence Interval:    ₹{result['lower_bound']:.2f} - ₹{result['upper_bound']:.2f}")
    print(f"Composite Score:        {result['composite_score']:.3f}")
    print(f"Overall Confidence:     {result['confidence']*100:.1f}%")
    print(f"Interpretation:         {result['interpretation']}")
    print("\nव्यक्तिगत स्कोर / Individual Scores:")
    for source, score in result['breakdown'].items():
        source_display = source.replace('_', ' ').title()
        print(f"  {source_display.ljust(20)}: {score:+.4f}")
    print("="*70)