# Import Generic Libraries
import pandas as pd
import numpy as np
import gradio as gr
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go
import mplfinance as mpf
import plotly.express as px
from plotly import subplots
import io
import os
import gc
from datetime import date, datetime, timedelta
# For AI Model
import tensorflow as tf
import keras
# Import Custom Classes
from models.model import (FourierTransform, SelfAttention, GlobalCrossAttention, FeedForward, EncoderLayer, Encoder, DecoderLayer, Decoder, TransformerLayer, GatedAttention, PreAttFeedForward, Inception)
#from models.model_close import GatedAttention as GatedAttention_close
#from models.model_close import PreAttFeedForward as PreAttFeedForward_close
# For NSE Data
#from nsepythonserver import equity_history # old
import yfinance as yf # new

# Global Constants
MODEL = {}
MODEL['open'] = "models/58Pct_Open_nsepy_inp128_out1_FourierTransform_withInception.keras"
MODEL['high'] = "models/62Pct_High_nsepy_inp128_out1_FourierTransform_withInception.keras"
MODEL['low'] = "models/62Pct_Low_nsepy_inp128_out1_FourierTransform_withInception.keras"
MODEL['close'] = "models/62Pct_Close_nsepy_inp128_out1_FourierTransform_withInception.keras"
REDUCTION_FACTOR = 8
SYMBOLS = ['BPCL','POWERGRID','NTPC','SUNPHARMA','TATACONSUM','ONGC','HINDALCO','ICICIBANK','SBIN','BHARTIARTL','WIPRO','ITC','AXISBANK','JSWSTEEL','COALINDIA','HDFCLIFE','TATAMOTORS']
SERIES = "EQ"
OHLC = ['open','high','low','close']
MAX_TIME_STEPS = 128
MAX_RANGE = 43
#START_DATE = datetime.strftime(datetime.now() - timedelta(203), '%d-%m-%Y') # for nsepythonserver
START_DATE = datetime.strftime(datetime.now() - timedelta(203), '%Y-%m-%d') # for yfinance
#yesterday = datetime.now() - timedelta(1)
#END_DATE = datetime.strftime(datetime.now() - timedelta(1), '%d-%m-%Y') # for nsepythonserver
END_DATE = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d') # for yfinance
#FUTURE_DATE = datetime.strftime(datetime.now(), '%d-%m-%Y') # for nsepythonserver
FUTURE_DATE = datetime.strftime(datetime.now(), '%Y-%m-%d') # for yfinance

# Data Preprocessing
class DataPreprocessing():
    def __init__(self, *, start_date, end_date, symbol, max_time_steps, max_range):
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.series = 'EQ'
        self.max_time_steps = max_time_steps
        self.max_range = max_range

    def data_preprocessing(self):
        # Load the dataset from nsepy
        #data_df = pd.DataFrame(equity_history(self.symbol,self.series,self.start_date,self.end_date))
        data_df = yf.download(tickers=f'{self.symbol}.NS',start=self.start_date,end=self.end_date)
        data_df.columns = data_df.columns.droplevel('Ticker')
    
        # Select desired column
        #data_df = data_df[['TIMESTAMP','mTIMESTAMP','CH_OPENING_PRICE','CH_TRADE_HIGH_PRICE',
        #                    'CH_TRADE_LOW_PRICE','CH_LAST_TRADED_PRICE']].sort_values(['TIMESTAMP'], ascending=[True]).copy()
        data_df = data_df[['Close','High','Low','Open']].sort_values(['Date'], ascending=[True]).copy()
        
        # Remove Duolicate
        data_df.drop_duplicates(ignore_index = False, inplace=True)
    
        # Set timestamp as index
        #data_df['mTIMESTAMP'] = pd.to_datetime(data_df.mTIMESTAMP)
        #data_df.set_index(data_df.mTIMESTAMP, verify_integrity=True, drop=True, inplace=True)

        # Drop unwanted columns
        #data_df.drop(labels=['TIMESTAMP','mTIMESTAMP'], axis=1, inplace=True)
    
        # Rename the columns
        #data_df.rename(columns={'CH_OPENING_PRICE':'open','CH_TRADE_HIGH_PRICE':'high',
        #                   'CH_TRADE_LOW_PRICE':'low','CH_LAST_TRADED_PRICE':'close'}, inplace=True)
        data_df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close'}, inplace=True)
        
        # Return the result
        return data_df


    def create_change_values(self, data_df):
        # Create past column
        req_cols = data_df.columns
        data_df_chg = data_df.copy()
    
        # Create past columns and change columns
        for col in req_cols:
          data_df_chg[str(col+'_chg')] = data_df[col] - data_df[col].shift(1)
    
        data_df_chg.drop(labels=['open', 'high', 'low',	'close'],axis=1, inplace=True)
    
        # Fill in null values
        data_df_chg.fillna(0.00,inplace=True)
    
        # return the result
        return data_df_chg

    def create_timeseries(self, data_df, ohlc_val):
        # Create future and target variavles
        sub_data_df = pd.DataFrame(data_df[str(ohlc_val+'_chg')].copy())
        sub_data_df.rename(columns={str(ohlc_val+'_chg'):0},inplace=True)
        
        shifted_columns = {}
        for i in np.arange(self.max_time_steps,0,-1):
            shifted_columns[i] = sub_data_df[0].shift(i)
        shifted_df = pd.DataFrame(shifted_columns)
        shifted_df = pd.concat([sub_data_df, shifted_df], axis=1)
    
        shifted_df.dropna(inplace=True)
    
        # Remove outliers
        reduced_matrix = []
        for i in shifted_df.values:
          reduced_arr = []
          for j in i:
            if (j >= self.max_range):
              reduced_arr.append(self.max_range)
            elif (j <= -self.max_range):
              reduced_arr.append(-self.max_range)
            else:
              reduced_arr.append(j)
          reduced_matrix.append(reduced_arr)
    
        # Convert matrix into dataframe
        reduced_data_df = pd.DataFrame(reduced_matrix)
        reduced_data_df = pd.concat([reduced_data_df, pd.Series(shifted_df.index)], axis=1, verify_integrity=True)
        #reduced_data_df.set_index('mTIMESTAMP', inplace=True)
        reduced_data_df.set_index('Date', inplace=True)
    
        # Return the result
        return reduced_data_df

# Data Generation Function
def create_dataset(ohlc, start_date, end_date, symbol, max_time_steps, max_range):
    # Create processed data
    # Defigning objects
    symbol_obj = DataPreprocessing(start_date=start_date, end_date=end_date, symbol=symbol, max_time_steps=max_time_steps, max_range=max_range)
    
    # Data Loading and Preprocessing
    processed_df = symbol_obj.data_preprocessing()
    
    # Creating Change Dataset
    change_df = symbol_obj.create_change_values(processed_df)
    
    # Creating TimeSeries
    timeseries_df_dict = {}
    for val in ohlc:
        # update it to dataframe
        timeseries_df_dict[val] = pd.DataFrame(symbol_obj.create_timeseries(change_df,val).values)
    
    return timeseries_df_dict, processed_df

# Load Models
custom_objects = {
    "FourierTransform": FourierTransform,
    "SelfAttention": SelfAttention,
    "GlobalCrossAttention": GlobalCrossAttention,
    "FeedForward": FeedForward,
    "EncoderLayer": EncoderLayer,
    "Encoder": Encoder,
    "DecoderLayer": DecoderLayer,
    "Decoder": Decoder,
    "TransformerLayer": TransformerLayer,
    "GatedAttention": GatedAttention,
    "PreAttFeedForward": PreAttFeedForward,
    "Inception": Inception,
}

# Load models
tf.keras.backend.clear_session()
reconstructed_model = {}
for val in OHLC:
    reconstructed_model[val] = keras.models.load_model(
        MODEL[val],
        custom_objects=custom_objects,
        compile=False  # Skip compiling to avoid optimizer issues
    )

# Make Predictions
def predict_the_future(stock):
    # Get Data
    input_df_dict, actual_df = create_dataset(ohlc=OHLC, start_date=START_DATE, end_date=END_DATE, symbol=stock, max_time_steps=MAX_TIME_STEPS, max_range=MAX_RANGE)
    
    # Make Predictions
    predictions = {}
    #diff_val = {}
    for val in OHLC:
        input_df = input_df_dict[val]
        input_val = input_df.tail(1).drop(input_df.columns[0], axis=1).copy().values
        pred_diff_val = reconstructed_model[val].predict(input_val).reshape(-1)
        actual_val = actual_df[val].tail(1).copy().values
        pred_val = actual_val + pred_diff_val
        #diff_val[val] = pred_diff_val
        predictions[val] = np.around(pred_val, decimals=2)
    
    # Format the OHLC
    predictions_list = [val for val in predictions.values()]
    predictions_list.sort(reverse=True)
    predictions['high']=predictions_list[0]
    predictions['open']=predictions_list[2]
    predictions['close']=predictions_list[1]
    predictions['low']=predictions_list[3]
    
    # Prepare the dataframes for plotting
    # Actual
    mc_hist = mpf.make_marketcolors(base_mpf_style='yahoo')
    actual_df['Type'] = [mc_hist for x in range(len(actual_df.index))] # for historical
    actual_df.index = pd.to_datetime(actual_df.index)
    # Prediction
    #predictions['mTIMESTAMP'] = FUTURE_DATE
    predictions['Date'] = FUTURE_DATE
    predictions_df = pd.DataFrame.from_dict(data=predictions)
    #predictions_df.set_index(predictions_df.mTIMESTAMP, verify_integrity=True, drop=True, inplace=True)
    predictions_df.set_index(predictions_df.Date, verify_integrity=True, drop=True, inplace=True)
    predictions_df.index = pd.to_datetime(predictions_df.index)
    mc_pred = mpf.make_marketcolors(base_mpf_style='blueskies')
    predictions_df['Type'] = [mc_pred for x in range(len(predictions_df.index))] # for forecasted
    # Combined
    combined_df = pd.concat([actual_df, predictions_df], axis=0, verify_integrity=True)
    combined_df.index = pd.to_datetime(combined_df.index)

    # Plot the combined candlestick chart
    mco = combined_df['Type'].values
    fig = mpf.plot(combined_df, type='candle', style='yahoo', marketcolor_overrides=mco, figscale=1, returnfig=True)

    return predictions['open'].item(), predictions['high'].item(), predictions['low'].item(), predictions['close'].item(), fig[0]

# Model Comparision
def model_comparisions(stock):
    # Get Data
    input_df_dict, actual_df = create_dataset(ohlc=OHLC, start_date=START_DATE, end_date=END_DATE, symbol=stock, max_time_steps=MAX_TIME_STEPS, max_range=MAX_RANGE)
    # Make Predictions
    selections = {}
    figs = {}
    for val in OHLC:
        # Extract the dataframe
        input_df = input_df_dict[val]
        input_val = input_df.drop(input_df.columns[0], axis=1).copy().values
        actual_df_val = actual_df[val].copy().to_frame()
        # Perform Predictions
        pred_diff_val = reconstructed_model[val].predict(input_val).reshape(-1)
        # derive the prediction values
        pred_val = actual_df_val.tail(pred_diff_val.shape[0]).values.reshape(-1) + pred_diff_val
        pred_val = [val.item() for val in pred_val]
        # Generate a dataframe to plot
        selected_df = actual_df_val.tail(pred_diff_val.shape[0]).copy()
        selected_df[val+'_pred'] = np.around(pred_val, decimals=2)
        selected_df.index = pd.to_datetime(selected_df.index)
        selected_df[val+'_MA5'] = selected_df[val].rolling(5).mean()
        selected_df[val+'_MA7'] = selected_df[val].rolling(7).mean()
        #selections[val] = selected_df
        # plot the candlesticks
        fig = go.Figure(data=[go.Scatter(x=selected_df.index.tolist(), y=selected_df[val], line=dict(color='green', width=1), name="Open Actual"),
                              go.Scatter(x=selected_df.index.tolist(), y=selected_df[val+'_pred'], line=dict(color='blue', width=1), name="Open Predicted"),
                              go.Scatter(x=selected_df.index.tolist(), y=selected_df[val+'_MA5'], line=dict(color='orange', width=1), name="Moving Averge 5 days"),
                              go.Scatter(x=selected_df.index.tolist(), y=selected_df[val+'_MA7'], line=dict(color='coral', width=1), name="Moving Averge 7 days")
                             ])
        fig.update_layout(title=f"Values and Predictions - {val} Dataset")
        figs[val] = fig
    
    return figs['open'], figs['high'], figs['low'], figs['close']

# Read the content of the README.md file
with open("about.md", "r") as file:
    about_lines = file.read()
    
# Display in GradIO
with gr.Blocks() as demo:
    gr.Markdown("# Indian Stock Market Prediction App")

    # About the App
    with gr.Tab("About the App"):
        gr.Markdown(about_lines)  
        
    # Market Prediction Tab
    with gr.Tab("Market Prediction"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, variant="panel"):
                # Stock Selection
                stock_input = gr.Dropdown(choices=SYMBOLS, label="Stock Picker:", info="Select the desired stock to predict future values!")
                # Submit the Stock name to find predictions
                predict_button = gr.Button("Make Prediction")
            with gr.Column(scale=1, variant="panel"):
                # Number field to display the results
                output_open = gr.Number(label="Predicted Value of open")
                output_high = gr.Number(label="Predicted Value of high")
                output_low = gr.Number(label="Predicted Value of low")
                output_close = gr.Number(label="Predicted Value of close")
        with gr.Row(variant="panel"):
            # Display the Charts
            graph_output_prediction = gr.Plot(label="Prediction Candle")
                                        
    # Set up the interface
    predict_button.click(predict_the_future, inputs=stock_input, outputs=[output_open, output_high, output_low, output_close, graph_output_prediction])

    # Comparing the models
    with gr.Tab("Model Comparision"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, variant="panel"):
                # Stock Selection
                stock_input = gr.Dropdown(choices=SYMBOLS, label="Stock Picker:", info="Select the desired stock to predict future values!")
            with gr.Column(scale=1, variant="panel"):
                # Submit the Stock name to find predictions
                evaluate_button = gr.Button("Evaluate Prediction")
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, variant="panel"):
                # Display the Charts
                graph_output_open = gr.Plot(label="Evaluations - Open")
            with gr.Column(scale=1, variant="panel"):
                # Display the Charts
                graph_output_high = gr.Plot(label="Evaluations - High")
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, variant="panel"):
                # Display the Charts
                graph_output_low = gr.Plot(label="Evaluations - Low")
            with gr.Column(scale=1, variant="panel"):
                # Display the Charts
                graph_output_close = gr.Plot(label="Evaluations - Close")
    # Set up the interface
    evaluate_button.click(model_comparisions, inputs=stock_input, outputs=[graph_output_open, graph_output_high, graph_output_low, graph_output_close])
        
if __name__ == "__main__":
    demo.launch()