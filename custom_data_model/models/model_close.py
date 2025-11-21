# Import generic libraries
import pandas as pd
import numpy as np
# Import Tensorflow and Keras for AI model
import tensorflow as tf
import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, Layer, MultiHeadAttention, LayerNormalization, Conv1D, Input, Add, Dense, Dropout, Flatten, Concatenate, Multiply, Softmax
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.math import rsqrt, minimum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.backend import set_value
from tensorflow.signal import stft, hann_window
from tensorflow.keras import initializers

initializer_for_relu = initializers.HeNormal(seed=1) # For layers with activation function Relu
initializer_for_sigmoid = initializers.GlorotNormal(seed=1) # For layers with activation function Sigmoid

# Defign Model Architecture
# Custom Gated Attention Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class GatedAttention(Layer):
    def __init__(self, units, regularizer_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.softmax = Softmax(axis=-1)
        self.query_dense = Dense(units,   # Project input to queries
                               activation='tanh',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(regularizer_rate),
                               bias_regularizer=l2(regularizer_rate),
                               activity_regularizer=l2(regularizer_rate),
                               kernel_initializer=initializer_for_sigmoid)
        self.key_dense = Dense(units,   # Project input to keys
                               activation='tanh',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(regularizer_rate),
                               bias_regularizer=l2(regularizer_rate),
                               activity_regularizer=l2(regularizer_rate),
                               kernel_initializer=initializer_for_sigmoid)
        self.value_dense = Dense(units,   # Project input to Value
                               activation='tanh',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(regularizer_rate),
                               bias_regularizer=l2(regularizer_rate),
                               activity_regularizer=l2(regularizer_rate),
                               kernel_initializer=initializer_for_sigmoid)
        self.gate_dense = Dense(units, # Project input to gate values
                              activation='sigmoid',
                              bias_initializer='zeros',
                              kernel_regularizer=l2(regularizer_rate),
                              bias_regularizer=l2(regularizer_rate),
                              activity_regularizer=l2(regularizer_rate),
                              kernel_initializer=initializer_for_sigmoid)
        
        self.multiply = Multiply()
        self.layer_norm = LayerNormalization(beta_initializer='zeros', gamma_initializer='ones',
                                        beta_regularizer=l2(regularizer_rate), gamma_regularizer=l2(regularizer_rate))
        
        self.querry = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.key = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.value = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        
        self.attention_scores = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.attention_weights = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.attention_output = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        
        self.gate = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.gated_attention_output = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.gated_ln_output = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "units": self.units,
                "regularizer_rate": self.regularizer_rate,
            }
          )
        return config

  #def build(self, input_shape):
  #  # Define weights
  #  self.kernel = self.add_weight(
  #      shape=(self.units,),
  #      initializer=initializer_for_relu,
  #      trainable=True
  #  )

    def call(self, magnitude):
        # Compute attention scores
        
        self.querry = self.query_dense(magnitude) # magnitude
        self.key = self.key_dense(magnitude) # magnitude
        self.value = self.value_dense(magnitude) # magnitude
        
        self.attention_scores = tf.matmul(self.querry, self.key, transpose_b=True) # Query, Key
        self.attention_scores /= tf.math.sqrt(tf.cast(tf.shape(self.key)[-1], tf.float32)) # Key
        
        # Softmax over the scores
        self.attention_weights = self.softmax(self.attention_scores)
        
        # Compute the attention output
        self.attention_output = tf.matmul(self.attention_weights, self.value) #Value
        
        # Compute the gate
        self.gate = self.gate_dense(magnitude)
        
        # Apply the gate to the attention output
        self.gated_attention_output = self.multiply([self.attention_output, self.gate])
        self.gated_ln_output = self.layer_norm(self.gated_attention_output)
        
        return self.gated_ln_output #, self.attention_weights

# Pre-Attention block - Feed forward network
@keras.saving.register_keras_serializable(package="CustomLayers")
class PreAttFeedForward(Layer):  
    def __init__(self, units, regularizer_rate, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)        
        self.units = units
        self.dropout_rate = dropout_rate
        self.regularizer_rate = regularizer_rate
        
        self.seq_sigmoid = Sequential([
            Dense(tf.get_static_value(tf.cast(units, dtype=tf.int32)),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='relu', kernel_initializer=initializer_for_relu),
            Dropout(dropout_rate),
            
            Dense(tf.get_static_value(tf.cast(units, dtype=tf.int32)),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='relu',
                  kernel_initializer=initializer_for_relu),
            Dropout(dropout_rate),
            
            Dense(units,
                  kernel_initializer=initializer_for_sigmoid,
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='sigmoid'
                  ) # NO Activation Function, to predict linear values as given in original paper
          ])
        
        self.seq_relu = Sequential([
            Dense(tf.get_static_value(tf.cast(units, dtype=tf.int32)),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='relu', kernel_initializer=initializer_for_relu),
            Dropout(dropout_rate),
            
            Dense(tf.get_static_value(tf.cast(units, dtype=tf.int32)),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='relu',
                  kernel_initializer=initializer_for_relu),
            Dropout(dropout_rate),
            
            Dense(units,
                  kernel_initializer=initializer_for_sigmoid,
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='relu'
                  ) # NO Activation Function, to predict linear values as given in original paper
          ])
        self.multiply = Multiply()
        self.dropout_layer = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(beta_initializer='zeros', gamma_initializer='ones',
                                            beta_regularizer=l2(regularizer_rate), gamma_regularizer=l2(regularizer_rate))
        
        self.gate_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.ln_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "units": self.units,
                "regularizer_rate": self.regularizer_rate,
            }
          )
        return config


    @classmethod
    def from_config(cls, config):
        config["seq_sigmoid"] = keras.layers.deserialize(config["seq_sigmoid"])
        config["seq_relu"] = keras.layers.deserialize(config["seq_relu"])
        return cls(**config)


    #def build(self, input_shape):
    #  # Define weights
    #  self.kernel = self.add_weight(
    #      shape=(self.feature_length,),
    #      initializer=initializer_for_relu,
    #      trainable=True
    #  )

    def call(self, x):
        # Gated attention
        self.gate_out = self.multiply([self.seq_relu(x), self.seq_sigmoid(x)])
        self.ln_out = self.layer_norm(self.gate_out)
        #x = tf.expand_dims(x, 1) # (batch_size, 1, feature_length)
        return self.ln_out