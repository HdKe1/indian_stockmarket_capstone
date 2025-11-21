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
# Fourier Transform Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class FourierTransform(Layer):
    def __init__(self, *, signal_length, frame_length, frame_step):
        super().__init__()
        self.signal_length = signal_length
        self.frame_length = frame_length
        self.frame_step = frame_step
    
        self.spectrogram = tf.Variable(trainable=True,
                                       initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.magnitude_x = tf.Variable(trainable=True,
                                       initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.angle_x = tf.Variable(trainable=True,
                                   initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
    
    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "signal_length": self.signal_length,
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
            }
          )
        return config

    def call(self, x):
        # Convert the waveform to a spectrogram via a STFT.
        self.spectrogram = tf.signal.stft(signals=x, frame_length=self.frame_length, frame_step=self.frame_step)
        self.magnitude_x = tf.math.abs(self.spectrogram)
        self.angle_x = tf.math.angle(self.spectrogram) # Disable it if using only magnitude as output
        return self.magnitude_x , self.angle_x

# Self-attention layer - Magnitude and Angle
@keras.saving.register_keras_serializable(package="CustomLayers")
class SelfAttention(Layer):
    def __init__(self, frequency_bins, time_bins, regularizer_rate, num_heads, dropout_rate=0.1, **kwargs):
        # Call the parent class (BaseAttention) constructor
        super().__init__(**kwargs)
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.regularizer_rate = regularizer_rate
        self.num_heads = num_heads
    
        self.mha = MultiHeadAttention(key_dim = frequency_bins,
                                      kernel_initializer = initializer_for_relu,
                                      num_heads = num_heads,
                                      dropout=dropout_rate,
                                      kernel_regularizer=l2(regularizer_rate),
                                      bias_regularizer=l2(regularizer_rate),
                                      activity_regularizer=l2(regularizer_rate)
                                      )
        self.layernorm = LayerNormalization(beta_initializer='zeros', gamma_initializer='ones',
                                          beta_regularizer=l2(regularizer_rate), gamma_regularizer=l2(regularizer_rate))
        self.add = Add()
    
        self.attn_output = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.add_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.norm_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "regularizer_rate": self.regularizer_rate,
                "num_heads": self.num_heads,
            }
          )
        return config

    def call(self, magnitude):
        magnitude = tf.ensure_shape(magnitude, [None, self.time_bins, self.frequency_bins])
        self.attn_output = self.mha( query=magnitude,  # The querys is what you're trying to find.
                                     key=magnitude,  # The keys what sort of information the dictionary has.
                                     value=magnitude # The value is that information.
                                     )
        self.add_out = self.add([magnitude, self.attn_output])
        self.norm_out = self.layernorm(self.add_out)
        return self.norm_out

# The global cross-attention layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class GlobalCrossAttention(Layer):
    def __init__(self, *, frequency_bins, time_bins, regularizer_rate, num_heads, dropout_rate=0.1, **kwargs):
        # Call the parent class (BaseAttention) constructor
        super().__init__(**kwargs)
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.regularizer_rate = regularizer_rate
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(key_dim = frequency_bins,
                              kernel_initializer = initializer_for_relu,
                              num_heads = num_heads,
                              dropout=dropout_rate,
                              kernel_regularizer=l2(regularizer_rate),
                              bias_regularizer=l2(regularizer_rate),
                              activity_regularizer=l2(regularizer_rate)
                              )
        self.layernorm = LayerNormalization(beta_initializer='zeros', gamma_initializer='ones',
                                        beta_regularizer=l2(regularizer_rate), gamma_regularizer=l2(regularizer_rate))
        self.add = Add()
        
        self.attn_output = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.add_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.norm_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
    
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "regularizer_rate": self.regularizer_rate,
                "num_heads": self.num_heads,
            }
          )
        return config
      
    def call(self, magnitude, angle):
        magnitude = tf.ensure_shape(angle, [None, self.time_bins, self.frequency_bins])
        angle = tf.ensure_shape(angle, [None, self.time_bins, self.frequency_bins])
        #attn_output, attn_scores = self.mha(
        self.attn_output = self.mha( query=magnitude,  # The querys is what you're trying to find.
                                     key=angle,  # The keys what sort of information the dictionary has.
                                     value=angle, # The value is that information.
                                     return_attention_scores=False
                                     )
    
        self.add_out = self.add([magnitude, self.attn_output])
        self.norm_out = self.layernorm(self.add_out)
        return self.norm_out

# Gated - Feed Forward Network Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class FeedForward(Layer):
    def __init__(self, *, frequency_bins, time_bins, regularizer_rate, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.seq = Sequential([
            Dense(tf.get_static_value(tf.cast(frequency_bins*time_bins, dtype=tf.int32)),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate),
                  activation='relu', kernel_initializer=initializer_for_relu),
            Dropout(dropout_rate),
            Dense(frequency_bins,
                  kernel_initializer=initializer_for_sigmoid,
                  bias_initializer='zeros',
                  kernel_regularizer=l2(regularizer_rate),
                  bias_regularizer=l2(regularizer_rate),
                  activity_regularizer=l2(regularizer_rate)
                  #activation='sigmoid' # NO Activation Function, to predict linear values as given in original paper
                  ) 
        ])
        self.add = Add()
        #self.multiply = Multiply()
        self.layer_norm = LayerNormalization(beta_initializer='zeros', gamma_initializer='ones',
                                            beta_regularizer=l2(regularizer_rate), gamma_regularizer=l2(regularizer_rate))
    
        self.ffn_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "regularizer_rate": self.regularizer_rate,
            }
          )
        return config

    def call(self, x):
        self.ffn_out = self.add([x, self.seq(x)])
        self.ffn_out = self.layer_norm(self.ffn_out)
        return self.ffn_out

# Complete Encoder Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class EncoderLayer(Layer):
    def __init__(self,*, frequency_bins, time_bins, num_heads, dropout_rate, regularizer_rate, **kwargs):
        super().__init__(**kwargs)
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.num_heads = num_heads
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.self_attention = SelfAttention(frequency_bins=frequency_bins,
                                            time_bins=time_bins,
                                            regularizer_rate=regularizer_rate,
                                            num_heads=num_heads,
                                            dropout_rate=dropout_rate
                                            )
    
        self.ffn = FeedForward(frequency_bins=frequency_bins, time_bins=time_bins, regularizer_rate=regularizer_rate, dropout_rate=dropout_rate)
    
        self.enc_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "num_heads": self.num_heads,
                "regularizer_rate": self.regularizer_rate,
                "dropout_rate": self.dropout_rate,
    
            }
          )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        config["self_attention"] = keras.layers.deserialize(config["self_attention"])
        config["ffn"] = keras.layers.deserialize(config["ffn"])
        return cls(**config)
      
    def call(self, angle):
        self.enc_out = self.self_attention(angle)
        self.enc_out = self.ffn(self.enc_out)
        return self.enc_out

@keras.saving.register_keras_serializable(package="CustomLayers")
class Encoder(Layer):
    def __init__(self, *, frequency_bins, time_bins, num_layers, num_heads, dropout_rate, regularizer_rate, **kwargs):
        super().__init__(**kwargs)
    
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.enc_layers = [ EncoderLayer(frequency_bins=frequency_bins,
                                         time_bins=time_bins,
                                         num_heads=num_heads,
                                         dropout_rate=dropout_rate,
                                         regularizer_rate=regularizer_rate
                                         ) for _ in range(num_layers)]
    
        self.enc_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "regularizer_rate": self.regularizer_rate,
                "dropout_rate": self.dropout_rate,
            }
          )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        config["enc_layers"] = keras.layers.deserialize(config["enc_layers"])
        return cls(**config)
      
    def call(self, angle):
        for i in range(self.num_layers):
            self.enc_out = self.enc_layers[i](angle)
        #self.last_attn_scores = self.enc_layers[-1].last_attn_scores
        return self.enc_out

# Complete Decoder Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class DecoderLayer(Layer):
    def __init__(self, *, frequency_bins, time_bins, num_heads, dropout_rate, regularizer_rate, **kwargs):
        super().__init__(**kwargs)
    
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.num_heads = num_heads
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.local_self_att = SelfAttention(
            frequency_bins=frequency_bins,
            time_bins=time_bins,
            regularizer_rate=regularizer_rate,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
    
        self.global_cross_att = GlobalCrossAttention(
            frequency_bins=frequency_bins,
            time_bins=time_bins,
            regularizer_rate=regularizer_rate,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
    
        self.ffn = FeedForward(frequency_bins=frequency_bins,
                               time_bins=time_bins,
                               regularizer_rate=regularizer_rate,
                               dropout_rate=dropout_rate)
    
        self.self_att_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.cross_att_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.fnn_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "num_heads": self.num_heads,
                "regularizer_rate": self.regularizer_rate,
                "dropout_rate": self.dropout_rate,
            }
          )
        return config

    @classmethod
    def from_config(cls, config):
        config["local_self_att"] = keras.layers.deserialize(config["local_self_att"])
        config["global_cross_att"] = keras.layers.deserialize(config["global_cross_att"])
        config["ffn"] = keras.layers.deserialize(config["ffn"])
        return cls(**config)
      
    def call(self, magnitude, angle):
        self.self_att_out = self.local_self_att(magnitude)
        self.cross_att_out = self.global_cross_att(magnitude=magnitude, angle=self.self_att_out)
        self.fnn_out = self.ffn(self.cross_att_out)
        return self.fnn_out

@keras.saving.register_keras_serializable(package="CustomLayers")
class Decoder(Layer):
    def __init__(self, *, frequency_bins, time_bins, num_heads, num_layers, regularizer_rate, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
    
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.decoder_layer = [ DecoderLayer(frequency_bins=frequency_bins,
                                            time_bins=time_bins,
                                            num_heads=num_heads,
                                            dropout_rate=dropout_rate,
                                            regularizer_rate=regularizer_rate
                                            ) for _ in range(num_layers)]
    
        self.decoder_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "regularizer_rate": self.regularizer_rate,
                "dropout_rate": self.dropout_rate,
            }
          )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        config["decoder_layer"] = keras.layers.deserialize(config["decoder_layer"])
        return cls(**config)
        
    def call(self, magnitude, angle):
        for i in range(self.num_layers):
            self.decoder_out  = self.decoder_layer[i](magnitude, angle)
        return self.decoder_out

# Transformer Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class TransformerLayer(Layer):
    def __init__(self, *, signal_length, frame_length, frame_step,
                   max_signal_length, min_signal_length,
                   frequency_bins, time_bins,
                   num_heads, num_layers, regularizer_rate, dropout_rate=0.1,
                   **kwargs):
        super().__init__(**kwargs)
    
        self.signal_length = signal_length
        self.frame_length = frame_length
        self.frame_step = frame_step
    
        self.max_signal_length = max_signal_length
        self.min_signal_length = min_signal_length
    
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
    
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.filter_size = tf.cast(tf.math.floor(max_signal_length/2), dtype=tf.int32)
    
        self.sft_layer = FourierTransform(signal_length=signal_length,
                                          frame_length=frame_length,
                                          frame_step=frame_step)
    
        self.encoder = Encoder(frequency_bins=frequency_bins,
                               time_bins=time_bins,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               dropout_rate=dropout_rate,
                               regularizer_rate=regularizer_rate)
    
        self.decoder = Decoder(frequency_bins=frequency_bins,
                               time_bins=time_bins,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               dropout_rate=dropout_rate,
                               regularizer_rate=regularizer_rate)
    
        self.flatten_layer = Flatten()
    
        self.angle_x = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.magnitude_x = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.enc_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.dec_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(2,),dtype=tf.float32))
        self.flat_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))


    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "signal_length": self.signal_length,
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
    
                "max_signal_length": self.max_signal_length,
                "min_signal_length": self.min_signal_length,
    
                "frequency_bins": self.frequency_bins,
                "time_bins": self.time_bins,
    
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "regularizer_rate": self.regularizer_rate,
                "dropout_rate": self.dropout_rate,
            }
          )
        return config


    @classmethod
    def from_config(cls, config):
        config["encoder"] = keras.layers.deserialize(config["encoder"])
        config["decoder"] = keras.layers.deserialize(config["decoder"])
        config["flatten_layer"] = keras.layers.deserialize(config["flatten_layer"])
        return cls(**config)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        self.magnitude_x, self.angle_x = self.sft_layer(inputs)
        self.enc_out = self.encoder(self.angle_x)  # (batch_size, time_bins, frequency_bins)
        self.dec_out = self.decoder(self.magnitude_x, self.enc_out)  # (batch_size, time_bins, frequency_bins)
        # Final linear layer output.
        self.flat_out = self.flatten_layer(self.dec_out) # (batch_size, time_bins * frequency_bins)
        # Return the output.
        return self.flat_out

# Custom Gated Attention Layer
@keras.saving.register_keras_serializable(package="CustomLayers")
class GatedAttention(Layer):
    def __init__(self, units, regularizer_rate=0.0001, **kwargs):
        super().__init__(**kwargs)
        
        self.units = units
        
        self.softmax = Softmax(axis=-1)
        self.query_dense = Dense(units,   # Project input to queries
                               activation='relu',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(regularizer_rate),
                               bias_regularizer=l2(regularizer_rate),
                               activity_regularizer=l2(regularizer_rate),
                               kernel_initializer=initializer_for_relu)
        self.key_dense = Dense(units,   # Project input to keys
                               activation='relu',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(regularizer_rate),
                               bias_regularizer=l2(regularizer_rate),
                               activity_regularizer=l2(regularizer_rate),
                               kernel_initializer=initializer_for_relu)
        self.value_dense = Dense(units,   # Project input to Value
                               activation='relu',
                               bias_initializer='zeros',
                               kernel_regularizer=l2(regularizer_rate),
                               bias_regularizer=l2(regularizer_rate),
                               activity_regularizer=l2(regularizer_rate),
                               kernel_initializer=initializer_for_relu)
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
    #    # Define weights
    #    self.kernel = self.add_weight(
    #        shape=(self.units,),
    #        initializer=initializer_for_relu,
    #        trainable=False
    #    )
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
        self.gate = tf.math.sigmoid(self.gate) #Value # Gate is a function of the queries
        
        # Apply the gate to the attention output
        self.gated_attention_output = self.attention_output * self.gate
        self.gated_ln_output = self.layer_norm(self.gate)
        
        return self.gated_ln_output

# Pre-Attention block - Feed forward network
@keras.saving.register_keras_serializable(package="CustomLayers")
class PreAttFeedForward(Layer):
    # Purpose of this layer is to reduce the output shape
    # Input = (batch_size, frequency_bins * time_bins * parallel layers)
    # output = (batch_size, 1, units)
    def __init__(self, units, regularizer_rate=0.0001, dropout_rate=0.1, **kwargs):
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
                  )
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
                  )
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
    #    # Define weights
    #    self.kernel = self.add_weight(
    #        shape=(self.units,),
    #        initializer=initializer_for_relu,
    #        trainable=False
    #    )
    
    def call(self, x):
        # Gated attention
        self.gate_out = self.multiply([self.seq_relu(x), self.seq_sigmoid(x)])
        self.ln_out = self.layer_norm(self.gate_out)
        return self.ln_out

# Inception Layer 
@keras.saving.register_keras_serializable(package="CustomLayers")
class Inception(Model):
    def __init__(self, *, signal_length, frame_length, frame_step,
                   max_signal_length, min_signal_length,
                   num_heads, num_layers, reduction_factor=8,
                   regularizer_rate, dropout_rate=0.1,
                   **kwargs):
        super().__init__(**kwargs)
    
        self.signal_length = signal_length
        self.frame_length = frame_length
        self.frame_step = frame_step
    
        self.max_signal_length = max_signal_length
        self.min_signal_length = min_signal_length
    
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.reduction_factor = reduction_factor
    
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
    
        self.iteration_len = tf.cast(tf.size(frame_length), dtype=tf.int32)
    
        self.time_bins = [tf.cast(((signal_length[i] - frame_length[i])/frame_step[i])+1 , dtype=tf.int32) for i in range(self.iteration_len)]
        self.frequency_bins = [tf.cast(tf.math.floor((frame_length[i]/2) +1), dtype=tf.int32) for i in range(self.iteration_len)]
        self.transformer_bins = tf.cast(tf.math.reduce_sum([self.time_bins[i] * self.frequency_bins[i] for i in range(self.iteration_len)])/self.reduction_factor, dtype=tf.int32)
    
    
        self.transformer = [ TransformerLayer(signal_length=signal_length[i], frame_length=frame_length[i], frame_step=frame_step[i],
                                              max_signal_length=max_signal_length, min_signal_length=min_signal_length,
                                              time_bins=tf.get_static_value(self.time_bins[i]),
                                              frequency_bins=tf.get_static_value(self.frequency_bins[i]),
                                              num_heads=num_heads[i], num_layers=num_layers[i], regularizer_rate=regularizer_rate,
                                              dropout_rate=dropout_rate
                                              ) for i in range(self.iteration_len)]
    
        self.pre_att_ffn = PreAttFeedForward(units=tf.get_static_value(self.transformer_bins), regularizer_rate=regularizer_rate, dropout_rate=dropout_rate)
        self.cust_att_layer = GatedAttention(units=tf.get_static_value(self.transformer_bins), regularizer_rate=regularizer_rate)
        self.final_layer = Dense(1) # STUPID !!!!!! - using activation='relu' will limit the output between 0 and infinity, it won't give -ve outputs
    
        self.sig_len = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.concat_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.transformer_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.preAtt_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.incAtt_out = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))
        self.logits = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal(seed=1)(shape=(1,),dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "signal_length" : self.signal_length,
                "frame_length" : self.frame_length,
                "frame_step" : self.frame_step,
    
                "max_signal_length" : self.max_signal_length,
                "min_signal_length" : self.min_signal_length,
    
                "num_heads" : self.num_heads,
                "num_layers" : self.num_layers,
                "regularizer_rate" : self.regularizer_rate,
                "dropout_rate" : self.dropout_rate,
              }
          )
        return config

    def call(self, inputs):
        for i in range(tf.get_static_value(self.iteration_len)):
            self.sig_len = tf.get_static_value(tf.cast(self.signal_length[i], dtype=tf.int32))
            inputs = inputs[:, -self.sig_len:]
            self.transformer_out = self.transformer[i](inputs)
            if i == 0:
                self.concat_out = self.transformer_out
            else:
                self.concat_out = tf.concat([self.concat_out, self.transformer_out], 1)

        # FFN to Reduce the dimentionality
        self.preAtt_out = self.pre_att_ffn(self.concat_out) # (batch_size, transformer_bins * iteration_len)
    
        # Inception Layer
        self.incAtt_out = self.cust_att_layer(self.preAtt_out)
        self.logits = self.final_layer(self.incAtt_out)  # (batch_size, target_len)
    
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del self.logits._keras_mask
        except AttributeError:
            pass
    
        # Return the final output and the attention weights.
        return self.logits
