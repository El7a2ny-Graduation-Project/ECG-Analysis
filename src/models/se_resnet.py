import tensorflow as tf
from tensorflow.keras import layers, models

def squeeze_excite_block(input_tensor, ratio=16):
    """Adds a Squeeze-and-Excitation block to the ResNet architecture."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, filters))(se)
    return layers.multiply([input_tensor, se])

def resnet_block(x, filters, kernel_size=3, stride=1):
    """Defines a deeper residual block with a squeeze-and-excitation module."""
    shortcut = x  # Skip connection
    
    # Pre-activation style (BN -> ReLU -> Conv)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", activation=None)(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding="same", activation=None)(x)
    
    x = squeeze_excite_block(x)  # Apply SE block

    # Ensure shortcut has the same shape as x (if necessary)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding="same")(shortcut)
    
    x = layers.Add()([shortcut, x])  # Add skip connection
    return x

def build_se_resnet(input_shape, num_classes):
    """Builds a deeper SE-ResNet model for ECG classification."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, strides=1, padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Deeper ResNet with more filters
    for filters in [64, 128, 256, 512, 1024]:
        x = resnet_block(x, filters)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, x)
