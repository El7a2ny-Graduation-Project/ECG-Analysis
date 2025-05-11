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
    """Defines a shallower residual block with a squeeze-and-excitation module."""
    shortcut = x  # Skip connection
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", activation=None)(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding="same", activation=None)(x)
    
    x = squeeze_excite_block(x)  # Apply SE block

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding="same")(shortcut)
    
    x = layers.Add()([shortcut, x])  # Add skip connection
    return x

def build_se_resnet(input_shape, num_classes, depth=34):
    """Builds a shallower SE-ResNet model for ECG classification."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, strides=1, padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Define a shallower architecture based on ResNet-34
    layers_per_block = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],  # Reduced depth to speed up training
    }
    
    num_blocks = layers_per_block.get(depth, [3, 4, 6, 3])
    
    filters_list = [64, 128, 256, 512]
    for i, num_layers in enumerate(num_blocks):
        for _ in range(num_layers):
            x = resnet_block(x, filters_list[i])
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, x)