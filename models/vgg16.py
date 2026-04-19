from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16(num_classes=10):
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model