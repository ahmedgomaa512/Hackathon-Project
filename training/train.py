from models.simple_cnn import build_simple_cnn
from models.vgg16 import build_vgg16
from models.resnet50 import build_resnet50
from tensorflow.keras.optimizers import Adam

X_train, y_train = None, None  # replace with CIFAR-10 or your dataset

models_dict = {
    "simple": build_simple_cnn(),
    "vgg16": build_vgg16(),
    "resnet50": build_resnet50()
}

for name, model in models_dict.items():
    print("Training", name)
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save(name + "_model.h5")