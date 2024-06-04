"""
Model definition.
"""

import pickle
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras import utils
from sklearn.model_selection import train_test_split
import yaml
from dvclive import Live

SEED = 42

def create_model(seed = SEED, sample_percentage = 1):
    """
    Creates and trains a Keras CNN model
    """
    utils.set_random_seed = seed

    with open('./data/processed/char_index.pkl', 'rb') as file:
        char_index = pickle.load(file)

    with open('./params.yml', encoding='utf-8') as file:
        params = yaml.load(file, yaml.Loader)

    with open('./data/processed/x_train.pkl', 'rb') as file:
        x_train = pickle.load(file)

    with open('./data/processed/y_train.pkl', 'rb') as file:
        y_train = pickle.load(file)

    with open('./data/processed/x_val.pkl', 'rb') as file:
        x_val = pickle.load(file)

    with open('./data/processed/y_val.pkl', 'rb') as file:
        y_val = pickle.load(file)

    model = Sequential()
    voc_size = len(char_index.keys())
    print(f"voc_size: {voc_size}")
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    x_train_sampled, _, y_train_sampled, _ = train_test_split(x_train, y_train, test_size=1 - sample_percentage, stratify=y_train, random_state=SEED)
    x_val_sampled, _, y_val_sampled, _ = train_test_split(x_val, y_val, test_size=1 - sample_percentage, stratify=y_val, random_state=SEED)

    model.fit(
        x_train_sampled,
        y_train_sampled,
        batch_size=params['batch_train'],
        epochs=params['epoch'],
        shuffle=True,
        validation_data=(x_val_sampled, y_val_sampled)
    )

    return model

if __name__ == "__main__":
    trained_model = create_model()

    trained_model.save('./models/model.keras')

    with Live() as live:
        live.log_artifact(
            str("./models/model.keras"),
            type="model",
            name="phishing-detection"
        )
