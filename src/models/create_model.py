"""
Model definition.
"""

import pickle
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import yaml
from dvclive import Live

def create_model():
    with open('./data/processed/char_index.pkl', 'rb') as f:
        char_index = pickle.load(f)

    with open('./params.yml', encoding='utf-8') as f:
        params = yaml.load(f, yaml.Loader)
    
    with open('./data/processed/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    
    with open('./data/processed/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    with open('./data/processed/x_val.pkl', 'rb') as f:
        x_val = pickle.load(f)
        
    with open('./data/processed/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)

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


    hist = model.fit(x_train, y_train,
                    batch_size=params['batch_train'],
                    epochs=params['epoch'],
                    shuffle=True,
                    validation_data=(x_val, y_val)
                    )

    model.save('./models/model.keras')

    with Live() as live:
        live.log_artifact(
            str("./models/model.keras"),
            type="model",
            name="phishing-detection"
        )

if __name__ == "__main__":
    create_model()
