import numpy as np

import tensorflow as tf

from recordingHelper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

# !! Modify this in the correct order
commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

tf.config.run_functions_eagerly(True)
model = tf.saved_model.load("vanilla_model")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    while True:
        command = predict_mic()
        if command == "stop":
            terminate()
            break