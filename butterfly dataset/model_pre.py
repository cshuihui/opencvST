from keras.models import load_model
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle


def image_process(path):
    image = cv2.imread(path)
    # cv2.imshow('1', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)
    return image


model = load_model('butterfly.keras')

result = model.predict([image_process('./train/Image_2.jpg')])

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

pred = np.argmax(result, axis=1)
label = label_encoder.inverse_transform(pred)

print(label)