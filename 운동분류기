from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


model = load_model('keras_model.h5')


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open('C:/20608박준호/123.jpg')

size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)


image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array


prediction = model.predict(data)
print(prediction)

p=prediction[0]
print(p)

for i in range(len(p)):
    if max(p)==p[i]: break

f=open("labels.txt","r",encoding='utf8')
label=f.readlines()
print(f"판정결과: {label[i][2:-1]}, {round(p[i]*100,2)}%")
