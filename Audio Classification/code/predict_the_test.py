from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import numpy as np 


model = load_model('test/audio_classification.h5')

# k = image.load_img('Spectrograms/storm_08.png')



class_labels = ['background', 'chainsaw', 'engine', 'storm']

k = Image.open('Spectrograms/storm_08.png')
k = k.resize((224, 224))

k = k.convert('RGB')


x_array = image.img_to_array(k)


x_array = np.expand_dims(x_array, axis=0)


y = model.predict(x_array)

for i, label in enumerate(class_labels):
    print(f'{label}: {y[0][i]}')