import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


model=tf.keras.models.load_model('my_model')

foto=image.load_img('hh.jpg',target_size=(150,150))
plt.imshow(foto)
plt.show()

x = image.img_to_array(foto)

x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0.5:
    print( "Ryu")
else:
    print("Ken")