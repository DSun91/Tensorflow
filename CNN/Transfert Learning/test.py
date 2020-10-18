import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


model=tf.keras.models.load_model('my_model_k_r')

#scores=model.evaluate_generator(generator=validation_generator,steps=1038/20)
#print(scores)
foto=image.load_img('kk.png',target_size=(150,150))
plt.imshow(foto)
plt.show()

x = image.img_to_array(foto)

x = np.expand_dims(x/255, axis=0)

images = np.vstack([x])


classes = model.predict(images)

print(classes[0])
if classes[0]>0.5:
    print( "Dog")
    print('Ryu')
else:
    print("Cat")
    print('Ken')