import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(96, 69)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(11, activation='softmax')
])

xdata = []
ydata = []
i=0

# for j in range(0,10):
while(i<=10):
    img = cv2.imread("imagedata/score"+str(i)+".png", cv2.IMREAD_GRAYSCALE)
    # ydatinit = [0,0,0,0,0,0,0,0,0,0]
    # ydatinit[i] = 0.999
    
    xdata.append(img/255)
    # ydata.append(ydatinit)
    ydata.append(i)
    
    # print(xdata[i].shape)
    
    i+=1

xdata = np.array(xdata)
# print(ydata)
ydata = np.array(ydata)

print(ydata)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(xdata, ydata, epochs=100)
model.save("woodcutter/digit_recognition_model.h5")

sample = cv2.imread("imagedata/score10.png", cv2.IMREAD_GRAYSCALE)
# print(sample[0].shape)

x = []
x.append(sample/255)
x = np.array(x)

result = np.argmax(model(x))
print(result)