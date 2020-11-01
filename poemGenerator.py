import numpy as np

import tensorflow as tf

text = (open("sonnets.txt").read())
text=text.lower()

characters = sorted(list(set(text)))
print(characters)

indexCharDict = {n:char for n, char in enumerate(characters)}
charIndexDict = {char:n for n, char in enumerate(characters)}

xRaw = []
yRaw = []
length = len(text)
chunk = 100

for i in range(0, length-chunk):
    sequence = text[i:i + chunk]
    label =text[i + chunk]
    xRaw.append([charIndexDict[char] for char in sequence])
    yRaw.append(charIndexDict[label])

xInput = np.reshape(xRaw, (len(xInput), chunk, 1))
yInput = xInput / float(len(characters))
yInput = tf.keras.utils.to_categorical(yRaw)

model = tf.keras.Sequential([
                             tf.keras.layers.LSTM(700, input_shape=(xInput.shape[1], xInput.shape[2]), return_sequences=True),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.LSTM(700, return_sequences=True),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.LSTM(700),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(yInput.shape[1], activation='softmax')
                          
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.fit(xInput, yInput, epochs=100, batch_size=50)

model.save_weights('generator.h5')

model.load_weights('generator.h5')

stringMapped = xRaw[99]
generatedChars = [indexCharDict[value] for value in stringMapped]
for i in range(400):
    x = np.reshape(string_mapped,(1,len(stringMapped), 1))
    x = x / float(len(characters))

    predictionIndex = np.argmax(model.predict(x, verbose=0))
    seq = [indexCharDict[value] for value in stringMapped]
    generatedChars.append(indexCharDict[predictionIndex])

    stringMapped.append(predictionIndex)
    stringMapped = stringMapped[1:len(stringMapped)]

generatedText=""
for char in generatedChars:
    generatedText += char
print(generatedText)

