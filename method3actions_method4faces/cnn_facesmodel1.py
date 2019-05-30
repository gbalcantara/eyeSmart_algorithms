from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam

classifier = Sequential()
classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Flatten())

classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(5, activation='softmax'))

INIT_LR = 0.0001
EPOCHS = 50
classifier.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
              metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
	'datasetfaces/cameratrain',
	target_size=(64,64),
	batch_size=32,
	class_mode='categorical')

test_set = test_datagen.flow_from_directory(
	'datasetfaces/cameratest',
	target_size=(64,64),
	batch_size=32,
	class_mode='categorical')

# print(training_set.shape)
#trainY = np.append(trainY,trainY,axis = 1)
# print(trainY.shape)

# print(test_set.shape)
#testY = np.append(testY,testY,axis = 1)
# print(testY.shape)

classifier.fit_generator(
	training_set,
	steps_per_epoch=EPOCHS,
	epochs=EPOCHS,
	validation_data=test_set,
	validation_steps=EPOCHS)

# serialize model to JSON
model_json = classifier.to_json()
with open("facesmodel1_camera.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("facesmodel1_camera.h5")
print("Saved model to disk")