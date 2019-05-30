# USAGE
# python3 classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png
# python3 classify.py --image examples/

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="actions",
	help="path to input image")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["image"]))
#print (imagePaths)

# load json and create model
json_file = open('actionmodel1_5action_mix.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("actionmodel1_5action_mix.h5")
print("Loaded model from disk")

j = 0

for (i, imagePath) in enumerate(imagePaths):
	# load the image
	print(imagePath)
	test_image = image.load_img(imagePath, target_size = (64,64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = loaded_model.predict(test_image)
	
	print(result)
	
	maxElement = np.amax(result)
	#print (maxElement)
	result1 = np.where(result == np.amax(result))
	#print(result1)
	print(result1[1])

	# y_prob = loaded_model.predict(test_image)
	# y_classes = y_prob.argmax(axis=-1)
	# print(y_classes)
	#training_set.class_indices
	if result1[1] == 0:
		prediction = 'sitting'
	elif result1[1] == 1:
		prediction = 'sleeping'
	elif result1[1] == 2:
		prediction = 'standing'
	elif result1[1] == 3:
		prediction = 'waving'
	elif result1[1] == 4:
		prediction = 'writing'
	print(prediction)

	fol1, fn1, t1 = imagePath.partition('/')
	fol2, fn2, t2 = t1.partition('_')
	print(fol2)

	if (prediction == fol2):
		print("true")
		j += 1
	else:
		print("false")
	print(j)

	# build the label and draw the label on the image
	#label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
	#output = imutils.resize(output, width=400)
	#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		#0.7, (0, 255, 0), 2)

	# show the output image
	#print("[INFO] {}".format(label))
	#cv2.imshow("Output", output)
	#cv2.waitKey(0)