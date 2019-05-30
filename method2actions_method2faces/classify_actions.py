# USAGE
# python3 classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png
# python3 classify.py --image examples/

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="5actions_online.model",
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", default="5actions_online.pickle",
	help="path to label binarizer")
ap.add_argument("-i", "--image", default="actions",
	help="path to input image")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["image"]))
#print (imagePaths)

print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

j = 0

for (i, imagePath) in enumerate(imagePaths):
	# load the image
	print(imagePath)
	image = cv2.imread(imagePath)
	output = image.copy()
 
	# pre-process the image for classification
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network and the label
	# binarizer
	

	# classify the input image
	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	# we'll mark our prediction as "correct" of the input image filename
	# contains the predicted label text (obviously this makes the
	# assumption that you have named your testing image files this way)
	filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
	correct = "correct" if filename.rfind(label) != -1 else "incorrect"
	print(label)
	fol1, fn1, t1 = imagePath.partition('/')
	fol2, fn2, t2 = t1.partition('_')
	print(fol2)

	if (label == fol2):
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