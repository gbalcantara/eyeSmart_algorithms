import numpy as np 
import argparse
from imutils import paths
from keras.preprocessing import image
from keras.models import load_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["image"]))
#print (imagePaths)
model = load_model("faces.model")

for (i, imagePath) in enumerate(imagePaths):
	print(imagePath)
	test_image = image.load_img(imagePath,target_size = (64,64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = model.predict(test_image)
	print(result)
	if result[0][0] > 0.5:
		prediction = 'cdz'
		print(prediction)

	elif result[0][1] > 0.5:
		prediction = 'jom'
		print(prediction)

	elif result[0][2] > 0.5:
		prediction = 'mnill'
		print(prediction)

	elif result[0][3] > 0.5:
		prediction = 'mty'
		print(prediction)

	else:
		prediction = 'ntig'
		print(prediction)