"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model

def main():
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('data/checkpoints/method1actionmix.inception.017-0.07.hdf5')

    # Get all our test images.
    images = glob.glob(os.path.join('actions', '*.jpg'))
    nb_images=len(images)
    j = 0
    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        image = images[sample]

        # Turn the image into an array.
        print(image)
        fol1, fn1, t1 = image.partition('/')
        fol2, fn2, t2 = t1.partition('_')
        print(fol2)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)
        #print(predictions)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1
            if(class_prediction[0]==fol2):
                print("true")
                j += 1
                print(j)
            else:
                print("false")
            print(j)
    print(j)


            
        

if __name__ == '__main__':
    main()
