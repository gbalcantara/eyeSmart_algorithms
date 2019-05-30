"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data import DataSet
import os.path

data = DataSet()

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'method1actionmix.inception.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'mixtrain'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
        #class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'mixtest'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
        #class_mode='binary')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(len(data.classes), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        #optimizer=SGD(lr=0.0001, momentum=0.9),
        optimizer=Adam(lr=0.0001),
        loss='binary_crossentropy',
        #loss='binary_crossentropy',
        metrics=['categorical_accuracy'])

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='InceptionV3.png', show_shapes=True, show_layer_names=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        validation_data=validation_generator,
        validation_steps=100,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def main(weights_file):
    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        model = train_model(model, 100, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 100, generators,
                        [checkpointer, early_stopper, tensorboard])

    #score = model.evaluate(testX, testY, verbose=0)
    #print(score)
    #print(score[1])

    #y_pred = model.predict(testX)
    #acc = sum([np.argmax(testY[i])==np.argmax(y_pred[i]) for i in range(100)])/100
    #print(acc)

if __name__ == '__main__':
    weights_file = None
    main(weights_file)
