from keras_preprocessing import image
import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.models import load_model
from keras.optimizers import SGD

datagen_train = image.ImageDataGenerator(rescale=1./255, 
    rotation_range=10, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    fill_mode = "nearest",
    validation_split=0.20)

datagen_test = image.ImageDataGenerator(rescale=1./255)

traindf = pd.read_csv('trainLabels.csv')
testdf = pd.read_csv('testLabels.csv')

def append_text(fn):
	return str(fn)+'.BMP'

traindf['ID'] = traindf['ID'].apply(append_text)

training = datagen_train.flow_from_dataframe(dataframe=traindf,
	directory="./train/",
	x_col="ID",
	y_col="Class",
	batch_size = 28,
	target_size = (64,64),
	class_mode = "categorical",
	subset='training',
	seed=42,
	shuffle=True)

validation = datagen_train.flow_from_dataframe(dataframe=traindf,
	directory="./train/",
	x_col="ID",
	y_col="Class",
	batch_size = 28,
	target_size = (64,64),
	class_mode = "categorical",
	subset='validation',
	seed=42,
	shuffle=True)

testing = datagen_test.flow_from_dataframe(dataframe=testdf,
	directory="./test/",
	x_col="ID",
	y_col=None,
	target_size = (64,64),
	batch_size = 20,
	class_mode = None,
	seed=42,
	shuffle=False)

def conv2D_model(input_shape_value):
    
    model = Sequential()
    
    model.add(Conv2D(512,(3,3), input_shape = input_shape_value))
    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(2048))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(62, activation='softmax'))

    sgd = SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])

    STEP_SIZE_TRAIN = training.n // training.batch_size
    STEP_SIZE_VALID = validation.n // validation.batch_size

    model.fit_generator(generator=training,
    	steps_per_epoch=STEP_SIZE_TRAIN,
    	epochs=5,
    	validation_data=validation,
    	validation_steps=STEP_SIZE_VALID)
    
    evaluator = model.evaluate_generator(generator=validation,
    	steps=STEP_SIZE_VALID)
    
    return model
    
model = conv2D_model((64,64,3))

print('Saving the model...')

model.save('best_model.h5')

print('Model Succesfully Trained.')

testing.reset()

model = load_model('best_model.h5')

STEP_SIZE_TEST = testing.n // testing.batch_size
predictions = model.predict_generator(testing, steps=STEP_SIZE_TEST, verbose=1)

predicted_class_indices = np.argmax(predictions, axis = 1)

labels = training.class_indices
labels = {value:key for key, value in labels.items()}
predictions = [labels[value] for value in predicted_class_indices]

filenames = [filename[:-4] for filename in testing.filenames]
results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
results.to_csv("Results.csv", index=False)
print('Results Saved !!')