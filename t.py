
#! remove this:

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1) # 28x28 pixel images with 1 colour channel (grayscale)

# Load the data and split it between train and test sets
# x = images, y = labels
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255 # 0-255 to 0-1
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1) # -1 means last dimension
x_test = np.expand_dims(x_test, -1)

import random
noise_probability = 0.7

def corrupt_label(y, err):
    found = np.where(err == y)
    if len(found) > 0:
        # select an element at random (index != found)
        noisy_label = random.choice(err)
        while noisy_label == y:
            noisy_label = random.choice(err)
        return noisy_label
    return y

# We corrupt the MNIST data with some common mistakes, such as 3-->8, 8-->3, 1-->{4, 7}, 5-->6 etc.
def corrupt_labels(y_train, noise_probability):
    num_samples = y_train.shape[0]
    err_es_1 = np.array([0, 2, 3, 5, 6, 8, 9])
    err_es_2 = np.array([1, 4, 7])

    corruptions = {}
    corrupted_indexes = {}

    for i in range(num_samples):
        # generate a random number between 0 and 1:
        p = random.random()

        #! if p > noise_probability, then we do not corrupt the label?
        if p < noise_probability:
            y = y_train[i]
            y_noisy = corrupt_label(y, err_es_1)
            if y_noisy == y:
                y_noisy = corrupt_label(y, err_es_2)

            key = str(y_train[i]) + '->' + str(y_noisy)
            corrupted_indexes[i] = i

            if key in corruptions:
                corruptions[key] += 1
            else:
                corruptions[key] = 0

            y_train[i] = y_noisy

    return corruptions, corrupted_indexes

corruptions, corrupted_indexes = corrupt_labels(y_train, noise_probability)
print ("Corruptions: " + str(corruptions))
print ("Corrupted indexes: {}".format(list(corrupted_indexes.keys())[0:10]))

# convert class vectors to binary class matrices
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

loss_list = []
acc_list = []

batch_size = 128
epochs = 3 # very high epochs might overfit the model.
validation_split=0.1


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def prune_points(x_train, y_train, pruned_indexes):
    num_samples = x_train.shape[0] # = 60000
    x_train_pruned = []
    y_train_pruned = []
    for i in range(num_samples):
        if not i in pruned_indexes:
            x_train_pruned.append(x_train[i])
            y_train_pruned.append(y_train[i])

    return np.array(x_train_pruned), np.array(y_train_pruned)

def trainAndEvaluateModel(x_train, y_train, x_test, y_test, model, pruned_indexes):

    if not pruned_indexes == None:
        x_train_pruned, y_train_pruned = prune_points(x_train, y_train, pruned_indexes)
    else:
        x_train_pruned = x_train
        y_train_pruned = y_train

    # start training the model:
    model.fit(x_train_pruned, y_train_pruned, batch_size=batch_size, epochs=epochs)
    
    #! remove this:
    loss, accuracy = model.evaluate(x_test, y_test)
    loss_list.append(loss) #! remove this
    acc_list.append(accuracy) #! remove this

    keras.backend.clear_session() # remove previous training weights


def myPrunedSubsetMethod(x_train, y_train, model):
    pruned_indexes = {}

    # TODO: implement algorithm here


    pruned_indexes[0] = 0 # update dict
    
    return pruned_indexes

# print(f"pruned_indexes: {pruned_indexes}")
for i in range(2):
    print(f"\nRunning iteration {i}")
    pruned_indexes = myPrunedSubsetMethod(x_train, y_train, model)
    # print(f"Pruned indexes: {pruned_indexes}\n")
    trainAndEvaluateModel(x_train, y_train_onehot, x_test, y_test_onehot, model, pruned_indexes)

print (f"Loss: {loss_list}")
print (f"Accuracy: {acc_list}")

import matplotlib.pyplot as plt
plt.plot(acc_list)
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.title('Accuracy vs Iteration')
plt.show()
