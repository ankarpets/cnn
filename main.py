#import the libraries
from logging import root
from tkinter.ttk import Label

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
from skimage.transform import resize
#import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

from keras.datasets import cifar10

def train_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(type(x_train))
    print(type(y_train))
    print(type(x_test))
    print(type(y_test))

    index = 10
    x_train[index]

    plt.imshow(x_train[index])
    plt.show()

    print('The image label is: ', y_train[index])

    #get the image classification
    classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #print the image class
    print('The image class is: ', classification[y_train[index][0]])

    #convert the labels into a set of 10 numbers to input into the neural network
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    #print the labels
    print(y_train_one_hot)

    #print the new label of the image/picture above
    print('The one label is: ', y_train_one_hot[index])

    #normalize the pixels to be values between 0 and 1
    x_train = x_train / 255
    x_test = x_test / 255

    #create the model architecture
    model = Sequential()

    #add the first layer (convolutional layer)
    model.add( Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))

    #add a pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))

    #add another convolutional layers
    model.add( Conv2D(32, (5,5), activation='relu'))

    #add another pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))

    #add a flattening layer
    model.add(Flatten())

    #add a layer with 1000 neurons
    model.add(Dense(1000, activation='relu'))

    #add a drop out layer
    model.add(Dropout(0.5))

    #add a layer with 500 neurons
    model.add(Dense(500, activation='relu'))

    #add a drop out layer
    model.add(Dropout(0.5))

    #add a layer with 250 neurons
    model.add(Dense(250, activation='relu'))

    #add a layer with 10 neurons
    model.add(Dense(10, activation='softmax'))

    #compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics=['accuracy']
                )
    #train the model
    hist = model.fit(x_train, y_train_one_hot,
                    batch_size = 256,
                    epochs = 10,
                    validation_split = 0.2)

    #evaluate the model using the test data set
    model.evaluate(x_test,y_test_one_hot)[1]
    #visualize the models accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    #visualize the models loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


    #To save this model
    model.save('my_model.h5')

#train_model()



def predict():
    # To load this model
    from keras.models import load_model
    model = load_model('my_model.h5')



    new_image = plt.imread(fname)
    plt.imshow(new_image)
    plt.show()

    #resize the image

    resized_image = resize(new_image, (32,32,3))
    plt.imshow(resized_image)
    plt.show()

    #get the models predictions
    predictions = model.predict(np.array([resized_image]))
    #show the predictions
    print(predictions)

    #sort the predictions from least to greatest
    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = predictions

    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    #show the sorted labels in order
    print(list_index)
    classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #print the first five most likely predictions
    for i in range (5):
        print(classification[list_index[i]], ':', round(predictions[0][list_index[i]]*100, 2), '%')




def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = askopenfilename(title='"pen')
    return filename


def open_img():
    # Select the Imagename  from a folder 
    x = openfilename()
    print(x)
    global fname
    fname = x
    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img 
    panel.image = img
    panel.grid(row=2)


root = Tk()
root.title("Image Loader")
# Set the resolution of window
# root.geometry("550x300 + 300 + 150")

# Allow Window to be resizable
root.resizable(width=True, height=True)

fname = ""

# Create a button and place it into the window using grid layout
btn1 = Button(root, text='open image', command=open_img).grid(
        row=1, columnspan=4)

btn12 = Button(root, text='make a prediction', command=predict).grid(
       row=3, columnspan=6)

btn3 = Button(root, text="Quit", command=root.destroy).grid(
        row=4, columnspan=8)


root.mainloop()




def main():
    print("it's a main function!")

if __name__ == "__main__":
    main()





