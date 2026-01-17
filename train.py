from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset/train", help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")

ap.add_argument("-m", "--model", type=str, default="output/skin_Model.h5", help="path to output model")
args = vars(ap.parse_args())

data = []
labels = []

#intial training  rate, number of epo,
# batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

#loop hai image paths ke upper
for imagePath in imagePaths:
    
    label = imagePath.split(os.path.sep)[-2]

    #load image , aur usko resize ya fix krke aspest ratio me 224x224 
    
    image = cv2.imread(imagePath)

    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    #update the dataa and labels lists accordingly
    data.append(image)
    labels.append(label)

 # print(data)
 #  print(label) isko hataye bcz ye time bdha raha tha


data = np.array(data) / 255.0
labels = np.array(labels)

#labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


#traing the data set and keeping, 20% of the dataset for testing and rest 80% for training (told by kostav sir)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

#iniliaze the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=15, fill_mode="nearest")
#now creating the model (ye hua mera VGG19 ka)
baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
#--------------------------------

# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(5, activation="softmax")(headModel)  
# place kiye the head FC model on top of the base model (this will become the actual model will train)  

model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile of model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions onn the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label karta hai largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# serialize the model to disk
print("[INFO] saving skin disease detector model...")
model.save(args["model"], save_format="h5") 

   

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))


# this one is forr accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# it shows the confusion matrix, accuracy, sensitivity,specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))




N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

try:
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
except:
    print("Error Drawing val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

try:
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
except:
    print("Error Drawing val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])    