import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt

from digit_dataset import DigitData
from digit_model import DigitNet, DigitClassifier


# Hyperparameters

input_size = 784 #28x28 pixels in images
hidden_size = 500 #can try out different sizes
num_classes = 10 #the classes present in the data - check below
num_epochs = 2 #for example purposes
batch_size = 100 #say
leraning_rate = 0.001 #can tune later


def runModel():

    data = DigitData(root = "./data")
    train_loader = data.getTrain(batch_size=batch_size)
    test_loader = data.getTest(batch_size=batch_size)


    model = DigitNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    digit_classifier = DigitClassifier(model=model, train_loader=train_loader, 
                                    test_loader=test_loader, learning_rate=leraning_rate)

    digit_classifier.trainModel(num_epochs=num_epochs)
    digit_classifier.testModel()

if __name__ == "__main__":
    runModel()