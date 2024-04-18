

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt



class DigitNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        super(DigitNet, self).__init__()

        self.input_size = input_size
        
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
    





class DigitClassifier:

    def __init__(self, model, train_loader, test_loader,  learning_rate):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.model = model

        pass

    def trainModel(self, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate)
        
        n_total_steps = len(self.train_loader)
        
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(self.device)
                labels = labels.to(self.device)
                
                # forward
                output = self.model(images)
                loss = criterion(output, labels)
                
                # backwards
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                
                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
    
    def testModel(self):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            
            for images, labels in self.test_loader:
                # again reshape data
                images = images.reshape(-1, 28*28).to(self.device)
                labels = labels.to(self.device)
                
                # get the outputs
                outputs = self.model(images)

                # make predictions with the outputs
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            # get the accuracy
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')