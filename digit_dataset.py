

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt





class DigitData:

    def __init__(self, root: str, download= False):
        
        self.train = torchvision.datasets.MNIST(root=root, train= True, 
                                           transform= transforms.ToTensor(),
                                           download= True)
        
        self.test = torchvision.datasets.MNIST(root=root, train= False,
                                          transform= transforms.ToTensor())
        


    def getTrain(self, batch_size: int, shuffle= False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset= self.train, batch_size=batch_size, shuffle= shuffle)
        

    def getTest(self, batch_size: int, shuffle= False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset= self.test, batch_size=batch_size, shuffle=shuffle)
    


    @staticmethod
    def dataHead(train_data: torch.utils.data.DataLoader, shape=False) -> tuple[torch.tensor]:
        examples = iter(train_data)
        samples, labels = next(examples)
        if shape:
            print(f"Samples shape: {samples.shape}, labels shape: {labels.shape}")
            
        return samples, labels
    

    def plotData(self) -> None:

        train_data = self.getTrain(batch_size= 100)
        samples = self.dataHead(train_data)

        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.imshow(samples[i][0], cmap='gray')
        
        plt.show()