from dfdetector import DFDetector
import os
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import _ranking
from sklearn.utils import multiclass
from sklearn.metrics._plot import precision_recall_curve
import metrics
import matplotlib.pyplot as plt
from pretrained_mods.mesonet import MesoInception4
from tqdm import tqdm
if __name__ == "__main__":
        #cnn lstm
    import torch
    import torch.nn as nn
    import torchvision.models as models

    # Imports
    import torch
    import torchvision
    import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
    import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
    import torch.nn.functional as F # All functions that don't have any parameters
    from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
    import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
    import torchvision.transforms as transforms # Transformations we can perform on our dataset

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 28
    hidden_size = 256
    num_layers = 2
    num_classes = 10 
    sequence_length = 28
    learning_rate = 0.005
    batch_size = 64
    num_epochs = 2


    # resnet + LSTM
    # efficientnetB1 + LSTM
    # rank90 solution 2x xception cnn method + effnetb1+lstm

    # adapted from https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2
    class ResNetLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(ResNetLSTM, self).__init__()
            self.resnet = models.resnet18(pretrained=False)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.resnet.fc = nn.Linear(in_features=512, out_features=input_size, bias=True)
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, 64)
            self.fc2 = nn.Linear(64,num_classes)

        def forward(self, x):
            # [32, 20, 3, 224, 224]
            #batch_size, num_frames, channels, height, width = x.size()
            batch_size, height, width = x.size()
            c_in = x.reshape(batch_size * 1, 1, height, width)
            c_out = self.resnet(c_in)
            r_in = c_out.view(batch_size, 1, -1)
            r_out, _ = self.lstm(r_in)
            out = self.fc1(r_out[:, -1, :]) # last hidden state
            result = self.fc2(out)
            return torch.sigmoid(result)

    # Load Data
    train_dataset = datasets.MNIST(root='dataset/', train=True, 
                                transform=transforms.ToTensor(), download=True)

    test_dataset = datasets.MNIST(root='dataset/', train=False, 
                                transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = ResNetLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()

    # Check accuracy on training & test to see how good our model
    def check_accuracy(loader, model):
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
            
        num_correct = 0
        num_samples = 0
        
        # Set model to eval
        model.eval()
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)
                
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            
            print(f'Got {num_correct} / {num_samples} with \
                accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
        # Set model back to train
        model.train()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
            
 

    # result = metrics.weighted_precision(y_true, y_pred)
    # print(result)
    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    #     dataset="uadfv", data_path="C:/Users/Chris/Desktop/fake_videos", method="mesonet",
    #     img_save_path="C:/Users/Chris/Desktop/fake_videos",epochs=1, batch_size=32, lr=0.001,folds=5,augmentation_strength="weak", fulltrain=False,faces_available=True,face_margin=0, seed=24)

    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    # dataset="celebdf", data_path='C:/Users/Chris/Desktop/Celeb-DF-v2', method="xception",
    # img_save_path="C:/Users/Chris/Desktop/Celeb-DF-v2",epochs=1, batch_size=32, lr=0.001, faces_available=False, face_margin=0)


# img save path and data path are the same -> redundancy can be removed!


#result = DFDetector.detect(video=video_file, method="xception", heatmap=False)

   # benchmark_result = DFDetector.benchmark(dataset="uadfv",data_path="C:/Users/Chris/Desktop/fake_videos", method="efficientnetb7")

# ->compare: whether to compare with the results of other methods that were precomputed by myself
