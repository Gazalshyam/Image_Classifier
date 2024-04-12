import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, trandormers,models

class Loaddata(object):

    @staticmethod
    def load_data(data_dir="./flowers"):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
    #: Define your transforms for the training, validation, and testing sets
        data_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
        train_transforms = transforms.Compose([
                              transforms.RandomRotation(25),
                              transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])
        test_transforms = transforms.Compose([
                              transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])
        validation_transforms=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
#  Load the datasets with ImageFolder
        image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
        train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
        test_datasets  = datasets.ImageFolder(test_dir, transform=test_transforms)

#  Using the image datasets and the trainforms, define the dataloaders
        dataloaders =  torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
        trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
        validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
        testloaders  = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

    return trainloaders , validloaders, testloaders, train_datasets
