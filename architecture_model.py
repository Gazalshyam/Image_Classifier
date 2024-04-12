import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
# model classifier for the model

class Architecturemodel(object):

    def model_architecure(self='vgg16',hardware='gpu',dropout=0.5,hidden_units=100,learning_rate=0.0025, epochs, train_loader,validation_loader):
        
        architecture=self

        if architecture.startswith('vgg'):
            input_size = {
            'vgg16': 25088,
            'vgg19': 25088
            }[architecture]
        elif architecture == 'alexnet':
            input_size = 9216
        elif architecture == 'inception':
            input_size = 2048
        elif architecture.startswith('resnet'):
            input_size = 512
        else:
            raise ValueError("Architecture not supported")
        
        if architecture.startswith('vgg'):
            model = models.vgg16(pretrained=True) if architecture == 'vgg16' else models.vgg19(pretrained=True)
        elif architecture == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif architecture == 'inception':
            model = models.inception_v3(pretrained=True)
        elif architecture.startswith('resnet'):
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError("Architecture not supported")
         
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 90),
                nn.ReLU(),
                nn.Linear(90, 80),
                nn.ReLU(),
                nn.Linear(80, 102),
                nn.LogSoftmax(dim=1)
            )
        
        model.classifier = classifier

        # Choose the architecture
        # architecture = 'vgg16'
        # Assuming other necessary variables are defined
        # Load pre-trained model
       # Freeze parameters so we don't backprop through them
       # Create classifier

        # Define criterion and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        if torch.cuda.is_available() and hardware == 'gpu':
            model.cuda()
        
        for epoch in range(epochs):
            running_loss=0

        print_every=10

        for i(inputs_train,training_labels) in enumerate(train_loader):
            steps+=1
            inputs_train=inputs_train.to(device)
            training_labels=training_labels.to(device)

            optimizer.zero_grad()
            outputs=model.forward(inputs_train)

            loss=criterion(outputs,training_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps% print_every==0:
                model.eval()
                validation_loss=0
                running_loss=0
                accuracy=0

                for validation_inputs,validation_labels in validation_loader:
                    optimizer.zero.grad()
                     validation_inputs, validation_labels = validation_inputs.to(device), validation_labels.to(device)
                    model.to(device)

                    with torch.no_grad():    
                        outputs = model.forward(validation_inputs)
                        valid_loss = criterion(outputs, validation_labels)
                        ps = torch.exp(outputs).data 
                        equality = (validation_labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                valid_loss = valid_loss / len(validation_loaders)
                train_loss = running_loss / len(train_loader)

                training_loss.append(train_loss)
                validation_loss.append(valid_loss)

                accuracy = accuracy /len(validloaders)

                print("Epoch Running: {}/{}... ".format(epoch+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valid_loss),
                      "Accuracy: {:.4f}".format(accuracy))

        return model,criterion,optimizer
