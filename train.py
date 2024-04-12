import argparse
from load_data import Loaddata
from architecture_model import Model_architecture
from save_and_test import  


parser_parse=argparse.ArgumentParser(description='Train')

//Adding the necessary arguments
parser_parse.add_argument('data_directory' default="flowers/")//by default the directory is flowers
//training and architecture related information
parser_parse.add_argument('--epochs' action ="store", type=int,default=15 dest="epochs")//saving the epochs to run
parser_parse.add_argument('--architecture' action="store",type=str,default="vgg16"dest="architecture")//by default the architecture is vgg16
parser_parse.add_argument('--hidden_units' action="store",type=int,default=100,dest="hidden_units" )
//model related information
parser_parse.add_argument('--gpu' action="store_true",default=False)
parser_parse.add_argument('--learning_rate'action="store",default=0.0025, dest="learning_rate")
parser_parse.add_argument('--dropout'action="store", default=0.5, dest="dropout")
parser_parse.add_argument('--save_directory' action="store" ,default="./checkpoint.pth",dest="save_directory")


training_parser=parser_parse.parse_args()

data_directory=training_parser.data_directory
epochs=training_parser.epochs
architecture=training_parser.architecture
hidden_units=training_parser.hidden_units

hardware= "gpu" if training_parser.gpu else "cpu"
model_learning_rate=training_parser.learning_rate
dropout=training_parser.dropout
path=training_parser.save_directory

//Loading Dataset
train_loader,validation_loader,test_loader,train_datasets=load_data.Loaddata(data_dir)

//Defining Architecture of the model
model,criterion,optimizer=architecture_model.Architecturemodel(architecture,hardware,dropout,hidden_units,model_learning_rate,epochs,train_loader,validation_loader)

//saving model and creating a file to save_checkpoint
model.class_to_idx = train_datasets.class_to_idx
checkpoint={
    'architecture':architecture,
    'dropout':dropout,
    'state_dict':model.state_dict
    'hidden_units':hidden_units,
    'learning_rate': learning_rate,
    'epochs':epochs,
    'optimizer_state_dict': optimizer.state_dict(),
}

torch.save(checkpoint, 'checkpoint.pth')
model_info = torch.load(path)
dropout=0.5
model = models.vgg19(pretrained=True)
model = model_base_setup(dropout)
model.load_state_dict(model_info['model_state_dict'])
model, model_info = load_checkpoint('checkpoint.pth')


