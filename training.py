
import argparse
from utils import Util

parser = argparse.ArgumentParser(description='Train')

ap.add_argument('data_dir',
                default="flowers/")
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=100)
ap.add_argument('--gpu', default=False, action='store_true')
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.0025)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

pa = ap.parse_args()

data_dir = pa.data_dir
path = pa.save_dir
learning_rate = pa.learning_rate
architecture = pa.arch
dropout = pa.dropout
hidden_units = pa.hidden_units
hardware = "gpu" if pa.gpu else "cpu"
epochs = pa.epochs
print_every = 10

print("Loading Dataset")
trainloaders, validationloaders, testloaders, train_dataset = Util.load_data(data_dir)

print("Setting up architecture")
model, criterion, optimizer = Util.model_setup(architecture, dropout, hidden_units, learning_rate, hardware)

print("Training model")
Util.train_network(trainloaders, validationloaders, model, criterion, optimizer, epochs, print_every, hardware)

print("Testing accuracy")
Util.test_accuracy(model, testloaders, hardware)

print("Saving model to disk")
Util.save_checkpoint(model, train_dataset.class_to_idx, path, architecture, hidden_units, dropout, learning_rate)

print("Model Done!")