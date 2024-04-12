import argparse
import json
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return adjustments(Image.open(image))
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

parser_parse=argparse.ArgumentParser(description='Train')
parser_parse.add_argument('input_image', default='flowers/test/1/image_06752.jpg' dest="input_image")
parser_parse.add_argument('checkpoint', default='checkpoint.pth' help="Used to store the checkpoint file")
parser_parse.add_argument('--top_k', default=5, dest="topk", action="store", type=int help="store the top k classes ")
parser_parse.add_argument('--category_names', default='cat_to_name.json',dest="category", action="store" help="category of the flowers")
parser_parse.add_argument('--gpu', default=False, action='store_true')


training_parser=parser_parse.parse_args()
image=training_parser.input_image
k=training_parser.topk
category_names=training_parser.category
hardware= "gpu" if parser.gpu else "cpu"
checkpoint=training_parser.checkpoint

train_loader,validation_loader,test_loader,train_datasets=load_data.Loaddata(data_dir)
//loading the checkpoint
architecture=checkpoint['architecture']
hidden_units = checkpoint['hidden_units']
dropout = checkpoint['dropout']
learning_rate=checkpoint['learning_rate']
model.load_state_dict(model_info['state_dict'])

predict(image,model,k)


