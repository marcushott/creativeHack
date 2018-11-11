from classification_nn import ClassificationNN
from solver import Solver
from data_utils import OutfitData
from data_utils import compute_norm_params

from torchvision import transforms

import torch.nn.functional as F
import torch
from torch.autograd import Variable
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

datapath = "../datasets/"


# compute parameters for the normalization
mean, dev = compute_norm_params(data_file=datapath+'train.json')
print("Mean and standard deviation over all channels:")
print(mean)
print(dev)

# data augmentation pipeline
data_transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomRotation(degrees=(-10,10)),
         #transforms.CenterCrop(65),
         transforms.Resize((75, 75)),
         transforms.ToTensor(),
])

#data_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
norm_transform = transforms.Normalize(mean=mean, std=dev)

train_data = OutfitData(data_file=datapath+'train.json', transform = None, norm_transform = norm_transform)
validate_data = OutfitData(data_file=datapath+'validate.json', transform = None, norm_transform = norm_transform)
test_data = OutfitData(data_file=datapath+'test.json', transform = None, norm_transform = norm_transform)
print("Number of training samples: ", train_data.__len__())
print("Number of validation samples: ", validate_data.__len__())
print("Number of test samples: ", test_data.__len__())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(validate_data, batch_size=25, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

model = ClassificationNN()
if (len(sys.argv) > 1 and sys.argv[1] == "load"):
    model.load_state_dict(torch.load("../model/vgg13.ckpt"))
else:
    solver = Solver(optim_args={"lr": 1e-5, "weight_decay": 0.004})
    solver.train(model, train_loader, val_loader, log_nth=0, num_epochs=12)

    # standard plotting - loss and accuracy
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(solver.train_loss_history, 'o')
    plt.plot(solver.val_loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    fig.savefig("../Images/loss.png")

torch.save(model.state_dict(), "../model/vgg13.ckpt")

# prepare a list to store the test results and set the model into the test-mode
test_results= [("image","true_label","model_p","model_label")]
model.eval()

for id, (img, target) in enumerate(test_loader):
    img = Variable(img)
    if model.is_cuda:
        img = img.cuda()
    
    prob = torch.nn.Softmax()
    outputs = prob(model.forward(img))
    outputs = outputs.data.cpu().numpy()
    
    test_results.append((test_data.data[id]["img"], target[0], outputs[0,1], int(outputs[0,1]>=0.5)))
    print(test_results[-1])

# a separate loader to output the wrong predictions of the trained model
val_check_loader = torch.utils.data.DataLoader(validate_data, batch_size=1, shuffle=False, num_workers=4) 
val_wrong = [("image","true_label")]
for id, (img, target) in enumerate(val_check_loader):
    img = Variable(img)
    
    if model.is_cuda:
        img = img.cuda()
        
    prob = torch.nn.Softmax()
    outputs = prob(model.forward(img))
    outputs = outputs.data.cpu().numpy()
    
    if ((outputs[0,1] > 0.5 and target[0] == 0) or (outputs[0,1] < 0.5 and target[0] == 1)):
        val_wrong.append((validate_data.data[id]["img"], target[0]))

model.train()

csvfile = "../datasets/output.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output)
    writer.writerows(test_results)

csvfile = "../datasets/wrong_predictions.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output)
    writer.writerows(val_wrong)
