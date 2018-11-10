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

data_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
norm_transform = transforms.Normalize(mean=mean, std=dev)

train_data = OutfitData(data_file=datapath+'train.json', transform = data_transform, norm_transform = norm_transform)
validate_data = OutfitData(data_file=datapath+'validate.json', transform = data_transform)#, norm_transform = norm_transform)
print("Number of training samples: ", train_data.__len__())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(validate_data, batch_size=5, shuffle=True, num_workers=4)

model = ClassificationNN()
solver = Solver(optim_args={"lr": 5e-6, "weight_decay": 0.004})
solver.train(model, train_loader, val_loader, log_nth=0, num_epochs=1)

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

test_data = OutfitData(data_file=datapath+'test.json', norm_transform=norm_transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4)

# prepare a list to store the test results and set the model into the test-mode
test_results= []
model.eval()

for inputs, id in test_loader:
    img = Variable(img)
    if model.is_cuda:
        img = img.cuda()
    
    prob = torch.nn.Softmax()
    outputs = prob(model.forward(img))
    outputs = outputs.data.cpu().numpy()
    
    test_results.append((id[0], outputs[0,1]))
    print(test_results[-1])

# a separate loader to output the wrong predictions of the trained model
val_check_loader = torch.utils.data.DataLoader(validate_data, batch_size=1, shuffle=False, num_workers=4) 
val_wrong = []
for id, (img, target) in enumerate(val_check_loader):
    img = Variable(img)
    
    print(model.is_cuda)
    if model.is_cuda:
        img = img.cuda()
        
    prob = torch.nn.Softmax()
    outputs = prob(model.forward(img))
    outputs = outputs.data.cpu().numpy()
    
    print(outputs)
    if ((outputs[0,1] > 0.5 and target[0] == 0) or (outputs[0,1] < 0.5 and target[0] == 1)):
        val_wrong.append((validate_data.data[id]["id"], target[0]))

model.train()

csvfile = "../datasets/output.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output)
    writer.writerows(test_results)

csvfile = "../datasets/wrong_predictions.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output)
    writer.writerows(val_wrong)
