import torch
import torch.nn as nn
import torchvision.models as models
import ssl


class ClassificationNN(nn.Module):

    def __init__(self, num_classes=2):
        super(ClassificationNN, self).__init__()
        # extract features with a pretrained model
        self.vgg = models.vgg19_bn(pretrained=True)
        # to freeze the model put requires_grad to False
        for param in self.vgg.parameters():
            param.requires_grad = True
        
        #for param in self.vgg.features.parameters():
        #    param.requires_grad = True
        
        # classifier on top
        self.classifier = nn.Sequential(
                    nn.Linear(512 * 3 * 3, 256),
                    nn.ReLU(True),
                    nn.Dropout(p=0.1),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(p=0.1),
                    nn.Linear(256, 2),
                )

    def forward(self, img):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.
        """ 
        img = self.vgg.features(img)
        print(img.shape)
        img = img.view(img.size(0), -1)
        #print()
        #img = self.vgg.classifier(img)
        img = self.classifier(img)
        return img
        #feat = self.classifier(feat)
        
        #return feat

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

