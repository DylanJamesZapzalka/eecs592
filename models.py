import torch
import torchvision


class BinaryPreTrainedNet(torch.nn.Module):

    def __init__(self, model_type):
        super(BinaryPreTrainedNet, self).__init__()
        self.activation = None
        if model_type == 'resnet':
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            self.model = torchvision.models.resnet18(weights=weights)
            self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
        else:
            raise ValueError('Model type ' + model_type + ' doesn\'t exist.')

        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

    def forward_logits(self, x):
        x = self.model(x)
        return x