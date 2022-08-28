import torch
from torchvision import models

class MyResNetModel(torch.nn.Module):
    def __init__(self):
        super(MyResNetModel, self).__init__()
        image_modules = list(models.resnet50(pretrained=True).children())[:-1]  # all layer except last layer
        self.modelA = torch.nn.Sequential(*image_modules)

    def forward(self, image):
        x = self.modelA(image)
        return x


batch_size = 1
class resnet_only(torch.nn.Module):
    def __init__(self, num_frames):
        # TODO: This is a draft for the model; to be revised
        # Some of these dimensions are arbitrary/incorrect. Change as appropriate.
        super(resnet_only, self).__init__()

        self.num_frames = num_frames

        # -- ResNet feature --
        resnet_rnn_in_dim = 2048
        self.resnet_rnn_out_dim = 1024
        resnet_lin_in_dim = self.resnet_rnn_out_dim
        resnet_lin_out_dim = 512

        self.resnet_rnn = torch.nn.RNN(resnet_rnn_in_dim, self.resnet_rnn_out_dim, batch_first=True)
        self.resnet_lin = torch.nn.Linear(resnet_lin_in_dim, resnet_lin_out_dim)

        # -- Combined --

        #   -- First layer --
        first_cat_in_dim = resnet_lin_out_dim
        first_cat_out_dim = int(first_cat_in_dim / 2)

        self.first_cat_lin = torch.nn.Linear(first_cat_in_dim, first_cat_out_dim)

        #   -- Second Layer --
        second_cat_in_dim = first_cat_out_dim
        second_cat_out_dim = int(second_cat_in_dim / 2)

        self.second_cat_lin = torch.nn.Linear(second_cat_in_dim, second_cat_out_dim)

        #   -- Third layer --
        third_cat_in_dim = second_cat_out_dim
        third_cat_out_dim = int(third_cat_in_dim / 2)

        self.third_cat_lin = torch.nn.Linear(third_cat_in_dim, third_cat_out_dim)

        #   -- Final layer --
        final_in_dim = third_cat_out_dim
        final_out_dim = 1

        self.final_lin = torch.nn.Linear(final_in_dim, final_out_dim)

    def forward(self, resnet):
        # Simplified 'draft' version; edit; currently uses all relu activations
        # TODO: Regularization/various things are missing.

        resnet_rnn_out = self.resnet_rnn(resnet.view(batch_size, self.num_frames, 2048))
        resnet_lin_out = torch.relu(self.resnet_lin(resnet_rnn_out[1].view(batch_size, self.resnet_rnn_out_dim)))

        cat_layer = resnet_lin_out

        first_cat_out = torch.relu(self.first_cat_lin(cat_layer))
        second_cat_out = torch.relu(self.second_cat_lin(first_cat_out))
        third_cat_out = torch.relu(self.third_cat_lin(second_cat_out))
        final_out = self.final_lin(third_cat_out)

        return final_out

