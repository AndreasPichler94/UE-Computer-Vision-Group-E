import torch
import torchvision
import sys
sys.path.append("./utils")
from train_deeplab import check_gpu_availability

class AosDeepLab(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AosDeepLab, self).__init__()

        self.model_name = "Deeplab"

        self.deeplab = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=True,
        )
        # replacing first block to allow 10 channel input
        first_layer_name, first_layer_module = next(iter(self.deeplab.backbone.named_children()))
        first_conv_layer = first_layer_module[0]

        new_first_conv = torch.nn.Conv2d(n_channels, 
                           first_conv_layer.out_channels, 
                           kernel_size=first_conv_layer.kernel_size, 
                           stride=first_conv_layer.stride, 
                           padding=first_conv_layer.padding, 
                           bias=False)

        new_first_block = torch.nn.Sequential(
            new_first_conv,
            first_layer_module[1],
            first_layer_module[2]
        )

        setattr(self.deeplab.backbone, first_layer_name, new_first_block)

        self.deeplab.classifier[4] = torch.nn.Conv2d(
            in_channels=self.deeplab.classifier[4].in_channels,
            out_channels=n_classes,
            kernel_size=self.deeplab.classifier[4].kernel_size,
            stride=self.deeplab.classifier[4].stride,
            padding=self.deeplab.classifier[4].padding,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        background_weight = 0.000281
        person_weight = 1.9997
        device = torch.device("cuda" if check_gpu_availability() else "cpu")
        class_weights = torch.FloatTensor([background_weight, person_weight]).to(device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.deeplab(x)
