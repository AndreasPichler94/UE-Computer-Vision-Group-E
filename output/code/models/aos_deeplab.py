import torch
import torchvision
import sys
sys.path.append("./utils")
from train import check_gpu_availability
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

class AosDeepLab(torch.nn.Module):
    def __init__(self, n_channels, n_classes, pixel_out = True):
        super(AosDeepLab, self).__init__()

        self.model_name = "Deeplab"

        self.pixel_out = pixel_out

        # self.deeplab = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        #     weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        # )
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        )
        # replacing first block to allow 10 channel input
        original_first_layer = self.deeplab.backbone.conv1
        original_weights = original_first_layer.weight.data
        mean_weights = original_weights.mean(dim=1, keepdim=True)
        new_weights = mean_weights.repeat(1, 10, 1, 1)

        new_first_layer = torch.nn.Conv2d(
            n_channels,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False if not original_first_layer.bias else True
        )

        new_first_layer.weight.data = new_weights

        if original_first_layer.bias is not None:
            new_first_layer.bias.data = original_first_layer.bias.data

        self.deeplab.backbone.conv1 = new_first_layer

        self.deeplab.classifier[4] = torch.nn.Conv2d(
            in_channels=self.deeplab.classifier[4].in_channels,
            out_channels=n_classes,
            kernel_size=self.deeplab.classifier[4].kernel_size,
            stride=self.deeplab.classifier[4].stride,
            padding=self.deeplab.classifier[4].padding,
            bias=True
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        if pixel_out:
            self.criterion = torch.nn.MSELoss()
        else:
            background_weight = 0.000281
            person_weight = 1.9997
            device = torch.device("cuda" if check_gpu_availability() else "cpu")
            class_weights = torch.FloatTensor([background_weight, person_weight]).to(device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.deeplab(x)
