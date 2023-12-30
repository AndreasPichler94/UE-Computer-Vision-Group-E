import torch
import torchvision


class AosDeepLab(torch.nn.Module):
    def __init__(self):
        super(AosDeepLab, self).__init__()
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=True,
        )
        self.num_classes = 2
        original_first_layer = self.deeplab.backbone.conv1
        self.deeplab.backbone.conv1 = torch.nn.Conv2d(
            10,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False,
        )

        self.deeplab.classifier[4] = torch.nn.Conv2d(
            in_channels=self.deeplab.classifier[4].in_channels,
            out_channels=self.num_classes,
            kernel_size=self.deeplab.classifier[4].kernel_size,
            stride=self.deeplab.classifier[4].stride,
            padding=self.deeplab.classifier[4].padding,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.deeplab(x)
