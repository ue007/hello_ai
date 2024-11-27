import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import onnx

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)  # Flatten the tensor
        x = self.classifier(x)
        return x

# 创建模型实例
model = AlexNet(num_classes=1000)

# 加载预训练的AlexNet模型（可选）
# model = torchvision.models.alexnet(pretrained=True)

# 测试模型在一些输入数据上的运行
dummy_input = torch.randn(1, 3, 224, 224)  # 输入尺寸：batch_size x channels x height x width

# 保存模型为ONNX格式
onnx_filename = "alexnet.onnx"
torch.onnx.export(model, dummy_input, onnx_filename, export_params=True, opset_version=11, do_constant_folding=True)

print(f"模型已保存为 ONNX 格式: {onnx_filename}")
