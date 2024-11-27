import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.onnx


# 定义训练集的数据转换，将图像转换为张量，并进行归一化
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 定义测试集的数据转换（与训练集类似）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 加载MNIST训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

# 加载MNIST测试集
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        test_loss, accuracy = test(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
        print(
            f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    # 保存训练好的模型
    torch.save(model.state_dict(), './save_model/lenet_model.pth')

    # 保存训练好的模型为ONNX格式
    # 首先需要一个示例输入数据，用于指定模型输入的形状等信息
    example_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(model,
                      example_input,
                      "./save_model/lenet_model.onnx",
                      verbose=True,  # 可以设置为True查看详细的转换过程信息
                      input_names=['input'],
                      output_names=['output'])

    # 绘制训练和测试损失曲线以及测试准确率曲线（可选，用于可视化训练效果）
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Testing Metrics')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
