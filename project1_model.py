import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# defining basic residual block
# do not modify
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# define the ResNet model here
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [1, 1, 1, 1])    # similar to ResNet9

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # define if GPU acceleration is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])

    transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    writer = SummaryWriter()
    steps = []
    lrs = []
    testlosses = []
    testaccs = []

    # model definition-ResNet
    net = project1_model()
    #print(net)
    print("total_count number of params: ", count_parameters(net))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    # SGD optimizer, momentum, weight decay
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.4, epochs=200, steps_per_epoch=len(train_loader), pct_start=0.2)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.4, mode='triangular2', step_size_up=19530)

    for epoch in range(200):
        # training
        print('\nEpoch #', epoch + 1)
        net.train()
        train_loss = 0
        correct_count = 0
        total_count = 0
        for batchI, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            outputs = net(input)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()
            steps.append(epoch)
            lrs.append(scheduler.get_lr()[0])

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_count += target.size(0)
            correct_count += predicted.eq(target).sum().item()

        accuracy = (100. * correct_count) / total_count
        avg_loss = train_loss / len(train_loader)
        writer.add_scalar('loss/train', avg_loss, epoch)
        writer.add_scalar('acc/train', accuracy, epoch)

        print('The training loss is ', avg_loss)
        print('Training accuracy is ', accuracy)

        # testing
        net.eval()
        test_loss = 0
        test_correct_count = 0
        test_total_count = 0
        with torch.no_grad():
            for batchI, (input, target) in enumerate(test_loader):
                input, target = input.to(device), target.to(device)
                outputs = net(input)
                loss = criterion(outputs, target)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total_count += target.size(0)
                test_correct_count += predicted.eq(target).sum().item()

            test_accuracy = (100. * test_correct_count) / test_total_count
            test_avg_loss = test_loss / len(test_loader)
            writer.add_scalar('loss/test', test_avg_loss, epoch)
            writer.add_scalar('acc/test', test_accuracy, epoch)

            print('The testing loss is .', test_avg_loss)
            print('Testing accuracy is: ', test_accuracy)

    model_path = './project1_model.pt'
    torch.save(net.state_dict(), model_path)

    plt.figure()
    plt.legend()
    plt.plot(steps, lrs, label='OneCycle')
    plt.savefig("learning rate.png")

if __name__ == '__main__':
    main()