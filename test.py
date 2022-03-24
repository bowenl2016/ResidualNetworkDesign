import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from project1_model import project1_model

print("This is the testing script")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = project1_model().to(device)
model_path = './project1_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

# Cifar10 tags
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("total_count number of params: ", count_parameters(model))

criterion = nn.CrossEntropyLoss().cuda()
# SGD optimizer, momentum, weight decay
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
steps = []
lrs = []

model.eval()
test_loss = 0
correct_count = 0
total_count = 0
with torch.no_grad():
    for batchI, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        outputs = model(input)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total_count += target.size(0)
        correct_count += predicted.eq(target).sum().item()

    print('The testing loss is .', test_loss/len(test_loader))
    print('Testing accuracy is: ' , (100. * correct_count) / total_count)