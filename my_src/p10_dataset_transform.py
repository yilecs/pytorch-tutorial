import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='../data', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, transform=dataset_transform, download=True)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter(log_dir='./logs/p10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)
writer.close()