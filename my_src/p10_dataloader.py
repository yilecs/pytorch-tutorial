import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中的第一个数据
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter(log_dir='../logs/dataloader')
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images('test_data_drop_last', imgs, step)
    step += 1
writer.close()