from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

img_path = '../data/datasets/train/ants_image/0013035.jpg'
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor = tensor_trans(img)
print(tensor)