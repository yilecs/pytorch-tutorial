from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = '../data/datasets/train'
    ants_label_dir = 'ants_image'
    bees_label_dir = 'bees_image'

    ants_dataset = MyDataset(root_dir, ants_label_dir)
    bees_dataset = MyDataset(root_dir, bees_label_dir)

    ant, ant_label = ants_dataset[0]
    bee, bee_label = bees_dataset[0]

    ant.show()
    print(ant_label)

    bee.show()
    print(bee_label)
