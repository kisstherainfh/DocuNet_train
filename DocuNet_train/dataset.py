from torch.utils.data import Dataset
import cv2
import glob
import numpy as np


class MyDataset(Dataset):
    def __init__(self, mode):
        super(MyDataset, self).__init__()
        self.mode = mode
        self.image = glob.glob('../images/data_gen/image/' + self.mode + '/*.jpg')

    def __getitem__(self, idx):
        file_path = self.image[idx]
        label_name = file_path.split('/')[-1].replace('jpg', 'npz')
        label_path = '../images/data_gen/label/' + self.mode + label_name

        inputs_bgr = cv2.imread(self.image[idx])
        inputs = cv2.cvtColor(inputs_bgr, cv2.COLOR_BGR2RGB)

        label = np.load(label_path)
        label_x = label['x']
        label_y = label['y']

        return inputs, label_x, label_y

    def __len__(self):
        return len(self.image)
