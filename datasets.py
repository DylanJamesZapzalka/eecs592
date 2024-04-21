import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SkinCancerDataset(Dataset):

    def __init__(self, csv_df, include_extra=False, include_group=False, include_aux=False):
        """
        Initialize the datset.
        :param csv_df: Pandas dataframe with image paths and labels
        :param include_extra: Indicates if should return img path
        :param include_group: Indicates if should return a group number
        :param include_aux: Indicates if should return a an auxiliary label
        """
        self.csv_df = csv_df
        self.include_extra = include_extra
        self.include_group = include_group
        self.include_aux = include_aux

    def __len__(self):
        return len(self.csv_df)

    def get_group_info(self):
        n_groups = 4
        group_counts = []
        for i in range(n_groups):
            group_counts.append(len(self.csv_df[(self.csv_df['group'] == i)]))
        group_counts = torch.tensor(group_counts)

        dataset_info = {'n_groups': n_groups, 'group_counts': group_counts}
        return dataset_info

    def __getitem__(self, idx):
        # Load main image
        img_transform = transforms.Compose([transforms.PILToTensor()])
        img_path = self.csv_df.iloc[idx]['new_img_location']
        full_img = Image.open(img_path)
        full_img = img_transform(full_img).to(dtype=torch.float32) / 255.0
        
        # Obtain label
        label = int(self.csv_df.iloc[idx]['cancer'])
        label = torch.tensor(label)

        # Obtain group
        group = self.csv_df.iloc[idx]['group']
        group = torch.tensor(group)

        # Obtain auxiliary label
        aux_label = self.csv_df.iloc[idx]['aux_label']
        aux_label = torch.tensor(aux_label)

        # Obtain bird segmentation
        lesion_segmentation = Image.open(self.csv_df.iloc[idx]['mediator_img_location'])
        lesion_segmentation = img_transform(lesion_segmentation).to(dtype=torch.float32) / 255.0

        if self.include_extra:
            return (full_img, label, lesion_segmentation, img_path)
        elif self.include_group:
            return (full_img, label, group)
        elif self.include_aux:
            return (full_img, label, aux_label)
        else:
            return (full_img, label, lesion_segmentation)



class TeacherDataset(Dataset):
    """
    Dataset that contains the logits generated by a teacher model and the covariates
    used to generate the logits
    """

    def __init__(self, x, logits):
        """
        Initialize the datset.
        :parm x: Data used to make a prediction
        :param logits: Logits produced by a teacher model 
        :param dataset: Name of dataset being used
        """
        self.x = x
        self.logits = logits

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):

        img_transform = transforms.Compose([transforms.PILToTensor()])
        img_path = self.x[idx]
        full_img = Image.open(img_path)
        x = img_transform(full_img).to(dtype=torch.float32) / 255.0

        # Obtain logit
        logit = self.logits[idx]

        # Dummy label for compatibility
        dummy_label = torch.tensor(0)
        
        return (x, logit, dummy_label)