import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pickle
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.norms import shape_norm, shape_grid_norm

    
class MNISTCustomTRNFS(Dataset):
    def __init__(self, files_df, base_path=None, 
                 return_size=None, rotation_range=None, normalize=True,
                 path_colname='path', label_colname='label', 
                 select_label=None, return_loc=False):
        """ MNIST: Per image transforms in the dataloader. 
        
        Do transform, adjustable padding to return_size, then normalize
        
        files_df: Pandas Dataframe containing the class and path of an image
        base_path: the path to append to the filename
        
        return_size:
        rotation_range: (list) range of degrees to rotate. if both equal, a fixed rotation.
        normalize: 
        path_colname: Name of column containing locations or filenames
        label_colname: Name of column containing labels
        select_label: if you only want one label returned
        return_loc: return location as well as the image and class
        """
        
        self.files = files_df
        self.base_path = base_path
        self.path_colname = path_colname
        self.label_colname = label_colname
        self.return_loc = return_loc
        self.return_size = return_size
        self.rotation_range = rotation_range
        self.normalize = normalize

        if isinstance(select_label, int):
            self.files = self.files.loc[self.files[self.label_colname] == int(select_label)]
            print(f'Creating dataloader with only label {select_label}')
            
    def pad_to_size(self, img, new_size):
        delta_width = new_size - img.size()[1]
        delta_height = new_size - img.size()[2]
        pad_width = delta_width //2
        pad_height = delta_height //2
        img = F.pad(img, (pad_height, pad_height, pad_width, pad_width), 'constant', 0)
        return img

    def __getitem__(self, index):
        if self.base_path:
            loc = str(self.base_path) +'/'+ self.files[self.path_colname].iloc[index]
        else:
            loc = self.files[self.path_colname].iloc[index]
        if self.label_colname:
            label = self.files[self.label_colname].iloc[index]
        else:
            label = 0
            
        img = Image.open(loc)
        if self.rotation_range:
            img = transforms.RandomRotation(self.rotation_range, expand=True)(img)
            
        img = transforms.ToTensor()(img)
        if self.return_size:
            img = self.pad_to_size(img, self.return_size)
        if self.normalize:
            img = transforms.Normalize((0.1307,), (0.3081,))(img)# this is wrong norm because imgs are padded

        if self.return_loc:
            return img, label, loc
        else:
            return img, label

    def __len__(self):
        return len(self.files)


def make_generators_MNIST_CTRNFS(files_dict_loc, batch_size, num_workers, 
                                 return_size=40, rotation_range=None, normalize=True,
                                 path_colname='path', label_colname='class', label=None, return_loc=False):
    with open(files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)

    folders = ['train', 'val']
    datasets = {}
    dataloaders = {}
    datasets = {x: MNISTCustomTRNFS(files_dict[x], base_path=None, 
                                       return_size=return_size, rotation_range=rotation_range, normalize=normalize,
                                       path_colname=path_colname, label_colname=label_colname, 
                                       select_label=label, return_loc=return_loc)
                for x in folders
               }

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                   for x in folders
                  }
    return dataloaders




#####.   $$$$$$$$.  OLD; delete




class FilesDFImageDataset(Dataset):
    def __init__(self, files_df, base_path=None, transforms=None, 
                 return_size=None, fixed_rotation=None,
                 path_colname='path', label_colname='label', 
                 select_label=None, return_loc=False, bw=False):
        """ 
        Dataset based pandas dataframe of locations & labels 
        Optionally filter by labels or return locations
        
        files_df: Pandas Dataframe containing the class and path of an image
        base_path: the path to append to the filename
        transforms: result of transforms.Compose()
        select_label: if you only want one label returned
        return_loc: return location as well as the image and class
        path_colname: Name of column containing locations or filenames
        label_colname: Name of column containing labels
        """
        
        self.files = files_df
        self.base_path = base_path
        self.transforms = transforms
        self.path_colname = path_colname
        self.label_colname = label_colname
        self.return_loc = return_loc
        self.bw = bw
#         self.return_size = return_size
        self.fixed_rotation = fixed_rotation

        if isinstance(select_label, int):
            self.files = self.files.loc[self.files[self.label_colname] == int(select_label)]
            print(f'Creating dataloader with only label {select_label}')
            
    def pad_to_size(self, img, new_size):
        delta_width = new_size - img.size()[1]
        delta_height = new_size - img.size()[2]
        pad_width = delta_width //2
        pad_height = delta_height //2
        img = F.pad(img, (pad_height, pad_height, pad_width, pad_width), 'constant', 0)
        return img

    def __getitem__(self, index):
        if self.base_path:
            loc = str(self.base_path) +'/'+ self.files[self.path_colname].iloc[index]
        else:
            loc = self.files[self.path_colname].iloc[index]
        if self.bw:
            img = Image.open(loc)
            if self.transforms is not None:
                img = self.transforms(img)
                if self.fixed_rotation:
                    img = TF.rotate(img, self.fixed_rotation)
#                 if self.return_size:
#                     img = self.pad_to_size(img, self.return_size)
#                     img = transforms.Normalize((0.1307,), (0.3081,))(img)# this is wrong norm because imgs are padded
        else:
            try:
                img = Image.open(loc).convert('RGB')
            except Exception as e:
                print(e)
                print('loc', loc)
                
        if self.label_colname:
            label = self.files[self.label_colname].iloc[index]
        else:
            label = 0
        # return the right stuff:
        if self.return_loc:
            return img, label, loc
        else:
            return img, label

    def __len__(self):
        return len(self.files)





# Maybe better off adding transfroms to the config, so not so much duplicated?
def make_generators_MNIST(files_dict_loc, batch_size, num_workers, return_size=40, rotation=None, crop_scale=None, 
                             path_colname='path', label_colname='class', label=None, return_loc=False):
    """ ! Only can use scaling or rotation once at a time
    use 6 padding for crop_scale so that transforms can be done without losing information. use crop_scale=.5
    """
    with open(files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)
        
    transform_list = [transforms.ToTensor()] # must wait till after padding to normalize
    # rotation use expand=True to prevent any rescaling of image, keeping pure rotation
    if rotation: 
        transform_list =  [transforms.RandomRotation(rotation, expand=True)] + transform_list 
        
    if crop_scale: # pad to size 40 for scaling down.
        transform_list = [
                         transforms.Pad(6, 6),
                         transforms.RandomResizedCrop(size=40, scale=(crop_scale, 1), ratio=(1,1))
                         ] + transform_list
    data_transforms ={}
    data_transforms['train'] = transforms.Compose(transform_list)
    data_transforms['val'] = transforms.Compose([
                                    transforms.Pad((6, 6)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    
    datasets = {}
    dataloaders = {}

    datasets = {x: FilesDFImageDataset(files_dict[x], base_path=None, transforms=data_transforms[x], 
                                       return_size=return_size,path_colname=path_colname, 
                                       label_colname=label_colname, select_label=label, 
                                       return_loc=return_loc, bw=True)
                                        
                for x in list(data_transforms.keys())}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                                                    for x in list(data_transforms.keys())}
    return dataloaders