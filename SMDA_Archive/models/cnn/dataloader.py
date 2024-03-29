import torch
import torchvision.transforms as vision_transforms
import numpy as np
from .dataset import MyDataset

def get_preprocessing_pipelines():
    preprocessing = {}
        
    # Initial variables
    input_size = 224
    video_means = [0.4489, 0.3352, 0.3106]  # [0.485, 0.456, 0.406]
    video_stds = [0.2380, 0.1965, 0.1962]  # [0.229, 0.224, 0.225]

    audio_mean = [0.485]  # Placeholder value, calculate from your dataset
    audio_std = [0.229] 
    
    # Train preprocess setting
    preprocessing['train'] = {
        "video": vision_transforms.Compose([
        vision_transforms.Resize((input_size, input_size)),
        vision_transforms.RandomHorizontalFlip(0.5),
        vision_transforms.ToTensor(),
        vision_transforms.Normalize(mean=video_means, std=video_stds)
        ]),
        "audio": vision_transforms.Compose([
        vision_transforms.Resize((input_size, input_size)),
        # vision_transforms.RandomHorizontalFlip(0.5),
        vision_transforms.ToTensor(),
        vision_transforms.Normalize(mean=audio_mean, std=audio_std)
        ])
    }
    
    # Valid preprocess setting
    preprocessing['val'] = {
        "video": vision_transforms.Compose([
        vision_transforms.Resize((input_size, input_size)),
        vision_transforms.ToTensor(),
        vision_transforms.Normalize(mean=video_means, std=video_stds)
        ]),
        "audio": vision_transforms.Compose([
        vision_transforms.Resize((input_size, input_size)),
        vision_transforms.ToTensor(),
        vision_transforms.Normalize(mean=audio_mean, std=audio_std)
        ])
    }
        
    # Test preprocess setting
    preprocessing['test'] = preprocessing['val']
    
    return preprocessing

def get_dataloaders(args, dataset_config, loader_config):
    preprocessing = get_preprocessing_pipelines()
    
    # create dataset object for each partition
    partitions = ['test'] if args.mode == "test" else ['train', 'test']
    datasets = {partition: MyDataset(
                data_path = dataset_config["data_path"],
                data_partition = partition,
                preprocessing_func = preprocessing[partition]
                ) for partition in partitions}
    dataset_loaders = {partition: torch.utils.data.DataLoader(
                        datasets[partition],
                        batch_size=1 if partition == 'test' else loader_config["batch_size"],
                        shuffle = (partition == 'train'),
                        pin_memory = loader_config["pin_memory"],
                        num_workers = loader_config["num_workers"],
                        worker_init_fn = np.random.seed(1)) for partition in partitions}
    return datasets, dataset_loaders