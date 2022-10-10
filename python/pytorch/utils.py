import torch
from torchvision import datasets, transforms

data_dir = "../data"

def prep_data(loader):
    dataiter = iter(loader)
    images, _ = dataiter.next()
    print("prep_data: images.type:",type(images))
    print("prep_data: images.shape:",images.shape)
    return images

def get_data(is_train=True, batch_size=128):
    kwargs = {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=is_train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return loader

def display_predictions(data):
    import pandas as pd
    from tabulate import tabulate
    df = pd.DataFrame(data).head(10)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
