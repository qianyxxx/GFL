# data_loader.py
from torch_geometric.datasets import WikiCS, Planetoid

def load_data(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif dataset_name == 'WikiCS':
        dataset = WikiCS(root='/tmp/WikiCS')
        data = dataset[0]
        data.num_classes = dataset.data.y.max().item() + 1
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return data