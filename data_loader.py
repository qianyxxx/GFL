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

    # 打印数据集的详细信息
    print(f'Dataset: {dataset_name}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {data.num_classes}')

    # 打印数据的维度
    print(f'Shape of feature matrix: {data.x.shape}')
    print(f'Shape of edge index: {data.edge_index.shape}')
    print(f'Shape of labels: {data.y.shape}')

    return data