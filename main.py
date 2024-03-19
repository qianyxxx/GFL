# main.py
import argparse
from data_loader import load_data
from model import GCN
from train_eval import train, evaluate
from client_server import Client, Server
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name (Cora or WikiCS)')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type (Adam or SGD)')
parser.add_argument('--clients', type=int, default=10, help='Number of clients')
parser.add_argument('--fed_algo', type=str, default='fedavg', help='Federated learning algorithm (fedavg or fedprox)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
args = parser.parse_args()

dataset_name = args.dataset
data = load_data(dataset_name)

model = GCN(data.num_features, 16, data.num_classes)

server = Server(model)
clients = [Client(data, server.send_model(), optim.Adam(model.parameters(), lr=args.lr)) for _ in range(args.clients)]

for epoch in range(args.epochs):
    updates = []
    client_models = [client.model for client in clients]
    for client in clients:
        update = client.train()
        updates.append(update)
    if args.fed_algo == 'fedavg':
        server.fedavg(updates)
    elif args.fed_algo == 'fedprox':
        server.fedprox(updates, client_models)
    else:
        raise ValueError(f'Unknown federated learning algorithm: {args.fed_algo}')
    model = server.send_model()
    for client in clients:
        client.receive_model(model)
    acc = evaluate(data, model)
    print(f'Epoch {epoch+1}, Validation Accuracy: {acc:.4f}')