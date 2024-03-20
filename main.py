# main.py
import argparse
from data_loader import load_data
from model import GCN
from train_eval import train, evaluate, cross_validate
from client_server import Client, Server
import torch.optim as optim
from metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='WikiCS', help='Dataset name (Cora or WikiCS)')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type (Adam or SGD)')
parser.add_argument('--clients', type=int, default=10, help='Number of clients')
parser.add_argument('--fed_algo', type=str, default='fedavg', help='Federated learning algorithm (fedavg or fedprox)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--local_epoch', type=int, default=5, help='Number of epochs per communication round')
parser.add_argument('--communication_rounds', type=int, default=10, help='Number of communication rounds')
args = parser.parse_args()

dataset_name = args.dataset
data = load_data(dataset_name)

# data = load_data('WikiCS')
# print(f'Shape of data: {data.x.shape}')
# print(f'Shape of train_mask: {data.train_mask.shape}')

model = GCN(data.num_features, 16, data.num_classes)

server = Server(model)
clients = [Client(data, server.send_model(), optim.Adam(model.parameters(), lr=args.lr)) for _ in range(args.clients)]

communication_round = 0  # 初始化通信轮次

for epoch in range(args.communication_rounds * args.local_epoch):  # 使用命令行参数定义训练轮次
    updates = []
    losses = []
    client_models = [client.model for client in clients]
    for client in clients:
        update, loss = client.train()
        updates.append(update)
        losses.append(loss)
    if args.fed_algo == 'fedavg':
        server.fedavg(updates)
    elif args.fed_algo == 'fedprox':
        server.fedprox(updates, client_models)
    else:
        raise ValueError(f'Unknown federated learning algorithm: {args.fed_algo}')
    if epoch % args.local_epoch == args.local_epoch - 1:  # 每 args.local_epoch 个epoch发送一次模型
        model = server.send_model()
        for client in clients:
            client.receive_model(model)
        communication_round += 1  # 增加通信轮次
        print(f'Communication round: {communication_round}/{args.communication_rounds}, Total local epochs completed: {epoch + 1}/{args.communication_rounds * args.local_epoch}')
        avg_loss = sum(losses) / len(losses)
        print(f'Average loss in this round: {avg_loss:.4f}')
        accuracy = cross_validate(data, model, optim.Adam(model.parameters(), lr=args.lr), args.local_epoch)
        print(f'10-fold Cross Validation Accuracy after this communication round: {accuracy:.4f}')

# 在所有通信轮次结束后进行一次交叉验证
accuracy = cross_validate(data, model, optim.Adam(model.parameters(), lr=args.lr), args.local_epoch)
print(f'10-fold Cross Validation Accuracy after all communication rounds: {accuracy:.4f}')

# 在所有通信轮次结束后计算并输出指标
model.eval()
with torch.no_grad():
    logits = model(data)
y_pred = logits.argmax(dim=1).cpu().numpy()  # 获取预测的类别
y_true = data.y.cpu().numpy()  # 获取真实的标签
accuracy = calculate_accuracy(y_true, y_pred)
precision = calculate_precision(y_true, y_pred)
recall = calculate_recall(y_true, y_pred)
f1 = calculate_f1(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')