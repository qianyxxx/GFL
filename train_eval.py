# train_eval.py
import torch.nn.functional as F
import torch
from sklearn.model_selection import KFold


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    # loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    loss = F.cross_entropy(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(data, model):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    preds = logits.argmax(dim=1)
    correct = preds.eq(data.y).sum().item()
    return correct / len(data.y)

def cross_validate(data, model, optimizer, epochs):
    kfold = KFold(n_splits=10, shuffle=True)
    accuracies = []

    for train_indices, test_indices in kfold.split(data.x):
        train_data = data.clone()  # 创建数据的副本
        test_data = data.clone()

        # 使用布尔掩码来选择训练和测试数据
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        test_mask[test_indices] = 1
        train_data.train_mask = train_mask
        test_data.test_mask = test_mask

        for epoch in range(epochs):
            train(train_data, model, optimizer)
        
        accuracy = evaluate(test_data, model)
        accuracies.append(accuracy)

    return sum(accuracies) / len(accuracies)