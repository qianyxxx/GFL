# train_eval.py
import torch.nn.functional as F
import torch

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    # loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    loss = F.cross_entropy(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# def evaluate(data, model):
#     model.eval()
#     with torch.no_grad():
#         logits = model(data)
#     preds = logits.argmax(dim=1)
#     correct = preds[data.val_mask].eq(data.y[data.val_mask]).sum().item()
#     return correct / len(data.val_mask)

def evaluate(data, model):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    preds = logits.argmax(dim=1)
    correct = preds.eq(data.y).sum().item()
    return correct / len(data.y)