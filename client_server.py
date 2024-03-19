# client_server.py
import torch.optim as optim
import torch
from train_eval import train

class Server:
    def __init__(self, model):
        self.model = model

    def send_model(self):
        return self.model

    def fedavg(self, updates):
        avg_update = {name: torch.zeros_like(param) for name, param in updates[0].items()}
        for update in updates:
            for name, param in update.items():
                avg_update[name] += param
        for name in avg_update.keys():
            avg_update[name] /= len(updates)
        for name, param in self.model.named_parameters():
            param.data = avg_update[name]

    def fedprox(self, updates, client_models, mu=0.01):
        avg_update = {name: torch.zeros_like(param) for name, param in updates[0].items()}
        for update in updates:
            for name, param in update.items():
                avg_update[name] += param
        for name in avg_update.keys():
            avg_update[name] /= len(updates)
        for i, client_model in enumerate(client_models):
            for name, param in client_model.named_parameters():
                avg_update[name] += mu * (param.data - avg_update[name])
        for name, param in self.model.named_parameters():
            param.data = avg_update[name]

class Client:
    def __init__(self, data, model, optimizer):
        self.data = data
        self.model = model
        self.optimizer = optimizer

    def train(self):
        loss = train(self.data, self.model, self.optimizer)
        return self.model.state_dict()

    def receive_model(self, model):
        self.model.load_state_dict(model.state_dict())