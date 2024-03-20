# Federated Learning Project

This project is a federated learning model implemented using PyTorch and PyTorch Geometric. It uses a Graph Convolutional Network (GCN) to process graph data and employs two federated learning algorithms: FedAvg and FedProx.

## File Structure

- [`main.py`](command:_github.copilot.openRelativePath?%5B%22main.py%22%5D "main.py"): The main program entry point, which includes code for parsing command-line arguments, training the model, and evaluating the results.
- [`model.py`](command:_github.copilot.openRelativePath?%5B%22model.py%22%5D "model.py"): Contains the definition of the Graph Convolutional Network (GCN) model.
- [`data_loader.py`](command:_github.copilot.openRelativePath?%5B%22data_loader.py%22%5D "data_loader.py"): Contains the data loading function, which can load either the WikiCS or Cora datasets.
- [`client_server.py`](command:_github.copilot.openRelativePath?%5B%22client_server.py%22%5D "client_server.py"): Contains the definitions of the Client and Server classes, which simulate the federated learning environment.
- [`train_eval.py`](command:_github.copilot.openRelativePath?%5B%22train_eval.py%22%5D "train_eval.py"): Contains functions for training and evaluating the model.
- [`metrics.py`](command:_github.copilot.openRelativePath?%5B%22metrics.py%22%5D "metrics.py"): Contains functions for calculating accuracy, precision, recall, and F1 score.

## Usage

1. Install dependencies: `pip install torch torch_geometric`
2. Run the main program: `python main.py --dataset WikiCS --optimizer Adam --clients 10 --fed_algo fedavg --lr 0.01 --local_epoch 5 --communication_rounds 10`

## Command-Line Arguments

- `--dataset`: The name of the dataset, either 'WikiCS' or 'Cora'. Default is 'WikiCS'.
- `--optimizer`: The type of optimizer, either 'Adam' or 'SGD'. Default is 'Adam'.
- `--clients`: The number of clients, must be a positive integer. Default is 10.
- `--fed_algo`: The federated learning algorithm, either 'fedavg' or 'fedprox'. Default is 'fedavg'.
- `--lr`: The learning rate, must be a positive float. Default is 0.01.
- `--local_epoch`: The number of local training epochs per communication round, must be a positive integer. Default is 5.
- `--communication_rounds`: The number of communication rounds, must be a positive integer. Default is 10.

## Notes

- Please ensure that your Python version is 3.6 or higher.
- Please ensure that you have installed all necessary dependencies.