
# Partial Model example modified from Sung Kim
# https://github.com/hunkim/PyTorchZeroToAll

import torch
import mlflow
import mlflow.pytorch

print("Torch Version:", torch.__version__)
print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
test_data =  [4.0, 5.0, 6.0]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def run(epochs, log_as_onnx):
    model = Model()
    print("model.type:",type(model))
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("Train:")
    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(f"  Epoch: {epoch}  Loss: {loss.data.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Predictions:")
    for v in test_data:
        tv = torch.Tensor([[v]])
        y_pred = model(tv)
        print(f"  {v}: {model(tv).data[0][0]}")

    with mlflow.start_run() as run:
        print("run_id:",run.info.run_id)
        mlflow.log_param("epochs", epochs)
        mlflow.pytorch.log_model(model, "pytorch-model")
        if log_as_onnx:
            import onnx_utils
            import onnx
            print("ONNX Version:", onnx.__version__)
            onnx_utils.log_model(model, "onnx-model", x_data)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", default=None, type=str)
    parser.add_argument("--epochs", dest="epochs", help="epochs", default=2, type=int)
    parser.add_argument("--log_as_onnx", dest="log_as_onnx", help="Log model as ONNX", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    run(args.epochs, args.log_as_onnx)
