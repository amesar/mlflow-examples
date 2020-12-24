#
# PyTorch MNIST example.
# Code based on https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py.
#
from argparse import ArgumentParser
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import utils

print("MLflow Version:", mlflow.__version__)
print("Torch Version:", torch.__version__)

def init(args):
    enable_cuda_flag = True if args.enable_cuda == "True" else False
    args.cuda = enable_cuda_flag and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

    def log_weights(self, step):
        pass


def train(model, optimizer, epoch, args, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar("train_loss", loss.data.item(), step, args.autolog)
            model.log_weights(step)


def test(model, epoch, args, train_loader, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    step = (epoch + 1) * len(train_loader)
    log_scalar("test_loss", test_loss, step, args.autolog)
    log_scalar("test_accuracy", test_accuracy, step, args.autolog)


def log_scalar(name, value, step, autolog):
    """Log a scalar value to both MLflow and TensorBoard"""
    if not autolog:
        mlflow.log_metric(name, value)


def do_train(args):
    model = Net()
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_loader = utils.get_data(True, args.batch_size)
    test_loader = utils.get_data(False, args.batch_size)

    with mlflow.start_run() as run:
        print("run_id:",run.info.run_id)
        mlflow.set_tag("mlflow_version",mlflow.__version__)
        mlflow.set_tag("torch_version",torch.__version__)
        mlflow.set_tag("autolog",args.autolog)

        # Perform the training
        for epoch in range(1, args.epochs + 1):
            train(model, optimizer, epoch, args, train_loader)
            test(model, epoch, args, train_loader, test_loader)

        if not args.autolog:
            for key, value in vars(args).items():
                mlflow.log_param(key, value)
            mlflow.pytorch.log_model(model, "pytorch-model")

        # Log model as ONNX
        if args.log_as_onnx:
            import onnx_utils
            import onnx
            print("ONNX Version:", onnx.__version__)
            dataiter = iter(test_loader)
            images, labels = dataiter.next()
            onnx_utils.log_model(model, "onnx-model", images)


def create_args():
    parser = ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--enable-cuda",
        type=str,
        choices=["True", "False"],
        default="True",
        help="enables or disables CUDA training",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--experiment-name", dest="experiment_name", help="Experiment name", default=None)
    parser.add_argument("--log-as-onnx", dest="log_as_onnx", help="Log model as ONNX", default=False, action='store_true')
    parser.add_argument("--autolog", dest="autolog", help="Autolog", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    return args


if __name__ == "__main__":
    args = create_args()
    init(args)
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    if args.autolog:
        print("AUTOLOG")
        mlflow.pytorch.autolog(log_every_n_epoch=1)
    do_train(args)
