import argparse
import pathlib
import re
import time
import datetime
import pandas as pd
from tempfile import mkdtemp
from tqdm import trange
import numpy as np 
import json 

import torch
from torch.utils.data import DataLoader

# relative import hacks (sorry)
import inspect
import os 
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user

from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet, LeNet
from log_utils import make_dir
from log_utils import (
    Logger,
    Timer,
    Constants,
    DefaultList,
)


parser = argparse.ArgumentParser(description='Fum BackDoor')
parser.add_argument(
    "--experiment", type=str, default="", metavar="E", help="experiment name"
)
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--model', default='lenet', help='model', choices=['lenet', 'badnet'])
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--save-dir', default='', help='')
parser.add_argument('--loss', default='ce', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')

# parser.add_argument("--scheduler", type=str, default="none", metavar="M", choices=["StepLR", "MultiStepLR"], help="optimizer (default: none)", )
# parser.add_argument("--lr_gamma", type=float, default=1.0, help="SCHEDULING, Multiplicative factor of learning rate decay , default: (1.0, no scheduling) ",)
# parser.add_argument("--lr_steps", metavar="N", type=int, nargs="+", default=[], help="List of epoch indices. Must be increasing/Period of learning rate decay",)
parser.add_argument("--seed", type=int, default=11, metavar="S", help="random seed (default: 11)")
parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA use")

# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if not args.experiment:
    args.experiment = args.model
runId = datetime.datetime.now().isoformat().replace(':', '-')
experiment_dir = pathlib.Path(f"{Constants.OUTPUT_DIR}/" + args.experiment)
experiment_dir.mkdir(parents=True, exist_ok=True)

if not args.load_local: 
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
else: 
    runPath = args.save_dir

device = torch.device("cuda" if args.cuda else "cpu")
print("Device: ", device)

sys.stdout = Logger("{}/run.log".format(runPath))
print("Expt:", runPath)
print("RunID:", runId)
command_line_args = sys.argv
command = " ".join(command_line_args)
print(f"The command that ran this script: {command}")

import loading_utils
loading_utils.set_reproducability(args.seed)

print("{}".format(args).replace(', ', ',\n'))


# create related path
pathlib.Path(f"{runPath}/checkpoints/").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{runPath}/logs/").mkdir(parents=True, exist_ok=True)
basic_model_path = f"{runPath}/checkpoints/{args.model}.pth"

with open("{}/args.json".format(runPath), "w") as fp:
    json.dump(args.__dict__, fp)
torch.save(args, "{}/args.rar".format(runPath))
print("args: \n", args)


print("\n# load dataset: %s " % args.dataset)
dataset_train, args.nb_classes, mean, std = build_poisoned_training_set(is_train=True, args=args)
dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

if args.model == 'badnet': 
    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
elif args.model == 'lenet': 
    model = LeNet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

if __name__ == "__main__":

    with Timer("BackDoor FUM") as t:
        start_time = time.time()
        if args.load_local:
            print("## Load model from : %s" % basic_model_path)
            model.load_state_dict(torch.load(basic_model_path), strict=True)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
            print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
            print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
        else:
            print(f"Start training for {args.epochs} epochs")
            stats = []
            for epoch in trange(args.epochs):
                train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
                test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
                print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
                
                # save model 
                torch.save(model.state_dict(), basic_model_path)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,}

                # save training stats
                stats.append(log_stats)
                df = pd.DataFrame(stats)
                df.to_csv(f"{runPath}/logs/log.csv", index=False, encoding='utf-8')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        ### Margin Distribution Testing...
        from margins.utils import get_dataset_loaders
        from margins.utils import train
        from margins.utils import generate_subspace_list
        from margins.utils import compute_margin_distribution
        from margins.utils import get_eval
        from margins.graphics import swarmplot

        make_dir(f'{runPath}/margins')
        model.eval()

        from margins.utils import TransformLayer
        print(mean, std, type(mean), type(std))
        trans = TransformLayer(mean=mean, std=std).to(device)

        SUBSPACE_DIM = 8
        DIM = 28
        SUBSPACE_STEP = 1

        subspace_list = generate_subspace_list(SUBSPACE_DIM, DIM, SUBSPACE_STEP, channels=1)
        NUM_SAMPLES_EVAL = 100

        testset = dataset_val_poisoned # dataset_val_clean # ?  
        eval_dataset, eval_loader, num_samples = get_eval(testset, num_samples=NUM_SAMPLES_EVAL, batch_size=NUM_SAMPLES_EVAL, seed=111)
        margins = compute_margin_distribution(model, trans, eval_loader, subspace_list,  f'{runPath}/margins/margins - poisoned.npy')
        swarmplot(margins, name = f'{runPath}/margins/margin distribiution - poisoned',color='tab:blue')
        
        testset =  dataset_val_clean# dataset_val_clean # ?  
        eval_dataset, eval_loader, num_samples = get_eval(testset, num_samples=NUM_SAMPLES_EVAL, batch_size=NUM_SAMPLES_EVAL, seed=111)
        margins = compute_margin_distribution(model, trans, eval_loader, subspace_list,  f'{runPath}/margins/margins - clean.npy')
        swarmplot(margins, name = f'{runPath}/margins/margin distribiution - clean',color='tab:blue')
