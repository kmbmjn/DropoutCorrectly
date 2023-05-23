from __future__ import print_function, division
import time
import copy
import os
import argparse
from termcolor import colored
import numpy as np
import cv2
import random
import pdb
import PIL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from my_model import *

np.random.seed(0)

parser = argparse.ArgumentParser(description="resnet_teacher")
parser.add_argument("--model_name", default="my_resnet18_pad", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--data_name", default="car", type=str)
parser.add_argument("--hp_lr", type=float, default=1e-2)
parser.add_argument("--hp_wd", type=float, default=5e-4)
parser.add_argument("--hp_bs", type=int, default=64)
parser.add_argument("--hp_ep", type=int, default=200)
parser.add_argument("--hp_opt", type=str, default="sgd")
parser.add_argument("--hp_sch", type=str, default="cos")
parser.add_argument("--hp_id", type=int, default=0)

args = parser.parse_args()


# read private_arguments such as num_workers
f = open("../p_command_multi/private_arguments.txt", "r")
lines = f.readlines()
for line in lines:
    line = line.strip()  # remove line change character.
    line_split = line.split(" ")
    locals()[line_split[0][2:]] = int(line_split[1])
f.close()


transform_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


transform = transforms.Compose(
    [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


if args.data_name == "car":
    dataset_aug = ImageFolder(
        root="~/dataset/my_stancars/cars_by_index/", transform=transform_aug
    )
    dataset = ImageFolder(
        root="~/dataset/my_stancars/cars_by_index/", transform=transform
    )
    num_classes = 196

if args.data_name == "dog":
    dataset_aug = ImageFolder(
        root="~/dataset/stan_dogs/Images/", transform=transform_aug
    )
    dataset = ImageFolder(root="~/dataset/stan_dogs/Images/", transform=transform)
    num_classes = 120

if args.data_name == "pet":
    dataset_aug = ImageFolder(
        root="~/dataset/oxford_pet/images/", transform=transform_aug
    )
    dataset = ImageFolder(root="~/dataset/oxford_pet/images/", transform=transform)
    num_classes = 37

if args.data_name == "fgvc":
    dataset_aug = ImageFolder(
        root="~/dataset/fgvc/fgvc-aircraft-2013b/", transform=transform_aug
    )
    dataset = ImageFolder(
        root="~/dataset/fgvc/fgvc-aircraft-2013b/", transform=transform
    )
    num_classes = 100

if args.data_name == "nab":
    dataset_aug = ImageFolder(
        root="~/dataset/nabirds/nabirds/", transform=transform_aug
    )
    dataset = ImageFolder(root="~/dataset/nabirds/nabirds/", transform=transform)
    num_classes = 555

if args.data_name == "cal":
    dataset_aug = ImageFolder(
        root="~/dataset/caltech101/images/", transform=transform_aug
    )
    dataset = ImageFolder(root="~/dataset/caltech101/images/", transform=transform)
    num_classes = 102

if args.data_name == "cub":
    dataset_aug = ImageFolder(root="~/dataset/cub/images/", transform=transform_aug)
    dataset = ImageFolder(root="~/dataset/cub/images/", transform=transform)
    num_classes = 200

if args.data_name == "food":
    dataset_aug = ImageFolder(root="~/dataset/food/merge/", transform=transform_aug)
    dataset = ImageFolder(root="~/dataset/food/merge/", transform=transform)
    num_classes = 101


# Shuffle the indices
len_dataset = len(dataset)  # 9144
len_train = int(len_dataset * 0.7)  # 6400
len_val = int(len_dataset * 0.15)  # 1371
len_test = int(len_dataset * 0.15)  # 1374

indices = np.arange(0, len_dataset)
np.random.shuffle(indices)  # shuffle the indicies

train_loader = torch.utils.data.DataLoader(
    dataset_aug,  # aug
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=hp_nw1,
    sampler=torch.utils.data.SubsetRandomSampler(indices[:len_train]),
)

val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=hp_nw2,
    sampler=torch.utils.data.SubsetRandomSampler(
        indices[len_train : len_train + len_val]
    ),
)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=hp_nw2,
    sampler=torch.utils.data.SubsetRandomSampler(
        indices[len_train + len_val : len_train + len_val + len_test]
    ),
)


dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

loader_sizes = {
    "train": len(train_loader),
    "val": len(val_loader),
    "test": len(test_loader),
}

dataset_sizes = {"train": len_train, "val": len_val, "test": len_test}

device = torch.device("cuda:" + str(args.hp_id) if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = float("-inf")  # 가장 작은 수

    for epoch in range(num_epochs):
        time_start_ep = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        is_now_best = False

        # Each epoch has a training and validation phase
        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()

                        optimizer.step()

                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase in ["val", "test"]:
                print("{}_acc: {:.6f}".format(phase, epoch_acc))

            if phase == "val" and epoch_acc > best_val_acc:
                print(colored("It is now best.", "green"))
                is_now_best = True
                best_val_acc = epoch_acc
                best_val_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "test" and is_now_best:
                best_test_acc = epoch_acc

        time_elapsed_ep = time.time() - time_start_ep
        total_time_expected = time_elapsed_ep * args.hp_ep
        remaining_time_expected = time_elapsed_ep * (
            args.hp_ep - epoch - 1
        )  # -1 for counting

        print(
            "Time elapsed for single epoch: {:.0f}m {:.0f}s".format(
                time_elapsed_ep // 60, time_elapsed_ep % 60
            )
        )
        print(
            "Total time expected: {:.0f}m {:.0f}s".format(
                total_time_expected // 60, total_time_expected % 60
            )
        )
        print(
            "Remaining time expected: {:.0f}m {:.0f}s".format(
                remaining_time_expected // 60, remaining_time_expected % 60
            )
        )
        print("")

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("")
    print("Final best val acc: {:6f}".format(best_val_acc))
    print("Final best test acc: {:6f}".format(best_test_acc))
    print("Final best epoch: ", best_val_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model
    return [model, best_val_acc.cpu().numpy()]


model_ft = get_my_model(args.model_name, num_classes)

## from torchsummary import summary
## summary(model_ft, (3, 224, 224))


## from torchsummaryX import summary
## summary(model_ft, torch.zeros((1, 3, 224, 224)))

###
print(model_ft)
###

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

if args.hp_opt == "sgd":
    optimizer_ft = optim.SGD(
        model_ft.parameters(), lr=args.hp_lr, momentum=0.9, weight_decay=args.hp_wd
    )

if args.hp_opt == "adam":
    optimizer_ft = optim.Adam(
        model_ft.parameters(), lr=args.hp_lr, weight_decay=args.hp_wd
    )

if args.hp_sch == "msl":
    hp_lr_decay_ratio = 0.2

    scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft,
        milestones=[
            args.hp_ep * 0.3,
            args.hp_ep * 0.6,
            args.hp_ep * 0.8,
        ],
        gamma=hp_lr_decay_ratio,
    )
if args.hp_sch == "cos":
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.hp_ep)

if args.mode in ["train"]:
    # return이 best model임
    model_ft, best_acc = train_model(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=args.hp_ep
    )

    save_dir = (
        "retrain_model_folder/pt_"
        + str(args.model_name)
        + "_lr"
        + str(args.hp_lr)
        + "_wd"
        + str(args.hp_wd)
    )

    # finally, save the best
    os.makedirs(save_dir, exist_ok=True)
    save_dirfile = save_dir + "/ba_" + str(best_acc)[:6]
    # torch.save(model_ft.state_dict(), save_dirfile)
