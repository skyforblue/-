import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time
import torch
import copy
import torch.nn as nn
from tqdm import tqdm


# cfg
# traindir = "data/train"
# validdir = "data/valid"
traindir = "/home/cyy/4T/project/roi_delivery/libs/pig_up_down/data/train"
validdir = "/home/cyy/4T/project/roi_delivery/libs/pig_up_down/data/valid"

# save weights
save_weight = "weights"
os.makedirs(save_weight,exist_ok=True)

# save log
save_log = "logs"
os.makedirs(save_log,exist_ok=True)
tb_writer = SummaryWriter(log_dir=save_log)

# class num
class_num = 3

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image transformations
normal_val = [0.5,0.5,0.5]
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.RandomApply([transforms.RandomRotation(degrees=30),
                                transforms.ColorJitter(brightness=0.1),
                                transforms.RandomHorizontalFlip()],p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(normal_val,
                             normal_val)  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.ToTensor(),
        transforms.Normalize(normal_val, normal_val)
    ]),
}

# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
}

dataset_sizes = {x: len(data[x]) for x in ['train', 'valid']}
class_names = data['train'].classes
print(dataset_sizes,class_names)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, class_num)
model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



# train
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase],desc=phase):
                # print(inputs.shape)

                # wrap them in Variable
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # print(outputs.data)
                # print("*"*30)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # statistics
                running_loss += loss.data.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print(f'Loss:{loss.data.item()*inputs.size(0)}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "train":
                tb_writer.add_scalar('train/loss',epoch_loss,epoch+1)
                tb_writer.add_scalar('train/acc',epoch_acc,epoch+1)
            elif phase == "valid":
                tb_writer.add_scalar('valid/loss',epoch_loss,epoch+1)
                tb_writer.add_scalar('valid/acc',epoch_acc,epoch+1)

        print()
        if epoch % 2 == 0:
            torch.save(model.state_dict(),f"{save_weight}/model_{epoch}.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    tb_writer.close()
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=101)
torch.save(model_ft.state_dict(),"best_model.pth")
print("Done!!!")