import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from maml import MAML
from train import adaptation, test
import pickle

import datetime
import os

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# dataset

# trainset = miniimagenet("/home/fukitani/MAML_Reproduction/data/miniimagenet/", ways=5, shots=5, test_shots=15, meta_train=True, download=False)
#trainloader = BatchMetaDataLoader(trainset, batch_size=2, num_workers=4, shuffle=True)
train_path = '/home/fukitani/MAML_anomaly/dataset_mini/Hazelnut/train'
train_set = ImageFolder(train_path, transform=torchvision.transforms.ToTensor())
#print(train_set.class_to_idx)  {'crack': 0, 'good': 1}
train_loader = DataLoader(dataset=train_set,batch_size=2,shuffle=True)

#testset = miniimagenet("/home/fukitani/MAML_Reproduction/data/miniimagenet/", ways=5, shots=5, test_shots=15, meta_test=True, download=False)
#testloader=BatchMetaDataLoader(testset, batch_size=2, num_workers=4, shuffle=True)
val_path = "/home/fukitani/MAML_anomaly/dataset_mini/Coffee_beans/train"
val_set = ImageFolder(val_path, transform=torchvision.transforms.ToTensor())
val_loader = DataLoader(val_set,batch_size=2,shuffle=True)

# training

num_epochs = 50 # batch sizeが2だと7751が上限(dataloaderの制限)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MAML().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

result_path = "/home/fukitani/MAML_anomaly/result/Coffee_beans"

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M/")
result_path = os.path.join(result_path, current_time)
os.makedirs(result_path, exist_ok=True)

trainiter = iter(train_loader)
evaliter = iter(val_loader)
# print(trainiter)
# print(evaliter)

train_loss_log = []
train_acc_log = []
val_loss_log = []
val_acc_log = []

for epoch in range(num_epochs):
    # train
    print('---------------------------------')
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    
    for i,data in enumerate(train_loader):
        train_x, train_y = data
        #print("train_x.shape : " , train_x.shape)
        train_x, train_y = train_x.to(device),train_y.to(device)

        for j , data in enumerate(val_loader):
            if j == i:
                val_x, val_y = data
                val_x, val_y = val_x.to(device),val_y.to(device)


        #trainbatch = trainiter.next()
        # trainbatch = next(trainiter)
        # valbatch = next(evaliter)
        #print(trainbatch[0])
        #print(trainbatch['train'])
        model.train()
        # loss, acc = adaptation(model, optimizer, trainbatch, loss_fn, lr=0.01, train_step=5, train=True, device=device)
        loss, acc = adaptation(model, optimizer, train_x, train_y ,val_x,val_y, loss_fn, lr=0.01, train_step=1, train=True, device=device)
        #print("train_loss : " , loss)
        model.eval()
        valloss, valacc,predicted = test(model, train_x, train_y, val_x, val_y, loss_fn, lr=0.01, train_step=1, device=device)

        
      
        # train_acc_log.append(acc)
    
    print("train_loss : ", (loss / len(train_set)))
    train_loss_log.append(loss.item())
    train_acc_log.append(acc)
    val_loss_log.append(valloss.item())
    val_acc_log.append(valacc)


    # # test
    # # evalbatch = evaliter.next()
    # evalbatch = evaliter.next()
    # model.eval()
    # valloss, valacc = test(model, evalbatch, loss_fn, lr=0.01, train_step=1, device=device)

    # val_loss_log.append(valloss.item())
    # val_acc_log.append(valacc)

    # print("train_loss = {:.4f}, train_acc = {:.4f}, val_loss = {:.4f}, val_acc = {:.4f}".format(loss.item(), acc, valloss.item(), valacc))


torch.save(model.state_dict(), result_path + 'model.pth')
all_result = {'train_loss': train_loss_log, 'train_acc': train_acc_log, 'test_loss': val_loss_log, 'test_acc': val_acc_log}

with open(result_path + 'train.pkl', 'wb') as f:
    pickle.dump(all_result, f)


#テキストファイルにパラメータを入力
text_name = "parameter.txt"
textfile_path = os.path.join(result_path, text_name)

try:
    with open(textfile_path, mode='x') as f:
        f.write("train_path = ")
        f.write(train_path)
        f.write(" \n")
        f.write("\ntest_path = ")
        f.write(val_path)
        f.write(" \n")
        f.write("\nnum_epochs = ")
        f.write(str(num_epochs))

except FileExistsError:
    pass