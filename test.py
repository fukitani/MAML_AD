from hashlib import sha1
from operator import gt
from random import shuffle
import re
import torch
import torchvision
import numpy as np
import pandas as pd
import csv

#from torchmeta.datasets.helpers import miniimagenet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
#from torchmeta.utils.data import BatchMetaDataLoader
from maml import MAML
from train import adaptation, test
import pickle
import os
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# testset = miniimagenet("data", ways=5, shots=5, test_shots=15, meta_test=True, download=True)
# testloader=BatchMetaDataLoader(testset, batch_size=2, num_workers=4, shuffle=True)
test_set = ImageFolder('/home/fukitani/MAML_anomaly/dataset_mini/Coffee_beans/test', transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_set,batch_size=1,shuffle=True)

val_set = ImageFolder('/home/fukitani/MAML_anomaly/dataset/Hazelnut/test', transform=torchvision.transforms.ToTensor())
val_loader = DataLoader(test_set,batch_size=1,shuffle=True)

evaliter = iter(test_loader)

model_dir = '/home/fukitani/MAML_anomaly/result/Coffee_beans/2022-09-26-18-04/'
model_path = os.path.join(model_dir, "model.pth")
print("model_path" , model_path)

model = MAML().to(device)
model.load_state_dict(torch.load(model_path))
loss_fn = torch.nn.CrossEntropyLoss().to(device)

test_loss_log = []
test_acc_log = []
bad_score = []
good_score = []
gt_label = []
predicted_label = []
result_list = []
correct = 0
total = 0
i = 0
predicted = 0
tp,fp,fn,tn = 0,0,0,0

for i,data in enumerate(test_loader):
    test_x, test_y = data
    test_x, test_y = test_x.to(device),test_y.to(device)

    for j , data in enumerate(val_loader):
        if j == i:
            val_x, val_y = data
            val_x, val_y = val_x.to(device),val_y.to(device)

    model.eval()
    print("------------------------------------------------------------------")
    #print("test_y :" , test_y)
    int_test_y = test_y[0]
    int_test_y = int_test_y.to("cpu").detach().numpy().copy()
    gt_label.append(int(int_test_y))
    

    
    # testloss, testacc, predicted = test(model,test_x, test_y, val_x, val_y, loss_fn, lr=0.01, train_step=40, device=device)
    # testloss, predicted , score = test(model,test_x, test_y, val_x, val_y, loss_fn, lr=0.01, train_step=40, device=device)
    testloss, testacc ,predicted = test(model,test_x, test_y, val_x, val_y, loss_fn, lr=0.01, train_step=40, device=device)

    total += test_y.size(0)
    correct += (predicted == test_y).sum()
    test_loss_log.append(testloss.item())
    test_acc_log.append(testacc)
    
    predicted = predicted.to("cpu").detach().numpy().copy()
    predicted_label.append(int(predicted))
    
    if int_test_y == 0: #真値が異常のとき
        tp += (predicted == int_test_y).sum()
        fn += (predicted != int_test_y).sum()

    if int_test_y == 1: #真値が正常のとき
        tn += (predicted == int_test_y).sum()
        fp += (predicted != int_test_y).sum()

    # if int_test_y == 0:
    #     bad_score.append(score)
    # if int_test_y == 1:
    #     good_score.append(score)


   # print("i {}: test_loss = {:.4f}, test_acc = {:.4f}".format(i, testloss.item(), testacc))

#all_result = {'test_loss': test_loss_log, 'test_acc': test_acc_log}
# with open(model_dir + 'test.pkl', 'wb') as f:
#     pickle.dump(all_result, f)
#print("bad_score\n" , bad_score)
#print("good_score\n" , good_score)


df = pd.DataFrame( { "真値" : gt_label , "予測結果" : predicted_label})
# CSV ファイル出力
df.to_csv(os.path.join(model_dir , "result.csv"),encoding="shift_jis")


# print("gt_label\n",gt_label)
# print("\npredicted_label\n",predicted_label)

print("finished-------------------------------")
print('Accuracy %d / %d = %f' % (correct, total, correct / total))
print("Precision : " , round(tp/(tp+fp),2))
print("Recall : " , round(tp/(tp+fn),2))

# print("Precision : " , tp/(tp+fp))
# print("Recall : " , tp/(tp+fn))
    


######################  画像と予測結果を出力  ##########################
for i,data in enumerate(test_loader):
    test_x, test_y = data
    test_x, test_y = test_x.to(device),test_y.to(device)

    for j , data in enumerate(val_loader):
        if j == i:
            val_x, val_y = data
            val_x, val_y = val_x.to(device),val_y.to(device)
# test_iter = iter(test_loader)
# val_iter = iter(val_loader)_, predicted = torch.max(outputs.data, 1)
ts = torchvision.transforms.ToPILImage()
im = ts(val_x[1])
predict = predicted[0].item()   
#print(predict)

if predict == 0 :
    print('予測結果:', 'crack')
elif predict == 1 :
    print('予測結果:', 'good')

plt.imshow(np.array(im))
plt.show()

# for i in range(20):
#     evalbatch = evaliter.next()
#     model.eval()
#     testloss, testacc = test(model, evalbatch, loss_fn, lr=0.01, train_step=40, device=device)

#     test_loss_log.append(testloss.item())
#     test_acc_log.append(testacc)

#     print("i {}: test_loss = {:.4f}, test_acc = {:.4f}".format(i, testloss.item(), testacc))

# all_result = {'test_loss': test_loss_log, 'test_acc': test_acc_log}

# with open(model_dir + 'test.pkl', 'wb') as f:
#     pickle.dump(all_result, f)