from cgi import print_environ
import torch
from collections import OrderedDict
from maml import MAML


def adaptation(model, optimizer, train_x,train_y, val_x,val_y, loss_fn, lr, train_step, train, device):
    predictions = []
    labels = []
    epoch_loss = 0
   
    # x_train, y_train = trainbatch    #x_train テンソル化した画像、y_train ラベル(0 : crack , 1 : good)
    #print("x_train.size(0) : ",x_train.size(0))  #バッチサイズ 2
    for idx in range(train_x.size(0)):
        weights = OrderedDict(model.named_parameters())
        #print("weights : " , weights)
        
        # k-shotでadaptation
        for iter in range(train_step):
            logits = model.adaptation(train_x, weights)
            #print("logits1 : " , logits)
            # print("input_y1 : " , input_y1)
            #print("input_y1 : " , train_y)
            #logits = model.adaptation(input_x, weights)
            loss = loss_fn(logits, train_y)
            #print("train_loss1" , loss)
            # loss = loss_fn(logits, input_y)
            gradients = torch.autograd.grad(loss, weights.values(), create_graph=train)

            weights = OrderedDict((name, param - lr * grad) for ((name, param), grad) in zip(weights.items(), gradients))

        # queryで評価
        logits = model.adaptation(val_x, weights)
        #print("logits2 : " , logits)
        #print("val_y1 : " , val_y)
        loss = loss_fn(logits, val_y)
        #print("train_loss2" , loss)
        if train:
            model.train()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #print("train_loss3" , loss)
            optimizer.step()
            #epoch_loss += loss.item() *  val_x.size(0)

        y_pred = logits.softmax(dim=1)
        predictions.append(y_pred)
        labels.append(train_y)
    
    #print("labels : " , labels)
    #print("predictions : " , predictions)
    y_pred = torch.cat(predictions)
    y_label = torch.cat(labels)
    batch_acc = torch.eq(y_pred.argmax(dim=-1), y_label).sum().item() / y_pred.shape[0]
    return loss, batch_acc

def test(model, test_x, test_y, val_x, val_y, loss_fn, lr, train_step, device):
    # x_train, y_train = batch    #x_train テンソル化した画像、y_train ラベル(0 : crack , 1 : good)
    # x_val, y_val = batch
    predictions = []
    labels = []


    for idx in range(test_x.size(0)):
        weights = OrderedDict(model.named_parameters())

        # k-shotでadaptation
        for iter in range(train_step):
            logits = model.adaptation(test_x, weights)
            loss = loss_fn(logits, test_y)
            gradients = torch.autograd.grad(loss, weights.values())

            weights = OrderedDict((name, param - lr * grad) for ((name, param), grad) in zip(weights.items(), gradients))

        # queryで評価
        with torch.no_grad():
            # input_x = x_val[idx].to(device)
            # input_y = y_val[idx].to(device)
            # logits = model.adaptation(input_x, weights)
            # loss = loss_fn(logits, input_y)
            logits = model.adaptation(val_x, weights)
            loss = loss_fn(logits, val_y)

            y_pred = logits.softmax(dim=1)
            #print("y_pred : " , y_pred)
            s, predicted = torch.max(y_pred, 1) 
            #score = s[0]
            # print("score : ", score)
            # score = score.to("cpu").detach().numpy().copy()
            
            predictions.append(y_pred)
            #print("predicted : " , predicted)
            labels.append(test_y)
         

    y_pred = torch.cat(predictions)
    #print("y_pred" , y_pred)
    y_label = torch.cat(labels)
    batch_acc = torch.eq(y_pred.argmax(dim=-1), y_label).sum().item() / y_pred.shape[0]
    #return loss, batch_acc ,predicted
    
    return loss, batch_acc ,predicted