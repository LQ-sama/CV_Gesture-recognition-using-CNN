# %%
from sklearn.metrics import f1_score
from torchvision import models
from torch import nn
import torch


# %%
# 得到resnet50，并在其之后加几个线性层，weights=models.ResNet50_Weights.DEFAULT表示用训练好的参数初始化
def get_net():    
   # net=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    net = models.resnet50(pretrained=True)
    net.add_module('fc1',module=nn.Linear(1000,64))
    net.add_module('relu',module=nn.ReLU())
    net.add_module('fc2',module=nn.Linear(64,25))
    return net

# %%
# 计算网络在对应loader上的准确率
def acc0(loader,model,device='cuda:0'):
    acc=0
    model.eval()
    for x,y in loader:
        x=x.to(device)
        y=y.to(device)
        y_hat=model(x)
        y_hat=torch.argmax(y_hat,dim=1)
        acc+=torch.sum(y==y_hat)
    return acc.item()/len(loader.dataset)

#下面是marco-f1指标
def acc(loader, model, device='cuda:0'):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            y_pred_batch = torch.argmax(output, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_batch.cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return macro_f1