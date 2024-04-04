# %%
import net
from torch import nn
import torch
import data

# %%
data.data_to_csv('./workers_ind/workers_data/train1',Shuffle=True)

data.csv_split_train_test_valid('./workers_ind/data.csv',train=0.6,test=0.2,valid=0.2,Shuffle=True)

# %%
train_transforms=data.train_transforms()
test_transforms=data.test_transforms()
trainloader=data.imgloader('./workers_ind/train.csv',transforms=train_transforms,batch_size=16,shuffle=True)
validloader=data.imgloader('./workers_ind/valid.csv',transforms=test_transforms,batch_size=1)
testloader=data.imgloader('./workers_ind/test.csv',transforms=test_transforms,batch_size=1)

# %%
device: str='cuda:0' if torch.cuda.is_available() else 'cpu'#没有n卡的话改为cpu即可

# %%
model=net.get_net()
model=model.to(device)
loss=nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=1e-2)
# 随着训练逐步减小学习率
lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optim,step_size=5,gamma=0.2)

# %%
epochs=10
print('第',0,'次验证集正确率:',net.acc(validloader,model,device=device))
for i in range(epochs):
    model.train()
    for x,y in trainloader:
        x=x.to(device)
        y=y.to(device)
        y_hat=model(x)
        l=loss(y_hat,y)
        l.backward()
        optim.step()
        optim.zero_grad()
    lr_scheduler.step()

    print('第',i+1,'次验证集正确率:',net.acc(validloader,model,device=device))

# %%
print('测试集准确率:',net.acc(testloader,model,device=device))


