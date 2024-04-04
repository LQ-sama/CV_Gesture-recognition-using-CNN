# %%
import os
import torch
from PIL import Image
import data
import pandas as pd
import numpy as np

# %%
number_to_label={
    0:'A', 1:'B', 2:'C', 3:'Five', 4:'Points', 5:'V'
}

# %%
#得到单张图片的输出标签，img:图片路径，model：模型
def get_img_type(img,model,device='cpu'):
    img=Image.open(img)
    test_transforms=data.test_transforms()
    img=test_transforms(img)
    if img.shape[0]==1:
        img=img.repeat(3,1,1)
    img=torch.unsqueeze(img,0)
    img=img.to(device)
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        label=model(img)
    label=torch.argmax(label,dim=1).item()
    label=number_to_label[label]
    return label

# %%
#得到一个图片文件夹下所有图片的标签，img_file:图片文件夹路径，返回值是一个列表，列表中的元素为[图片名称，图片种类]
def get_img_file_type(img_file,model,device='cpu'):
    type_list=[]
    for i in os.listdir(img_file):
        img_path=img_file+'/'+i
        img_name=i
        img_label=get_img_type(img_path,model,device=device)
        type_list.append([img_name,img_label])
    return type_list

# %%
#把type_list存到csv文件中
def name_and_label_to_csv(type_list,csv='./type.csv'):
    type_list=np.array(type_list)
    type_list=pd.DataFrame({"img_name":type_list[:,0],"img_label":type_list[:,1]})
    type_list.to_csv(csv,index=False)

# %%
model=torch.load('./model.pth')
print(get_img_type('./Hand_Posture_Hard_Stu/A/A-complex01.png',model))
type_list=get_img_file_type('./Hand_Posture_Hard_Stu/A/',model,device='cuda:0')
print(type_list)
name_and_label_to_csv(type_list,'./type.csv')

# %%



