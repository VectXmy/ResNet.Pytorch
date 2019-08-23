__author__="xu hongtao"
__email__="xxxmy@foxmail.com"

import resnet
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch



transformer=transforms.Compose([
                    transforms.Resize((250,250)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(10,(10,10)),
                    transforms.RandomResizedCrop((224,224),scale=(0.85,1.0)),
                    transforms.ToTensor()
                    ])
train_dataset=ImageFolder("./data",transform=transformer)
print(train_dataset.class_to_idx)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)

model=resnet.ResNet18(class_num=3).cuda()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,weight_decay=1e-4,momentum=0.9)
criterion=torch.nn.CrossEntropyLoss().cuda()
model.train()

for epoch in range(30):
    for input,target in train_dataloader:
        input=torch.Tensor(input).cuda()
        target=torch.Tensor(target).long().cuda()
        out=model(input)
        loss=criterion(out,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
