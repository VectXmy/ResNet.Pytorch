__author__="xu hongtao"
__email__="xxxmy@foxmail.com"

from res_block import basic_2d,bottlneck_2d
import torch.nn as nn
import torch.nn.functional as F
import torch



class ResNet(nn.Module):
    def __init__(self,
                block,#残差块
                blocks,#每个stage的残差块数量
                include_top=True,#是否包含分类层头部
                class_num=1000,#分类类别个数
                per_block_exp=1):#使用的残差块通道扩展倍数，basic块是1，bottleneck块是4
        super(ResNet,self).__init__()
        self.pre=nn.Sequential(
                            nn.Conv2d(3, 64, (7,7), stride=(2,2), padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d((3,3), stride=(2,2),padding=1)
                            )

        self.blocks=blocks
        self.include_top=include_top
        self.class_num=class_num
        self.per_block_exp=per_block_exp

        layers=[]
        
        in_features=64
        out_features=64

        for stage_id,iterations in enumerate(self.blocks):
            for block_id in range(iterations): 
                         
                if block_id==0 and stage_id>0:
                    in_features=out_features*self.per_block_exp//2
                elif block_id==0 and stage_id==0:
                    in_features=out_features
                elif block_id>0 :
                    in_features=out_features*self.per_block_exp
                
                
                # self.__dict__["stage%s_block%s"%(str(stage_id),str(block_id))]=block(in_features,out_features,stage=stage_id,block=block_id)
                layers.append(block(in_features,out_features,stage=stage_id,block=block_id))


            if stage_id!=(len(self.blocks)-1):
                out_features*=2
            
        
        self.layers=nn.Sequential(*layers)
        

        if self.include_top:
            assert self.class_num>0
            self.fc_with_softmax=nn.Sequential(
                            nn.Linear(out_features*self.per_block_exp,self.class_num),
                            nn.Softmax()
                            )
        self.initialize_weights()

    def forward(self,x):
        x=self.pre(x)
        
        if self.include_top:
            x=self.layers(x)
            x=F.adaptive_avg_pool2d(x,1)
            x=x.view(x.size(0),-1)
            x=self.fc_with_softmax(x)
            return x
        else:
            output=[]
            n=0
            for i ,num in enumerate(self.blocks):
                x=self.layers[n:n+num](x)
                n+=num
                # print("x:%d"%(i),x.size(),n)
                output.append(x)
            
            return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                torch.nn.init.normal_(m.weight.data,0,0.01)
                m.bias.data.zero_()
                
class  ResNet18(ResNet):
    def __init__(self,
            blocks=None,#每个stage的残差块数量
            include_top=True,
            class_num=1000,
            ):
        if blocks is None:
            blocks=[2,2,2,2]
        super(ResNet18,self).__init__(basic_2d,
                                    blocks,
                                    include_top=include_top,
                                    class_num=class_num,
                                    per_block_exp=1)
class ResNet34(ResNet):
    def __init__(self,
            blocks=None,#每个stage的残差块数量
            include_top=True,
            class_num=1000,
            ):
        if blocks is None:
            blocks=[3,4,6,3]
        super(ResNet34,self).__init__(basic_2d,
                                    blocks,
                                    include_top=include_top,
                                    class_num=class_num,
                                    per_block_exp=1)

class ResNet50(ResNet):
    def __init__(self,
            blocks=None,#每个stage的残差块数量
            include_top=True,
            class_num=1000,
            ):
        if blocks is None:
            blocks=[3,4,6,3]
        super(ResNet50,self).__init__(bottlneck_2d,
                                    blocks,
                                    include_top=include_top,
                                    class_num=class_num,
                                    per_block_exp=4)

class ResNet101(ResNet):
    def __init__(self,
            blocks=None,#每个stage的残差块数量
            include_top=True,
            class_num=1000,
            ):
        if blocks is None:
            blocks=[3,4,23,3]
        super(ResNet101,self).__init__(bottlneck_2d,
                                    blocks,
                                    include_top=include_top,
                                    class_num=class_num,
                                    per_block_exp=4)
if __name__=="__main__":
    import cv2 ,torch
    import numpy as np
     
    input1=np.ones((128,128,3))
    input1=input1[np.newaxis,...]
    input1=torch.Tensor(input1)
    input1=input1.permute(0,3,1,2)
    net=ResNet18(include_top=True,class_num=3)

    out=net(input1)
    print(out)
    