import os
import torch
print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import read_bci_data


class DeepConvNet(nn.Module):
    def __init__(self,activation=nn.ELU()):
        super(DeepConvNet,self).__init__()
        self.conv0=nn.Conv2d(1,25,kernel_size=(1,5))
        channels=[25,25,50,100,200]
        kernel_sizes=[None,(2,1),(1,5),(1,5),(1,5)]
        for i in range(1,len(channels)):
            setattr(self,'conv'+str(i),nn.Sequential(
                nn.Conv2d(channels[i-1],channels[i],kernel_size=kernel_sizes[i]),
                nn.BatchNorm2d(channels[i],eps=1e-5,momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            ))
        self.classify=nn.Linear(8600,2)
    def forward(self,X):
        out=self.conv0(X)
        out=self.conv1(out)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=out.view(out.shape[0],-1)
        neurons=out.shape[0]
        out=self.classify(out)
        return out

'''
Train & Evaluate
loss function: cross entropy
optimizer: adam
learning rate: 0.001
weight decay: 0.01
epoch: 300
activation fuction: ReLU,LeakyReLU,ELU
'''
lr=0.001
epochs=300

def train_with_different_activation(loader_train,loader_test,activations,device):
    """
    Args:
        loader_train: training dataloader
        loader_test: testing dataloader
        activations: {ReLU,LeakyReLU,ELU} pytorch layer
        
        device: pytorch device gpu,cpu
    Return:
        dataframe: with column 'epoch','ReLU_train','ReLU_test','LeakyReLU_train'...
        best_model_wts: models' weight with the best evaluated accuracy
    """
    Loss=nn.CrossEntropyLoss()
    df=pd.DataFrame()
    df['epoch']=range(1,epochs+1)
    best_model_wts={'ReLU':None,'LeakyReLU':None,'ELU':None}
    best_evaluated_acc={'ReLU':0,'LeakyReLU':0,'ELU':0}
    for name,activation in activations.items():
        """
        train model with an specific activation function
        """
        model=DeepConvNet(activation)
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.01)
        acc_train=list()
        acc_test=list()
        for epoch in range(1,epochs+1):
            """
            train
            """
            model.train()
            total_loss=0
            correct=0
            for idx,(data,target) in enumerate(loader_train):
                data=data.to(device,dtype=torch.float)
                target=target.to(device,dtype=torch.long) #target type has to be 'long'
                predict=model(data)
                loss=Loss(predict,target)
                total_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(target).sum().item()
                """
                update
                """
                optimizer.zero_grad()
                loss.backward()  # bp
                optimizer.step()
            total_loss/=len(loader_train.dataset)
            correct=100.*correct/len(loader_train.dataset)
            if epoch%10==0:
                print(f'epcoh{epoch:>3d}  loss:{total_loss:.4f}  acc:{correct:.1f}%')
            acc_train.append(correct)
            """
            test
            """
            model.eval()
            correct=evaluate(model,loader_test,device)
            acc_test.append(correct)
            # update best_model_wts
            if correct>best_evaluated_acc[name]:
                best_evaluated_acc[name]=correct
                best_model_wts[name]=copy.deepcopy(model.state_dict())
        df[name+'_train']=acc_train
        df[name+'_test']=acc_test

    return df,best_model_wts


def evaluate(model,loader_test,device):
    model.eval()
    correct=0
    for idx,(data,target) in enumerate(loader_test):
        data=data.to(device,dtype=torch.float)
        target=target.to(device,dtype=torch.long)
        predict=model(data)
        correct+=predict.max(dim=1)[1].eq(target).sum().item()
    
    correct=100.*correct/len(loader_test.dataset)
    return correct


def plot(dataframe):
    fig=plt.figure(figsize=(10,6))
    for name in dataframe.columns[1:]:
        plt.plot('epoch',name,data=dataframe)
    plt.legend()
    return fig

if __name__ == '__main__':

	X_train,y_train,X_test,y_test=read_bci_data()
	dataset=TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
	loader_train=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=4)
	dataset=TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test))
	loader_test=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=4)
	print(f'test dataset:\n{dataset[:3]}')
	
	
	randi=int(np.random.randint(0,X_train.shape[0],1))
	print(f'sample_id:{randi}')
	plt.figure(figsize=(10,2))
	plt.plot(X_train[randi,0,0])
	plt.figure(figsize=(10,2))
	plt.plot(X_train[randi,0,1])
	
	activations={'ReLU':nn.ReLU(),'LeakyReLU':nn.LeakyReLU(),'ELU':nn.ELU()}
	df,best_model_wts=train_with_different_activation(loader_train,loader_test,activations,device)
	for name,model_wts in best_model_wts.items():
	    torch.save(model_wts,os.path.join('deepconvnet_models',name+'111.pt'))


	figure=plot(df)
	figure.savefig('deepconvnet1.png')
	for column in df.columns[1:]:
		print(f'{column} max acc: {df[column].max()}')
