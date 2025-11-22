import torch
import torch.nn as nn


def conv_dilation_1d_norelu(in_channel, out_channel, kernel_size=3, strid=1, groups=1, dilation=1,padding=-1):  
    
    if padding == -1:
        padding = (kernel_size + (kernel_size-1)*(dilation-1) -1) // 2
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size, strid, padding=padding, groups=groups, bias=False, dilation=dilation),  
        nn.BatchNorm1d(out_channel),  
    )

def conv_dilation_1d_dim2(in_channel, out_channel, kernel_size=3, strid=1, groups=1, dilation=1,padding=-1):  
    
    if padding == -1:
        padding = (kernel_size + (kernel_size-1)*(dilation-1) -1) // 2
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size, strid, padding=padding, groups=groups, bias=False, dilation=dilation),  
        nn.BatchNorm1d(out_channel),  
        nn.ReLU6(),
        nn.Conv1d(out_channel, out_channel, kernel_size, 1, padding=padding, groups=groups, bias=False, dilation=dilation),  
        nn.BatchNorm1d(out_channel),  
    )
            
class MSFEM(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3, strid=None, groups=1, dilation_list:list=[1,3,5,7],padding=-1):
        super(MSFEM,self).__init__()
        if out_channel/in_channel==2:
            self.downsample = True
        elif out_channel/in_channel==1 or in_channel==1:
            self.downsample = False
        else:            
            breakpoint()

        if strid==None:
            if self.downsample:
                strid=2
            else:
                strid=1

        self.conv = 0
        self.conv_len = len(dilation_list)

        for d_i in range(self.conv_len):
            exec('self.conv{} = conv_dilation_1d_dim2(in_channel, out_channel, kernel_size, strid, groups, dilation=dilation_list[d_i],padding=-1)'.format(d_i))             
        self.relu = nn.ReLU6()
        
        if self.downsample:
            self.ds_layer = conv_dilation_1d_norelu(in_channel, out_channel, 1, strid, groups, dilation=1,padding=-1)
    def forward(self,x0):
        x1 = eval('self.conv{}'.format(0))(x0)
        for i in range(1,self.conv_len):
            x1 += eval('self.conv{}'.format(i))(x0)
        if self.downsample:
            x0 = self.ds_layer(x0)            
        x2 = x1+x0
        x2 = self.relu(x2)
        return x2
    
class SomnoNet_encoder(nn.Module):
    def __init__(self, num_classes=5,sample=200):
        super().__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(1,sample*3//2), stride=(1,sample*3//4))
        self.conv_res0 = MSFEM(1,16,3,dilation_list=[1,3,5])
        self.conv_res1 = MSFEM(16,32,3,dilation_list=[1,3,5])
        self.conv_res2 = MSFEM(32,32,3,dilation_list=[1,3,5])
        self.pool1 = nn.MaxPool1d(2,2)
        self.pool2 = nn.MaxPool1d(2,2)
        self.pool3 = nn.MaxPool1d(2,2)       
        self.avgpool1=nn.AdaptiveAvgPool1d(1)
    def forward(self, x): 
        x_shape = x.shape
        x0 = x.chunk(x.shape[0], 0)
        x0 = torch.cat(x0, 1).squeeze(0)                
        x1 = self.unfold(x0.unsqueeze(1))
        x1 = torch.transpose(x1,1,2)                    
        x1_shape = x1.shape
        x_1 = x1.chunk(x1.shape[0], 0)
        x_1 = torch.cat(x_1, 1).squeeze(0).unsqueeze(1)   
        x2 = self.conv_res0(x_1)
        x2 = self.pool1(x2)                                 
        x3 = self.conv_res1(x2)
        x3 = self.pool2(x3)  
        x4 = self.conv_res2(x3)                               
        x6 = self.avgpool1(x4).squeeze(2)                   
        x7 = x6.unsqueeze(0).chunk(x1_shape[0], 1)
        x7 = torch.cat(x7)                                  
        x8 = x7.unsqueeze(0).chunk(x_shape[0], 1)   
        x8 = torch.cat(x8)                                
        return (x8) 
    
class SomnoNet_decoder_voting(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()  
        self.fc1 = nn.Linear(32,5)   
        self.avg_pool = nn.AdaptiveAvgPool1d(1)     
    def forward(self,x):                            
        x1 = self.fc1(x.squeeze(0))
        x2 = x1.transpose(1,2)
        x2 = self.avg_pool(x2).squeeze(-1).unsqueeze(0)
        return x2
    
class SomnoNet_decoder_vector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.fc1 = nn.Linear(32,5)   
        self.avg_pool = nn.AdaptiveAvgPool1d(1)     
    def forward(self,x):                                
        x1 = x.squeeze(0)                               
        x2 = x1.transpose(1,2)                          
        x2 = self.avg_pool(x2).squeeze(-1)              
        x2.requires_grad_(True)
        x3 = self.fc1(x2)                               
        x4 = x3.unsqueeze(0)                            
        return x4, x2
    
class SomnoNet_decoder_timeSeries(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.rnn0 = nn.GRU(32,32,num_layers=1, bidirectional=True,batch_first=True)
        self.fc1 = nn.Linear(64,5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)     
    def forward(self,x):                            
        x.requires_grad_(True)
        x0 = x.chunk(x.shape[1],1)    
        x0 = torch.cat(x0,2).squeeze(1)             
        x1,_ = self.rnn0(x0)                        
        x3 = x1.unsqueeze(0).chunk(x.shape[0],1)
        x3 = torch.cat(x3)
        x4 = x3.chunk(x.shape[1],2)
        x4 = torch.cat(x4,1)
        x5 = x4.transpose(2,3).squeeze(0)
        x6 = self.avg_pool(x5).squeeze(-1)
        x7 = self.fc1(x6).unsqueeze(0)
        return x7, x
    
class SomnoNet(nn.Module):
    def __init__(self, num_classes=5,data_freq=200):
        super().__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(1,data_freq*3//2), stride=(1,data_freq*3//4))
        self.conv_res0 = MSFEM(1,16,3,dilation_list=[1,3,5])
        self.conv_res1 = MSFEM(16,32,3,dilation_list=[1,3,5])
        self.conv_res2 = MSFEM(32,64,3,dilation_list=[1,3,5])
        self.pool1 = nn.MaxPool1d(2,2)
        self.pool2 = nn.MaxPool1d(2,2)
        self.pool3 = nn.MaxPool1d(2,2)
        self.pool4 = nn.MaxPool1d(2,2)
        self.fc1 = nn.Linear(128,32)
        self.fc1_2 = nn.Linear(32,num_classes)
        self.avg1_3 = nn.AdaptiveAvgPool1d(1)
        self.batch1 = nn.BatchNorm1d(32)
        self.batch1_2 = nn.BatchNorm1d(39)
        self.avgpool1=nn.AdaptiveAvgPool1d(1)
        self.lstm_0 = nn.GRU(input_size=64, hidden_size=64,num_layers = 5,bidirectional=True,batch_first=True)
        self.drp = nn.Dropout1d()
        self.lstm = nn.GRU(input_size=32, hidden_size=32,num_layers = 1, bidirectional=False,batch_first=True)
    def forward(self, x):  
        x_shape = x.shape
        x0 = x.chunk(x.shape[0], 0)
        x0 = torch.cat(x0, 1).squeeze(0)                
        x1 = self.unfold(x0.unsqueeze(1))
        x1 = torch.transpose(x1,1,2)                    
        x1_shape = x1.shape
        x_1 = x1.chunk(x1.shape[0], 0)
        x_1 = torch.cat(x_1, 1).squeeze(0).unsqueeze(1)   
        x2 = self.conv_res0(x_1)
        x3 = self.conv_res1(x2)
        x4 = self.conv_res2(x3)                               
        x6 = self.avgpool1(x4).squeeze(2)                   
        x6, _ = self.lstm_0(x6)
        x6 = self.batch1(self.fc1(x6))                      
        x7 = x6.unsqueeze(0).chunk(x1_shape[0], 1)
        x7 = torch.cat(x7)                                  
        x9, _ = self.lstm(x7)                          
        x10 = self.batch1_2(self.fc1_2(x9))            
        x11 = torch.transpose(x10, 1, 2)                    
        x12 = self.avg1_3(x11).squeeze(2)                    
        x13 = x12.unsqueeze(0).chunk(x_shape[0], 1)   
        x13 = torch.cat(x13)                                
        return (x13)     