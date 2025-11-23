import numpy as np
from collections import defaultdict
import torch, os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from scipy.fftpack import fft as sp_fft
import torch.fft as pt_fft
from scipy.signal import butter, lfilter, freqz
import glob, math
import logging
logger = logging.getLogger("default_log")



def split_minibatch(x,minibatch=20,pad=0):  
    x_shape = list(x.shape)  
    assert x_shape[0]==1
    if x_shape[0]<minibatch:
        quit_len = x.shape[1]%minibatch
        if quit_len != 0:
            if pad==0:
                x = x[:,:-quit_len]
            elif pad==1:
                pad_len = minibatch-quit_len
                x_shape[1] = pad_len
                x = torch.cat([torch.zeros(x_shape,dtype=x.dtype),x],1)
            else:
                breakpoint()
        x = x.chunk(x.shape[1]//minibatch, 1)    
        x = torch.cat(x) 
    if pad==1:     
        return x,pad_len
    else:
        return x


def freeze(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False

def net_param(net):
    pre_total = sum([param.nelement() for param in net.parameters()])    
    if pre_total<1e6:
        return("Number of parameter: %.2fK" % (pre_total/1e3))
    else:
        return("Number of parameter: %.2fM" % (pre_total/1e6)) 

def net_param_k(net):
    pre_total = sum([param.nelement() for param in net.parameters()])    
    return pre_total/1e3
   
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def save_seq_ids(fname, ids):
    """Save sequence of IDs into txt file."""
    with open(fname, "w") as f:
        for _id in ids:
            f.write(str(_id) + "\n")

def load_seq_ids_fname(fname):
    """Load sequence of IDs from txt file."""
    ids = []
    with open(fname, "r") as f:
        for line in f:
            ids.append(int(line.strip()))
    ids = np.asarray(ids)
    return ids

def load_seq_ids(dataset_dir):
    data_path_list = glob.glob(os.path.join(dataset_dir,"*.npz"))
    data_path_list.sort()
    data_path_list_new=[]
    for i in data_path_list:
        data_path_list_new.append(os.path.basename(i).replace('tr','').replace('.npz',''))   
    return data_path_list_new


def print_n_samples_each_class(labels):
    """Print the number of samples in each class."""

    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        logger.info("{}: {}".format(str(c), n_samples))


def compute_portion_each_class(labels):
    """Determine the portion of each class."""

    n_samples = len(labels)
    unique_labels = np.unique(labels)
    class_portions = np.zeros(len(unique_labels), dtype=np.float32)
    for c in unique_labels:
        n_class_samples = len(np.where(labels == c)[0])
        class_portions[c] = n_class_samples/float(n_samples)
    return class_portions


def get_balance_class_oversample(x, y):
    """Balance the number of samples of all classes by (oversampling).

    The process is as follows:
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """

    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def get_balance_class_sample(x, y):
    """Balance the number of samples of all classes by sampling.

    The process is as follows:
        1. Find the class that has the smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        sample_idx = np.random.choice(idx, size=n_min_classes, replace=False)
        balance_x.append(x[sample_idx])
        balance_y.append(y[sample_idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y

def compute_adjustment(train_loader, tro, GPU): # len(train_loader) 1751
    """compute the base probabilities"""
    # breakpoint()
    label_freq = {}
    for i, target in enumerate(train_loader[1]):
        if GPU:
            target = target #target.shape torch.Size([20, 1])
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    # breakpoint()
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                # breakpoint()
                self.alpha = Variable(torch.tensor(alpha)*torch.ones(class_num, 1))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        # breakpoint()

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Focal_Loss():
	def __init__(self,weight,gamma=2):
		super(Focal_Loss,self).__init__()
		self.gamma=gamma
		self.weight=weight
	def forward(self,preds,labels):
		"""
		preds:softmax输出结果
		labels:真实值
		"""
		eps=1e-7
		y_pred =preds.view((preds.size()[0],preds.size()[1],-1)) #B*C*H*W->B*C*(H*W)
		
		target=labels.view(y_pred.size()) #B*C*H*W->B*C*(H*W)
		
		ce=-1*torch.log(y_pred+eps)*target
		floss=torch.pow((1-y_pred),self.gamma)*ce
		floss=torch.mul(floss,self.weight)
		floss=torch.sum(floss,dim=1)
		return torch.mean(floss)


class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0,save_epoch_nums=0):
        self.save_epoch_nums = save_epoch_nums
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.test_acc_fine = np.Inf  
        self.delta = delta
    def __call__(self,epoch, num_epochs,train_loss,val_loss,test_loss,test_acc,model,path):
        print("epoch={}/{}---".format(epoch, num_epochs),"[ train_loss={:.3f}".format(train_loss),"val_loss={:.3f}".format(val_loss),"test_loss={:.3f} ]".format(test_loss),"test_acc={:.3f}".format(test_acc))
        score = -val_loss
        if self.best_score is None:
            if epoch>self.save_epoch_nums:
                self.best_score = score
                result = self.save_checkpoint(val_loss,test_acc,model,path)
                return result
        elif score < self.best_score+self.delta:
            if epoch>self.save_epoch_nums:

                self.counter+=1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter>=self.patience:
                    self.early_stop = True
            return 0
        else:
            self.best_score = score
            if epoch>self.save_epoch_nums:
                result = self.save_checkpoint(val_loss,test_acc,model,path)
            else:
                result = 0
            self.counter = 0
            return result
    def save_checkpoint(self,val_loss,test_acc,model,path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            print(f'test acc decreased ({self.test_acc_fine:.6f} --> {test_acc:.6f})')
            print('Saving model ...')
        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss
        self.test_acc_fine = test_acc
        return self.test_acc_fine
    
class SPP100(nn.Module):
    def __init__(self,c1,c2,c3):
        super(SPP100, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=c1,stride=1,padding=c1 // 2)
        self.pool2 = nn.MaxPool1d(kernel_size=c2, stride=1, padding=c2 // 2)
        self.pool3 = nn.MaxPool1d(kernel_size=c3, stride=1, padding=c3 // 2)
    def forward(self,x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x,x1,x2,x3],dim=1)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1,nop=0):
        super(Focus, self).__init__()
        self.nop = nop
        self.conv = nn.Conv1d(c1 * 4, c2, k, 1)
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = x[..., : x.size(dim=-1)-self.nop]
        x = self.conv(torch.cat([x[..., ::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]], 1))
        return x

class atten_celoss(object):
    def __init__(self,classes_1_idx,classes_2_idx):
        self.c1_idx = classes_1_idx
        self.c2_idx = classes_2_idx
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.eps = 1e-6
    def __call__(self, pred, target):
        pred_value = torch.argmax(pred,-1)
        F1 = (pred_value == self.c1_idx) * (target == self.c2_idx)
        F2 = (pred_value == self.c2_idx) * (target == self.c1_idx)
        weight = F1+F2
        result = torch.sum(self.criterion(pred,target)*weight)/(torch.sum(weight)+self.eps)
        return result
    
class EarlyStopping_cmatrix():
    def __init__(self,patience=7,verbose=False,delta=0,save_epoch_nums=0,save_model=False,num_epochs=None):
        self.save_epoch_nums = save_epoch_nums
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.test_acc_fine = np.Inf  
        self.delta = delta
        self.save_data = False
        self.save_model=save_model
        self.num_epochs = num_epochs

    def __call__(self, val_loss,model,path,epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_data = False
            self.save_checkpoint(val_loss, model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
            self.save_data = False
            if epoch == self.num_epochs-1:
                self.save_data = True
                self.save_checkpoint(val_loss, model,path)

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,path)
            self.counter = 0
            # self.save_data = True
        # if epoch == self.num_epochs-1:
        #     self.save_checkpoint(val_loss, model,path)

    def save_checkpoint(self, val_loss, model, path):
        # print('!!!!!!!!!!!!!!')
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            # print(f'test acc decreased ({self.test_acc_fine:.6f} --> {test_acc:.6f})')
            print('Saving model ...')
        self.model = model
        if self.save_model:
            torch.save(self.model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss
        self.save_data = True
        # self.test_acc_fine = test_acc
import csv
from datetime import datetime

def save_eval_data(model, save_dir,save_name='eval_data',date_time=None,test_loader=None,test_sids=None):
    
    if date_time == None:
        date_time = datetime.now().strftime('%Y%m%d%H')
    save_name = save_name+date_time+'.csv'
    save_path = os.path.join(save_dir,save_name)

    eval_result = []
    with torch.no_grad():
        model.eval()
        for idx,(x,y) in enumerate(test_loader):
            x = torch.transpose(x,1,2)       
            x = x.to(device='cuda:0')
            y = y.to(device='cuda:0')
            outputs = model(x)
                   
            outputs = outputs.reshape(-1, outputs.shape[len(outputs.shape) - 1])
            y = y.reshape(-1)
            pred = F.softmax(outputs, dim=1)
            pred = torch.argmax(pred, dim=1)
            test_acc = (pred == y).sum() / len(y)
            eval_result.append([test_sids[idx],str(np.round(test_acc.cpu().item(),3))])
    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for result_i in eval_result:
            writer.writerow(result_i)
            
def draw_cm(pred_list,label_list,c_matrix_dir,test_sids,test_sid_nums,classes):
    matrix_png_list=[]
    show_png_list=[]
    assert len(pred_list)==len(label_list)==len(test_sid_nums)
    for human_i in range(len(test_sids)):
        
        m_sub_dir = os.path.join(c_matrix_dir,test_sids[human_i])
        if not os.path.exists(m_sub_dir):
            os.makedirs(m_sub_dir)          
        # image_path = str(c_matrix_dir)+'/'+str(test_sids[human_i])
        if not os.path.exists(c_matrix_dir):
            os.makedirs(c_matrix_dir)
        # marix
        C_i = confusion_matrix(label_list[human_i], pred_list[human_i],labels=np.arange(classes))
        plt.matshow(C_i, cmap=plt.cm.Blues)
        # plt.colorbar()
        for i in range(len(C_i)):
            for j in range(len(C_i)):
                plt.annotate(C_i[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')     
        # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
        # num_label = [0,1,2,3,4,5]
        # ann_labels = ['W','N1','N2','N3','REM','Move']
        num_label = [0,1,2,3,4]
        trick_label = [2,3,4,5,6]
        ann_labels = ['W','N1','N2','N3','REM']
        trick_show = ['N3','N2','N1','REM','Wake']

        plt.xticks(num_label,ann_labels)
        plt.yticks(num_label,ann_labels)

        plt.title('test_sids:{}'.format(test_sids[human_i]))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # name_list=['W','N1','N2','N3','REM','move']
        # .xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))
        cm_png_dir = '{0}/{1}/matrix_{1}.png'.format(c_matrix_dir,test_sids[human_i])
        plt.savefig(cm_png_dir)
        plt.close()


        # cycle
        len_y = len(label_list[human_i])
        # breakpoint()
        draw_label = y2draw_y(label_list[human_i].numpy().tolist())
        draw_pred = y2draw_y(pred_list[human_i].numpy().tolist())
        error = np.where(np.array(draw_label)-np.array(draw_pred)!=0)[0]
        error_idx = np.append(np.insert(np.where(np.diff(error)!=1)[0],0,-1),len(error)-1)
        assert len(draw_label) == len(draw_pred)
        x = np.linspace(0,len_y,len(draw_label))
        # breakpoint()
        fig, ax = plt.subplots(2,1)
        # draw_label 
        fig.suptitle('test_sids:{}'.format(test_sids[human_i]),fontsize=75)
        ax[0].set_title('Label show',fontsize=50)
        ax[0].plot(x,draw_label,linewidth=3)
        ax[0].set_xlim(0,len_y)
        ax[0].set_xticks(np.linspace(0,len_y,2))
        ax[0].set_ylim(1.5,7)
        ax[0].set_yticks(trick_label)
        ax[0].set_yticklabels(trick_show)
        ax[0].tick_params(labelsize=35)
        ax[0].fill([0,len_y,len_y,0],[1.5,1.5,3,3],facecolor='BLUE',alpha=0.3)
        ax[0].fill([0,len_y,len_y,0],[3,3,4,4],facecolor='BLUE',alpha=0.25)
        ax[0].fill([0,len_y,len_y,0],[4,4,5,5],facecolor='BLUE',alpha=0.2)
        ax[0].fill([0,len_y,len_y,0],[5,5,6,6],facecolor='BLUE',alpha=0.15)
        ax[0].fill([0,len_y,len_y,0],[6,6,7,7],facecolor='BLUE',alpha=0.1)
        # ax[0].fill([0,len_y,len_y,0],[6.5,6.5,7,7],facecolor='BLUE',alpha=0.05)
        ax[0].grid(True,color="blue", linestyle='--',linewidth="0.5",axis="y")

        # draw_pred
        ax[1].set_title('Pred show',fontsize=50)
        ax[1].plot(x,draw_pred,linewidth=3)
        ax[1].set_xlim(0,len_y)
        ax[1].set_xticks(np.linspace(0,len_y,2))
        ax[1].set_ylim(1.5,7)
        ax[1].set_yticks(trick_label)
        ax[1].set_yticklabels(trick_show)
        ax[1].tick_params(labelsize=35)
        ax[1].fill([0,len_y,len_y,0],[1.5,1.5,3,3],facecolor='BLUE',alpha=0.3)
        ax[1].fill([0,len_y,len_y,0],[3,3,4,4],facecolor='BLUE',alpha=0.25)
        ax[1].fill([0,len_y,len_y,0],[4,4,5,5],facecolor='BLUE',alpha=0.2)
        ax[1].fill([0,len_y,len_y,0],[5,5,6,6],facecolor='BLUE',alpha=0.15)
        ax[1].fill([0,len_y,len_y,0],[6,6,7,7],facecolor='BLUE',alpha=0.1)
        # ax[1].fill([0,len_y,len_y,0],[6.5,6.5,7,7],facecolor='BLUE',alpha=0.05)

        ## draw error
        for error_idx_idx in range(len(error_idx)-1):
            x_begin=x[error[error_idx[error_idx_idx]+1]]
            x_end=x[error[error_idx[error_idx_idx+1]]]
            # y_begin=draw_pred[error[error_idx[error_idx_idx]+1]]-1
            y_begin=1.5
            y_end=draw_pred[error[error_idx[error_idx_idx]+1]]
            # print(x_begin,x_end,y_begin,y_end)
            # breakpoint()
            ax[1].fill([x_begin,x_end,x_end,x_begin],\
                        [y_begin,y_begin,y_end,y_end],facecolor='darkslategray',alpha=0.5)


        ax[1].grid(True,color="blue", linestyle='--',linewidth="0.5",axis="y")
        fig.set_size_inches(100, 10)
        show_png_dir = '{0}/{1}/show_cycle_{1}.png'.format(c_matrix_dir,test_sids[human_i])
        plt.tight_layout()
        plt.savefig(show_png_dir)
        plt.close()

        matrix_png_list.append(cm_png_dir)
        show_png_list.append(show_png_dir)
# sum cm
    
    C_i = confusion_matrix(np.concatenate(label_list), np.concatenate(pred_list),labels=np.arange(classes))
    plt.matshow(C_i, cmap=plt.cm.Blues)
    # plt.colorbar()
    for i in range(len(C_i)):
        for j in range(len(C_i)):
            plt.annotate(C_i[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center') 
    plt.xticks(num_label,ann_labels)
    plt.yticks(num_label,ann_labels)

    plt.title('test_sum')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # name_list=['W','N1','N2','N3','REM','move']
    # .xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))
    cm_png_dir = '{0}/matrix_sum.png'.format(c_matrix_dir)
    plt.savefig(cm_png_dir)
    plt.close()

    return matrix_png_list, show_png_list


def y2draw_y(y,freq=30):
    draw_y = []
    for i in range(len(y)):
        if y[i]==0:
            for j in range(freq):
                draw_y.append(6)
        elif y[i]==1:
            for j in range(freq):
                draw_y.append(4)
        elif y[i]==2:
            for j in range(freq):
                draw_y.append(3)        
        elif y[i]==3:
            for j in range(freq):       
                draw_y.append(2)
        elif y[i]==4:
            for j in range(freq):
                draw_y.append(5)
        elif y[i]==5:
            for j in range(freq):
                draw_y.append(6.5)                
        else:
            for j in range(freq):
                draw_y.append(1)
    assert len(y) == len(draw_y)/freq
    return draw_y


# idx=0
def draw_epoch(data,idx,fs=100):
    fig = plt.figure(dpi=100,
                    constrained_layout=True)

    gs = GridSpec(len(data.files)-1, 1, figure=fig)
    for idx in range(len(data.files)-1):
        ax = fig.add_subplot(gs[idx, 0])
        ax.set_xlim(0,30)
        major_ticks_top = np.linspace(0,30,30+1)
        ax.set_xticks(major_ticks_top)
        ax.grid(color='b', ls = '-.', axis = 'x', which ='major', lw = 5,alpha=0.3)
        plt.title(str(data.files[idx]),fontsize=50)
        signal=data[str(data.files[idx])][idx]
        signal_t=np.linspace(0,30,len(signal))
        plt.tick_params(labelsize=40)
        plt.plot(signal_t,signal,linewidth=3)
    fig.set_size_inches(100, 10*(len(data.files)-1))

def fft(data,method:str='scipy',fs=100,ret=2):
    assert method in ['scipy','pytorch']
    if method == 'scipy':
        FT = sp_fft(data)
    else:
        FT_tensor=pt_fft.fft(torch.tensor(data))
        FT = np.array(FT_tensor)
    if len(data.shape)==1:
        amp_FT2 = abs(FT)/len(data)*2
        label_FT = np.linspace(0,int(len(data)/2)-1,int(len(data)/2))    # 生成频率坐标
        amp_FT = amp_FT2[0:int(len(data)/2)]
        fre_FT = label_FT/len(data)*fs
        if ret ==3:
            pha_FT = np.angle(FT)                         # 计算相位角并去除2pi跃变
            pha_FT[np.abs(FT) < 1] = 0
            pha_FT=pha_FT[0:int(len(data)/2)]  
            return fre_FT,amp_FT,pha_FT

    elif len(data.shape)==2:
        amp_FT2 = abs(FT)/len(data[0])*2
        label_FT = np.linspace(0,int(len(data[0])/2)-1,int(len(data[0])/2))    # 生成频率坐标
        amp_FT = amp_FT2[:,0:int(len(data[0])/2)]
        fre_FT = label_FT/len(data[0])*fs
        pha_FT = np.angle(FT)                         # 计算相位角并去除2pi跃变
        if ret==3:
            pha_FT[np.abs(FT) < 1] = 0
            pha_FT=pha_FT[:,0:int(len(data[0])/2)]  
            return fre_FT,amp_FT,pha_FT
    return fre_FT,amp_FT

def sam(model, lossFunc, optimizer, optim_mode, tail_classes, signal, targets, loss):
    if optim_mode == 'sgd':
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
    elif optim_mode == 'sam':
        loss = loss.mean()
        loss.backward()
        optimizer.first_step()
        
        logits = model(signal)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = lossFunc(logits, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.second_step()
        
    elif optim_mode == 'imbsam':
        tail_mask = torch.where((targets[:, None] == tail_classes[None, :].to(targets.device)).sum(1) == 1, True, False)
        head_loss = loss[~tail_mask].sum() / targets.size(0) 
        head_loss.backward(retain_graph=True)
        optimizer.first_step()
                            
        tail_loss = loss[tail_mask].sum() / targets.size(0) 
        tail_loss.backward()
        optimizer.second_step()
                            
        logits = model(signal)
        tail_loss = lossFunc(logits[tail_mask], targets[tail_mask]).sum() / targets.size(0) 
        tail_loss.backward()
        optimizer.third_step()
        loss = head_loss + tail_loss
        
    return loss

class SAM():
    
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
        
    @torch.no_grad()
    def first_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
        
    @torch.no_grad()
    def second_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        
class ImbSAM:
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_normal = self.state[p].get("grad_normal")
            if grad_normal is None:
                grad_normal = torch.clone(p).detach()
                self.state[p]["grad_normal"] = grad_normal
            grad_normal[...] = p.grad[...]
        self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def third_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
            p.grad.add_(self.state[p]["grad_normal"])
        self.optimizer.step()
        self.optimizer.zero_grad()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs = 100, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def edfdata_filter(data_list , lowcut, highcut, fs = 100, order=5):
    new_data_list = []
    for data_i in data_list:
        new_data_i = butter_bandpass_filter(data_i, lowcut, highcut, fs, order)
        new_data_list.append(new_data_i)
    return new_data_list

def sleep_periods_filter(signal,label, w_edge_mins = 30): # sub* [B*3000], sub* [B]

    new_signal = []
    new_labels = []
    for sub_idx in range(len(signal)):
        nw_idx = np.where(label[sub_idx] != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(label[sub_idx]): end_idx = len(label[sub_idx]) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        new_signal_i = signal[sub_idx][select_idx]
        new_label_i = label[sub_idx][select_idx]
        new_signal.append(new_signal_i)
        new_labels.append(new_label_i)
    return new_signal,new_labels

def CEloss_2dim(pred, target):  
    logprobs = F.log_softmax(pred, dim=1)
    loss = -1*torch.sum(target*logprobs, 1)  
    return loss.mean() 

def one_hot(arr,classes):
    w = len(arr)
    h = classes
    # h = np.max(arr)+1
    z = np.zeros([w, h])  # 四行七列
    for i in range(w):  # 4
        j = int(arr[i])  # 拿到数组里面的数字
        z[i][j] = 1
    return z

def smoothing_label(array_1d_list,window_size=2,classes=6,method='avg'):
    assert method in ['avg','weight','step']
    array_2d_list = []
    for array_idx in range(len(array_1d_list)):
        array_1d = array_1d_list[array_idx]
        array_1d = one_hot(array_1d,classes)
        if window_size > 0 :
            if method == 'avg':
                smoothing_method = '_smoothing_avg'
            elif method == 'weight':
                smoothing_method = '_smoothing_weight'
            elif method == 'step':
                smoothing_method = '_smoothing_step'
            array_2d = []
            for i in range(array_1d.shape[1]):
                array_i = array_1d[:,i]
                array_2d.append(eval(smoothing_method)(array_i,window_size))
            array_2d = np.transpose(np.array(array_2d),(1,0))
        else:
            array_2d = array_1d
        array_2d_list.append(array_2d)
    return array_2d_list

def _smoothing_avg(array_i,window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(array_i, window, mode='valid')

def _smoothing_weight(array_i,window_size):
    weights = np.arange(1, window_size + 1)
    weights = weights / np.sum(weights)
    return np.convolve(array_i, weights, mode='same')

def _smoothing_step(array_i,window_size=2,eps=0.1):
    diff_array = np.diff(array_i)
    plus_idx = np.where(diff_array==1)[0]
    minus_idx = np.where(diff_array==-1)[0]
    assert window_size%1==0 and window_size>=0
    for i in range(window_size-1):
        enhance_idx = np.clip(np.hstack((plus_idx-i,minus_idx+1+i)),0,len(array_i)-1)
        weaken_idx = np.clip(np.hstack((plus_idx+1+i,minus_idx-i)),0,len(array_i)-1)
        array_i[enhance_idx]+=eps   
        array_i[weaken_idx]-=eps      
    return array_i

def cos_mask(x_flatten,y_flatten,batch_size,beta):
    data_len = x_flatten.shape[0]  # batch*minibatch
    minibatch = int(data_len/batch_size)
    assert minibatch%1 == 0
    similarity_matrix = F.cosine_similarity(x_flatten.unsqueeze(1), x_flatten.unsqueeze(0), dim=2)
    sameclass_mask = torch.ones_like(similarity_matrix) * (y_flatten.expand(data_len, data_len).eq(y_flatten.expand(data_len, data_len).t()))
    diffclass_mask = torch.ones_like(similarity_matrix) - sameclass_mask
    sameminibatch_mask = torch.ones_like(similarity_matrix)
    for i in range(batch_size):
        sameminibatch_mask[i*minibatch:(i+1)*minibatch,i*minibatch:(i+1)*minibatch]=0 # same minibatch mask\
    diff_mask = diffclass_mask*sameminibatch_mask
    diff_matrix = similarity_matrix*diff_mask
    
    diff_loss = torch.sum(-torch.log((diff_matrix+1)/2+1e-6)-1)/torch.sum(diff_mask)
    if beta!=0:
        diffminibatch_mask = torch.ones_like(similarity_matrix)-sameminibatch_mask-torch.eye(len(similarity_matrix),len(similarity_matrix),device=similarity_matrix.device)
        same_mask = sameclass_mask*diffminibatch_mask
        same_matrix = similarity_matrix*same_mask        
        same_loss = torch.sum(torch.expm1((same_matrix+1)/2))/torch.sum(same_mask)   
    return diff_loss+beta*same_loss

def gen_same_sec_mask(sec_size_list):
    mask_1d = np.array([])
    for i in range(len(sec_size_list)):
        mask_1d = np.append(mask_1d,np.ones(sec_size_list[i])*i)
    mask_1d = torch.tensor(mask_1d).unsqueeze(0)
    mask_2d = torch.ones(len(mask_1d), len(mask_1d)) * (mask_1d.eq(mask_1d.t()))
    return mask_2d



def cos_mask_windows(x_flatten,y,minibatch_size,beta):
    y_flatten = y

    data_len = x_flatten.shape[0]  # batch*minibatch
    batch_size = data_len/minibatch_size
    if batch_size%1 == 0:
        batch_size = int(batch_size)
    else:
        breakpoint()
    x = x_flatten.unsqueeze(0).chunk(batch_size, 1) 
    x = torch.cat(x)                              # [16, 20, 5, 20]
    y = y_flatten.unsqueeze(0).chunk(batch_size, 1)  
    y = torch.cat(y)                              # [16, 20]

    # breakpoint()
    unimportant_tensor = torch.zeros(1,x.shape[-2]).to(x.device)
    sec_size_list = []
    for sec_idx in range(len(x)):
        sec_unimportant_tensor = torch.zeros(1,x.shape[-2]).to(x.device)
        sec_x = x[sec_idx]                        # [20, 5, 20]
        sec_y = y[sec_idx]                        # [20]
        for epoch_idx in range(len(sec_x)):
            epoch_x = sec_x[epoch_idx]
            epoch_y = sec_y[epoch_idx]
            x_important = epoch_x[epoch_y]
            epoch_unimportant_mask = torch.where(F.relu(x_important)/torch.max(x_important)>0.5,0,1)
            for i in range(len(epoch_unimportant_mask)):
                if epoch_unimportant_mask[i]==1:
                    sec_unimportant_tensor = torch.cat((sec_unimportant_tensor,epoch_x[:,i].unsqueeze(0)))
        sec_unimportant_tensor = sec_unimportant_tensor[1:]
        sec_size_list.append(len(sec_unimportant_tensor))
        unimportant_tensor = torch.cat((unimportant_tensor,sec_unimportant_tensor))
    unimportant_tensor = unimportant_tensor[1:]
    similarity_matrix = F.cosine_similarity(unimportant_tensor.unsqueeze(1), unimportant_tensor.unsqueeze(0), dim=2)

    same_sec_mask = gen_same_sec_mask(sec_size_list)
    self_mask = 1-torch.eye(len(similarity_matrix))
    diff_mask = (same_sec_mask*self_mask).to(device=similarity_matrix.device)
    diff_matrix = similarity_matrix*diff_mask+(torch.ones_like(diff_mask)-diff_mask)
    diff_loss = torch.sum(-torch.log((diff_matrix+1)/2+1e-6))/torch.sum(diff_mask)

    return diff_loss

class Reg_hook(object):
    def __init__(self):
       self.feature_map  = []
    def forward_hook(self,module, inp, outp):
        self.feature_map.append(outp)    # 把输出装入字典feature_map

class Sleep_cam(object):
    def __init__(self,net_layer,dro_list=None):
        self.feature_map  = []
        net_layer.register_forward_hook(self.hook)
        self.dro_list = dro_list
    def hook(self, module, inp, outp):
        self.feature_map.append(outp)    # 把输出装入字典feature_map

    def gen_cam(self,x,y,cam_net,device=None):
        with torch.no_grad():
            cam_net = cam_net.to(device=device)
            cam_net = cam_net.eval()
            x = x.to(device=device)
            y = y.to(device=device)
            if self.dro_list == None:
                pred_y = cam_net(x)
            else:
                pred_y = cam_net(x,dro_list=self.dro_list)
            hook_map = torch.transpose(self.feature_map[0],1,2)
            self.feature_map  = []
            pred_y = torch.argmax(pred_y,-1)[0].cpu().detach().numpy()
            y = y.squeeze().cpu().detach().numpy()
            assert len(hook_map) == len(y) == len(pred_y)
            pred_cam = []
            label_cam = []
            for i in range(len(hook_map)):
                pred_cam.append(hook_map[i][pred_y[i]].cpu().detach().numpy())
                label_cam.append(hook_map[i][y[i]].cpu().detach().numpy())
            torch.cuda.empty_cache()
            return pred_cam, label_cam, pred_y, y

class Sleep_genius_cam(object):
    def __init__(self,cam_layer,saved_net,recurrent_len,patch,device=None):
        self.cam_layer = cam_layer
        self.saved_net = saved_net.to(device=device)
        self.saved_net = self.saved_net.eval()
        self.recurrent_len = recurrent_len
        self.device=device
        self.patch = patch
    def gen_cam(self,x,y):
        with torch.no_grad():
            self.x = x.to(device=self.device)
            self.y = y.to(device=self.device)
            self.pred_y,_,_ = self.saved_net(self.x)
            self.feture_map = eval('self.saved_net.'+self.cam_layer)
            self.pred_y = torch.argmax(self.pred_y,-1)[0].cpu().detach().numpy()
            self.y = self.y.squeeze().cpu().detach().numpy()
            assert len(self.feture_map) == len(self.y) == len(self.pred_y)
            self.pred_cam = []
            self.label_cam = []
            for i in range(len(self.feture_map)):
                self.pred_cam.append(self.feture_map[i,:,self.pred_y[i]].cpu().detach().numpy())
                self.label_cam.append(self.feture_map[i][self.y[i]].cpu().detach().numpy())
            # return pred_cam, label_cam, pred_y, y
    def save_picture(self,cam_dir,sub_i,cam_idx):
            cam_tensor = self.feture_map[cam_idx].cpu().detach().numpy()
            signal = self.x[0][cam_idx][0].cpu().detach().numpy()
            pred_y = self.pred_y[cam_idx]
            pred_cam = cam_tensor[:,pred_y][-self.patch:]
            label_y = self.y[cam_idx]
            label_cam = cam_tensor[:,label_y][-self.patch:]

            if cam_idx>=self.recurrent_len:
                last_signal,last_pred_cam,last_label_cam  = [],[],[]
                for last_i in range(self.recurrent_len-1):
                    last_signal.append(self.x[0][cam_idx-last_i][0].cpu().detach().numpy())
                    last_pred_cam.append(cam_tensor[:,pred_y][last_i*self.patch:(last_i+1)*self.patch])
                    last_label_cam.append(cam_tensor[:,label_y][last_i*self.patch:(last_i+1)*self.patch])

            pred_class_name = classes_num2true[str(pred_y)]
            label_class_name = classes_num2true[str(label_y)]
            if pred_class_name==label_class_name:
                TF='T'
            else: 
                TF='F'
            signal_t = np.arange(len(signal))

            fig = plt.figure(dpi=100,
                            constrained_layout=True)
            
            if cam_idx>=self.recurrent_len:
                gs = GridSpec(9*(self.recurrent_len), 9, figure=fig)
                for last_i in range(self.recurrent_len-1):
                    exec('ax_last_{0}_1 = fig.add_subplot(gs[9*{0}:9*{0}+7, :])'.format(last_i))
                    plt.title('origin_signal--last_{}'.format(last_i-self.recurrent_len+1),fontsize=20)   
                    plt.plot(signal_t,last_signal[last_i],linewidth=3)
                    plt.xlim(signal_t[0],signal_t[-1])
                    exec('ax_last_{0}_2 = fig.add_subplot(gs[9*{0}+7, :])'.format(last_i))
                    plt.title('pred_attention--{}'.format(pred_class_name),fontsize=15)
                    plt.imshow(last_pred_cam[last_i][np.newaxis,:], cmap="Blues", aspect="auto")                
                    exec('ax_last_{0}_3 = fig.add_subplot(gs[9*{0}+8, :])'.format(last_i))
                    plt.title('label_attention--{}'.format(label_class_name),fontsize=15)
                    plt.imshow(last_label_cam[last_i][np.newaxis,:], cmap="Blues", aspect="auto") 
                ax1 = fig.add_subplot(gs[9*self.recurrent_len-9:9*self.recurrent_len-2, :])
                plt.title('origin_signal',fontsize=20)   
                plt.plot(signal_t,signal,linewidth=3)
                plt.xlim(signal_t[0],signal_t[-1])
                ax2 = fig.add_subplot(gs[9*self.recurrent_len-2, :])
                plt.title('pred_attention--{}'.format(pred_class_name),fontsize=15)
                plt.imshow(pred_cam[np.newaxis,:], cmap="Blues", aspect="auto")
                ax3 = fig.add_subplot(gs[9*self.recurrent_len-1, :])
                plt.title('label_attention--{}'.format(label_class_name),fontsize=15)
                plt.imshow(label_cam[np.newaxis,:], cmap="Blues", aspect="auto")
                fig.set_size_inches(30, 5*(self.recurrent_len))
            else:
                gs = GridSpec(9, 9, figure=fig)
                ax1 = fig.add_subplot(gs[0:7, :])
                plt.title('origin_signal',fontsize=20)   
                plt.plot(signal_t,signal,linewidth=3)
                plt.xlim(signal_t[0],signal_t[-1])
                ax2 = fig.add_subplot(gs[7, :])
                plt.title('pred_attention--{}'.format(pred_class_name),fontsize=15)
                plt.imshow(pred_cam[np.newaxis,:], cmap="Blues", aspect="auto")
                ax3 = fig.add_subplot(gs[8, :])
                plt.title('label_attention--{}'.format(label_class_name),fontsize=15)
                plt.imshow(label_cam[np.newaxis,:], cmap="Blues", aspect="auto")
                fig.set_size_inches(30, 5)
            show_cam_path = '{0}/{1}/epoch_{2}_{3}.png'.format(cam_dir,sub_i,cam_idx,TF)
            plt.savefig(show_cam_path)
            plt.close('all')
            return 0 
            
        
        
        
classes_num2true={'0':'W','1':'N1','2':'N2','3':'N3','4':'R'}

pre_channel = ['eeg_f3', 'eeg_f4', 'eeg_c3', 'eeg_c4', 'eeg_o1', 'eeg_o2', 'eog', 'emg']

def show_cam_on_signal_8(cam_input,pred_cam,pred_class,label_cam,label_class,c_matrix_dir,fold_idx,sub_i,epoch_i):
    ch=len(cam_input)
    cam_len=20
    cam_input = cam_input.squeeze() # [ch, 6000]
    pred_cam = pred_cam.squeeze()   # [160]
    label_cam = label_cam.squeeze() # [160]   
    if type(pred_class)==torch.Tensor:pred_class=pred_class.cpu()
    if type(label_class)==torch.Tensor:label_class=label_class.cpu()
    pred_class = np.array(pred_class).squeeze()
    label_class = np.array(label_class).squeeze()
    pred_class = classes_num2true[str(pred_class)]
    label_class = classes_num2true[str(label_class)]
    if pred_class==label_class:
        TF='T'
    else: 
        TF='F'
    breakpoint()
    cam_input_t = np.arange(len(cam_input[0]))
    fig = plt.figure(dpi=100, constrained_layout=True)
    gs = GridSpec(9*ch, 9*ch, figure=fig)
    axs = [0]*ch*3
    for i in range(ch):
        axs[0+3*i] = fig.add_subplot(gs[0+i*9:7+i*9, :])
        plt.title(pre_channel[i],fontsize=20)   
        plt.plot(cam_input_t,cam_input[i],linewidth=3)
        plt.xlim(cam_input_t[0],cam_input_t[-1])
        axs[1+3*i] = fig.add_subplot(gs[7+i*9, :])
        plt.title('pred_attention--{}'.format(pred_class),fontsize=15)
        plt.imshow(pred_cam[i*cam_len:(i+1)*cam_len][np.newaxis,:], cmap="Blues", aspect="auto")
        axs[2+3*i] = fig.add_subplot(gs[8+i*9, :])
        plt.title('label_attention--{}'.format(label_class),fontsize=15)
        plt.imshow(label_cam[i*cam_len:(i+1)*cam_len][np.newaxis,:], cmap="Blues", aspect="auto")

    fig.set_size_inches(30, 5*ch)
    show_png_path = '{0}/{1}/epoch_{2}_{3}__{4}_{5}.png'.format(c_matrix_dir,fold_idx,sub_i,epoch_i,label_class,TF)
    plt.savefig(show_png_path)
    plt.close('all')
    return 0    

class Dropout_channel(object):
    def __init__(self, ch_num, drop_p):
        super(Dropout_channel, self).__init__()        
        assert 0 <= drop_p <= 1
        self.__p = drop_p     # 置零概率
        self.ch_num = ch_num
    def go(self):
        if self.__p == 0:
            return [1]*self.ch_num
        if self.__p == 1:
            return [0]*self.ch_num
        while 1:
            rand_ = np.random.random(self.ch_num)
            rand_list = [1 if i>self.__p else 0 for i in rand_]
            if sum(rand_list)>=1:
                return rand_list        
    def eval(self):
        return [1]*self.ch_num 


def rms_v(array,window,strip):
    win_zero = np.zeros(window)
    rms_list=[]
    i = 0
    while i*strip+window<len(array):
        rms_i = math.sqrt(np.mean(np.square(np.subtract(array[i*strip:i*strip+window],win_zero))))
        rms_list.append(rms_i)
        i += 1
    return rms_list


def trans_data(data, dim, torch_trans=False):
    is_numpy = isinstance(data, np.ndarray)
    is_torch  = isinstance(data, torch.Tensor)

    if not (is_numpy or is_torch):
        raise TypeError("data must be numpy.ndarray or torch.Tensor")

    orig_type = "numpy" if is_numpy else "torch"

    cur_dim = data.ndim
    if cur_dim > dim:
        print(f"[error] input shape: {cur_dim}, target shape: {dim}")
    else:
        for _ in range(dim - cur_dim):
            data = data[np.newaxis, ...] if is_numpy else data.unsqueeze(0)
        cur_dim = data.ndim

    if torch_trans:
        if is_numpy:
            # numpy -> torch
            data = torch.from_numpy(data)
        else:
            # torch -> numpy
            data = data.detach().cpu().numpy()
    return data
    
def gen_showMap(show_map):
    new_show_map = [ 0 for i in range(len(show_map)+1)]
    for i in range(len(show_map)):
        new_show_map[i]+=show_map[i]
        new_show_map[i+1]+=show_map[i]
    new_show_map[0]+=show_map[0]
    new_show_map[-1]+=show_map[-1]
    new_show_map = np.array(new_show_map)
    return new_show_map