import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as sio


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    
    return torch.softmax(m , -1)


def attention(Q, K, V):
    #Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K) #(batch_size, dim_attn, seq_length)
    
    return  torch.matmul(a,  V) #(batch_size, seq_length, seq_length)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        
        #Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.heads = nn.ModuleList(self.heads)
        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                      
        
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        
        x = self.fc(a)
        
        return x
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
        #self.fc2 = nn.Linear(5, dim_val)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        
        x = self.fc1(x)
        #print(x.shape)
        #x = self.fc2(x)
        
        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x     
    
def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length
    
    t = torch.zeros(batch_size,1).uniform_(0,20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size,1) + t
    
    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

def create_inout_sequences(input_data, output_data, tw):
    xs = []
    ys = []

    for i in range(len(input_data)-tw-1):
        xya = input_data[i:(i+tw), 3]
        xyv = input_data[i:(i+tw), 2]
        xy = input_data[i:(i+tw), 1]
        xu = input_data[i+1:(i+tw)+1, 0]
        x = np.vstack([xu,xy,xyv,xya])
        y = output_data[i+tw,:]

        xs.append(x)
        ys.append(y)

    return np.transpose(np.array(xs),[0,2,1]), np.array(ys)

def load_flatbeam_data(train_window):
    data = sio.matlab.loadmat('../data/FlatBeam_NLResponse_RandVibe.mat')['out']
    y = data[0]['def'][0].T[:, :, np.newaxis][:, 10, :]
    yv = data[0]['vel'][0].T[:, :, np.newaxis][:, 10, :]
    ya = data[0]['acc'][0].T[:, :, np.newaxis][:, 10, :]
    u = data[0]['fext'][0].T[:, :, np.newaxis][:, 10, :]
    t = data[0]['t'][0][0]


    umax = np.max(u)
    ymax = np.max(np.abs([y, yv, ya]), axis=1)

    u_norm = u / umax
    y_norm = y / ymax[0]
    yv_norm = yv / ymax[1]
    ya_norm = ya / ymax[2]

    N = len(u)
    ntrain = int(N * 0.5)
    start = 50

    utrain = u_norm[start:ntrain]
    ytrain = y_norm[start:ntrain]
    yvtrain = yv_norm[start:ntrain]
    yatrain = ya_norm[start:ntrain]
    ttrain = t[start:ntrain]
    intrain = np.hstack([utrain, ytrain, yvtrain, yatrain])
    outtrain = np.hstack([ytrain, yvtrain, yatrain])

    utest = u_norm[ntrain:start + ntrain + 500]
    ytest = y_norm[ntrain:start + ntrain + 500]
    yvtest = yv_norm[ntrain:start + ntrain + 500]
    yatest = ya_norm[ntrain:start + ntrain + 500]
    ttest = t[ntrain:start + ntrain + 500]
    intest = np.hstack([utest, ytest, yvtest, yatest])
    outtest = np.hstack([ytest, yvtest, yatest])

    ufull = np.concatenate([utrain, utest], axis=0)
    yfull = np.concatenate([ytrain, ytest], axis=0)
    infull = np.concatenate([intrain, intest], axis=0)
    outfull = np.concatenate([outtrain, outtest], axis=0)

    train_window = train_window

    x_train, y_train = create_inout_sequences(intrain, outtrain, train_window)
    x_val, y_val = create_inout_sequences(intest, outtest, train_window)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()


    x_full, y_full = create_inout_sequences(infull, outfull, train_window)

    x_full = torch.from_numpy(x_full).float()
    y_full = torch.from_numpy(y_full).float()

    print(x_train.shape, y_train.shape)
    print(x_full.shape, y_full.shape)

    return x_full, y_full, x_train, y_train, x_val, y_val


def RK_check(x, dx, dt, scalers, M=2, alpha=None, beta=None):
    if beta is None:
        beta = [0., 1.5, -0.5]
    if alpha is None:
        alpha = [-1., 1., 0.]
    x = x * scalers[:-1].T
    dx = dx * scalers[1:].T

    Y = alpha[0] * x[M:, :] + dt * beta[0] * dx[M:, :]

    for m in range(1, M + 1):
        Y = Y + alpha[m] * x[M - m:-m, :] + dt * beta[m] * dx[M - m:-m, :]

    return Y / scalers[:-1].T * 1e-1