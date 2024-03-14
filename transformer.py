#Author: Damian Grunert
#Date: 25-02-2024
#Content: A modular implementation of the default transformer architecture 
#The model can be applied to one sequence at a time, with a modular-length input- and output-sequence, and then conditioned on a following value
#using the Pytorch - Dynamic Computational Graph for Per-Sequence-Training and Prediction
#Purely demonstrational, find the transformer_improved.py file for a practical implementation

import torch
import torch.nn as nn
from typing import Type
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parameter import Parameter
import math
import copy

#"-------------------------------- Low-Level Transformer Methods ------------------------------------------------"
#Implements the layer-normalization-layer in the transformer architecture.
#Not a class as it isnt parameterized.
def layerNorm(x):
    std_mean = torch.std_mean(x, dim = 0)
    return (x - std_mean[1]) / (std_mean[0] + 0.0001)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_k = d_model // h #Can be chosen otherwise
        #Output matrix processing the concatenated attention heads
        self.WO = Parameter(torch.rand(self.d_k * h, d_model))

        #Learnt, general parameters to yield Q, K and V from an Input for all Attention Heads combined
        self.WQ_comb = Parameter(torch.rand(d_model, self.d_k))
        self.WK_comb = Parameter(torch.rand(d_model, self.d_k))
        self.WV_comb = Parameter(torch.rand(d_model, self.d_k))

        #List of weight matrices for each attention head
        self.WQ = nn.ParameterList(Parameter(torch.rand(self.d_k, self.d_k)) for i in range(h))
        self.WK = nn.ParameterList(Parameter(torch.rand(self.d_k, self.d_k)) for i in range(h))
        self.WV = nn.ParameterList(Parameter(torch.rand(self.d_k, self.d_k)) for i in range(h))
        
    def attention(self, Q, K, V):
        return nn.functional.softmax(((Q @ K.t())*(1 / math.sqrt(self.d_k))), dim=1) @ V

    def forward(self, XQ, XK, XV):
        #Encoder has the same, decoder different inputs for the attention evaluation (from which Q, K, V are constructed)
        heads = [self.attention(XQ @ self.WQ_comb @ self.WQ[i], XK @ self.WK_comb @ self.WK[i], XV @ self.WV_comb @ self.WV[i]) for i in range(self.h)]
        return torch.cat(heads, dim = 1) @ self.WO

class MaskedMultiHeadAttention(MultiHeadAttention):
    def attention(self, Q, K, V):
        mask = torch.tril(torch.ones(Q.shape[0], K.shape[0]))
        return (nn.functional.softmax((Q @ K.t()), dim=1)* mask *(1 / math.sqrt(self.d_k))) @ V

#Implementation of one single ANN layer as used in most transformers with an additional residual connection
class ANNLayer(nn.Module):
    def __init__(self, d_ff, d_model):
        super(ANNLayer, self).__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.weights_1, self.bias_1 = Parameter(nn.init.kaiming_normal_(torch.empty(d_model, d_ff))), Parameter(nn.init.zeros_(torch.empty(d_ff)))
        self.weights_2, self.bias_2 = Parameter(nn.init.kaiming_normal_(torch.empty(d_ff, d_model))), Parameter(nn.init.zeros_(torch.empty(d_model)))
    
    def forward(self, X):
        return nn.functional.relu(X @ self.weights_1 + self.bias_1) @ self.weights_2 + self.bias_2

#"-------------------------------- Higher-Level Transformer-Components ------------------------------------------------"
def postionalEncoding(X):
    #Creates a 'transposed vector' of the sequence positions from 0 to the amount of sequence members
    pos_vec = torch.arange(0, X.shape[0]).unsqueeze(1)
    div_vec = torch.float_power(torch.tensor(10000), torch.arange(0, X.shape[1], 2) / X.shape[1] * (-1))
    pos_enc = torch.empty(X.shape[0], X.shape[1])
    pos_enc[:, 0::2] = torch.sin(pos_vec * div_vec)
    pos_enc[:, 1::2] = torch.cos(pos_vec * div_vec)
    return X + pos_enc

class EmbeddingLayer(nn.Module):
    def __init__(self, d_input, d_model):
        super(EmbeddingLayer, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
    def forward(self, x):
        pass

class EncoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model):
        super(EncoderLayer, self).__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.attention = MultiHeadAttention(n_head, d_model)
        self.ann = ANNLayer(dim_ann, d_model)

    #We are assuming a post-LN-Architecture
    def forward(self, x):
        attended_val = self.attention(x, x, x)
        attended_res = attended_val + x
        attended_norm = layerNorm(attended_res)

        ann_val = self.ann(attended_norm)
        ann_res = ann_val + attended_norm
        ann_norm = layerNorm(ann_res)
        return ann_norm

class DecoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model, has_encoder):
        super(DecoderLayer, self).__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.has_encoder = has_encoder
        self.attention = MaskedMultiHeadAttention(n_head, d_model)
        if has_encoder:
            self.enc_attention = MultiHeadAttention(n_head, d_model)
        self.ann = ANNLayer(dim_ann, d_model)

    def forward(self, x, enc_val = None):
        attended_val = self.attention(x, x ,x)
        attended_res = attended_val + x
        attended_norm = layerNorm(attended_res)
        
        if self.has_encoder:
            attended_val_enc = self.enc_attention(attended_norm, enc_val, enc_val)
            attended_res_enc = attended_val_enc + attended_norm
            attended_norm = layerNorm(attended_res_enc)
        
        ann_val = self.ann(attended_norm)
        ann_res = ann_val + attended_norm
        ann_norm = layerNorm(ann_res)
        return ann_norm

class OutputLayer(nn.Module):
    def __init__(self, d_model, d_output):
        super(OutputLayer, self).__init__()
        self.d_model = d_model
        self.d_output = d_output
        pass
    def forward(self, x):
        pass

#Complete transformer-architecture, entailing a variable number of encoder and decoder layers
class Transformer(nn.Module):
    def __init__(self, d_input: int, d_output: int, d_model: int, n_head: int, dim_ann: int, 
                 embedding_layer: Type[EmbeddingLayer], enc_layers: int, dec_layers: int, output_layer: Type[OutputLayer]):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dim_ann = dim_ann
        self.d_input = d_input
        self.d_output = d_output

        #Adding the used layers and registering them as submodules for the optimizer to track the weights
        self.input_embedding_layer = embedding_layer(self.d_input, self.d_model)
        self.output_embedding_layer = embedding_layer(self.d_output, self.d_model)
        if enc_layers > 0:
            self.encoder = nn.ModuleList([EncoderLayer(dim_ann, n_head, d_model) for i in range(enc_layers)])
        if dec_layers > 0:
            self.decoder = nn.ModuleList([DecoderLayer(dim_ann, n_head, d_model, enc_layers > 0) for i in range(dec_layers)])
        self.output_layer = output_layer(d_model, d_output)

    def forward(self, X, output):
        #Encoder-Part:
        if self.enc_layers > 0:
            #Embedding-Layer
            curr_repr = self.input_embedding_layer(X)
            #Positional Encoding Layer
            curr_repr = postionalEncoding(curr_repr)
            #Encoder-Block
            for i in range(self.enc_layers):
                curr_repr = self.encoder[i](curr_repr)

        #Decoder-Part:
        if self.dec_layers > 0:
            #Begin of sequence vector, arbitrarily choosen to be the one vector
            output = torch.cat([torch.ones(1, output.shape[1]), output], dim = 0)
            #Re-assigning the (existing) encoder values and current input values, using embedding
            if self.enc_layers > 0:
                enc_repr, curr_repr = curr_repr, self.output_embedding_layer(output)
            else:
                enc_repr, curr_repr = None, self.output_embedding_layer(output)
            #Positional encoding to the existing inputs
            curr_repr = postionalEncoding(curr_repr)
            #Decoder-Block
            for i in range(self.dec_layers):
                curr_repr = self.decoder[i](curr_repr, enc_repr)
        #Output-Function is applied to the d_seq \times d_model matrix to a d_seq \times d_output model, with the last vector being returned as the final output
        return self.output_layer(curr_repr)[-1, :]

#"-------------------------------- Implementation Of Common High-Level Components ------------------------------------------------"
class LinearEmbedding(nn.Module):
    def __init__(self, d_input, d_model):
        super(LinearEmbedding, self).__init__()
        self.weight = Parameter(torch.rand(d_input, d_model))
    def forward(self, x):
        return x @ self.weight

class LinearOutput(nn.Module):
    def __init__(self, d_model, d_output):
        super(LinearOutput, self).__init__()
        self.weight = Parameter(torch.rand(d_model, d_output))
    def forward(self, x):
        return x @ self.weight

#"------------------------------------------------ Optimization Algorithms ------------------------------------------------"
#Linear learning-rate warmup followed by exponential decay to ensure convergence in difficulty training transformers
#Essentially provides a wrapper for a basic optimization algorithm to schedule the learning rate
class NoamOptimizer:
    def __init__(self, warmup_steps, d_model, optimizer) -> None:
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.optimizer = optimizer

    def step(self):
        #Increments the step number
        self.step_num += 1
        #Sets the new learning rate according to the noam-rule
        lr = self.d_model**(-0.5) * min(self.step_num**(-0.5), self.step_num * self.warmup_steps**(-1.5))
        #Changes all learning-rates for different weight groups to the one given above
        for weight_group in self.optimizer.param_groups:
            weight_group['lr'] = lr
        #Facilitates a step in the original optimizer, with the changed learning rate
        self.optimizer.step()
    
    #For if you want to re-assign the progress of an optimization algorithm from previous state
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    #For if you want to save the current optimizers progress for later use
    def get_state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
#"-------------------------------- Unit-Tests ------------------------------------------------"
def test_pos_encoding():
    pass

def test_embedding():
    pass

def test_ann():
    pass

def test_multihead_attention():
    pass

def test_output_layer():
    pass

def test_encoder():
    pass

def test_decoder():
    pass

#Simple unit test to check the general functioning of the transformer by learning noise
def test_transformer():
    data = torch.rand(50, 3, 1000)
    labels = torch.rand(1, 1000)
    prev_outputs = torch.rand(10, 1, 1000)
    x = Transformer(3, 1, 128, 1, 8, LinearEmbedding, 1, 1, LinearOutput)
    num = 0
    optim = torch.optim.Adam(x.parameters(), lr=0.0001) #NoamOptimizer(1000, x.d_model, torch.optim.Adam(x.parameters(), lr=0))
    loss_func = nn.MSELoss()
    for j in range(50):
        for i in range(data.shape[2]):
            prediction = x.forward(data[:, :, i], prev_outputs[:, :, i])
            loss = loss_func(prediction, labels[:, i])
            loss.backward()
            optim.step()
            if i % 1000 == 0:
                print(float(loss))
    for i in x.parameters():
        print(i)



if __name__ == "__main__":
    test_transformer()
