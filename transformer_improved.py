#Author: Damian Grunert
#Date: 25-02-2024
#Content: A modular implementation of an improved transformer architecture incorporating possible improvements, such as:
# - Vectorized and parallelizable mini-batch training and prediction using a hyperparameterized maximum sequence length padded with zero vectors in the front for both sequences
# (The fixed sequence length is to allow for parallelized training through a common tensor and to control the length of long-term interactions)
# - Being able to train entire sequences using a start token formed by a one-vector and the masked self attention mechanism so that one can condition the decoder output on the entire given output ---
# - Pre-LN-Architecture ---
# - GeLU activation function ---
# - The DropOut-Mechanism for the residuals, ANN hidden layer and attention sublayer, however not the embedding ---
# - Removing the bias in feedforward-layers ---
# - Adding a bias in the WO-Weight
# - Xavier-Initialization for the Weight-Matrices (so that they are initilaized with unit standard deviation) and Kaiming-Initialization for the ANN-Matrices
# - Initialization of the bias weights to be zero instead of one
# - QoL: Being able to save and retrieve the parameters of the model ---

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

def dropout_func(x, dropout):
    dropout_chances = torch.rand(x.shape)
    dropout_mask = dropout_chances > dropout
    return (x * dropout_mask) / (1-dropout)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, has_enc = False):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_k = d_model // h #Can be chosen otherwise
        self.has_enc = has_enc
        #Output matrix processing the concatenated attention heads
        self.WO = Parameter(nn.init.xavier_uniform_(torch.empty(self.d_k * h, d_model)))
        self.WO_bias = Parameter(nn.init.zeros_(torch.empty(d_model)))

        #Learnt, general parameters to yield Q, K and V from an Input for all Attention Heads combined
        self.WQ_comb = Parameter(nn.init.xavier_uniform_(torch.empty(d_model, self.d_k)))
        self.WK_comb = Parameter(nn.init.xavier_uniform_(torch.empty(d_model, self.d_k)))
        self.WV_comb = Parameter(nn.init.xavier_uniform_(torch.empty(d_model, self.d_k)))

        #List of weight matrices for each attention head
        self.WQ = nn.ParameterList(Parameter(nn.init.xavier_uniform_(torch.empty(self.d_k, self.d_k))) for i in range(h))
        self.WK = nn.ParameterList(Parameter(nn.init.xavier_uniform_(torch.empty(self.d_k, self.d_k))) for i in range(h))
        self.WV = nn.ParameterList(Parameter(nn.init.xavier_uniform_(torch.empty(self.d_k, self.d_k))) for i in range(h))
        
    def attention(self, Q, K, V, dropout = 0, mask_indices = None):
        padding_mask = torch.zeros(V.shape[0], Q.shape[1], K.shape[1])
        if mask_indices != None:
            for i in range(V.shape[0]):   #Really dont want to use for-loops, but hey - for the masking
                padding_mask[i, mask_indices[i], :] = -1e+19
                padding_mask[i, :, mask_indices[i]] = -1e+19
        return dropout_func(nn.functional.softmax(((Q @ K.mT + padding_mask)*(1 / math.sqrt(self.d_k))), dim=1), dropout) @ V
    
    #No different from the regular attention mechanism, but in the way masking of padded values is done (so that values )
    def attention_enc_dec(self, Q, K, V, dropout = 0, input_mask_indices = None, output_mask_indices = None):
        padding_mask = torch.zeros(V.shape[0], Q.shape[1], K.shape[1])
        if input_mask_indices != None:
            for i in range(V.shape[0]):
                padding_mask[i, :, input_mask_indices[i]] = -1e+19
        if output_mask_indices != None:
            for i in range(V.shape[0]):
                padding_mask[i, output_mask_indices[i], :] = -1e+19

        return dropout_func(nn.functional.softmax(((Q @ K.mT + padding_mask)*(1 / math.sqrt(self.d_k))), dim=1), dropout) @ V
    
    def forward(self, XQ, XK, XV, dropout = 0, mask_indices = None, mask_output_indices = None):
        if self.has_enc == False:
            #Encoder has the same, decoder different inputs for the attention evaluation (from which Q, K, V are constructed)
            heads = [self.attention(XQ @ self.WQ_comb @ self.WQ[i], XK @ self.WK_comb @ self.WK[i], XV @ self.WV_comb @ self.WV[i], dropout, mask_indices) for i in range(self.h)]
        else:
            heads = [self.attention_enc_dec(XQ @ self.WQ_comb @ self.WQ[i], XK @ self.WK_comb @ self.WK[i], XV @ self.WV_comb @ self.WV[i], dropout, mask_indices, mask_output_indices) for i in range(self.h)]

        return torch.cat(heads, dim = 2) @ self.WO + self.WO_bias

class MaskedMultiHeadAttention(MultiHeadAttention):
    def attention(self, Q, K, V, dropout = 0, mask_indices = None):
        #Causal mask used in the decoder to prevent attention to future outputs
        causal_mask = (torch.tril(torch.ones(Q.shape[1], K.shape[1])) == 0) * (-1e+19) #Fills all values above the diagonal with large negative values to be ignored during softmax
        padding_mask = torch.zeros(V.shape[0], Q.shape[1], K.shape[1])
        #Padding-mask for the values in the decoder 
        if mask_indices != None:
            for i in range(V.shape[0]):
                padding_mask[i, mask_indices[i], :] = -1e+19
                padding_mask[i, :, mask_indices[i]] = -1e+19
        mask = torch.minimum(causal_mask, padding_mask) #If a position is masked due to either one, we mask it in the combined mask
        return dropout_func(nn.functional.softmax((Q @ K.mT) + mask, dim=1)*(1 / math.sqrt(self.d_k)), dropout) @ V

#Implementation of one single ANN layer as used in most transformers with an additional residual connection
class ANNLayer(nn.Module):
    def __init__(self, d_ff, d_model):
        super(ANNLayer, self).__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.weights_1 = Parameter(nn.init.kaiming_normal_(torch.empty(d_model, d_ff)))
        self.weights_2 = Parameter(nn.init.kaiming_normal_(torch.empty(d_ff, d_model)))
    
    def forward(self, X, dropout = 0):
        return dropout_func(nn.functional.gelu(X @ self.weights_1), dropout) @ self.weights_2

#"-------------------------------- Higher-Level Transformer-Components ------------------------------------------------"
def postionalEncoding(X):
    #Creates a 'transposed vector' of the sequence positions from 0 to the amount of sequence members
    pos_vec = torch.arange(0, X.shape[1]).unsqueeze(1)
    div_vec = torch.float_power(torch.tensor(10000), torch.arange(0, X.shape[2], 2) / X.shape[2] * (-1))
    pos_enc = torch.empty(X.shape[1], X.shape[2])
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

    #We are assuming a pre-LN-Architecture
    def forward(self, x, dropout = 0, mask_indices = None):
        norm_val = layerNorm(x)
        attended_val = self.attention(norm_val, norm_val ,norm_val, dropout, mask_indices)
        attended_res = dropout_func(attended_val, dropout) + norm_val

        ann_norm = layerNorm(attended_res)
        ann_val = self.ann(ann_norm, dropout)
        ann_res = dropout_func(ann_val, dropout) + ann_norm
        return ann_res

class DecoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model, has_encoder):
        super(DecoderLayer, self).__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.has_encoder = has_encoder
        self.attention = MaskedMultiHeadAttention(n_head, d_model)
        if has_encoder:
            self.enc_attention = MultiHeadAttention(n_head, d_model, has_enc=True)
        self.ann = ANNLayer(dim_ann, d_model)

    def forward(self, x, enc_val = None, dropout = 0, mask_indices_input = None, mask_indices_output = None):
        norm_val = layerNorm(x)
        attended_val = self.attention(norm_val, norm_val ,norm_val, dropout = dropout, mask_indices = mask_indices_input)
        attended_res = dropout_func(attended_val, dropout) + norm_val
        
        if self.has_encoder:
            norm_val_enc = layerNorm(attended_res)
            attended_val_enc = self.enc_attention(norm_val_enc, enc_val, enc_val, dropout, mask_indices_input, mask_indices_output)
            attended_res = attended_val_enc + norm_val_enc
        
        ann_norm = layerNorm(attended_res)
        ann_val = self.ann(ann_norm, dropout)
        ann_res = dropout_func(ann_val, dropout) + ann_norm
        return ann_res

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
                 embedding_layer: Type[EmbeddingLayer], enc_layers: int, dec_layers: int, output_layer: Type[OutputLayer], max_seq_length):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dim_ann = dim_ann
        self.d_input = d_input
        self.d_output = d_output
        #For now, we assume input = output length
        self.max_seq_length = max_seq_length

        #Adding the used layers and registering them as submodules for the optimizer to track the weights
        self.input_embedding_layer = embedding_layer(self.d_input, self.d_model)
        self.output_embedding_layer = embedding_layer(self.d_output, self.d_model)
        if enc_layers > 0:
            self.encoder = nn.ModuleList([EncoderLayer(dim_ann, n_head, d_model) for i in range(enc_layers)])
        if dec_layers > 0:
            self.decoder = nn.ModuleList([DecoderLayer(dim_ann, n_head, d_model, enc_layers > 0) for i in range(dec_layers)])
        self.output_layer = output_layer(d_model, d_output)

    #Input: batchsize x seq_length x 
    #Output: Indices of pure zero rows used for masking
    def masked_rows_indices(self, input):
        zero_rows_mask = torch.all(input == 0, dim=-1)
        return [torch.nonzero(mask, as_tuple=False)[:, 0].tolist() for mask in zero_rows_mask]

    #Assuming a batch input of dimensions batch_size x inp_seq_len x d_input and batchsize x out_seq_len x d_output respectively, predicts all values for the sequence (may be used to predict the last or all sequence values)
    def forward(self, X, output, dropout = 0, add_token = True):
        #Encoder-Part:
        if self.enc_layers > 0:
            #Data-preprocessing: figuring out the binary mask for both input and output by the zero-padding-vector by saving the amount of zero vectors in batchsize x 1-array
            input_mask_rows = self.masked_rows_indices(X)
            #Embedding-Layer
            curr_repr = self.input_embedding_layer(X)
            #Positional Encoding Layer
            curr_repr = postionalEncoding(curr_repr)
            #Encoder-Block
            for i in range(self.enc_layers):
                curr_repr = self.encoder[i](curr_repr, dropout, mask_indices = input_mask_rows)
        else:
            input_mask_rows = None

        #Decoder-Part:
        if self.dec_layers > 0:
            if add_token:
                #Begin of sequence vector, arbitrarily choosen to be the trivial one-vector
                output = torch.cat([torch.ones(output.shape[0], 1, output.shape[2]), output], dim = 1)
            #Mask indices for the output sequence
            output_mask_rows = self.masked_rows_indices(output)
            #Re-assigning the (existing) encoder values and current input values, using embedding
            if self.enc_layers > 0:
                enc_repr, curr_repr = curr_repr, self.output_embedding_layer(output)
            else:
                enc_repr, curr_repr = None, self.output_embedding_layer(output)
            #Positional encoding to the existing inputs
            curr_repr = postionalEncoding(curr_repr)
            #Decoder-Block
            for i in range(self.dec_layers):
                curr_repr = self.decoder[i](curr_repr, enc_repr, dropout = dropout, mask_indices_input = input_mask_rows, mask_indices_output = output_mask_rows)
        #Output-Function is applied to the d_seq \times d_model matrix to a d_seq \times d_output model, with the last vector being returned as the final output
        output = self.output_layer(curr_repr)
        for i in range(output.shape[0]):
            output[i, output_mask_rows[i], :] = 0
        #Return everything but the arbitrarily formed start-of-sequence-token and mask the padding values to zero so that they have no error during training
        return output[:, 1:, :]
    
    #Returns the last n predictions batched as a list of torch arrays
    def get_prediction(self, X, output, dropout = 0, add_token = True, num_tokens = 1):
        transform_outputs = self.forward(X, output, dropout = 0, add_token = True)

        result = torch.zeros(transform_outputs.shape[0], transform_outputs.shape[1], transform_outputs.shape[2])

        for i in range(transform_outputs.shape[0]):
            # Find indices of non-zero values along the second dimension
            nonzero_indices = (transform_outputs[i, :, :] != 0).nonzero(as_tuple=True)[0]
            # Check if there are enough non-zero values, if not replace everything up them
            if len(nonzero_indices) >= num_tokens:
                # Get the last n non-zero indices
                last_n_indices = nonzero_indices[-num_tokens:]
                # Assign the corresponding values to the result tensor
                result[i, :, :] = transform_outputs[i, last_n_indices, :]
            else:

                last_n_indices = nonzero_indices[-num_tokens:]
                result[i, :, :] = transform_outputs[i, last_n_indices, :]
    
        return result
    
    #Assumes a list of tensor-inputs of variable size d_seq x d_input/output and trims them down or pads them w/ zeros to conform with the sequence length for encoder and decoder
    def pad_inputs(self, inputs):
        #Removing all elements in sequences longer than the max-sequence-length
        for i in range(len(inputs)):
            if inputs[i].shape[0] > self.max_seq_length:
                inputs[i] = inputs[i][0:self.max_seq_length]
        #Padding the last value to the maximum for the inputs
        inputs[-1] = torch.cat([inputs[-1], torch.zeros(self.max_seq_length - inputs[-1].shape[0], inputs[-1].shape[1])], dim = 0)

        #Padding the remaining sequences with length <= max_seq_length to the desired dimensions, outputs a batch x seq x d torch tensor
        return torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    
    #Retrieves the parameters (Wrapper)
    def retrieve_weights(self):
        return self.state_dict()
    
    #Assigns the parameters (Wrapper)
    def assign_weights(self, state_dict):
        self.load_state_dict(state_dict)

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
    data = [torch.rand(4, 3) for i in range(100)]
    outputs = [torch.rand(4, 1) for i in range(100)]
    x = Transformer(3, 1, 2, 1, 2, LinearEmbedding, 1, 1, LinearOutput, max_seq_length=5)
    data, outputs = x.pad_inputs(data), x.pad_inputs(outputs)
    optim = torch.optim.Adam(x.parameters(), lr=0.000075) #NoamOptimizer(1000, x.d_model, torch.optim.Adam(x.parameters(), lr=0))
    loss_func = nn.MSELoss()
    for j in range(10000):
        prediction = x.forward(data, outputs, dropout = 0.0)
        loss = loss_func(prediction, outputs)
        loss.backward()
        optim.step()
        if j % 100 == 0:
            print(float(loss))
    assert(float(loss) <= 0.15)

if __name__ == "__main__":
    test_transformer()