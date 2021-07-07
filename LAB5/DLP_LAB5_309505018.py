from __future__ import unicode_literals, print_function, division
import os
import torch
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from plot import plot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 512  # LSTM hidden size
vocab_size = 29  # The number of vocabulary:vocab_size==input_size ,containing:SOS,EOS,UNK,a-z
teacher_forcing_ratio = 0.5
LR = 0.05
epochs = 50
decoder_type='simple'

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

################### DataTransformer #########################################
import json


class DataTransformer:
    def __init__(self):
        self.char2idx=self.build_char2idx()
        self.idx2char=self.build_idx2char()
        self.MAX_LENGTH=0  # max length of the training data word(contain 'EOS')

    def build_char2idx(self):
        """
        {'SOS':0,'EOS':1,'UNK':2,'a':3,'b':4 ... 'z':28}
        """
        dictionary={'SOS':0,'EOS':1,'UNK':2}
        dictionary.update([(chr(i+97),i+3) for i in range(0,26)])
        return dictionary

    def build_idx2char(self):
        """
        {0:'SOS',1:'EOS',2:'UNK',3:'a',4:'b' ... 28:'z'}
        """
        dictionary={0:'SOS',1:'EOS',2:'UNK'}
        dictionary.update([(i+3,chr(i+97)) for i in range(0,26)])
        return dictionary

    def sequence2indices(self,sequence,add_eos=True):
        """
        :param sequence(string): a char sequence
        :param add_eox(boolean): whether add 'EOS' at the end of the sequence
        :return: int sequence
        """
        indices=[]
        for c in sequence:
            indices.append(self.char2idx[c])
        if add_eos:
            indices.append(self.char2idx['EOS'])
        self.MAX_LENGTH = max(self.MAX_LENGTH, len(indices))
        return indices

    def indices2sequence(self,indices):
        """
        :param indices: int sequence (without EOS_token)
        :return: string
        """
        re=""
        for i in indices:
            re+=self.idx2char[i]
        return re

    def build_training_set(self,path):
        """
        :return:
            int_list: [[input,target],[input,target]....]  (input & target are all int sequence)
            str_list: [[input,target],[input,target]....]  (input & target are all string)
        """
        int_list=[]
        str_list=[]
        with open(path,'r') as file:
            dict_list=json.load(file)
            for dict in dict_list:
                target=self.sequence2indices(dict['target'])
                for input in dict['input']:
                    int_list.append([self.sequence2indices(input,add_eos=True),target])
                    str_list.append([input,dict['target']])
        return int_list,str_list

############################################################

################### Model definition #########################################
#Encoder
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        """
        output of rnn is not used in simple decoder,but used in attention decoder
        :param input_size: 29 (containing:SOS,EOS,UNK,a-z)
        :param hidden_size: 256
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden_state, cell_state):
        """
        batch_size here is 1
        :param input: tensor
        :param hidden_state: (num_layers*num_directions=1,batch_size=1,vec_dim=256)
        :param cell_state: (num_layers*num_directions=1,batch_size=1,vec_dim=256)
        """
        embedded = self.embedding(input).view(1, 1, -1)  # view(1,1,-1) due to input of rnn must be (seq_len,batch,vec_dim)
        output,(hidden_state,cell_state) = self.rnn(embedded, (hidden_state,cell_state) )
        return output,hidden_state,cell_state

    def init_h0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def init_c0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Simple Decoder
class SimpleDecoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(SimpleDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, cell_state):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden_state,cell_state) = self.rnn(output, (hidden_state,cell_state) )
        output = self.softmax(self.out(output[0]))
        return output,hidden_state,cell_state

    def init_h0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def init_c0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)

############################################################


################### Training #########################################
def train(decoder_type,input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length,teacher_forcing_ratio,device):
    """
    train by one (input,target) pair
    :param decoder_type: 'simple' or 'attention'
    :param input_tensor: (time1,1) tensor for encoder
    :param target_tensor: (time2,1) tensor for decoder
    :param max_length: word maximum length in training data
    :return: loss value
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    """
    encoder forwarding
    """
    encoder_hidden_state=encoder.init_h0()
    encoder_cell_state=encoder.init_c0()
    for ei in range(input_length):
        encoder_output,encoder_hidden_state,encoder_cell_state=encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
        # encoder_output: (time,batch,num_directions*hidden_size)
        encoder_outputs[ei]=encoder_outputs[0,0]

    """
    decoder forwarding
    """
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden_state=encoder_hidden_state
    decoder_cell_state=encoder_cell_state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if decoder_type=='simple':
                decoder_output,decoder_hidden_state,decoder_cell_state=decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            else:  # decoder_type=='attention'
                decoder_output,decoder_hidden_state,decoder_cell_state,_=decoder(decoder_input,decoder_hidden_state,decoder_cell_state,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing
            # don't care decoder_output is EOS or not

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if decoder_type == 'simple':
                decoder_output, decoder_hidden_state, decoder_cell_state = decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            else:  # decoder_type=='attention'
                decoder_output, decoder_hidden_state, decoder_cell_state, _ = decoder(decoder_input,decoder_hidden_state,decoder_cell_state,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length

def trainIters(decoder_type,encoder,decoder,training_pairs,learning_rate,max_length,teacher_forcing_ratio,device):
    """
    :param decoder_type: 'simple' or 'attention'
    :param training_pairs: [(input,target),(input,target)....(input,target)] (input=(input_len,1)tensor, target=(target_len,1)tensor)
    """
    assert decoder_type=='simple' or decoder_type=='attention','no such decoder_type'
    loss_total=0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    random.shuffle(training_pairs)  # shuffle training_pairs
    for input_tensor,target_tensor in training_pairs:
        loss = train(decoder_type,input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length,teacher_forcing_ratio,device)
        loss_total+=loss

    return loss_total/len(training_pairs)

def evaluate(decoder_type,input_tensor,encoder,decoder,max_length,device):
    """
    :param decoder_type: 'simple' or 'attention'
    :param input_tensor: (time,1) tensor for encoder
    :param max_length: word maximum length in training data
    :return: predicted indices list
    """
    predicted=[]
    input_length=input_tensor.size(0)
    encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)
    """
    encoder forwarding
    """
    encoder_hidden_state=encoder.init_h0()
    encoder_cell_state=encoder.init_c0()
    for ei in range(input_length):
        encoder_output,encoder_hidden_state,encoder_cell_state=encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
        encoder_outputs[ei]=encoder_output[0,0]
    """
    decoder forwarding
    """
    decoder_input=torch.tensor([[SOS_token]],device=device)
    decoder_hidden_state=encoder_hidden_state
    decoder_cell_state=encoder_cell_state
    for di in range(max_length):
        if decoder_type=='simple':
            decoder_output,decoder_hidden_state,decoder_cell_state=decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
        else:  # decoder_type=='attention'
            decoder_output,decoder_hidden_state,decoder_cell_state,_=decoder(decoder_input,decoder_hidden_state,decoder_cell_state,encoder_outputs)
        topv,topi=decoder_output.data.topk(1)
        decoder_input=topi.squeeze().detach()
        if decoder_input.item()==EOS_token:
            break
        else:
            predicted.append(decoder_input.item())
    return predicted

def evaluateAll(decoder_type,encoder,decoder,testing_pairs,max_length,device):
    """
    :param decoder_type: 'simple' or 'attention'
    :param testing_pairs: [(input,target),(input,target)....(input,target)] (input=(input_len,1)tensor, target=(target_len,1)tensor)
    :param max_length: word maximum length in training data
    :return: [predicted1,predicted2,......] a list of integer list
    """
    assert decoder_type=='simple' or decoder_type=='attention','no such decoder_type'
    predicted_list=[]
    for input_tensor,target_tensor in testing_pairs:
        predicted_list.append(evaluate(decoder_type,input_tensor,encoder,decoder,max_length,device))
    return predicted_list

###############################################################

#################### plot #######################################
import matplotlib.pyplot as plt

def plot(loss,bleu):
    fig=plt.figure(figsize=(8,6))
    plt.plot(loss,label='loss')
    plt.plot(bleu,label='BLEU-4')
    plt.legend()
    return fig

###############################################################
if __name__=='__main__':
    """
    load training data
    """
    # training data
    datatransformer = DataTransformer()
    training_list,_ = datatransformer.build_training_set(path='train.json')
    training_tensor_list = []
    # convert list to tensor
    for training_pair in training_list:
        input_tensor = torch.tensor(training_pair[0], device=device).view(-1, 1)
        target_tensor = torch.tensor(training_pair[1], device=device).view(-1, 1)
        training_tensor_list.append((input_tensor, target_tensor))
    # testing data
    testing_list,testing_input=datatransformer.build_training_set(path='test.json')
    testing_tensor_list=[]
    # convert list to tensor
    for testing_pair in testing_list:
        input_tensor=torch.tensor(testing_pair[0],device=device).view(-1,1)
        target_tensor=torch.tensor(testing_pair[1],device=device).view(-1,1)
        testing_tensor_list.append((input_tensor,target_tensor))
    """
    model
    """
    encoder=EncoderRNN(vocab_size,hidden_size).to(device)
    if decoder_type=='simple':
        decoder=SimpleDecoderRNN(vocab_size,hidden_size).to(device)
    else:
        decoder=AttentionDecoderRNN(vocab_size,hidden_size,datatransformer.MAX_LENGTH).to(device)
    """
    train
    """
    loss_list=[]
    BLEU_list=[]
    best_score=0
    best_encoder_wts,best_decoder_wts=None,None
    for epoch in range(1,epochs+1):
        loss=trainIters(decoder_type,encoder,decoder,training_tensor_list,learning_rate=0.05,max_length=datatransformer.MAX_LENGTH,teacher_forcing_ratio=0.5,device=device)
        print(f'epoch{epoch:>2d} loss:{loss:.4f}')
        predicted_list=evaluateAll(decoder_type,encoder,decoder,testing_tensor_list,max_length=datatransformer.MAX_LENGTH,device=device)
        # test all testing data
        score=0
        for i,(input,target) in enumerate(testing_input):
            predict=datatransformer.indices2sequence(predicted_list[i])
            print(f'input:  {input}')
            print(f'target: {target}')
            print(f'pred:   {predict}')
            print('============================')
            score+=compute_bleu(predict,target)
        score/=len(testing_input)
        print(f'BLEU-4: {score:.2f}')

        loss_list.append(loss)
        BLEU_list.append(score)
        # update best model wts
        if score>best_score:
            best_score=score
            best_encoder_wts=copy.deepcopy(encoder.state_dict())
            best_decoder_wts=copy.deepcopy(decoder.state_dict())

    # save model
    torch.save(best_encoder_wts,os.path.join('models',f'encoder_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.pt'))
    torch.save(best_decoder_wts,os.path.join('models',f'{decoder_type}decoder_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.pt'))
    # plot
    figure=plot(loss_list,BLEU_list)
    figure.show()
    figure.savefig(os.path.join('result',f'{decoder_type}_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.png'))
    plt.waitforbuttonpress(0)




