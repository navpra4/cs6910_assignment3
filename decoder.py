import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 31




class DecoderRNN(nn.Module):
    def __init__(self, hidden_size,embedding_size ,output_size,num_layers=3,bidirectional=True,cell_type="LSTM"):
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(output_size, embedding_size)
        if(cell_type=='LSTM'):
          self.gru = nn.LSTM(embedding_size, hidden_size, batch_first=True,num_layers=num_layers,bidirectional=bidirectional)
        elif cell_type=='GRU':
          self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True,num_layers=num_layers,bidirectional=bidirectional)
        elif cell_type=='RNN':
          self.gru = nn.RNN(embedding_size, hidden_size, batch_first=True,num_layers=num_layers,bidirectional=bidirectional)

        self.out = nn.Linear(hidden_size* (2 if bidirectional else 1), output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))

        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,embedding_size,num_layers=1,dropout_p=0.1,bidirectional=False,cell_type='GRU'):
        super(AttnDecoderRNN, self).__init__()


        self.cell_type = cell_type

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = BahdanauAttention(hidden_size)
        if(cell_type=='GRU'):
          self.gru = nn.GRU(embedding_size+ hidden_size, hidden_size,num_layers=num_layers, batch_first=True)
        elif(cell_type=='LSTM'):
          self.gru = nn.LSTM(embedding_size+ hidden_size, hidden_size,num_layers=num_layers, batch_first=True)
        else:
          self.gru = nn.RNN(embedding_size+ hidden_size, hidden_size,num_layers=num_layers, batch_first=True)
        #self.gru = nn.GRU(embedding_size+ hidden_size, hidden_size, batch_first=True)
        #self.gru = nn.GRU(2* hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        if self.cell_type == 'LSTM':
            h_n, c_n = encoder_hidden
            decoder_hidden = (h_n, c_n)
        else:
            decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):

        embedded =  self.dropout(self.embedding(input))

        if self.cell_type == 'LSTM':
            h_n, c_n = hidden
            #query = h_n.permute(1, 0, 2)
            query = h_n[-1].unsqueeze(0).permute(1, 0, 2)
        else:
            #query = hidden.permute(1, 0, 2)
            query = hidden[-1].unsqueeze(0).permute(1, 0, 2)

        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        if self.cell_type == 'LSTM':
            output, (h_n, c_n) = self.gru(input_gru, hidden)
            hidden = (h_n, c_n)
        else:
            output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights