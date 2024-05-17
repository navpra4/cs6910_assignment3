import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#MAX_LENGTH = 31

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,embedding_size, dropout_p=0.1,num_layers=3,bidirectional=True,cell_type="LSTM"):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_size)
        if(cell_type=='LSTM'):
          self.gru = nn.LSTM(embedding_size, hidden_size, batch_first=True,num_layers=num_layers,bidirectional=bidirectional)
        elif cell_type=='GRU':
          self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True,num_layers=num_layers,bidirectional=bidirectional)
        elif cell_type=='RNN':
          self.gru = nn.RNN(embedding_size, hidden_size, batch_first=True,num_layers=num_layers,bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
      embedded = self.dropout(self.embedding(input))
      output, hidden = self.gru(embedded)
      return output, hidden
    



    
