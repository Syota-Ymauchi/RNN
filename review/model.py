import torch
import torch.nn as nn


from mymodels import MyRNN, MyUGRNN, MyGRU, MyLSTM 

class Model(nn.Module):
    """
    自分で実装したクラスをインポートして使用する
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, rnn_type='LSTM', bidirection=False):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        input_size = embedding_dim
        self.num_direction = 2 if bidirection == True else 1

        if self.rnn_type == 'RNN':
            self.rnn = MyRNN(input_size, hidden_size, bidirection=bidirection)

        elif self.rnn_type == 'LSTM':
            self.rnn = MyLSTM(input_size, hidden_size, bidirection=bidirection)

        elif self.rnn_type == 'GRU':
            self.rnn = MyLSTM(input_size, hidden_size, bidirection=bidirection)

        elif self.rnn_type == 'UGRNN':
            self.rnn = MyLSTM(input_size, hidden_size, bidirection=bidirection)

        else:
            raise ValueError('Unsupported RNN type. Choose from ["LSTM", "RNN", "GRU", "UGRNN"]')

        # 最終層の全結合層
        self.fc = nn.Linear(hidden_size*self.num_direction, output_size)
    
    def forward(self, input, h_0=None, c_0=None):
        
        input = self.embedding(input)

        if self.rnn_type == 'LSTM':
            output_seq, (h_n, c_n) = self.rnn(input, h_0, c_0)
            out = self.fc(output_seq)

            return out, (h_n, c_n)
            
        else:
            output_seq, h_n = self.rnn(input, h_0)
            out = self.fc(output_seq)

            return out, h_n

class PytorchModel(nn.Module):
    """
    pytorchで実装されたクラスを使用する
    """
    def __init__(self, vovab_size, embedding_dim, hidden_size, output_size, rnn_type='LSTM', bidirectional=False, num_layers=1):
        super().__init__()
        self.num_direction = 2 if bidirectional == True else 1
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        print(self.rnn_type)

        # embedding layer
        self.embedding = nn.Embedding(vovab_size, embedding_dim, padding_idx=0)

        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=self.num_layers)
        
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=self.num_layers)
        
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=self.num_layers)

        else:
            raise ValueError('Unsupported RNN type. Choose from ["LSTM", "RNN", "GRU"]')        
        
        # 最終層の全結合層
        self.fc = nn.Linear(self.hidden_size*self.num_direction, output_size)
    
    def forward(self, input, h_0=None, c_0=None):
        # 初期隠れ状態の生成
        if h_0 is None:
            h_0 = torch.zeros(self.num_direction*self.num_layers, input.size(0), self.hidden_size)
            c_0 = torch.zeros(self.num_direction*self.num_layers, input.size(0), self.hidden_size)

        
        # embedding
        embedded_input = self.embedding(input)

        if self.rnn_type != 'LSTM':
            output_seq, _ = self.rnn(embedded_input, h_0)
        
        else:
            output_seq, _ = self.rnn(embedded_input, (h_0, c_0))
        
        out = self.fc(output_seq)

        return out, _





    






