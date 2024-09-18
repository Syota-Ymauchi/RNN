import torch

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='LSTM', bidirectional=False, print_shape=False):
        super().__init__()
        self.num_direction = 2 if bidirectional else 1
        self.print_shape = print_shape


        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional) 
        
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional) 

        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional) 

       
        else:
            raise ValueError('Unsapported RNN type. Choose from ["LSTM", "RNN", "GUR"')

        self.fc = nn.Linear(hidden_size * self.num_direction, output_size)
            
            
            
     

    def forward(self, x):
        output_seq, _ = self.rnn(x)

        # bidirectionが実行されたshape確認用
        if self.print_shape:
            print(output_seq.shape)

        # output_seq : [batch_size, seq_len, hidden_size*num_direction]
        output_seq = output_seq[:, -1, :]

        out = self.fc(output_seq)
        return out