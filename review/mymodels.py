import torch
import torch.nn as nn

# bidirection対応に変更する
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirection=False):
        """
        embedding layerを使用する時はinput_sizeにembedding dimを引数として入れる
        """
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_direction = 2 if bidirection == True else 1
    
        # 全結合層
        self.hidden = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x : h] -> input dim = embedding dim + hidden dim
        if self.num_direction == 2:
            self.backward_hidden = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: h] -> input dim = embedding dim + hidden dim
       
        # 活性化関数
        self.tanh = nn.Tanh()
        
    
    def forward(self, input, h_0=None):
  
        batch_size, seq_len , _ = input.size() # input : [batch_size, seq_len(embedding_dim), input_size(vocab_size)]

        if h_0 is None:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
        else:
            h_0 = h_0
        
        h = h_0.squeeze(0) # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        
        output = []
        for i in range(seq_len):
            combined = torch.cat((input[:,i,:], h), dim=1)  
            h = self.tanh(self.hidden(combined))
            
            output.append(h.unsqueeze(1)) # h : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
   
        # backward   
        if self.num_direction == 2:
            h_backeard_0 = h_0
            out_backward = []
  
            h_backward = h_backeard_0.squeeze(0)
            for i in reversed(range(seq_len)):
                combined = torch.cat((input[:,i,:], h_backward), dim=1)  
                h_backward =  self.tanh(self.backward_hidden(combined))
        
                out_backward.append(h_backward.unsqueeze(1)) # h_backward : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
         
    
            out_backward = out_backward[::-1] # [t=T, T-1, ... ,0] -> [0, 1, 2, ... , T]
          
            out_backward = torch.cat(out_backward, dim=1) # [batch_size, seq_len, hidden_size]
            output = torch.cat(output, dim=1)   
            output_seq = torch.cat((output, out_backward), dim=2) #[batch_size, seq_len, hidden_size*2]
            h_n = torch.cat((h.unsqueeze(0), h_backward.unsqueeze(0)), dim=0) # [2, batch_size, hidden_size]
            
            return output_seq, h_n

        # not use bidirection
        else:
            h_n = h.unsqueeze(0) # [batch_size, hidden_size] -> [1, batch_size, hidden_size]    
            output_seq = torch.cat(output, dim=1) # [batch_size, seq_len, hidden_size]
    
            return output_seq, h_n

class MyUGRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirection=False):
        """
        embedding layerを使用する時はinput_sizeにembedding dimを引数として入れる
        """
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_direction = 2 if bidirection == True else 1
        
    
        # 全結合層
        self.hidden_candidate = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: h] -> input dim = embedding dim + hidden dim
        self.update_gate = nn.Linear(input_size+self.hidden_size, self.hidden_size)
        if self.num_direction == 2:
            self.backward_hidden = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: h] -> input dim = embedding dim + hidden dim

        # 活性化関数
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, h_0=None):
  
        batch_size, seq_len , _ = input.size() # input : [batch_size, seq_len(embedding_dim), input_size(vocab_size)]

        if h_0 is None:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
        else:
            h_0 = h_0
        
        h = h_0.squeeze(0) # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        
        output = []
        for i in range(seq_len):
            combined = torch.cat((input[:,i,:], h), dim=1)  
            h_candidate = self.tanh(self.hidden_candidate(combined))
            update_gate = self.sigmoid(self.update_gate(combined))
            h = update_gate * h_candidate + (1 - update_gate) * h 
            output.append(h.unsqueeze(1)) # h : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
         # backward   
        if self.num_direction == 2:
            h_backeard_0 = h_0
            out_backward = []
  
            h_backward = h_backeard_0.squeeze(0)
            for i in reversed(range(seq_len)):
                combined = torch.cat((input[:,i,:], h_backward), dim=1)  
                h_backward =  self.tanh(self.backward_hidden(combined))
        
                out_backward.append(h_backward.unsqueeze(1)) # h_backward : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
         
    
            out_backward = out_backward[::-1] # [t=T, T-1, ... ,0] -> [0, 1, 2, ... , T]
          
            out_backward = torch.cat(out_backward, dim=1) # [batch_size, seq_len, hidden_size]
            output = torch.cat(output, dim=1)   
            output_seq = torch.cat((output, out_backward), dim=2) #[batch_size, seq_len, hidden_size*2]
            h_n = torch.cat((h.unsqueeze(0), h_backward.unsqueeze(0)), dim=0) # [2, batch_size, hidden_size]
            
            return output_seq, h_n

        # not use bidirection

        h_n = h.unsqueeze(0) # [batch_size, hidden_size] -> [1, batch_size, hidden_size]    
        output_seq = torch.cat(output, dim=1) # [batch_size, seq_len, hidden_size]

        return output_seq, h_n



class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirection=False):
        """
        embedding layerを使用する時はinput_sizeにembedding dimを引数として入れる
        """
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_direction = 2 if bidirection == True else 1

    
        # 全結合層
        self.hidden_candidate = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: W.T] -> input dim = input(embedding) dim + hidden dim
        self.update_gate = nn.Linear(input_size+self.hidden_size, self.hidden_size)
        self.reset_gate = nn.Linear(input_size+self.hidden_size, self.hidden_size)
        if self.num_direction == 2:
            self.backward_hidden = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: h] -> input dim = embedding dim + hidden dim


        # 活性化関数
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, h_0=None):
  
        batch_size, seq_len , _ = input.size() # input : [batch_size, seq_len(embedding_dim), input_size(vocab_size)]

        if h_0 is None:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
        else:
            h_0 = h_0
        
        h = h_0.squeeze(0) # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        
        output = []
   
        for i in range(seq_len):
            combined = torch.cat((input[:,i,:], h), dim=1) 

            # リセットゲートを計算してから combined_reset を作成(h_candidateで使用)
            reset_gate = self.sigmoid(self.reset_gate(combined)) 
            combined_reset = torch.cat((input[:,i,:], h*reset_gate), dim=1) 

            update_gate = self.sigmoid(self.update_gate(combined))
            h_candidate = self.tanh(self.hidden_candidate(combined_reset))
            h = update_gate * h_candidate + (1 - update_gate) * h 
            output.append(h.unsqueeze(1)) # h : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        # backward   
        if self.num_direction == 2:
            h_backeard_0 = h_0
            out_backward = []
  
            h_backward = h_backeard_0.squeeze(0)
            for i in reversed(range(seq_len)):
                combined = torch.cat((input[:,i,:], h_backward), dim=1)  
                h_backward =  self.tanh(self.backward_hidden(combined))
        
                out_backward.append(h_backward.unsqueeze(1)) # h_backward : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
         
    
            out_backward = out_backward[::-1] # [t=T, T-1, ... ,0] -> [0, 1, 2, ... , T]
          
            out_backward = torch.cat(out_backward, dim=1) # [batch_size, seq_len, hidden_size]
            output = torch.cat(output, dim=1)   
            output_seq = torch.cat((output, out_backward), dim=2) #[batch_size, seq_len, hidden_size*2]
            h_n = torch.cat((h.unsqueeze(0), h_backward.unsqueeze(0)), dim=0) # [2, batch_size, hidden_size]
            
            return output_seq, h_n

        # not use bidirection

        h_n = h.unsqueeze(0) # [batch_size, hidden_size] -> [1, batch_size, hidden_size]    
        output_seq = torch.cat(output, dim=1) # [batch_size, seq_len, hidden_size]

        return output_seq, h_n




class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirection=False):
        """
        embedding layerを使用する時はinput_sizeにembedding dimを引数として入れる
        """
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_direction = 2 if bidirection == True else 1

    
        # 全結合層
        self.cell_candidate = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: W.T] -> input dim = input(embedding) dim + hidden dim
        self.update_gate = nn.Linear(input_size+self.hidden_size, self.hidden_size)
        self.forget_gate = nn.Linear(input_size+self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(input_size+self.hidden_size, self.hidden_size)
        if self.num_direction == 2:
            self.backward_hidden = nn.Linear(input_size+self.hidden_size, self.hidden_size) #[x: h] -> input dim = embedding dim + hidden dim




        # 活性化関数
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, h_0=None, c_0=None):
  
        batch_size, seq_len , _ = input.size() # input : [batch_size, seq_len(embedding_dim), input_size(vocab_size)]

        if h_0 is None:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
        else:
            h_0 = h_0
        
        if c_0 is None:
            c_0 = torch.zeros(1, batch_size, self.hidden_size)
        else:
            c_0 = c_0
        
        h = h_0.squeeze(0) # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        c = c_0.squeeze(0) # [1, batch_size, hidden_size] -> [batch_size, hidden_size]


        
        output = []
   
        for i in range(seq_len):
            combined = torch.cat((input[:,i,:], h), dim=1) 
            c_candidate = self.tanh(self.cell_candidate(combined))
            forget_gate = self.sigmoid(self.forget_gate(combined))
            update_gate = self.sigmoid(self.update_gate(combined))
            output_gate = self.sigmoid(self.output_gate(combined))
            c = update_gate * c_candidate + forget_gate * c
            h = output_gate * self.tanh(c) # [batch_size, hidden_size]
            output.append(h.unsqueeze(1)) # [batch_size, 1, hidden_size]
        
        # backward   
        if self.num_direction == 2:
            h_backeard_0 = h_0
            out_backward = []
  
            h_backward = h_backeard_0.squeeze(0)
            for i in reversed(range(seq_len)):
                combined = torch.cat((input[:,i,:], h_backward), dim=1)  
                h_backward =  self.tanh(self.backward_hidden(combined))
        
                out_backward.append(h_backward.unsqueeze(1)) # h_backward : [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
         
    
            out_backward = out_backward[::-1] # [t=T, T-1, ... ,0] -> [0, 1, 2, ... , T]
          
            out_backward = torch.cat(out_backward, dim=1) # [batch_size, seq_len, hidden_size]
            output = torch.cat(output, dim=1)   
            output_seq = torch.cat((output, out_backward), dim=2) #[batch_size, seq_len, hidden_size*2]
            h_n = torch.cat((h.unsqueeze(0), h_backward.unsqueeze(0)), dim=0) # [2, batch_size, hidden_size]
            
            return output_seq, h_n

        # not use bidirection

        h_n = h.unsqueeze(0) # [batch_size, hidden_size] -> [1, batch_size, hidden_size]   
        c_n = c.unsqueeze(0) # [batch_size, hidden_size] -> [1, batch_size, hidden_size]  
        output_seq = torch.cat(output, dim=1) # [batch_size, seq_len, hidden_size]

        return output_seq, (h_n, c_n)



        
        
            
        

        
        


        
