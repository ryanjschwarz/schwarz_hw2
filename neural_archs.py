import torch
from torch import nn


# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.
class DAN(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, embeddings):
        # TODO: Declare DAN architecture
        super(DAN, self).__init__()   

        self.embeddings = embeddings
        self.linear_layer_one = nn.Linear(embedding_size, hidden_size)
        self.linear_layer_two = nn.Linear(hidden_size, 20)
        self.linear_layer_three = nn.Linear(embedding_size, hidden_size)
        self.linear_layer_four = nn.Linear(hidden_size, 20)
        self.linear = nn.Linear(40,output_size)
        self.sigmoid = nn.Sigmoid()
        self.flat = nn.Flatten()


    def forward(self, in_one, in_two):
        # TODO: Implement DAN forward pass
        embedded_layer_one = self.embeddings(in_one)
        pooled_one = torch.mean(embedded_layer_one, dim=1)
        hidden_layer_one = torch.nn.functional.relu(self.linear_layer_one(pooled_one.float()))
        output_layer_one = self.linear_layer_two(hidden_layer_one)
        embedded_layer_two = self.embeddings(in_two)
        pooled_two = torch.mean(embedded_layer_two, dim=1)
        hidden_layer_two = torch.nn.functional.relu(self.linear_layer_three(pooled_two.float()))
        output_layer_two = self.linear_layer_four(hidden_layer_two)
        
        output_combined = torch.cat((output_layer_one, output_layer_two),dim=1)
        output_combined_fin = self.flat(output_combined)
        output = self.sigmoid(self.linear(output_combined_fin))
       
        return output



class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embeddings, flag_bdir):
        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True, )
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn2 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.flat = torch.nn.Flatten()
        self.linear = torch.nn.Linear(2*hidden_size*num_layers, output_size)
        self.embeddings = embeddings
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        
        #unused
        rnn_loss = torch.nn.CrossEntropyLoss()


    def forward(self, in_one, in_two):
        # TODO: Implement RNN forward pass
        embedded = self.embeddings(in_two)
        in_one_embed = self.embeddings(in_one)
        in_two_embed = self.embeddings(in_two)
        output, hidden = self.rnn(embedded)

        #even though this brick isnt used, the dimensions warp incorrectly without it
        output_one, brick = self.rnn1(in_one_embed.float())
        output_two, brick = self.rnn2(in_two_embed.float())
        output_one_mean = torch.mean(output_one,dim = 1)
        output_two_mean = torch.mean(output_two, dim = 1)

        combined_out = torch.cat((output_one_mean, output_two_mean), dim = 1)
        flat_out = self.flat(combined_out)
        linear_out = self.sigmoid(self.linear(flat_out))
        
        return linear_out


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embeddings, flag_bdir):
        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(2*hidden_size*num_layers*(int(flag_bdir) + 1), output_size) #(1860, _) 
        self.embeddings = embeddings
        self.sigmoid = nn.Sigmoid()



    def forward(self, in_one, in_two):
        batch_size = in_one.size(0)
        in_one = self.embeddings(in_one)
        in_two = self.embeddings(in_two)
        #print(in_one.size())
        
        h1 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size)
        c1 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size)
        h2 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size)
        c2 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size)
        #print(h1.size())
        
        output_one, _ = self.lstm1(in_one.float(), (h1.float(), c1.float()))
        output_two, _ = self.lstm2(in_two.float(), (h2.float(), c2.float()))
        #print(output_one.size())
        #print(output_two.size())
        
        combined_out = torch.cat((output_one, output_two), dim=2)
        #print(combined_out.size())
        
        flat_out = self.flat(combined_out)
        #print(flat_out.size())
        linear_out = self.sigmoid(self.linear(flat_out))
        #print(linear_out.size())
        
        return linear_out
