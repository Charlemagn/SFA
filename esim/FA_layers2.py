import torch
import math
import torch.nn as nn

class FABlock(nn.Module):
    def __init__(self, seq_len:int, input_dim:int, dim:int)->None:
        super(FABlock, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.dim = int(max(dim, 2 * (8.33 * math.log(self.seq_len) // 2)))
        self.squeeze_avg = nn.AvgPool1d(self.seq_len)
        self.dense_z = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.dim))
        self.dense_e = nn.Sequential(nn.Linear(in_features=self.dim, out_features=self.input_dim)
                      , nn.BatchNorm1d(num_features=self.input_dim, eps=1e-6, momentum=0.1, affine=True)
                      , nn.Sigmoid())
        
    def forward(self, inputs)->torch.Tensor:
        assert inputs.shape[-2] == self.seq_len, "squence length must be correct"
        assert inputs.shape[-1] == self.input_dim, "feature demension must be correct"
        squeeze_avg = self.squeeze_avg(inputs.clone().transpose(-2,-1)).squeeze(-1)
        dense_z = self.dense_z(squeeze_avg)
        dense_e = self.dense_e(dense_z)
        output = inputs * dense_e.unsqueeze(1)

        return output


class SFABlock(nn.Module):
    def __init__(self,
                 seq_len:int,
                 input_dim:int,
                 dim1:int,
                 dim2:int,
                 seq2seq_layers_typ:nn.Module=nn.GRU,
                 num_stacked_layers:int=3,
                 dropout_rate:float=0.2)-> None:
        super(SFABlock, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.dim1 = int(max(dim1, 2 * (8.33 * math.log(self.seq_len) // 2)))
        self.dim2 = int(max(dim2, 2 * (8.33 * math.log(self.seq_len) // 2)))
        self.seq2seq_layers = seq2seq_layers_typ
        self.num_stacked_layers = num_stacked_layers
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.auto_encoder_start = nn.Conv1d(in_channels=self.input_dim, out_channels=self.dim1, kernel_size=1, padding='same')
        self.auto_encoder_end = nn.Conv1d(in_channels=self.dim1, out_channels=self.input_dim, kernel_size=1, padding='same')
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.global_max_pooling = nn.AdaptiveMaxPool2d((1,1))
        self.dense_z = nn.Sequential(nn.Linear(in_features=self.dim1, out_features=self.dim2)
                                     ,nn.Tanh())
        self.dense_select_weights = nn.ModuleList([nn.Sequential(nn.Linear(in_features=self.dim2, out_features=self.dim1)
                                                   ,nn.BatchNorm1d(num_features=self.dim1, eps=1e-6, momentum=0.1, affine=True)
                                                   ,nn.Sigmoid()) for _ in range(self.num_stacked_layers)])
        self.seq2seq_layers = nn.ModuleList([self.seq2seq_layers(input_size=self.dim1, hidden_size=self.dim1 // 2, bidirectional=True, batch_first=True) for _ in
                         range(self.num_stacked_layers)])

    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        assert inputs.shape[-2] == self.seq_len, "squence length must be correct"
        assert inputs.shape[-1] == self.input_dim, "feature demension must be correct"
        inputs_ = self.auto_encoder_start(inputs.clone().transpose(-1,-2))
        seq2seq_branches = []
        hidden_states = []
        for layer in range(self.num_stacked_layers):
            output, hidden = self.seq2seq_layers[layer](inputs_.transpose(-1,-2), hidden_states[-1] if layer > 0 else None)
            if self.dropout_rate > 0:
                output = self.dropout(output)
            seq2seq_branches.append(output)
            hidden_states.append(hidden)
        fuse_branches = torch.stack(seq2seq_branches, dim=1).permute(0,3,1,2)
        squeeze_avg = self.global_avg_pooling(fuse_branches)
        squeeze_max = self.global_max_pooling(fuse_branches)
        z = self.dense_z((squeeze_avg + squeeze_max).squeeze(-1).squeeze(-1))

        select_weights = [self.dense_select_weights[i](z) for i in range(self.num_stacked_layers)]
        select_weights_norm = [torch.exp(weight)/torch.sum(torch.exp(torch.stack(select_weights))) for weight in select_weights]

        weighted_added_branches = torch.sum(torch.stack([branch * weight.unsqueeze(1) for branch, weight in zip(seq2seq_branches, select_weights_norm)]), dim=0)
        output = self.auto_encoder_end(weighted_added_branches.clone().transpose(-1,-2))

        return output.transpose(-1,-2)


if __name__ == '__main__':
    batch_size = 32
    seq_len = 100
    feature_dim = 300
    dim1 = 128
    dim2 = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(batch_size, seq_len, feature_dim).to(device)
    # fa_block = FABlock(seq_len, feature_dim, dim1).to(device)
    sfa_block = SFABlock(seq_len, feature_dim, dim1, dim2).to(device)
    outputs = sfa_block(inputs)
    print(inputs.device)
    print(outputs.device)
    assert inputs.shape == outputs.shape




