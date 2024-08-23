import torch
import math
import torch.nn as nn

class FABlock(nn.Module):
    def __init__(self, dim:int)->None:
        super(FABlock, self).__init__()
        self.dim = dim

    def forward(self, inputs)->torch.Tensor:
        self.dim = int(max(self.dim, 2 * (8.33 * math.log(inputs.shape[-2]) // 2)))
        squeeze_avg = nn.AvgPool1d(inputs.shape[-2])(inputs.clone().transpose(-2,-1)).squeeze(-1)
        dense_z = nn.Sequential(nn.Linear(in_features=squeeze_avg.shape[-1], out_features=self.dim)
                      , nn.Tanh())(squeeze_avg)
        dense_e = nn.Sequential(nn.Linear(in_features=self.dim, out_features=squeeze_avg.shape[-1])
                      , nn.BatchNorm1d(num_features=self.dim1, eps=1e-6, momentum=0.1, affine=True)
                      , nn.Sigmoid())(dense_z)
        output = inputs * dense_e.unsqueeze(1)

        return output


class SFABlock(nn.Module):
    def __init__(self,
                 dim1:int,
                 dim2:int,
                 seq2seq_layers_typ:nn.Module=nn.GRU,
                 num_stacked_layers:int=3,
                 dropout_rate:float=0.2)-> None:
        super(SFABlock, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.seq2seq_layers = seq2seq_layers_typ
        self.num_stacked_layers = num_stacked_layers
        self.dropout_rate = dropout_rate

    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        self.dim1 = int(max(self.dim1, 2 * (8.33 * math.log(inputs.shape[-2]) // 2)))
        self.dim2 = int(max(self.dim2, 2 * (8.33 * math.log(inputs.shape[-2]) // 2)))
        global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        global_max_pooling = nn.AdaptiveAvgPool2d((1,1))
        dense_z = nn.Sequential(nn.Linear(in_features=self.dim1, out_features=self.dim2)
                                     ,nn.Tanh())
        dense_select_weights = [nn.Sequential(nn.Linear(in_features=self.dim2, out_features=self.dim1)
                                                   ,nn.BatchNorm1d(num_features=self.dim1, eps=1e-6, momentum=0.1, affine=True)
                                                   ,nn.Sigmoid()) for _ in range(self.num_stacked_layers)]
        seq2seq_layers = nn.ModuleList([self.seq2seq_layers(input_size=self.dim1, hidden_size=self.dim1 // 2, bidirectional=True, batch_first=True) for _ in
                         range(self.num_stacked_layers)])
        inputs_ = nn.Conv1d(in_channels=inputs.shape[-1], out_channels=self.dim1, kernel_size=1, padding='same')(inputs.clone().transpose(-1,-2))
        seq2seq_branches = []
        hidden_states = []
        for layer in range(self.num_stacked_layers):
            output, hidden = seq2seq_layers[layer](inputs_.transpose(-1,-2), hidden_states[-1] if layer > 0 else None)
            if self.dropout_rate > 0:
                output = nn.Dropout(self.dropout_rate)(output)
            seq2seq_branches.append(output)
            hidden_states.append(hidden)
        fuse_branches = torch.stack(seq2seq_branches, dim=1).permute(0,3,1,2)
        squeeze_avg = global_avg_pooling(fuse_branches)
        squeeze_max = global_max_pooling(fuse_branches)
        z = dense_z((squeeze_avg + squeeze_max).squeeze(-1).squeeze(-1))

        select_weights = [dense_select_weights[i](z) for i in range(self.num_stacked_layers)]
        select_weights_norm = [torch.exp(weight)/torch.sum(torch.exp(torch.stack(select_weights))) for weight in select_weights]

        weighted_added_branches = torch.sum(torch.stack([branch * weight.unsqueeze(1) for branch, weight in zip(seq2seq_branches, select_weights_norm)]), dim=0)
        output = nn.Conv1d(in_channels=weighted_added_branches.shape[-1], out_channels=inputs.shape[-1], kernel_size=1, padding='same')(weighted_added_branches.clone().transpose(-1,-2))

        return output.transpose(-1,-2)


if __name__ == '__main__':
    batch_size = 32
    seq_len = 100
    feature_dim = 300
    dim1 = 128
    dim2 = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(batch_size, seq_len, feature_dim).to(device)
    sfa_block = SFABlock(dim1, dim2).to(device)
    outputs = sfa_block(inputs)
    print(inputs.device)
    print(outputs.device)
    assert inputs.shape == outputs.shape




