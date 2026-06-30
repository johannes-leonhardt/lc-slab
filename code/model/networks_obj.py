import torch
import torch.nn as nn
from torch_scatter import scatter
import torch_geometric.nn as pygnn
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.pool import graclus

# Network implementations

class MLP(nn.Module):

    def __init__(self, n_in, n_out, n_hidden, n_layers):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layer = n_layers

        self.input_layer = _SingleLinear(n_in, n_hidden)
        self.hidden_layers = nn.ModuleList([_SingleLinear(n_hidden, n_hidden) for _ in range(n_layers)])
        self.output_layer = pygnn.Linear(n_hidden, n_out)

    def forward(self, graph):

        x, seg = graph.x, graph.seg
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = torch.stack([x[graph.batch == i][seg[i]].permute(2, 0, 1) for i in torch.unique(graph.batch)])

        return x, None
    
ANN = MLP # Alias

class GNN(nn.Module):

    def __init__(self, n_in, n_out, n_hidden, n_layers, conv, conv_kwargs):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layer = n_layers

        self.input_layer = _SingleLinear(n_in, n_hidden)
        self.hidden_layers = nn.ModuleList([_SingleConv(n_hidden, n_hidden, conv, conv_kwargs) for _ in range(n_layers)])
        self.output_layer = pygnn.Linear(n_hidden, n_out)

    def forward(self, graph):

        x, edge_index, seg = graph.x, graph.edge_index, graph.seg
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        x = self.output_layer(x)
        x = torch.stack([x[graph.batch == i][seg[i]].permute(2, 0, 1) for i in torch.unique(graph.batch)])

        return x, None

class GraphUNet(nn.Module):

    def __init__(self, n_in, n_out, n_hidden, n_layers, conv, conv_kwargs):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layer = n_layers
        self.conv = conv
        self.conv_kwargs = conv_kwargs
        
        self.input_layer = _DoubleConv(n_in, n_hidden, conv, conv_kwargs)
        self.down_layers, self.up_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            self.down_layers.append(_Down(n_hidden * (2 ** i), n_hidden * (2 ** (i + 1)), conv, conv_kwargs))
        for i in reversed(range(n_layers)):
            self.up_layers.append(_Up(n_hidden * (2 ** (i + 1)), n_hidden * (2 ** i), conv, conv_kwargs))
        self.output_layer = pygnn.Linear(n_hidden, n_out)

    def forward(self, graph):

        x, edge_index, batch, pos, seg = graph.x, graph.edge_index, graph.batch, graph.pos, graph.seg
        x = self.input_layer(x, edge_index)
        xs, edge_indices, perms = [], [], []
        for down_layer in self.down_layers:
            xs.append(x)
            edge_indices.append(edge_index)
            x, edge_index, batch, pos, perm = down_layer(x, edge_index, batch, pos)
            perms.append(perm)
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, xs[-i-1], edge_indices[-i-1], perms[-i-1])
        x = self.output_layer(x)
        x = torch.stack([x[graph.batch == i][seg[i]].permute(2, 0, 1) for i in torch.unique(graph.batch)])

        return x, None
    
# Network building blocks
    
class LinearConv(nn.Module):

    def __init__(self, n_in, n_out):

        super().__init__()

        self.conv1 = pygnn.Linear(n_in, n_out)

    def forward(self, x, edge_index):
        
        x = self.conv1(x)

        return x

class _SingleLinear(nn.Module):

    def __init__(self, n_in, n_out):

        super().__init__()

        self.conv1 = pygnn.Linear(n_in, n_out)
        self.bn1 = pygnn.BatchNorm(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x
    
class _SingleConv(nn.Module):

    def __init__(self, n_in, n_out, conv, conv_kwargs):

        super().__init__()

        self.conv1 = conv(n_in, n_out, **conv_kwargs)
        self.bn1 = pygnn.BatchNorm(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)

        return x
    
class _Down(nn.Module):

    def __init__(self, n_in, n_out, conv, conv_kwargs):

        super().__init__()
        self.pool = _GraclusPool()
        self.double_conv = _DoubleConv(n_in, n_out, conv, conv_kwargs)

    def forward(self, x, edge_index, batch, pos):

        x, edge_index, batch, pos, perm = self.pool(x, edge_index, batch, pos)
        x = self.double_conv(x, edge_index)

        return x, edge_index, batch, pos, perm
    
class _Up(nn.Module):

    def __init__(self, n_in, n_out, conv, conv_kwargs):

        super().__init__()
        self.linear = pygnn.Linear(n_in, n_in//2)
        self.double_conv = _DoubleConv(n_in, n_out, conv, conv_kwargs)

    def forward(self, x, x_res, edge_index_res, perm):

        x = self.linear(x)
        x_up = x[perm]
        x = torch.cat((x_res, x_up), dim=-1)
        x = self.double_conv(x, edge_index_res)

        return x
    
class _DoubleConv(nn.Module):

    def __init__(self, n_in, n_out, conv, conv_kwargs):

        super().__init__()

        self.conv1 = conv(n_in, n_out, **conv_kwargs)
        self.conv2 = conv(n_out, n_out, **conv_kwargs)
        self.bn1 = pygnn.BatchNorm(n_out)
        self.bn2 = pygnn.BatchNorm(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class _GraclusPool(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x, edge_index, batch, pos):
            
        edge_index_wo_self_loops, _ = remove_self_loops(edge_index)
        weights = 1 / torch.sum((x[edge_index_wo_self_loops[0]] - x[edge_index_wo_self_loops[1]]) ** 2, dim=1)
        perm = graclus(edge_index_wo_self_loops, weights, num_nodes=x.shape[0])
        _, perm = torch.unique(perm, return_inverse=True)
        x = scatter(x, perm, dim=0, reduce="mean")
        edge_index = torch.unique(perm[edge_index], dim=1)
        batch = scatter(batch, perm, dim=0, reduce="min")
        pos = scatter(pos, perm, dim=0, reduce="mean")
        
        return x, edge_index, batch, pos, perm