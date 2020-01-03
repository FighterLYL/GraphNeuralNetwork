import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import torch.nn.init as init


class StackGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, 
                 use_bias=False, activation=F.relu):
        """对得到的每类评分使用级联的方式进行聚合
        
        Args:
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 评分的类别数，比如1~5分，值为5
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(StackGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.use_bias = use_bias
        self.activation = activation
        assert output_dim % num_support == 0
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim, ))
        self.reset_parameters()
        self.weight = self.weight.view(input_dim, output_dim // 5, 5)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):
        """StackGCNEncoder计算逻辑
        
        Args:
            user_supports (list of torch.sparse.FloatTensor): 
                归一化后每个评分等级对应的用户与商品邻接矩阵
            item_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的商品与用户邻接矩阵
            user_inputs (torch.Tensor): 用户特征的输入
            item_inputs (torch.Tensor): 商品特征的输入
        
        Returns:
            [torch.Tensor]: 用户的隐层特征
            [torch.Tensor]: 商品的隐层特征
        """
        assert len(user_supports) == len(item_supports) == self.num_support
        user_hidden = []
        item_hidden = []
        for i in range(self.num_support):
            tmp_u = torch.matmul(self.weight[..., i], user_inputs)
            tmp_v = torch.matmul(self.weight[..., i], item_inputs)
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u)
            user_hidden.append(tmp_user_hidden)
            item_hidden.append(tmp_item_hidden)
        
        user_hidden = torch.cat(user_hidden, dim=1)
        item_hidden = torch.cat(item_hidden, dim=1)
        
        user_outputs = self.activation(user_hidden)
        item_outputs = self.activation(item_hidden)
        
        if self.use_bias:
            user_outputs += self.bias
            item_outputs += self.bias_item
        
        return user_outputs, item_outputs
    

class SumGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, 
                 use_bias=False, activation=F.relu):
        """对得到的每类评分使用求和的方式进行聚合
        
        Args:
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 评分的类别数，比如1~5分，值为5
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(SumGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.use_bias = use_bias
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim * num_support))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim, ))
        self.reset_parameters()
        self.weight = self.weight.view(input_dim, output_dim, 5)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):
        """SumGCNEncoder计算逻辑
        
        Args:
            user_supports (list of torch.sparse.FloatTensor): 
                归一化后每个评分等级对应的用户与商品邻接矩阵
            item_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的商品与用户邻接矩阵
            user_inputs (torch.Tensor): 用户特征的输入
            item_inputs (torch.Tensor): 商品特征的输入
        
        Returns:
            [torch.Tensor]: 用户的隐层特征
            [torch.Tensor]: 商品的隐层特征
        """
        assert len(user_supports) == len(item_supports) == self.num_support
        user_hidden = 0
        item_hidden = 0
        for i in range(self.num_support):
            tmp_u = torch.matmul(self.weight[..., i], user_inputs)
            tmp_v = torch.matmul(self.weight[..., i], item_inputs)
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u)
            user_hidden += tmp_user_hidden
            item_hidden += tmp_item_hidden
        
        user_outputs = self.activation(user_hidden)
        item_outputs = self.activation(item_hidden)
        
        if self.use_bias:
            user_outputs += self.bias
            item_outputs += self.bias_item
        
        return user_outputs, item_outputs


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, activation=F.relu,
                 share_weights=False):
        """非线性变换层
        
        Args:
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
            share_weights (bool, optional): 用户和商品是否共享变换权值. Defaults to False.
        
        """
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = activation
        self.share_weights = share_weights
        self.linear_user = nn.Linear(input_dim, output_dim, bias=use_bias)
        if self.share_weights:
            self.linear_item = self.linear_user
        else:
            self.linear_item = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, user_inputs, item_inputs):
        """前向传播
        
        Args:
            user_inputs (torch.Tensor): 输入的用户特征
            item_inputs (torch.Tensor): 输入的商品特征
        
        Returns:
            [torch.Tensor]: 输出的用户特征
            [torch.Tensor]: 输出的商品特征
        """
        user_outputs = self.linear_user(user_inputs)
        item_outputs = self.linear_item(item_inputs)

        if self.activation:
            user_outputs = self.activation(user_outputs)
            item_outputs = self.activation(item_outputs)

        return user_outputs, item_outputs


class Decoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        """解码器
        
        Args:
            input_dim (int): 输入的特征维度
            num_classes (int): 总共的评分级别数，eg. 5
        """
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        weights = []
        for i in range(self.num_classes):
            weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
            weights.append(weight)
        self.reset_parameters()
        
    def reset_parameters(self):
        for weight in self.weights:
            init.kaiming_uniform_(weight)
    
    def forward(self, user_inputs, item_inputs, user_indices, item_indices):
        """计算非归一化的分类输出
        
        Args:
            user_inputs (torch.Tensor): 用户的隐层特征
            item_inputs (torch.Tensor): 商品的隐层特征
            user_indices (torch.LongTensor): 
                所有交互行为中用户的id索引，与对应的item_indices构成一条边,shape=(num_edges, )
            item_indices (torch.LongTensor): 
                所有交互行为中商品的id索引，与对应的user_indices构成一条边,shape=(num_edges, )
        
        Returns:
            [torch.Tensor]: 未归一化的分类输出，shape=(num_edges, num_classes)
        """
        user_inputs = user_inputs[user_indices]
        item_inputs = item_inputs[item_indices]
        outputs = []
        for weight in self.weights:
            tmp = torch.matmul(user_inputs, weight)
            out = tmp * item_inputs
            outputs.append(out)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
