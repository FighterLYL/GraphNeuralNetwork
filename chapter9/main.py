"""基于 MovieLens-100K 数据的GraphAutoEncoder"""
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F

from dataset import MovielensDataset
from autoencoder import StackGCNEncoder, FullyConnected, Decoder


######hyper
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
LEARNING_RATE = 0.015
EPOCHS = 1000
NODE_INPUT_DIM = 2625
SIDE_FEATURE_DIM = 41
GCN_HIDDEN_DIM = 500
SIDE_HIDDEN_DIM = 10
ENCODE_HIDDEN_DIM = 75
NUM_BASIS = 4
DROPOUT_RATIO = 0.55
WEIGHT_DACAY = 0.
######hyper


SCORES = torch.tensor([[1, 2, 3, 4, 5]]).to(DEVICE)


def to_torch_sparse_tensor(x, device):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    data = x.data

    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values,
                                                x.shape).to(device)

    return th_sparse_tensor


def tensor_from_numpy(x, device):

    return torch.from_numpy(x).to(device)


class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim,
                 gcn_hidden_dim, side_hidden_dim,
                 encode_hidden_dim,
                 num_support=5, num_classes=5, num_basis=3):
        super(GraphMatrixCompletion, self).__init__()
        self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support, DROPOUT_RATIO)
        self.dense1 = FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                     use_bias=True)
        self.dense2 = FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                     dropout=DROPOUT_RATIO, activation=lambda x: x)
        self.decoder = Decoder(encode_hidden_dim, num_basis, num_classes,
                               dropout=DROPOUT_RATIO, activation=lambda x: x)

    def forward(self, user_supports, item_supports,
                user_inputs, item_inputs,
                user_side_inputs, item_side_inputs,
                user_edge_idx, item_edge_idx):
        user_gcn, movie_gcn = self.encoder(user_supports, item_supports, user_inputs, item_inputs)
        user_side_feat, movie_side_feat = self.dense1(user_side_inputs, item_side_inputs)

        user_feat = torch.cat((user_gcn, user_side_feat), dim=1)
        movie_feat = torch.cat((movie_gcn, movie_side_feat), dim=1)

        user_embed, movie_embed = self.dense2(user_feat, movie_feat)

        edge_logits = self.decoder(user_embed, movie_embed, user_edge_idx, item_edge_idx)

        return edge_logits


data = MovielensDataset()
user2movie_adjacencies, movie2user_adjacencies, \
    user_side_feature, movie_side_feature, \
    user_identity_feature, movie_identity_feature, \
    user_indices, movie_indices, labels, train_mask = data.build_graph(
        *data.read_data())

user2movie_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in user2movie_adjacencies]
movie2user_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in movie2user_adjacencies]
user_side_feature = tensor_from_numpy(user_side_feature, DEVICE).float()
movie_side_feature = tensor_from_numpy(movie_side_feature, DEVICE).float()
user_identity_feature = tensor_from_numpy(user_identity_feature, DEVICE).float()
movie_identity_feature = tensor_from_numpy(movie_identity_feature, DEVICE).float()
user_indices = tensor_from_numpy(user_indices, DEVICE).long()
movie_indices = tensor_from_numpy(movie_indices, DEVICE).long()
labels = tensor_from_numpy(labels, DEVICE)
train_mask = tensor_from_numpy(train_mask, DEVICE)


model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                              SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, num_basis=NUM_BASIS).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)
model_inputs = (user2movie_adjacencies, movie2user_adjacencies,
                user_identity_feature, movie_identity_feature,
                user_side_feature, movie_side_feature, user_indices, movie_indices)

def train():
    test_result = []
    model.train()
    for e in range(EPOCHS):
        logits = model(*model_inputs)
        loss = criterion(logits[train_mask], labels[train_mask])
        rmse = expected_rmse(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新

        tr = test()
        test_result.append(tr)
        model.train()
        print(f"Epoch {e:04d}: TrainLoss: {loss.item():.4f}, TrainRMSE: {rmse.item():.4f}, "
              f"TestRMSE: {tr[0]:.4f}, TestLoss: {tr[1]:.4f}")

    test_result = np.asarray(test_result)
    idx = test_result[:, 0].argmin()
    print(f'test min rmse {test_result[idx]} on epoch {idx}')


@torch.no_grad()
def test():
    model.eval()
    logits = model(*model_inputs)
    test_mask = ~train_mask
    loss = criterion(logits[test_mask], labels[test_mask])
    rmse = expected_rmse(logits[test_mask], labels[test_mask])
    return rmse.item(), loss.item()


def expected_rmse(logits, label):
    true_y = label + 1  # 原来的评分为1~5，作为label时为0~4
    prob = F.softmax(logits, dim=1)
    pred_y = torch.sum(prob * SCORES, dim=1)
    
    diff = torch.pow(true_y - pred_y, 2)
    
    return torch.sqrt(diff.mean())


if __name__ == "__main__":
    train()
