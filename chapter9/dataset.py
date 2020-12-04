import os
import urllib.request
from zipfile import ZipFile
from io import StringIO

import numpy as np
import pandas as pd
import scipy.sparse as sp


def globally_normalize_bipartite_adjacency(adjacencies, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    print('{} normalizing bipartite adj'.format(
        ['Asymmetrically', 'Symmetrically'][symmetric]))

    adj_tot = np.sum([adj for adj in adjacencies])
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(
            degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def get_adjacency(edge_df, num_user, num_movie, symmetric_normalization):
    user2movie_adjacencies = []
    movie2user_adjacencies = []
    train_edge_df = edge_df.loc[edge_df['usage'] == 'train']
    for i in range(5):
        edge_index = train_edge_df.loc[train_edge_df.ratings == i, [
            'user_node_id', 'movie_node_id']].to_numpy()
        support = sp.csr_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                                shape=(num_user, num_movie), dtype=np.float32)
        user2movie_adjacencies.append(support)
        movie2user_adjacencies.append(support.T)

    user2movie_adjacencies = globally_normalize_bipartite_adjacency(user2movie_adjacencies,
                                                                    symmetric=symmetric_normalization)
    movie2user_adjacencies = globally_normalize_bipartite_adjacency(movie2user_adjacencies,
                                                                    symmetric=symmetric_normalization)

    return user2movie_adjacencies, movie2user_adjacencies


def get_node_identity_feature(num_user, num_movie):
    """one-hot encoding for nodes"""
    identity_feature = np.identity(num_user + num_movie, dtype=np.float32)
    user_identity_feature, movie_indentity_feature = identity_feature[
        :num_user], identity_feature[num_user:]

    return user_identity_feature, movie_indentity_feature


def get_user_side_feature(node_user: pd.DataFrame):
    """用户节点属性特征，包括年龄，性别，职业"""
    age = node_user['age'].to_numpy().astype('float32')
    age /= age.max()
    age = age.reshape((-1, 1))
    gender_arr, gender_index = pd.factorize(node_user['gender'])
    gender_arr = np.reshape(gender_arr, (-1, 1))
    occupation_arr = pd.get_dummies(node_user['occupation']).to_numpy()

    user_feature = np.concatenate([age, gender_arr, occupation_arr], axis=1)

    return user_feature


def get_movie_side_feature(node_movie: pd.DataFrame):
    """电影节点属性特征，主要是电影类型"""
    movie_genre_cols = ['Action', 'Adventure', 'Animation',
                        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']
    movie_genre_arr = node_movie.loc[:,
                                     movie_genre_cols].to_numpy().astype('float32')
    return movie_genre_arr


def convert_to_homogeneous(user_feature: np.ndarray, movie_feature: np.ndarray):
    """通过补零将用户和电影的属性特征对齐到同一维度"""
    num_user, user_feature_dim = user_feature.shape
    num_movie, movie_feature_dim = movie_feature.shape
    user_feature = np.concatenate(
        [user_feature, np.zeros((num_user, movie_feature_dim))], axis=1)
    movie_feature = np.concatenate(
        [np.zeros((num_movie, user_feature_dim)), movie_feature], axis=1)

    return user_feature, movie_feature


def normalize_feature(feature):
    row_sum = feature.sum(1)
    row_sum[row_sum == 0] = np.inf
    normalized_feat = feature / row_sum.reshape(-1, 1)
    return normalized_feat


class MovielensDataset(object):
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.maybe_download()

    @staticmethod
    def build_graph(edge_df: pd.DataFrame, user_df: pd.DataFrame,
                    movie_df: pd.DataFrame, symmetric_normalization=False):
        node_user = edge_df[['user_node']
                            ].drop_duplicates().sort_values('user_node')
        node_movie = edge_df[['movie_node']
                             ].drop_duplicates().sort_values('movie_node')
        node_user.loc[:, 'user_node_id'] = range(len(node_user))
        node_movie.loc[:, 'movie_node_id'] = range(len(node_movie))

        edge_df = edge_df.merge(node_user, on='user_node', how='left')\
            .merge(node_movie, on='movie_node', how='left')

        node_user = node_user.merge(user_df, on='user_node', how='left')
        node_movie = node_movie.merge(movie_df, on='movie_node', how='left')
        num_user = len(node_user)
        num_movie = len(node_movie)

        # adjacency
        user2movie_adjacencies, movie2user_adjacencies = get_adjacency(edge_df, num_user, num_movie,
                                                                       symmetric_normalization)

        # node property feature
        user_side_feature = get_user_side_feature(node_user)
        movie_side_feature = get_movie_side_feature(node_movie)
        user_side_feature = normalize_feature(user_side_feature)
        movie_side_feature = normalize_feature(movie_side_feature)
        user_side_feature, movie_side_feature = convert_to_homogeneous(user_side_feature,
                                                                       movie_side_feature)

        # one-hot encoding for nodes
        user_identity_feature, movie_indentity_feature = get_node_identity_feature(
            num_user, num_movie)

        # user_indices, movie_indices, labels, train_mask
        user_indices, movie_indices, labels = edge_df[[
            'user_node_id', 'movie_node_id', 'ratings']].to_numpy().T
        train_mask = (edge_df['usage'] == 'train').to_numpy()

        return user2movie_adjacencies, movie2user_adjacencies, \
            user_side_feature, movie_side_feature, \
            user_identity_feature, movie_indentity_feature, \
            user_indices, movie_indices, labels, train_mask

    def read_data(self):
        data_dir = os.path.join(self.data_root, "ml-100k")
        # edge data
        edge_train = pd.read_csv(os.path.join(data_dir, 'u1.base'), sep='\t',
                                 header=None, names=['user_node', 'movie_node', 'ratings', 'timestamp'])
        edge_train.loc[:, 'usage'] = 'train'
        edge_test = pd.read_csv(os.path.join(data_dir, 'u1.test'), sep='\t',
                                header=None, names=['user_node', 'movie_node', 'ratings', 'timestamp'])
        edge_test.loc[:, 'usage'] = 'test'
        edge_df = pd.concat((edge_train, edge_test),
                            axis=0).drop(columns='timestamp')
        edge_df.loc[:, 'ratings'] -= 1
        # item feature
        sep = r'|'
        movie_file = os.path.join(data_dir, 'u.item')
        movie_headers = ['movie_node', 'movie_title', 'release_date', 'video_release_date',
                         'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, encoding='latin1')
        # user feature
        users_file = os.path.join(data_dir, 'u.user')
        users_headers = ['user_node', 'age',
                         'gender', 'occupation', 'zip_code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, encoding='latin1')
        return edge_df, users_df, movie_df

    def maybe_download(self):
        save_path = os.path.join(self.data_root)
        if not os.path.exists(save_path):
            self.download_data(self.url, save_path)
        if not os.path.exists(os.path.join(self.data_root, "ml-100k")):
            zipfilename = os.path.join(self.data_root, "ml-100k.zip")
            with ZipFile(zipfilename, "r") as zipobj:
                zipobj.extractall(os.path.join(self.data_root))
                print("Extracting data from {}".format(zipfilename))

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        print("Downloading data from {}".format(url))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        request = urllib.request.urlopen(url)
        filename = os.path.basename(url)
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(request.read())
        return True


if __name__ == "__main__":
    data = MovielensDataset()
    user2movie_adjacencies, movie2user_adjacencies, \
        user_side_feature, movie_side_feature, \
        user_identity_feature, movie_indentity_feature, \
        user_indices, movie_indices, labels, train_mask = data.build_graph(
            *data.read_data())
