# GraphNeuralNetwork
《深入浅出图神经网络：GNN原理解析》配套代码

### 环境依赖
```
python>=3.6
jupyter
scipy
numpy
matplotlib
torch>=1.2.0
```

### Getting Start

#### Chapter5：基于GCN的节点分类（使用Cora数据集）

##### 文件说明

| 文件           | 说明                                 |
| -------------- | ------------------------------------ |
| GCN_Cora.ipynb | 基于Cora数据集的节点分类notebook     |
| GCN_Cora.py    | 同GCN_Cora.ipynb，只是文件格式不一样 |

##### 运行示例

```shell
cd chapter5
# 对于文件 GCN_Cora.ipynb，启动 jupyter notebook 进行查看
# 对于文件 GCN_Cora.py
python3 GCN_Cora.py
```

#### Chapter7：GraphSage示例（使用Cora数据集）

##### 文件说明

| 文件        | 说明                          |
| :---------- | ----------------------------- |
| main.py     | 基于Cora数据集的GraphSage示例 |
| net.py      | 主要是GraphSage定义           |
| data.py     | 主要是Cora数据集准备          |
| sampling.py | 简单的采样接口                |

##### 运行示例

```shell
cd chapter7
python3 main.py
```

#### Chapter8：图分类

##### 依赖包

```shell
torch_scatter [pip install --verbose --no-cache-dir torch-scatter]
```

##### 文件说明

| 文件              | 说明                             |
| ----------------- | -------------------------------- |
| self_attn_pool.py | 自注意力机制的池化层和模型的定义 |

##### 运行示例

> TODO: 待补充示例

#### Chapter9：图自编码器

##### 文件说明

| 文件           |                 |
| -------------- | --------------- |
| autoencoder.py | GCN自编码器定义 |

##### 运行示例

> TODO: 待补充示例

