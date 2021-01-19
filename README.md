# GraphNeuralNetwork
《深入浅出图神经网络：GNN原理解析》配套代码

### 关于勘误

>由于作者水平有限，时间仓促，书中难免会有一些错误或不准确的地方，给读者朋友造成了困扰，表示抱歉。
仓库中提供了目前已经发现的一些问题的[勘误](./勘误.pdf),在此向指正这些错误的读者朋友表示感谢。

* 在5.4节图滤波器的介绍中，存在一些描述错误和概念模糊的问题，可能给读者理解造成偏差，勘误中对相关问题进行了更正

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

* [x] [Chapter5: 基于GCN的节点分类](./chapter5)
* [x] [Chapter7: GraphSage示例](./chapter7)
* [x] [Chapter8: 图分类示例](./chapter8)
* [x] [Chapter9: 图自编码器](./chapter9)

### FAQ

1. Cora数据集无法下载

Cora数据集地址是：[kimiyoung/planetoid](https://github.com/kimiyoung/planetoid/tree/master/data)。
~~仓库中提供了一份使用到的cora数据，可以分别将它放在 `chapter5/cora/raw` 或者 `chapter7/cora/raw` 目录下。~~
新代码直接使用本地数据.
