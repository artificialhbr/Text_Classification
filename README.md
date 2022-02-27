# Text Classification with CNN

使用卷积神经网络进行中文文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

## 环境

- Python 3 
- torch 1.8.0
- numpy
- scikit-learn
- scipy

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```


数据集划分如下：
- 训练集: 5000*10
- 验证集: 500*10
- 测试集: 1000*10

从原数据集生成子集的过程请参看`helper`下的两个脚本。其中，`copy_data.sh`用于从每个分类拷贝6500个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行该文件后，得到三个数据文件：

- data.train.txt: 训练集(50000条)
- data.dev.txt: 验证集(10000条)
- data.test.txt: 测试集(10000条)


## CNN卷积神经网络

### 配置项

CNN可配置的参数如下所示，在`TextCNN.py`中。

```python
class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
```

### CNN模型

具体参看`TextCNN.py`的实现。



### 训练与验证

运行 `python run_cnn.py train`，可以开始训练。

> 若之前进行过训练，请把tensorboard/textcnn删除，避免TensorBoard多次训练结果重叠。

```
Configuring CNN model...
Configuring TensorBoard and Saver...
Loading training and validation data...
Time usage: 0:00:14
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  10.94%, Val Loss:    2.3, Val Acc:   8.92%, Time: 0:00:01 *
Iter:    100, Train Loss:   0.88, Train Acc:  73.44%, Val Loss:    1.2, Val Acc:  68.46%, Time: 0:00:04 *
Iter:    200, Train Loss:   0.38, Train Acc:  92.19%, Val Loss:   0.75, Val Acc:  77.32%, Time: 0:00:07 *
Iter:    300, Train Loss:   0.22, Train Acc:  92.19%, Val Loss:   0.46, Val Acc:  87.08%, Time: 0:00:09 *
Iter:    400, Train Loss:   0.24, Train Acc:  90.62%, Val Loss:    0.4, Val Acc:  88.62%, Time: 0:00:12 *
Iter:    500, Train Loss:   0.16, Train Acc:  96.88%, Val Loss:   0.36, Val Acc:  90.38%, Time: 0:00:15 *
Iter:    600, Train Loss:  0.084, Train Acc:  96.88%, Val Loss:   0.35, Val Acc:  91.36%, Time: 0:00:17 *
Iter:    700, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.26, Val Acc:  92.58%, Time: 0:00:20 *
Epoch: 2
Iter:    800, Train Loss:   0.07, Train Acc:  98.44%, Val Loss:   0.24, Val Acc:  94.12%, Time: 0:00:23 *
Iter:    900, Train Loss:  0.092, Train Acc:  96.88%, Val Loss:   0.27, Val Acc:  92.86%, Time: 0:00:25
Iter:   1000, Train Loss:   0.17, Train Acc:  95.31%, Val Loss:   0.28, Val Acc:  92.82%, Time: 0:00:28
Iter:   1100, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.23, Val Acc:  93.26%, Time: 0:00:31
Iter:   1200, Train Loss:  0.081, Train Acc:  98.44%, Val Loss:   0.25, Val Acc:  92.96%, Time: 0:00:33
Iter:   1300, Train Loss:  0.052, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  93.58%, Time: 0:00:36
Iter:   1400, Train Loss:    0.1, Train Acc:  95.31%, Val Loss:   0.22, Val Acc:  94.12%, Time: 0:00:39
Iter:   1500, Train Loss:   0.12, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  93.58%, Time: 0:00:41
Epoch: 3
Iter:   1600, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:   0.26, Val Acc:  92.34%, Time: 0:00:44
Iter:   1700, Train Loss:  0.018, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  93.46%, Time: 0:00:47
Iter:   1800, Train Loss:  0.036, Train Acc: 100.00%, Val Loss:   0.28, Val Acc:  92.72%, Time: 0:00:50
No optimization for a long time, auto-stopping...
```

在验证集上的最佳效果为94.12%，且只经过了3轮迭代就已经停止。



### 测试

运行 `python run_cnn.py test` 在测试集上进行测试。

```
Configuring CNN model...
Loading test data...
Testing...
Test Loss:   0.14, Test Acc:  96.04%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.96      0.99      0.97      1000
         房产       1.00      1.00      1.00      1000
         家居       0.95      0.91      0.93      1000
         教育       0.95      0.89      0.92      1000
         科技       0.94      0.97      0.95      1000
         时尚       0.95      0.97      0.96      1000
         时政       0.94      0.94      0.94      1000
         游戏       0.97      0.96      0.97      1000
         娱乐       0.95      0.98      0.97      1000

avg / total       0.96      0.96      0.96     10000

Confusion Matrix...
[[991   0   0   0   2   1   0   4   1   1]
 [  0 992   0   0   2   1   0   5   0   0]
 [  0   1 996   0   1   1   0   0   0   1]
 [  0  14   0 912   7  15   9  29   3  11]
 [  2   9   0  12 892  22  18  21  10  14]
 [  0   0   0  10   1 968   4   3  12   2]
 [  1   0   0   9   4   4 971   0   2   9]
 [  1  16   0   4  18  12   1 941   1   6]
 [  2   4   1   5   4   5  10   1 962   6]
 [  1   0   1   6   4   3   5   0   1 979]]
Time usage: 0:00:05
```

在测试集上的准确率达到了96.04%，且各类的precision, recall和f1-score都超过了0.9。

从混淆矩阵也可以看出分类效果十分优秀。

