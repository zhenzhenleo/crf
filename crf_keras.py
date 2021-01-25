# -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    ref: https://spaces.ac.cn/archives/5542
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量，并使用element-wise乘积的求和sum来替代矩阵乘积。
        args: 
            inputs: tensor of shape (batch_size, output_dim+1)。CRF前面编码模型t+1时刻的输出得分分布，最后一列表示该样本是否已经截止，即t+1位置是否为padding；
            states: tensor of shape [(batch_size, output_dim)]。到前一时刻（记为t时刻）为止，各个终点路径得分。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # 到前一时刻（记为t时刻）为止，各个终点路径得分，(batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # 转移概率矩阵，(1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]  # 如果mask为1则直接拿前一时刻的得分
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        args: 
            inputs: tensor of shape (batch_size, seq_len, num_labels)，CRF前编码模型输出得分分布
            labels: tensor of shape (batch_size, seq_len, num_labels)，实际分布
        return:
            tensor of shape (batch_size, 1)
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分（标签对应的loss）
        labels1 = K.expand_dims(labels[:, :-1], 3)  # (batch_size, seq_len, num_labels, 1)
        labels2 = K.expand_dims(labels[:, 1:], 2)  # (batch_size, seq_len, 1, num_labels)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)  # 转移矩阵loss
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        """
        y_true: tensor of shape (batch_size, seq_len, num_labels)或(batch_size, seq_len, num_labels+1)，后者增加了一个padding标签表示mask
        y_pred: tensor of shape (batch_size, seq_len, num_labels)
        return:
            tensor of shape (batch_size, 1)
        """
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]
        else:
            mask = K.ones_like(y_pred[:, :, :1])
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态，[(batch_size, num_labels)]
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)
