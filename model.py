import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F

class LSTM(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim：隐藏层维度
        # num_class:多分类个数
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        # lstm层
        self.output = nn.Linear(hidden_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, lengths):
        # 前向计算函数
        # inputs:输入
        # lengths:打包的序列长度
        # print(f"输入为：{inputs.size()}")
        embeds = self.embedding(inputs)
        # 注意这儿是词向量层，不是词袋词向量层
        # print(f"词向量层输出为：{embeds.size()}")
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        # LSTM需要定长序列，使用该函数将变长序列打包
        # print(f"经过打包为：{x_pack.size()}")
        hidden, (hn, cn) = self.lstm(x_pack)
        # print(f"经过lstm计算后为：{hn.size()}")
        outputs = self.output(hn[-1])
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = -1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return log_probs