import torch
from torch.utils.data import Dataset
import pandas as pd
import jieba
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import os

# 该类用于实现token到索引的映射
class Vocab:

    def __init__(self, stopwordspath:str):
        # 构造函数
        # tokens：全部的token列表
        print("初始化词表类。。。")
        self.idx_to_token = list()
        # 将token存成列表，索引直接查找对应的token即可
        self.token_to_idx = dict()
        # 将索引到token的映射关系存成字典，键为索引，值为对应的token
        self.unk = -1
        self.stopwords = open(stopwordspath).read().split('\n')
        print("初始化词表类成功。。。")

    def build(self, data):
        # 构建词表
        # cls：该类本身
        # data: 输入的文本数据
        # min_freq：列入token的最小频率
        # reserved_tokens：额外的标记token
        # stop_words：停用词文件路径
        print("开始创建词表。。。\n前五个停用词：")
        token_freqs = defaultdict(int) # 用于统计各个token的频率
        stopwordslist = self.stopwords
        print(stopwordslist[:5])
        token_freqs["<unk>"] = 1e10 # 未知标记的频率设置为一个很大的数
        for i in tqdm(range(data.shape[0]), desc=f"创建词表"): # tqdm用于显示进度条
            for token in jieba.lcut(data.iloc[i]["review"]): # jieba分词
                if token in stopwordslist:
                    continue
                # 如果不是停用词，且不是空字符，则将该token加入词表
                else: token_freqs[token] += 1
        # 将词频按照从大到小排序
        token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
        print("前五个词频：", token_freqs[:5])
        # 构建idx_to_token和token_to_idx
        idx_to_token = [token for token, freq in token_freqs]
        token_to_idx = {token: 1 + idx for idx, token in enumerate(idx_to_token)}
        token_to_idx["<unk>"] = 0
        # 将词表中的token存入类中
        self.idx_to_token = idx_to_token
        self.token_to_idx = token_to_idx
        self.unk = -1 # 未知标记
        return self

    def __len__(self):
        # 返回词表的大小
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入token对应的索引，不存在则返回<unk>返回的索引
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标签对应的索引值
        ids = []
        for token in tokens:
            if token in self.token_to_idx:
                ids.append(self.token_to_idx[token])
            elif token not in self.stopwords:
                # ids.append(self.unk)
                continue
            else: # 如果是停用词，则放置为0
                ids.append(0)
        return ids

    def convert_ids_to_tokens(self, ids):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in ids]


# 实现 token 映射与构建训练集、测试集
def build_data(data_path:str, stop_words:str, setting:str):
    # train_data_path：训练集路径
    # test_data_path：测试集路径
    # stop_words：停用词表路径
    data = pd.read_csv(data_path)
    # test_data = pd.read_csv(test_data_path)
    print("读取文件数据。。。")
    print("开始构建词表。。。")
    # 如果已经存在词表，可以直接加载
    if os.path.exists("./data/data.vocab"):
        print("词表正在加载。。。")
        # 加载txt词表文件data/vocab.txt
        train_vocab = torch.load("./data/data.vocab")
        print("加载词表成功。。。")
    else :
        print("词表正在创建。。。")
        train_vocab = Vocab(stop_words)
        train_vocab.build(data)
        # 保存词表为txt
        torch.save(train_vocab, "./data/data.vocab")
        print("创建词表成功。。。")
    
    print("词表大小：", len(train_vocab))
    print("前五个idx_to_token词：", end="")
    print(train_vocab.idx_to_token[:5])
    # 构建词表
    print("开始构建数据。。。")

    if setting == "train":
        if os.path.exists("./data/data.train"):
            print("训练数据正在加载。。。")
            # 加载数据
            data = torch.load("./data/data.train")
            print("训练加载数据成功。。。")
        else:
            print("训练数据正在构建。。。")
            data = [(train_vocab.convert_tokens_to_ids(jieba.lcut(sentence)), 1) for sentence in data[data["label"] == 1]["review"]]\
                +[(train_vocab.convert_tokens_to_ids(jieba.lcut(sentence)), 0) for sentence in data[data["label"] == 0]["review"]]
            print("构建训练训练成功。。。")
            # 保存训练集
            torch.save(data, "./data/data.train")
            print("保存训练数据成功。。。")
        print("-----数据大致情况-----")
        print("训练集大小：", len(data))
        print("---------------------")
    
    if setting == "test":
        if os.path.exists("./data/data.test"):
            print("测试数据正在加载。。。")
            # 加载数据
            data = torch.load("./data/data.test")
            print("加载测试数据成功。。。")
        else:
            print("测试数据正在构建。。。")
            data = [(train_vocab.convert_tokens_to_ids(jieba.lcut(sentence)), 1) for sentence in data[data["label"] == 1]["review"]]\
                +[(train_vocab.convert_tokens_to_ids(jieba.lcut(sentence)), 0) for sentence in data[data["label"] == 0]["review"]]
            print("构建测试数据成功。。。")
            # 保存测试集
            torch.save(data, "./data/data.test")
            print("保存测试数据成功。。。")
        print("-----数据大致情况-----")
        print("测试集大小：", len(data))
        print("---------------------")
    return data, train_vocab


class MyDataset(Dataset):

    def __init__(self, data) -> None:
        # data：使用词表映射之后的数据
        self.data = data

    def __len__(self):
        # 返回样例的数目
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
def collate_fn(examples):
    # 从独立样本集合中构建各批次的输入输出
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    # 获取每个序列的长度
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 目标targets为该批次所有样例输出结果构成的张量
    inputs = pad_sequence(inputs, batch_first=True)
    # 将用pad_sequence对批次类的样本进行补齐
    return inputs, lengths, targets

def single_tokens_to_ids(sentences:str):
    # 将单个句子转换为索引
    vocab = torch.load("./data/data.vocab")
    return vocab.convert_tokens_to_ids(jieba.lcut(sentences))