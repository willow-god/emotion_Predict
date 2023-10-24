from dataset import build_data, MyDataset, collate_fn
from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.nn as nn
from model import LSTM
from tqdm import tqdm

# 超参数设置
embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 50
num_class = 2

# 读取数据
print("---------------开始构建数据---------------")
train_data, vocab = build_data("./data/train.csv",'./data/stopwords.txt', setting="train")
print("---------------构建数据成功---------------")
print("---------------训练数据大小---------------")
print("训练集：", len(train_data))
print("词表大小：", len(vocab))
print("---------------开始加载数据---------------");

# 加载数据
train_dataset = MyDataset(train_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
print("---------------加载数据成功---------------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(len(vocab) + 1, embedding_dim, hidden_dim, num_class)
model.to(device)
# 加载模型

nll_loss = nn.NLLLoss()
# 负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Adam优化器
print("model structure:", model)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        # print(inputs.size()) 
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    torch.save(model.state_dict(), "./model/emo_predict_model_epoch{}.pkl".format(epoch))
    print(f"Loss:{total_loss:.2f}")

# 保存模型
torch.save(model.state_dict(), "./model/emo_predict_model_end12.pkl")
print("---------------模型保存成功---------------")
