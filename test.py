from dataset import build_data, MyDataset, collate_fn
from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.nn as nn
from model import LSTM
from tqdm import tqdm

embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 20
num_class = 2

test_data, vocab = build_data("./data/test.csv", './data/stopwords.txt', setting="test")
print("测试集样例：", test_data[0])

test_dataset = MyDataset(test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(len(vocab) + 1, embedding_dim, hidden_dim, num_class)
state_dict = torch.load("./model/emo_predict_model_end12.pkl")
model.load_state_dict(state_dict)
model.to(device)
model.eval()
# 测试模型
model.eval()
total = 0
correct = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    log_probs = model(inputs, lengths)
    _, y_pred = log_probs.max(1)
    total += 1
    correct += (y_pred == targets).sum().item()

print(f"准确率:{correct / total:.2f}")
model.eval()
print(model)
print("---------------模型测试成功---------------")
print("样例测试：")
print("输入：", test_data[1][0])
print("输出：", test_data[1][1])
data = test_data[1]
inputs = torch.tensor(data[0]).unsqueeze(0).to(device)
lengths = torch.tensor([len(data[0])]).to(device)
log_probs = model(inputs, lengths)
_, y_pred = log_probs.max(1)
print("预测结果：", y_pred.item())
print("---------------模型测试成功---------------")