# emotion_Predict
---
这个项目将实现一个简单的基于LSTM的情感分析神经网络，并将其使用可视化显示出来

# struction
---
以下是模型数据及参数量大小：
```txt
LSTM(
    (embedding): Embedding(222170, 128)
    (lstm): LSTM(128, 24, batch_first=True)
    (output): Linear(in_features=24, out_features=2, bias=True)
)
```
以上模型主要使用LSTM进行推理计算，实现二分类

# environments
---
```txt
jieba==0.42.1
pandas==2.0.3
torch==2.1.0+cu118
tqdm==4.66.1
tqdm==4.65.0
```
请安装以上包，并配置相关虚拟环境，其他环境不保证成功运行，使用以下脚本实现安装环境：
```cmd
pip install -r requirements.txt
```

# dataset
---
本次实验总共使用了两个数据集，你可以在下面两个链接中进行下载：
```HTML
影评数据：https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
微博数据：https://github.com/logan-zou/Pytorch_practice/blob/main/dataset/weibo_senti_100k.csv
其他数据：自己找，按照微博数据格式保存为(./data/train.csv)即可
```
下载之后需要进行相关操作进行处理，并置于对应目录：(./data/train.csv)，将数据集手动切分，创建(./data/test.csv)文件，并放置一定数量数据，具体数据自己划分即可，下面我们将进行数据预处理

# Data preprocess
---
· 其中第一个数据正常直接使用

· 第二个数据需要按照阈值划分，并将其他列数据删除，只需要留下star和commit列，其中star可以按照阈值划分，阈值自定义即可，运行[main.ipynb](./main.ipynb)实现影评数据预处理

最终数据结构如下：
![data](./data.png)

# train model
---
请在目录下运行以下文件：
```cmd
python train.py
```
在运行之前请确保数据正确，如(./data/train.csv)和(./data/test.csv)数据，以及(./model)目录，用于存放训练的模型

<font color=Red>注意，默认训练轮数为50，每次训练都会保存相关模型文件，请确保内存大于6GB，如果内存不够，请修改(./train.py)脚本中以下部分：</font>

修改以下epoch减少训练轮数以减少模型生成数量
```python
# 超参数设置
embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 50 # 训练轮数
num_class = 2
```
注释此部分中模型保存部分以拒绝生成过程中的模型文件，注意循环外面的保存模型不要注释
```python
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
    torch.save(model.state_dict(), "./model/emo_predict_model_epoch{}.pkl".format(epoch)) # 保存每次的模型
    print(f"Loss:{total_loss:.2f}")
```

# test model
---
在运行train.py后，在(./model)目录应该会生成一个模型文件(./model/emo_predict_model_end12.pk1)，并且会生成字典文件(./data/data.vocab)并保存，如果都有，将测试文件放到目录(./data/test.py)，运行以下脚本文件进行测试
```cmd
python test.py
```
最终会得到相关准确率等数据，可以按照相关需求进行修改

# test demo
---
下面文件可以展示一个良好的效果，实现可视化，在模型文件等数据文件都生成了的情况下，运行以下脚本文件：
```cmd
python main.py
```
最终可以实现可视化
![demo](./demo.png)

# source
---
```txt
from{
    the part of this project comes from https://blog.csdn.net/UIBE_day_day_up/article/details/127973787
    thinks for author
}
```


