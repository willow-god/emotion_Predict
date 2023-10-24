# emotion_Predict
这个项目将实现一个简单的基于LSTM的情感分析神经网络，并将其使用可视化显示出来

# environments
---
```txt
jieba==0.42.1
pandas==2.0.3
torch==2.1.0+cu118
tqdm==4.66.1
tqdm==4.65.0
```
请安装以上包，并配置相关虚拟环境，其他环境不保证成功运行

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
![data](./img/data.jpg)

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


