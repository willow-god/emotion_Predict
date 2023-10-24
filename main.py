# 模型效果可视化
# 作者：Liushen
# 时间：2023/10/23
from dataset import build_data, MyDataset, collate_fn, single_tokens_to_ids
from model import LSTM
import torch
from torch.utils.data import DataLoader
import tkinter as tk

def predict(input_text):
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
    state_dict = torch.load("./model/emo_predict_model.pkl")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 测试模型
    model.eval()
    print("样例测试：")
    print("输入：", input_text)
    data = single_tokens_to_ids(input_text)
    print("转换后：", data)
    inputs = torch.tensor(data).unsqueeze(0).to(device)
    lengths = torch.tensor([len(data)]).to(device)
    log_probs = model(inputs, lengths)
    _, y_pred = log_probs.max(1)
    print("预测结果：", y_pred.item())
    result = y_pred.item()
    if result == 1:
        result = "你的心情很嗨皮哦！"
    else:
        result = "你的心情有点糟糕哦！"
    return result

def show_result(input_text, output_box):
    print(input_text)
    output_box.delete(0, tk.END)
    output_text = predict(input_text)
    output_box.insert(tk.END, str(output_text))
    

def main():
    window = tk.Tk()
    window.title('情感分析')
    window.geometry('500x80')
    # 设置窗口大小不可变
    window.resizable(False, False)
    # 设置窗口永远在最前面
    window.wm_attributes('-topmost', 1)

    # 添加文本框
    input_box = tk.Entry(window, width=400)
    input_box.insert(0, '请输入要分析的文本')
    input_box.pack() # 将小部件放置到主窗口中

    # 添加事件处理函数
    def clear_input(event):
        input_box.delete(0, tk.END)

    # 绑定事件处理函数
    input_box.bind("<FocusIn>", clear_input)

    # 添加输出文本框
    output_box = tk.Entry(window, width=400)
    # 设置不可编辑，但是可以通过insert方法插入值
    # output_box['state'] = 'readonly'
    output_box.pack()

    # 添加按钮
    button = tk.Button(window, text='分析情感', width=200, height=100, command=lambda: show_result(input_box.get(), output_box))
    button.pack()

    window.mainloop()

if __name__ == "__main__":
    main()