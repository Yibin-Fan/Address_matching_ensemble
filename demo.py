import threading
import json
import tkinter as tk
from tkinter import ttk
import jieba
import torch
import torch.nn.functional as F
from define_esim import ESIM


class AddressMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("地址匹配系统")

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 输入框
        self.input_label = ttk.Label(self.main_frame, text="请输入地址:")
        self.input_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.address_input = ttk.Entry(self.main_frame, width=50)
        self.address_input.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # 搜索按钮
        self.search_button = ttk.Button(self.main_frame, text="查找匹配", command=self.start_find_match)
        self.search_button.grid(row=1, column=2, padx=5, pady=5)

        # 结果显示区域
        self.result_label = ttk.Label(self.main_frame, text="匹配结果:")
        self.result_label.grid(row=2, column=0, sticky=tk.W, pady=5)

        self.result_text = tk.Text(self.main_frame, height=4, width=50)
        self.result_text.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # 加载模型和词典
        self.result_text.insert(tk.END, "正在加载模型，请稍候...\n")
        threading.Thread(target=self.load_resources).start()

    def load_resources(self):
        self.model, self.device = self.load_model()
        self.word_dict = self.load_word_dict()
        self.result_text.insert(tk.END, "模型加载完成！\n")

    def load_word_dict(self):
        return json.load(open('data/dict/word_dict.json', 'r', encoding='utf-8'))

    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embedding_matrix = torch.randn(44018, 200).numpy()
        max_sequence_length = 128

        model = ESIM(
            vocab_size=44018,
            embedding_dim=200,
            embedding_matrix=embedding_matrix,
            max_sequence_length=max_sequence_length,
            hidden_dim=128
        )

        checkpoint = torch.load('result/best_esim_model.pth', map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model, device

    def start_find_match(self):
        # 启动一个线程来执行查找匹配
        threading.Thread(target=self.find_match).start()

    def find_match(self):
        query = self.address_input.get()
        if not query.strip():
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "请输入地址")
            return

        query_indices = address_to_index(query, self.word_dict)
        best_match = None
        highest_score = -1

        with torch.no_grad():
            query_tensor = torch.LongTensor(query_indices).unsqueeze(0).to(self.device)

            with open('data/dataset/demo/unique_addresses.txt', 'r', encoding='utf-8') as f:
                for address in f:
                    address = address.strip()
                    if address:
                        addr_indices = address_to_index(address, self.word_dict)
                        addr_tensor = torch.LongTensor(addr_indices).unsqueeze(0).to(self.device)

                        output = self.model(query_tensor, addr_tensor)
                        score = output.cpu().item()

                        if score > highest_score:
                            highest_score = score
                            best_match = address

        self.result_text.delete(1.0, tk.END)
        if best_match:
            self.result_text.insert(tk.END, f"最佳匹配: {best_match}\n相似度: {highest_score:.4f}")
        else:
            self.result_text.insert(tk.END, "未找到匹配结果")


def address_to_index(address_text, word_dict):
    indices = []
    words = jieba.cut(address_text)
    for word in words:
        index = word_dict.get(word, 0)
        indices.append(index)
    max_len = 128
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices.extend([0] * (max_len - len(indices)))
    return indices


def main():
    root = tk.Tk()
    app = AddressMatcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()