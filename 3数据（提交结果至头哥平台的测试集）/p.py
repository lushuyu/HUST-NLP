import torch
import numpy as np
from model import BiLSTM_CRF
from tqdm import tqdm

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载词汇表和标签表
word2idx = np.load('word2idx.npy', allow_pickle=True).item()
tag2idx = np.load('tag2idx.npy', allow_pickle=True).item()
idx2tag = np.load('idx2tag.npy', allow_pickle=True).item()

# 加载模型
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
model = BiLSTM_CRF(len(word2idx), tag2idx, EMBEDDING_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load('bilstm_crf_model.pth'))
model.eval()

# 读取并预处理新文档
def read_and_preprocess(file_path, word2idx):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = list(line.strip())  # 逐字处理
            if words:
                X = [word2idx.get(w, word2idx["UNK"]) for w in words]
                sentences.append((words, X))
    return sentences

# 处理文档
sentences = read_and_preprocess('train_data.txt', word2idx)

# 预测函数
def predict(model, sentences, device):
    model.eval()
    results = []
    with torch.no_grad():
        for words, indices in tqdm(sentences):
            x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            mask = torch.ones_like(x, dtype=torch.uint8).to(device)
            feats = model(x)
            preds = model.predict(feats, mask)[0]
            results.append((words, preds))
    return results

# 预测
results = predict(model, sentences, device)

# 输出预测结果到文件
with open('ner_results.txt', 'w', encoding='utf-8') as output:
    for words, preds in results:
        for i in range(len(words)):
            if words[i].strip():  # 确保单词不是空白
                output.write(f"{words[i]} {idx2tag[preds[i]]}\n")

print("命名实体识别结果已保存到 ner_results.txt")
