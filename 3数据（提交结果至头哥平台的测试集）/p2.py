import torch
import numpy as np
from transformers import BertTokenizer
from model import BERT_BiLSTM_CRF
from tqdm import tqdm

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载标签表
tag2idx = np.load('tag2idx.npy', allow_pickle=True).item()
idx2tag = np.load('idx2tag.npy', allow_pickle=True).item()

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载模型
HIDDEN_DIM = 64
bert_model_name = 'bert-base-chinese'
model = BERT_BiLSTM_CRF(bert_model_name, tag2idx, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load('bert_bilstm_crf_model.pth'))
model.eval()

# 读取并预处理新文档
def read_and_preprocess(file_path, tokenizer):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = list(line.strip())  # 逐字处理
            if words:
                tokenized_sentence = tokenizer.tokenize(" ".join(words))
                input_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
                sentences.append((words, input_ids))
    return sentences

# 处理文档
sentences = read_and_preprocess('train_data.txt', tokenizer)

# 预测函数
def predict(model, sentences, device):
    model.eval()
    results = []
    with torch.no_grad():
        for words, indices in tqdm(sentences):
            preds = []
            for i in range(0, len(indices), 512):
                input_ids = indices[i:i+512]
                x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                mask = (x != 0).type(torch.uint8).to(device)
                feats = model(x)
                segment_preds = model.predict(feats, mask)[0]
                segment_preds = segment_preds[:len(input_ids)]  # 确保预测结果长度不超过输入长度
                preds.extend(segment_preds)
            preds = preds[:len(words)]  # 确保预测结果长度不超过原始单词长度
            results.append((words, preds))
    return results

# 预测
results = predict(model, sentences, device)

# 输出预测结果到文件
with open('ner_results.txt', 'w', encoding='utf-8') as output:
    for words, preds in results:
        if len(words) != len(preds):
            print(f"Mismatch in length: {len(words)} words vs {len(preds)} preds")
        for i in range(len(words)):
            if i < len(preds) and words[i].strip():  # 确保索引不越界且单词不是空白
                output.write(f"{words[i]} {idx2tag[preds[i]]}\n")
        output.write("\n")  # 添加换行以分隔句子

print("命名实体识别结果已保存到 ner_results.txt")
