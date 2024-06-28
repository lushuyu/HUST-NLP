import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from model import BERT_BiLSTM_CRF

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
X_train = np.load('train_data.npy', allow_pickle=True)
X_val = np.load('val_data.npy', allow_pickle=True)
X_test = np.load('test_data.npy', allow_pickle=True)
y_train = np.load('train_labels.npy', allow_pickle=True)
y_val = np.load('val_labels.npy', allow_pickle=True)
y_test = np.load('test_labels.npy', allow_pickle=True)
word2idx = np.load('word2idx.npy', allow_pickle=True).item()
tag2idx = np.load('tag2idx.npy', allow_pickle=True).item()
idx2word = np.load('idx2word.npy', allow_pickle=True).item()
idx2tag = np.load('idx2tag.npy', allow_pickle=True).item()

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

# Tokenize and align labels
def preprocess_for_bert(sentences, labels):
    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs)
        for sent, labs in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
    input_ids = [ids[:128] + [0] * (128 - len(ids)) for ids in input_ids]

    label_ids = [[tag2idx.get(l) for l in lab] for lab in labels]
    label_ids = [lab[:128] + [tag2idx["PAD"]] * (128 - len(lab)) for lab in label_ids]

    return input_ids, label_ids

# 将索引转换回对应的单词和标签
def idx_to_word(sentences, idx2word):
    return [[idx2word.get(idx) for idx in sentence] for sentence in sentences]

def idx_to_tag(labels, idx2tag):
    return [[idx2tag.get(idx) for idx in label] for label in labels]

X_train = idx_to_word(X_train, idx2word)
X_val = idx_to_word(X_val, idx2word)
X_test = idx_to_word(X_test, idx2word)
y_train = idx_to_tag(y_train, idx2tag)
y_val = idx_to_tag(y_val, idx2tag)
y_test = idx_to_tag(y_test, idx2tag)

X_train, y_train = preprocess_for_bert(X_train, y_train)
X_val, y_val = preprocess_for_bert(X_val, y_val)
X_test, y_test = preprocess_for_bert(X_test, y_test)

# 转换数据为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_data, batch_size=16)

# 初始化模型
HIDDEN_DIM = 64
bert_model_name = 'bert-base-chinese'

model = BERT_BiLSTM_CRF(bert_model_name, tag2idx, HIDDEN_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 训练模型
for epoch in range(5):  # 5个epoch
    model.train()
    for sentences, tags in train_loader:
        mask = (sentences != 0).type(torch.uint8).to(device)
        model.zero_grad()
        feats = model(sentences)
        loss = model.loss(feats, tags, mask)
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sentences, tags in val_loader:
            mask = (sentences != 0).type(torch.uint8).to(device)
            feats = model(sentences)
            val_loss += model.loss(feats, tags, mask).item()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}")

# 保存模型
torch.save(model.state_dict(), 'bert_bilstm_crf_model.pth')

# 预测函数
def predict(model, sentences, device):
    model.eval()
    with torch.no_grad():
        mask = (sentences != 0).type(torch.uint8).to(device)
        feats = model(sentences)
        predictions = model.predict(feats, mask)
    return predictions

# 评估模型
predictions = predict(model, X_test_tensor, device)

# 将预测结果转换为标签
y_true = [[idx2tag[idx] for idx in sent] for sent in y_test_tensor.cpu().numpy()]
y_pred = [[idx2tag[idx] for idx in sent] for sent in predictions]

# 打印分类报告
print(classification_report(y_true, y_pred))
