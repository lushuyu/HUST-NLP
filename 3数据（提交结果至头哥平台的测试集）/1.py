import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    sentences = []
    tags = []
    sentence = []
    tag = []

    for line in data:
        if line.strip():
            word, ner_tag = line.strip().split()
            sentence.append(word)
            tag.append(ner_tag)
        else:
            sentences.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []

    return sentences, tags

def preprocess_data(sentences, tags, max_len=100):
    words = list(set(word for sentence in sentences for word in sentence))
    tags_list = list(set(tag for tag_seq in tags for tag in tag_seq))

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    tag2idx = {t: i + 1 for i, t in enumerate(tags_list)}
    tag2idx["PAD"] = 0

    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: t for t, i in tag2idx.items()}

    X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in sentences]
    X = [s[:max_len] + [word2idx["PAD"]] * (max_len - len(s)) for s in X]

    y = [[tag2idx.get(t) for t in ts] for ts in tags]
    y = [ts[:max_len] + [tag2idx["PAD"]] * (max_len - len(ts)) for ts in y]

    return X, y, word2idx, tag2idx, idx2word, idx2tag

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    sentences, tags = load_data('RMRB_NER_CORPUS.txt')
    X, y, word2idx, tag2idx, idx2word, idx2tag = preprocess_data(sentences, tags)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    np.save('train_data.npy', X_train)
    np.save('val_data.npy', X_val)
    np.save('test_data.npy', X_test)
    np.save('train_labels.npy', y_train)
    np.save('val_labels.npy', y_val)
    np.save('test_labels.npy', y_test)
    np.save('word2idx.npy', word2idx)
    np.save('tag2idx.npy', tag2idx)
    np.save('idx2word.npy', idx2word)
    np.save('idx2tag.npy', idx2tag)
