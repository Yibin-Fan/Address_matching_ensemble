import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gensim.models.word2vec
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import matplotlib.pyplot as plt
from define_esim import ESIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextMatchDataset(Dataset):
    def __init__(self, text1_path, text2_path, label_path, max_len=128):
        self.max_len = max_len
        self.text1_data = self._load_text_data(text1_path)
        self.text2_data = self._load_text_data(text2_path)
        with open(label_path, 'r') as f:
            self.labels = np.array([float(line.strip()) for line in f])

    def _load_text_data(self, file_path):
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                nums = [int(x) for x in line.strip().split()]
                if len(nums) > self.max_len:
                    nums = nums[:self.max_len]
                else:
                    nums = nums + [0] * (self.max_len - len(nums))
                sequences.append(nums)
        return np.array(sequences, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'addr1': torch.LongTensor(self.text1_data[idx]),
            'addr2': torch.LongTensor(self.text2_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }


def load_combined_embeddings(w2v_path, glove_path, vocab_file):
    # 加载Word2Vec模型
    w2v_model = gensim.models.Word2Vec.load(w2v_path)

    # 加载GloVe向量
    glove_vectors = {}
    vector_size = None
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = np.array([float(x) for x in tokens[1:]], dtype=np.float32)
            if vector_size is None:
                vector_size = len(vector)
            glove_vectors[word] = vector

    # 读取交集词表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]

    # 初始化合并后的嵌入矩阵
    embedding_dim = w2v_model.vector_size + vector_size
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))  # +1 为填充符0

    # 合并词向量
    word_to_idx = {}
    for i, word in enumerate(vocab, 1):  # 从1开始，0留给填充符
        if word in w2v_model.wv and word in glove_vectors:
            w2v_vector = w2v_model.wv[word]
            glove_vector = glove_vectors[word]
            embedding_matrix[i] = np.concatenate([w2v_vector, glove_vector])
            word_to_idx[word] = i

    return embedding_matrix, word_to_idx


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            text1 = batch['addr1'].to(device)
            text2 = batch['addr2'].to(device)
            labels = batch['label'].to(device)

            outputs = model(text1, text2)
            predictions.extend((outputs > 0.5).float().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return precision, recall, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_f1 = 0
    # Initialize history dictionary to store metrics
    history = {
        'loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            text1 = batch['addr1'].to(device)
            text2 = batch['addr2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(text1, text2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        # Store metrics in history
        history['loss'].append(avg_loss)
        history['precision'].append(val_precision)
        history['recall'].append(val_recall)
        history['f1'].append(val_f1)

        # Log all metrics
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]')
        logger.info(f'Training Loss: {avg_loss:.4f}')
        logger.info(f'Validation Precision: {val_precision:.4f}')
        logger.info(f'Validation Recall: {val_recall:.4f}')
        logger.info(f'Validation F1: {val_f1:.4f}')

        # Save model if F1 score improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'precision': val_precision,
                'recall': val_recall,
            }, save_path)
            logger.info("best model found and saved")

        logger.info('-' * 50)

    # Plot training history
    plot_training_history(history, num_epochs)

    return history

def plot_training_history(history, num_epochs):
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)

    # Plot all metrics
    plt.plot(epochs, history['precision'], 'b-', label='Precision')
    plt.plot(epochs, history['recall'], 'g-', label='Recall')
    plt.plot(epochs, history['f1'], 'r-', label='F1')
    plt.plot(epochs, history['loss'], 'y-', label='Loss')

    plt.title('Training Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('result/diagram.png')
    plt.close()


# 训练主函数
def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    MAX_LEN = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Using device: {DEVICE}')

    # Load datasets
    train_dataset = TextMatchDataset(
        'data/dataset/train/addr1_tokenized.txt',
        'data/dataset/train/addr2_tokenized.txt',
        'data/dataset/train/labels.txt',
        max_len=MAX_LEN
    )

    val_dataset = TextMatchDataset(
        'data/dataset/valid/addr1_tokenized.txt',
        'data/dataset/valid/addr2_tokenized.txt',
        'data/dataset/valid/labels.txt',
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load word embeddings
    embedding_matrix, word_to_idx = load_combined_embeddings(
        'model/word2vec.model',
        'GloVe/vectors.txt',
        'data/vocab/intersection_vocab.txt'
    )
    vocab_size = len(word_to_idx) + 1  # +1 为填充符
    embedding_dim = embedding_matrix.shape[1]

    # Initialize model (only once)
    model = ESIM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        max_sequence_length=MAX_LEN
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path='result/best_esim_model.pth'
    )

    logger.info("Training completed. Training history plot saved as 'result.png'")

if __name__ == '__main__':
    main()