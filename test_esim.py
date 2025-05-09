import torch
from torch.utils.data import DataLoader
import logging
from train_esim import TextMatchDataset, evaluate, load_combined_embeddings
from define_esim import ESIM


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model():
    # Configuration
    BATCH_SIZE = 32
    MAX_LEN = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'result/best_esim_model.pth'

    # Load test dataset
    test_dataset = TextMatchDataset(
        'data/dataset/test/addr1_tokenized.txt',
        'data/dataset/test/addr2_tokenized.txt',
        'data/dataset/test/labels.txt',
        max_len=MAX_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load word embeddings
    embedding_matrix, word_to_idx = load_combined_embeddings(
        'model/word2vec.model',
        'GloVe/vectors.txt',
        'data/vocab/intersection_vocab.txt'
    )
    vocab_size = len(word_to_idx) + 1  # +1 为填充符
    embedding_dim = embedding_matrix.shape[1]

    # Initialize model
    model = ESIM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        max_sequence_length=MAX_LEN
    ).to(DEVICE)

    # Load trained model weights
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from epoch {checkpoint['epoch']} with best F1: {checkpoint['best_f1']:.4f}")

    with torch.no_grad():

        # Evaluate model on test set
        test_precision, test_recall, test_f1 = evaluate(model, test_loader, DEVICE)

        # Print results
        logger.info('-' * 50)
        logger.info('Test Results:')
        logger.info(f'Precision: {test_precision:.4f}')
        logger.info(f'Recall: {test_recall:.4f}')
        logger.info(f'F1 Score: {test_f1:.4f}')
        logger.info('-' * 50)


if __name__ == '__main__':
    test_model()