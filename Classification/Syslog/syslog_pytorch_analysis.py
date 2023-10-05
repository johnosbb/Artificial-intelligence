import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import Vectors

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define fields for text and labels
TEXT = Field(tokenize='spacy', lower=True)
LABEL = Field(sequential=False, dtype=torch.float)

# Define fields mapping to your CSV columns (adjust these according to your dataset)
fields = [("text", TEXT), ("label", LABEL)]

# Load the IMDb dataset
train_data, test_data = TabularDataset.splits(
    path='./data/Syslog', train='train.csv', test='test.csv', format='csv', fields=fields)

# Build the vocabulary based on the train dataset
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Create iterators for the train, validation, and test datasets
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort=False)

# Define a simple text classification model


class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = torch.mean(embedded, dim=0)
        hidden = self.fc(pooled)
        hidden = self.dropout(hidden)
        output = self.out(hidden)
        return output


# Hyperparameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
DROPOUT = 0.5

# Initialize the model and optimizer
model = SimpleTextClassifier(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Move the model to the GPU if available
model = model.to(device)
criterion = criterion.to(device)

# Training loop
# ... (rest of the training and evaluation code remains the same)
