import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import io

# --- 1. Data Preprocessing (Generalized) ---

class SDTMPreprocessor:
    """
    Handles the preprocessing of SDTM data for the transformer model.
    Converts a DataFrame into tokenized sequences and builds a vocabulary.
    This class is generalized to handle various SDTM domains.
    """
    def __init__(self, subject_id_col, sequence_col, continuous_cols=None, bin_counts=None):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            "[PAD]": 0,  # Padding
            "[SOS]": 1,  # Start of Subject Sequence
            "[EOS]": 2,  # End of Subject Sequence
            "[SEP]": 3,  # Separator between values in a record
        }
        self.token_counter = len(self.special_tokens)
        
        # Domain-specific configuration
        self.subject_id_col = subject_id_col
        self.sequence_col = sequence_col
        self.continuous_cols = continuous_cols if continuous_cols else []
        self.bin_counts = bin_counts if bin_counts else {}
        self.columns = []

    def _add_to_vocab(self, item):
        """Adds a new item to the vocabulary if it's not already present."""
        if item not in self.vocab:
            self.vocab[item] = self.token_counter
            self.reverse_vocab[self.token_counter] = item
            self.token_counter += 1

    def fit(self, df):
        """Fits the preprocessor on the data to build the vocabulary."""
        self.columns = df.columns.tolist()
        df_processed = df.copy()

        # Discretize continuous columns
        for col in self.continuous_cols:
            if col in df_processed.columns:
                num_bins = self.bin_counts.get(col, 10) # Default to 10 bins
                df_processed[f'{col}_binned'] = pd.cut(df_processed[col], bins=num_bins, labels=False, include_lowest=True)

        # Build vocabulary from all columns
        for col in self.columns:
            if col in self.continuous_cols:
                binned_col_name = f"{col}_binned"
                unique_values = df_processed[binned_col_name].dropna().unique()
                for val in unique_values:
                    self._add_to_vocab(f"{col}__{int(val)}") # Ensure value is int for consistency
            else:
                unique_values = df_processed[col].astype(str).unique()
                for val in unique_values:
                    self._add_to_vocab(f"{col}__{val}")
        
        for token, index in self.special_tokens.items():
            self.vocab[token] = index
            self.reverse_vocab[index] = token

    def transform(self, df):
        """Transforms the DataFrame into a list of tokenized sequences, one per subject."""
        if not self.vocab:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        df_processed = df.copy()
        
        # Apply the same binning
        for col in self.continuous_cols:
             if col in df_processed.columns:
                num_bins = self.bin_counts.get(col, 10)
                df_processed[f'{col}_binned'] = pd.cut(df_processed[col], bins=num_bins, labels=False, include_lowest=True)

        tokenized_subjects = []
        for _, group in df_processed.groupby(self.subject_id_col):
            subject_sequence = [self.vocab['[SOS]']]
            
            group_to_process = group
            if self.sequence_col and self.sequence_col in group.columns:
                group_to_process = group.sort_values(self.sequence_col)
                
            for _, row in group_to_process.iterrows():
                for col in self.columns:
                    if col in self.continuous_cols:
                        binned_col_name = f"{col}_binned"
                        value = row[binned_col_name]
                        # Handle potential NaNs from binning
                        token_str = f"{col}__{int(value)}" if pd.notna(value) else f"{col}__nan"
                    else:
                        value = str(row[col])
                        token_str = f"{col}__{value}"
                    
                    subject_sequence.append(self.vocab.get(token_str, self.vocab['[PAD]']))
                    subject_sequence.append(self.vocab['[SEP]'])
            
            subject_sequence.pop() 
            subject_sequence.append(self.vocab['[EOS]'])
            tokenized_subjects.append(subject_sequence)
            
        return tokenized_subjects

    def get_vocab_size(self):
        return len(self.vocab)

# --- 2. PyTorch Dataset and Dataloader (Unchanged) ---

class SDTMDataset(Dataset):
    """Custom PyTorch Dataset for SDTM sequences."""
    def __init__(self, sequences, max_seq_len):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            seq = seq + [0] * (self.max_seq_len - len(seq))
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq

# --- 3. Transformer Model Architecture (Unchanged) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TabularTransformer(nn.Module):
    """A GPT-style decoder-only transformer for tabular data generation."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=src_mask, memory_mask=src_mask)
        output = self.fc_out(output)
        return output

# --- 4. Training and Synthesis Functions (Unchanged logic) ---

def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    """Function to train the model."""
    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            mask = model._generate_square_subsequent_mask(seq_len).to(device)
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs.view(-1, model.fc_out.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    print("Training finished.")

def synthesize_data(model, preprocessor, device, max_len=200, num_subjects=5):
    """Generates synthetic data using the trained model."""
    model.eval()
    synthetic_subjects_tokens = []
    sos_token = preprocessor.vocab['[SOS]']
    eos_token = preprocessor.vocab['[EOS]']
    
    print(f"\nSynthesizing {num_subjects} subjects...")
    with torch.no_grad():
        for i in range(num_subjects):
            subject_seq = [sos_token]
            for _ in range(max_len - 1):
                input_tensor = torch.tensor([subject_seq], dtype=torch.long).to(device)
                seq_len = input_tensor.size(1)
                mask = model._generate_square_subsequent_mask(seq_len).to(device)
                output = model(input_tensor, mask)
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                if next_token == eos_token:
                    subject_seq.append(next_token)
                    break
                subject_seq.append(next_token)
            if subject_seq[-1] != eos_token: subject_seq.append(eos_token)
            synthetic_subjects_tokens.append(subject_seq)

    records = []
    for i, subject_tokens in enumerate(synthetic_subjects_tokens):
        current_record = {preprocessor.subject_id_col: f"SYNTH-{i+1:03d}"}
        # Skip [SOS] and [EOS]
        for token in subject_tokens[1:-1]:
            token_str = preprocessor.reverse_vocab.get(token)
            if token_str == '[SEP]':
                if len(current_record) > 1: records.append(current_record)
                current_record = {preprocessor.subject_id_col: f"SYNTH-{i+1:03d}"}
            elif token_str:
                parts = token_str.split('__')
                if len(parts) == 2:
                    col, val = parts
                    if col != preprocessor.subject_id_col: current_record[col] = val
        if len(current_record) > 1: records.append(current_record)
    return pd.DataFrame(records)

# --- 5. Dummy Data Generation ---
def generate_dummy_data(domain='EX', num_subjects=100):
    """Generates dummy SDTM data for various domains."""
    if domain == 'EX':
        data = []
        for i in range(num_subjects):
            usubjid = f"SUBJ-{i+1:03d}"
            start_day = 1
            for seq in range(1, np.random.randint(2, 10)):
                dose = np.random.choice([25, 50, 100])
                duration = np.random.randint(7, 28)
                end_day = start_day + duration - 1
                data.append({'usubjid': usubjid, 'exseq': seq, 'exdose': dose, 'exstdy': start_day, 'exendy': end_day})
                start_day = end_day + 1
        return pd.DataFrame(data)
    
    elif domain == 'AE':
        data = []
        terms = ['HEADACHE', 'NAUSEA', 'FATIGUE', 'DIZZINESS']
        severities = ['MILD', 'MODERATE', 'SEVERE']
        for i in range(num_subjects):
            usubjid = f"SUBJ-{i+1:03d}"
            for seq in range(1, np.random.randint(1, 5)):
                start_day = np.random.randint(1, 50)
                duration = np.random.randint(1, 10)
                data.append({
                    'usubjid': usubjid, 'aeseq': seq, 'aeterm': np.random.choice(terms), 
                    'aesev': np.random.choice(severities), 'aestdy': start_day, 'aeendy': start_day + duration
                })
        return pd.DataFrame(data)

    elif domain == 'DS':
        data = []
        terms = ['COMPLETED', 'ADVERSE EVENT', 'PHYSICIAN DECISION', 'LOST TO FOLLOW-UP']
        for i in range(num_subjects):
            usubjid = f"SUBJ-{i+1:03d}"
            data.append({
                'usubjid': usubjid, 'dsseq': 1, 'dsterm': np.random.choice(terms),
                'dsstdy': np.random.randint(50, 100)
            })
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Domain '{domain}' not supported for dummy data generation.")

# --- 6. Main Execution Block ---

if __name__ == '__main__':
    # --- Configuration ---
    DOMAIN = 'AE'  # <-- CHANGE THIS TO 'EX', 'AE', 'DS', etc.
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 8
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    EPOCHS = 25
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Domain-specific settings ---
    if DOMAIN == 'EX':
        SUBJECT_ID_COL, SEQUENCE_COL = 'usubjid', 'exseq'
        CONTINUOUS_COLS = ['exdose', 'exstdy', 'exendy']
        BIN_COUNTS = {'exdose': 10, 'exstdy': 30, 'exendy': 30}
    elif DOMAIN == 'AE':
        SUBJECT_ID_COL, SEQUENCE_COL = 'usubjid', 'aeseq'
        CONTINUOUS_COLS = ['aestdy', 'aeendy']
        BIN_COUNTS = {'aestdy': 30, 'aeendy': 30}
    elif DOMAIN == 'DS':
        SUBJECT_ID_COL, SEQUENCE_COL = 'usubjid', 'dsseq'
        CONTINUOUS_COLS = ['dsstdy']
        BIN_COUNTS = {'dsstdy': 20}
    else:
        raise ValueError(f"Configuration for domain '{DOMAIN}' not found.")

    # --- Generate and Load Data ---
    print(f"\nGenerating dummy training data for {DOMAIN} domain...")
    train_df = generate_dummy_data(domain=DOMAIN, num_subjects=200)
    print("Dummy data generated. Shape:", train_df.shape)
    print("Sample data:\n", train_df.head())

    # --- Preprocess Data ---
    preprocessor = SDTMPreprocessor(
        subject_id_col=SUBJECT_ID_COL, sequence_col=SEQUENCE_COL,
        continuous_cols=CONTINUOUS_COLS, bin_counts=BIN_COUNTS
    )
    preprocessor.fit(train_df)
    tokenized_data = preprocessor.transform(train_df)
    vocab_size = preprocessor.get_vocab_size()
    print(f"\nVocabulary size: {vocab_size}")

    # --- Create Dataset and Dataloader ---
    sdtm_dataset = SDTMDataset(tokenized_data, MAX_SEQ_LEN)
    dataloader = DataLoader(sdtm_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Initialize and Train Model ---
    model = TabularTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=preprocessor.vocab['[PAD]'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    train_model(model, dataloader, criterion, optimizer, device, epochs=EPOCHS)

    # --- Synthesize and Display New Data ---
    synthetic_df = synthesize_data(model, preprocessor, device, num_subjects=3)
    
    print(f"\n--- Generated Synthetic SDTM {DOMAIN} Data ---")
    if not synthetic_df.empty:
        synthetic_df = synthetic_df.reindex(columns=preprocessor.columns, fill_value=np.nan)
        print(synthetic_df)
    else:
        print("No data was synthesized. The model may need more training or adjustments.")

