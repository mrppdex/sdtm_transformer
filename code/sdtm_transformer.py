
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.api.types import is_numeric_dtype
from torch.utils.data import DataLoader, Dataset


# --- 1. Data Preprocessing ---

@dataclass
class SequenceExample:
    """Container that stores the encoded sequence for one subject."""

    categorical: np.ndarray  # Shape: (seq_len, num_categorical)
    continuous: np.ndarray  # Shape: (seq_len, num_continuous)


class SDTMPreprocessor:
    """Prepares SDTM domain data for the seq2seq transformer."""

    def __init__(
        self,
        subject_id_col: str,
        sequence_col: Optional[str],
        continuous_cols: Optional[List[str]] = None,
        auto_detect_continuous: bool = True,
        min_bins: int = 5,
        max_bins: int = 50,
    ) -> None:
        self.subject_id_col = subject_id_col
        self.sequence_col = sequence_col
        self.auto_detect_continuous = auto_detect_continuous
        self.user_continuous_cols = set(continuous_cols or [])
        self.min_bins = min_bins
        self.max_bins = max_bins

        self.columns: List[str] = []
        self.column_types: Dict[str, str] = {}
        self.categorical_cols: List[str] = []
        self.continuous_cols: List[str] = []

        self.categorical_vocabs: Dict[str, Dict[str, int]] = {}
        self.categorical_reverse_vocabs: Dict[str, Dict[int, str]] = {}
        self.categorical_pad_idx: int = 0

        self.continuous_stats: Dict[str, Tuple[float, float]] = {}

        self.first_row_categorical_dist: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.first_row_continuous_stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        if self.subject_id_col not in df.columns:
            raise ValueError("subject_id_col not present in dataframe")

        df_processed = df.copy()
        self.columns = [col for col in df_processed.columns if col != self.subject_id_col]

        detected_continuous = set()
        if self.auto_detect_continuous:
            for col in self.columns:
                if is_numeric_dtype(df_processed[col]) and df_processed[col].nunique(dropna=True) > 1:
                    detected_continuous.add(col)

        self.continuous_cols = sorted(
            (self.user_continuous_cols | detected_continuous) - {self.subject_id_col}
        )
        self.categorical_cols = [
            col for col in self.columns if col not in self.continuous_cols
        ]

        self.column_types = {
            col: ("continuous" if col in self.continuous_cols else "categorical")
            for col in self.columns
        }

        for col in self.categorical_cols:
            vocab = {"[PAD]": self.categorical_pad_idx}
            reverse_vocab = {self.categorical_pad_idx: "[PAD]"}
            unique_values = (
                df_processed[col].astype(str).fillna("<NA>").replace("nan", "<NA>")
            ).unique()
            next_index = 1
            for val in unique_values:
                if val not in vocab:
                    vocab[val] = next_index
                    reverse_vocab[next_index] = val
                    next_index += 1
            self.categorical_vocabs[col] = vocab
            self.categorical_reverse_vocabs[col] = reverse_vocab

        for col in self.continuous_cols:
            series = df_processed[col].astype(float)
            mean = float(series.mean()) if not series.dropna().empty else 0.0
            std = float(series.std()) if series.std() not in (0, np.nan) else 1.0
            if not np.isfinite(std) or std == 0:
                std = 1.0
            self.continuous_stats[col] = (mean, std)

        first_rows = df_processed
        if self.sequence_col and self.sequence_col in df_processed.columns:
            first_rows = df_processed.sort_values(self.sequence_col)
        first_rows = first_rows.groupby(self.subject_id_col).head(1)

        for col in self.categorical_cols:
            counts = (
                first_rows[col]
                .astype(str)
                .fillna("<NA>")
                .replace("nan", "<NA>")
                .value_counts()
            )
            if counts.empty:
                values = np.array(["<NA>"])
                probs = np.array([1.0])
            else:
                values = counts.index.to_numpy()
                probs = (counts / counts.sum()).to_numpy()
            self.first_row_categorical_dist[col] = (values, probs)

        for col in self.continuous_cols:
            series = first_rows[col].astype(float)
            mean = float(series.mean()) if not series.dropna().empty else 0.0
            std = float(series.std()) if series.std() not in (0, np.nan) else 1.0
            if not np.isfinite(std) or std == 0:
                std = 1.0
            self.first_row_continuous_stats[col] = (mean, std)

    def _encode_categorical_value(self, col: str, value: object) -> int:
        vocab = self.categorical_vocabs[col]
        key = str(value) if pd.notna(value) else "<NA>"
        if key == "nan":
            key = "<NA>"
        return vocab.get(key, self.categorical_pad_idx)

    def _encode_continuous_value(self, col: str, value: object) -> float:
        mean, std = self.continuous_stats[col]
        if pd.isna(value):
            return 0.0
        return float((float(value) - mean) / std)

    def encode_row(self, row: pd.Series) -> Tuple[List[int], List[float]]:
        cat_values = [self._encode_categorical_value(col, row.get(col)) for col in self.categorical_cols]
        cont_values = [self._encode_continuous_value(col, row.get(col)) for col in self.continuous_cols]
        return cat_values, cont_values

    def transform(self, df: pd.DataFrame) -> List[SequenceExample]:
        if not self.columns:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        df_processed = df.copy()
        if self.sequence_col and self.sequence_col in df_processed.columns:
            df_processed = df_processed.sort_values([self.subject_id_col, self.sequence_col])

        sequences: List[SequenceExample] = []
        for _, group in df_processed.groupby(self.subject_id_col):
            cat_rows: List[List[int]] = []
            cont_rows: List[List[float]] = []
            for _, row in group.iterrows():
                cat_values, cont_values = self.encode_row(row)
                cat_rows.append(cat_values)
                cont_rows.append(cont_values)

            cat_array = (
                np.array(cat_rows, dtype=np.int64)
                if self.categorical_cols
                else np.zeros((len(cont_rows), 0), dtype=np.int64)
            )
            cont_array = (
                np.array(cont_rows, dtype=np.float32)
                if self.continuous_cols
                else np.zeros((len(cat_rows), 0), dtype=np.float32)
            )
            sequences.append(SequenceExample(categorical=cat_array, continuous=cont_array))

        return sequences

    def denormalize_continuous(self, col: str, value: float) -> float:
        mean, std = self.continuous_stats[col]
        return value * std + mean

    def decode_categorical(self, col: str, index: int) -> str:
        return self.categorical_reverse_vocabs[col].get(index, "<UNK>")

    def sample_first_row(self) -> Tuple[List[int], List[float], Dict[str, object]]:
        sampled_row: Dict[str, object] = {}
        categorical_indices: List[int] = []
        continuous_values: List[float] = []

        for col in self.categorical_cols:
            values, probs = self.first_row_categorical_dist[col]
            sampled_value = np.random.choice(values, p=probs)
            sampled_row[col] = sampled_value if sampled_value != "<NA>" else np.nan
            categorical_indices.append(self._encode_categorical_value(col, sampled_value))

        for col in self.continuous_cols:
            mean, std = self.first_row_continuous_stats[col]
            sampled_value = np.random.normal(mean, std)
            sampled_row[col] = sampled_value
            continuous_values.append(self._encode_continuous_value(col, sampled_value))

        return categorical_indices, continuous_values, sampled_row

    def build_dataframe_from_sequences(
        self, sequences: List[List[Dict[str, object]]]
    ) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        for subject_idx, subject_rows in enumerate(sequences, start=1):
            usubjid = f"SYNTH-{subject_idx:03d}"
            for seq_idx, row in enumerate(subject_rows, start=1):
                record = {self.subject_id_col: usubjid}
                if self.sequence_col:
                    record[self.sequence_col] = seq_idx
                for col, value in row.items():
                    record[col] = value
                records.append(record)
        return pd.DataFrame(records)


class SDTMDataset(Dataset):
    """PyTorch dataset that serves encoder/decoder examples for each subject."""

    def __init__(self, sequences: List[SequenceExample]):
        self.sequences = sequences
        if sequences:
            self.num_categorical = sequences[0].categorical.shape[1]
            self.num_continuous = sequences[0].continuous.shape[1]
        else:
            self.num_categorical = 0
            self.num_continuous = 0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        cat = torch.from_numpy(seq.categorical) if self.num_categorical > 0 else torch.empty((seq.continuous.shape[0], 0), dtype=torch.long)
        cont = torch.from_numpy(seq.continuous) if self.num_continuous > 0 else torch.empty((seq.categorical.shape[0], 0), dtype=torch.float32)

        seq_len = cat.shape[0] if self.num_categorical > 0 else cont.shape[0]
        if seq_len == 0:
            raise ValueError("Encountered an empty sequence during dataset construction.")

        encoder_cat = cat[:1].clone()
        encoder_cont = cont[:1].clone()
        decoder_cat = cat.clone()
        decoder_cont = cont.clone()

        target_cat = torch.zeros_like(decoder_cat)
        target_cont = torch.zeros_like(decoder_cont)
        target_cat_mask = torch.zeros_like(decoder_cat, dtype=torch.bool)
        target_cont_mask = torch.zeros_like(decoder_cont, dtype=torch.bool)
        stop_targets = torch.zeros(decoder_cat.shape[0], dtype=torch.float32)

        for t in range(seq_len):
            if t + 1 < seq_len:
                if self.num_categorical > 0:
                    target_cat[t] = decoder_cat[t + 1]
                    target_cat_mask[t] = True
                if self.num_continuous > 0:
                    target_cont[t] = decoder_cont[t + 1]
                    target_cont_mask[t] = True
            else:
                stop_targets[t] = 1.0

        return {
            "encoder_categorical": encoder_cat,
            "encoder_continuous": encoder_cont,
            "decoder_categorical": decoder_cat,
            "decoder_continuous": decoder_cont,
            "target_categorical": target_cat,
            "target_continuous": target_cont,
            "target_categorical_mask": target_cat_mask,
            "target_continuous_mask": target_cont_mask,
            "stop_targets": stop_targets,
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]):
        batch_size = len(batch)
        num_categorical = self.num_categorical
        num_continuous = self.num_continuous

        max_enc_len = max(
            (item["encoder_categorical"].shape[0] if num_categorical > 0 else item["encoder_continuous"].shape[0])
            for item in batch
        )
        max_dec_len = max(
            (item["decoder_categorical"].shape[0] if num_categorical > 0 else item["decoder_continuous"].shape[0])
            for item in batch
        )

        if num_categorical > 0:
            encoder_cat = torch.zeros((batch_size, max_enc_len, num_categorical), dtype=torch.long)
            decoder_cat = torch.zeros((batch_size, max_dec_len, num_categorical), dtype=torch.long)
            target_cat = torch.zeros((batch_size, max_dec_len, num_categorical), dtype=torch.long)
            target_cat_mask = torch.zeros((batch_size, max_dec_len, num_categorical), dtype=torch.bool)
        else:
            encoder_cat = torch.empty((batch_size, max_enc_len, 0), dtype=torch.long)
            decoder_cat = torch.empty((batch_size, max_dec_len, 0), dtype=torch.long)
            target_cat = torch.empty((batch_size, max_dec_len, 0), dtype=torch.long)
            target_cat_mask = torch.empty((batch_size, max_dec_len, 0), dtype=torch.bool)

        if num_continuous > 0:
            encoder_cont = torch.zeros((batch_size, max_enc_len, num_continuous), dtype=torch.float32)
            decoder_cont = torch.zeros((batch_size, max_dec_len, num_continuous), dtype=torch.float32)
            target_cont = torch.zeros((batch_size, max_dec_len, num_continuous), dtype=torch.float32)
            target_cont_mask = torch.zeros((batch_size, max_dec_len, num_continuous), dtype=torch.bool)
        else:
            encoder_cont = torch.empty((batch_size, max_enc_len, 0), dtype=torch.float32)
            decoder_cont = torch.empty((batch_size, max_dec_len, 0), dtype=torch.float32)
            target_cont = torch.empty((batch_size, max_dec_len, 0), dtype=torch.float32)
            target_cont_mask = torch.empty((batch_size, max_dec_len, 0), dtype=torch.bool)

        stop_targets = torch.zeros((batch_size, max_dec_len), dtype=torch.float32)
        stop_mask = torch.zeros((batch_size, max_dec_len), dtype=torch.bool)

        for i, item in enumerate(batch):
            enc_len = item["encoder_categorical"].shape[0] if num_categorical > 0 else item["encoder_continuous"].shape[0]
            dec_len = item["decoder_categorical"].shape[0] if num_categorical > 0 else item["decoder_continuous"].shape[0]

            if num_categorical > 0:
                encoder_cat[i, :enc_len] = item["encoder_categorical"]
                decoder_cat[i, :dec_len] = item["decoder_categorical"]
                target_cat[i, :dec_len] = item["target_categorical"]
                target_cat_mask[i, :dec_len] = item["target_categorical_mask"]
            if num_continuous > 0:
                encoder_cont[i, :enc_len] = item["encoder_continuous"]
                decoder_cont[i, :dec_len] = item["decoder_continuous"]
                target_cont[i, :dec_len] = item["target_continuous"]
                target_cont_mask[i, :dec_len] = item["target_continuous_mask"]

            stop_targets[i, :dec_len] = item["stop_targets"]
            stop_mask[i, :dec_len] = True

        encoder_padding_mask = torch.zeros((batch_size, max_enc_len), dtype=torch.bool)
        decoder_padding_mask = ~stop_mask

        return {
            "encoder_categorical": encoder_cat,
            "encoder_continuous": encoder_cont,
            "decoder_categorical": decoder_cat,
            "decoder_continuous": decoder_cont,
            "target_categorical": target_cat,
            "target_continuous": target_cont,
            "target_categorical_mask": target_cat_mask,
            "target_continuous_mask": target_cont_mask,
            "stop_targets": stop_targets,
            "stop_mask": stop_mask,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_padding_mask": decoder_padding_mask,
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TabularSeq2SeqTransformer(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: List[int],
        num_continuous: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_categorical = len(categorical_cardinalities)
        self.num_continuous = num_continuous

        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model, padding_idx=0) for cardinality in categorical_cardinalities]
        )
        self.continuous_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_continuous)]
        )

        self.encoder_positional = PositionalEncoding(d_model, dropout)
        self.decoder_positional = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.categorical_heads = nn.ModuleList(
            [nn.Linear(d_model, cardinality) for cardinality in categorical_cardinalities]
        )
        self.continuous_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_continuous)])
        self.stop_head = nn.Linear(d_model, 1)

    def _embed_inputs(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = categorical.shape[0], categorical.shape[1]
        device = categorical.device if categorical.numel() > 0 else continuous.device
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        for idx, embed in enumerate(self.categorical_embeddings):
            col_values = categorical[:, :, idx]
            embeddings = embeddings + embed(col_values)

        for idx, projection in enumerate(self.continuous_embeddings):
            col_values = continuous[:, :, idx].unsqueeze(-1)
            embeddings = embeddings + projection(col_values)

        return embeddings

    def forward(
        self,
        encoder_categorical: torch.Tensor,
        encoder_continuous: torch.Tensor,
        decoder_categorical: torch.Tensor,
        decoder_continuous: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_inputs = self._embed_inputs(encoder_categorical, encoder_continuous)
        decoder_inputs = self._embed_inputs(decoder_categorical, decoder_continuous)

        encoder_inputs = self.encoder_positional(encoder_inputs)
        decoder_inputs = self.decoder_positional(decoder_inputs)

        memory = self.transformer_encoder(
            encoder_inputs,
            src_key_padding_mask=src_key_padding_mask,
        )

        tgt_mask = self.generate_square_subsequent_mask(decoder_inputs.size(1)).to(decoder_inputs.device)

        decoded = self.transformer_decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        categorical_logits = [head(decoded) for head in self.categorical_heads]
        continuous_preds = [head(decoded).squeeze(-1) for head in self.continuous_heads]
        stop_logits = self.stop_head(decoded).squeeze(-1)

        return {
            "categorical_logits": categorical_logits,
            "continuous_preds": continuous_preds,
            "stop_logits": stop_logits,
        }

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = torch.tensor(0.0, device=outputs["stop_logits"].device)
    breakdown: Dict[str, float] = {}

    categorical_losses = []
    for idx, logits in enumerate(outputs["categorical_logits"]):
        targets = batch["target_categorical"][:, :, idx]
        mask = batch["target_categorical_mask"][:, :, idx]
        if mask.any():
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask_flat = mask.reshape(-1)
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
            categorical_losses.append(loss)
    if categorical_losses:
        cat_loss = torch.stack(categorical_losses).mean()
        total_loss = total_loss + cat_loss
        breakdown["categorical"] = float(cat_loss.detach().cpu())

    continuous_losses = []
    for idx, preds in enumerate(outputs["continuous_preds"]):
        targets = batch["target_continuous"][:, :, idx]
        mask = batch["target_continuous_mask"][:, :, idx]
        if mask.any():
            preds_flat = preds.reshape(-1)
            targets_flat = targets.reshape(-1)
            mask_flat = mask.reshape(-1)
            loss = F.mse_loss(preds_flat[mask_flat], targets_flat[mask_flat])
            continuous_losses.append(loss)
    if continuous_losses:
        cont_loss = torch.stack(continuous_losses).mean()
        total_loss = total_loss + cont_loss
        breakdown["continuous"] = float(cont_loss.detach().cpu())

    stop_logits = outputs["stop_logits"]
    stop_targets = batch["stop_targets"]
    stop_mask = batch["stop_mask"]
    if stop_mask.any():
        logits_flat = stop_logits.reshape(-1)
        targets_flat = stop_targets.reshape(-1)
        mask_flat = stop_mask.reshape(-1)
        stop_loss = F.binary_cross_entropy_with_logits(logits_flat[mask_flat], targets_flat[mask_flat])
        total_loss = total_loss + stop_loss
        breakdown["stop"] = float(stop_loss.detach().cpu())

    return total_loss, breakdown


def train_model(
    model: TabularSeq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
) -> None:
    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                encoder_categorical=batch["encoder_categorical"],
                encoder_continuous=batch["encoder_continuous"],
                decoder_categorical=batch["decoder_categorical"],
                decoder_continuous=batch["decoder_continuous"],
                src_key_padding_mask=batch["encoder_padding_mask"],
                tgt_key_padding_mask=batch["decoder_padding_mask"],
                memory_key_padding_mask=batch["encoder_padding_mask"],
            )
            loss, _ = compute_losses(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        avg_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    print("Training finished.")


def synthesize_data(
    model: TabularSeq2SeqTransformer,
    preprocessor: SDTMPreprocessor,
    device: torch.device,
    num_subjects: int = 5,
    max_steps: int = 20,
    stop_threshold: float = 0.5,
) -> pd.DataFrame:
    model.eval()
    generated_subjects: List[List[Dict[str, object]]] = []

    with torch.no_grad():
        for _ in range(num_subjects):
            cat_indices, cont_values, sampled_row = preprocessor.sample_first_row()
            subject_rows = [sampled_row.copy()]

            decoder_cat = torch.tensor([cat_indices], dtype=torch.long, device=device).unsqueeze(0)
            decoder_cont = torch.tensor([cont_values], dtype=torch.float32, device=device).unsqueeze(0)

            for _ in range(max_steps):
                encoder_cat = decoder_cat[:, :1, :]
                encoder_cont = decoder_cont[:, :1, :]

                outputs = model(
                    encoder_categorical=encoder_cat,
                    encoder_continuous=encoder_cont,
                    decoder_categorical=decoder_cat,
                    decoder_continuous=decoder_cont,
                    src_key_padding_mask=torch.zeros((1, encoder_cat.size(1)), dtype=torch.bool, device=device),
                    tgt_key_padding_mask=torch.zeros((1, decoder_cat.size(1)), dtype=torch.bool, device=device),
                    memory_key_padding_mask=torch.zeros((1, encoder_cat.size(1)), dtype=torch.bool, device=device),
                )

                stop_logit = outputs["stop_logits"][0, -1]
                stop_prob = torch.sigmoid(stop_logit).item()
                if stop_prob > stop_threshold:
                    break

                next_cat_indices: List[int] = []
                next_cont_values: List[float] = []
                next_row: Dict[str, object] = {}

                for idx, logits in enumerate(outputs["categorical_logits"]):
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    predicted_index = torch.argmax(probs).item()
                    next_cat_indices.append(predicted_index)
                    value = preprocessor.decode_categorical(preprocessor.categorical_cols[idx], predicted_index)
                    next_row[preprocessor.categorical_cols[idx]] = np.nan if value == "<NA>" else value

                for idx, preds in enumerate(outputs["continuous_preds"]):
                    predicted_value = preds[0, -1].item()
                    denorm = preprocessor.denormalize_continuous(preprocessor.continuous_cols[idx], predicted_value)
                    next_cont_values.append(predicted_value)
                    next_row[preprocessor.continuous_cols[idx]] = denorm

                subject_rows.append(next_row)

                next_cat_tensor = torch.tensor([next_cat_indices], dtype=torch.long, device=device).unsqueeze(0)
                next_cont_tensor = torch.tensor([next_cont_values], dtype=torch.float32, device=device).unsqueeze(0)
                decoder_cat = torch.cat([decoder_cat, next_cat_tensor], dim=1)
                decoder_cont = torch.cat([decoder_cont, next_cont_tensor], dim=1)

            generated_subjects.append(subject_rows)

    return preprocessor.build_dataframe_from_sequences(generated_subjects)


def generate_dummy_data(domain: str = "EX", num_subjects: int = 100) -> pd.DataFrame:
    if domain == "EX":
        data = []
        for i in range(num_subjects):
            usubjid = f"SUBJ-{i+1:03d}"
            start_day = 1
            for seq in range(1, np.random.randint(2, 10)):
                dose = np.random.choice([25, 50, 100])
                duration = np.random.randint(7, 28)
                end_day = start_day + duration - 1
                data.append(
                    {
                        "usubjid": usubjid,
                        "exseq": seq,
                        "exdose": dose,
                        "exstdy": start_day,
                        "exendy": end_day,
                    }
                )
                start_day = end_day + 1
        return pd.DataFrame(data)

    if domain == "AE":
        data = []
        terms = ["HEADACHE", "NAUSEA", "FATIGUE", "DIZZINESS"]
        severities = ["MILD", "MODERATE", "SEVERE"]
        for i in range(num_subjects):
            usubjid = f"SUBJ-{i+1:03d}"
            for seq in range(1, np.random.randint(1, 5)):
                start_day = np.random.randint(1, 50)
                duration = np.random.randint(1, 10)
                data.append(
                    {
                        "usubjid": usubjid,
                        "aeseq": seq,
                        "aeterm": np.random.choice(terms),
                        "aesev": np.random.choice(severities),
                        "aestdy": start_day,
                        "aeendy": start_day + duration,
                    }
                )
        return pd.DataFrame(data)

    if domain == "DS":
        data = []
        terms = ["COMPLETED", "ADVERSE EVENT", "PHYSICIAN DECISION", "LOST TO FOLLOW-UP"]
        for i in range(num_subjects):
            usubjid = f"SUBJ-{i+1:03d}"
            data.append(
                {
                    "usubjid": usubjid,
                    "dsseq": 1,
                    "dsterm": np.random.choice(terms),
                    "dsstdy": np.random.randint(50, 100),
                }
            )
        return pd.DataFrame(data)

    raise ValueError(f"Domain '{domain}' not supported for dummy data generation.")


if __name__ == "__main__":
    DOMAIN = "AE"
    BATCH_SIZE = 8
    D_MODEL = 128
    NHEAD = 8
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    DIM_FEEDFORWARD = 256
    EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if DOMAIN == "EX":
        SUBJECT_ID_COL, SEQUENCE_COL = "usubjid", "exseq"
        CONTINUOUS_COLS = ["exdose", "exstdy", "exendy"]
    elif DOMAIN == "AE":
        SUBJECT_ID_COL, SEQUENCE_COL = "usubjid", "aeseq"
        CONTINUOUS_COLS = ["aestdy", "aeendy"]
    elif DOMAIN == "DS":
        SUBJECT_ID_COL, SEQUENCE_COL = "usubjid", "dsseq"
        CONTINUOUS_COLS = ["dsstdy"]
    else:
        raise ValueError(f"Configuration for domain '{DOMAIN}' not found.")

    print()
    print(f"Generating dummy training data for {DOMAIN} domain...")
    train_df = generate_dummy_data(domain=DOMAIN, num_subjects=200)
    print("Dummy data generated. Shape:", train_df.shape)
    print("Sample data:\n", train_df.head())

    preprocessor = SDTMPreprocessor(
        subject_id_col=SUBJECT_ID_COL,
        sequence_col=SEQUENCE_COL,
        continuous_cols=CONTINUOUS_COLS,
    )
    preprocessor.fit(train_df)
    sequences = preprocessor.transform(train_df)

    categorical_cardinalities = [len(preprocessor.categorical_vocabs[col]) for col in preprocessor.categorical_cols]
    num_continuous = len(preprocessor.continuous_cols)

    dataset = SDTMDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)

    model = TabularSeq2SeqTransformer(
        categorical_cardinalities=categorical_cardinalities,
        num_continuous=num_continuous,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_model(model, dataloader, optimizer, device, epochs=EPOCHS)

    synthetic_df = synthesize_data(model, preprocessor, device, num_subjects=3, max_steps=10)
    print()
    print("--- Generated Synthetic Data ---")
    if not synthetic_df.empty:
        print(synthetic_df)
    else:
        print("No data generated. Try training for more epochs or adjusting hyperparameters.")
