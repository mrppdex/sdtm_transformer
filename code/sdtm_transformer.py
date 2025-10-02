import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.api.types import is_numeric_dtype
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Data Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class EncodedSubject:
    """Container holding the encoded information for a single subject."""

    encoder_cat: np.ndarray
    encoder_cat_mask: np.ndarray
    encoder_num: np.ndarray
    encoder_num_mask: np.ndarray
    decoder_input_cat: np.ndarray
    decoder_input_cat_mask: np.ndarray
    decoder_input_num: np.ndarray
    decoder_input_num_mask: np.ndarray
    target_cat: np.ndarray
    target_cat_mask: np.ndarray
    target_num: np.ndarray
    target_num_mask: np.ndarray
    stop_targets: np.ndarray
    target_length: int


class SDTMSeq2SeqPreprocessor:
    """Preprocess SDTM data for a Transformer based seq2seq generator."""

    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[START]": 1,
        "[MISSING]": 2,
        "[UNK]": 3,
    }

    def __init__(
        self,
        subject_id_col: str,
        sequence_col: Optional[str],
        continuous_cols: Optional[List[str]] = None,
        auto_detect_continuous: bool = True,
    ) -> None:
        self.subject_id_col = subject_id_col
        self.sequence_col = sequence_col
        self.auto_detect_continuous = auto_detect_continuous
        self.user_continuous_cols = set(continuous_cols or [])

        self.columns: List[str] = []
        self.continuous_cols: List[str] = []
        self.categorical_cols: List[str] = []

        self.categorical_vocabs: Dict[str, Dict[str, int]] = {}
        self.categorical_reverse_vocabs: Dict[str, Dict[int, str]] = {}
        self.categorical_pad_idx: Dict[str, int] = {}
        self.categorical_start_idx: Dict[str, int] = {}

        self.continuous_stats: Dict[str, Tuple[float, float]] = {}

        self.max_target_len: int = 1
        self.initial_rows: List[pd.Series] = []

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------
    def _detect_continuous(self, df: pd.DataFrame) -> List[str]:
        detected = set(self.user_continuous_cols)
        if not self.auto_detect_continuous:
            return sorted(detected)

        for col in df.columns:
            if col in (self.subject_id_col, self.sequence_col):
                continue
            if is_numeric_dtype(df[col]):
                detected.add(col)
        return sorted(detected)

    def _init_categorical_vocab(self, values: List[str]) -> Dict[str, int]:
        vocab = dict(self.SPECIAL_TOKENS)
        next_index = max(vocab.values()) + 1
        for value in values:
            if value not in vocab:
                vocab[value] = next_index
                next_index += 1
        return vocab

    def _build_reverse_vocab(self, vocab: Dict[str, int]) -> Dict[int, str]:
        return {index: token for token, index in vocab.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "SDTMSeq2SeqPreprocessor":
        self.columns = [
            col
            for col in df.columns
            if col not in (self.subject_id_col, self.sequence_col)
        ]

        continuous_candidates = self._detect_continuous(df)
        self.continuous_cols = [
            col for col in self.columns if col in continuous_candidates
        ]
        self.categorical_cols = [
            col for col in self.columns if col not in self.continuous_cols
        ]

        for col in self.categorical_cols:
            unique_values = (
                df[col].astype(str).fillna("[MISSING]").unique().tolist()
            )
            vocab = self._init_categorical_vocab(unique_values)
            self.categorical_vocabs[col] = vocab
            self.categorical_reverse_vocabs[col] = self._build_reverse_vocab(vocab)
            self.categorical_pad_idx[col] = vocab["[PAD]"]
            self.categorical_start_idx[col] = vocab["[START]"]

        for col in self.continuous_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            mean = float(series.mean()) if not series.isna().all() else 0.0
            std = float(series.std()) if series.std() not in (None, 0.0) else 1.0
            if std == 0.0 or math.isnan(std):
                std = 1.0
            self.continuous_stats[col] = (mean, std)

        self.max_target_len = 1
        self.initial_rows = []

        for _, group in df.groupby(self.subject_id_col):
            group_sorted = (
                group.sort_values(self.sequence_col)
                if self.sequence_col and self.sequence_col in group.columns
                else group
            )
            self.initial_rows.append(group_sorted.iloc[0][self.columns])
            target_len = max(1, len(group_sorted) - 1)
            self.max_target_len = max(self.max_target_len, target_len)

        return self

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_categorical_value(self, col: str, value: object) -> Tuple[int, int]:
        vocab = self.categorical_vocabs[col]
        if pd.isna(value):
            return vocab["[MISSING]"], 0
        value_str = str(value)
        if value_str not in vocab:
            return vocab["[UNK]"], 1
        idx = vocab[value_str]
        mask = 0 if value_str == "[MISSING]" else 1
        return idx, mask

    def _encode_continuous_value(self, col: str, value: object) -> Tuple[float, float]:
        mean, std = self.continuous_stats[col]
        if pd.isna(value):
            return 0.0, 0.0
        normalized = (float(value) - mean) / std
        return normalized, 1.0

    def _encode_row(
        self, row: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cat_indices, cat_mask = [], []
        for col in self.categorical_cols:
            idx, mask = self._encode_categorical_value(col, row.get(col))
            cat_indices.append(idx)
            cat_mask.append(mask)

        num_values, num_mask = [], []
        for col in self.continuous_cols:
            value, mask = self._encode_continuous_value(col, row.get(col))
            num_values.append(value)
            num_mask.append(mask)

        return (
            np.asarray(cat_indices, dtype=np.int64),
            np.asarray(cat_mask, dtype=np.float32),
            np.asarray(num_values, dtype=np.float32),
            np.asarray(num_mask, dtype=np.float32),
        )

    def _start_tokens(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cat_indices = [self.categorical_start_idx[col] for col in self.categorical_cols]
        cat_mask = [0.0 for _ in self.categorical_cols]
        num_values = [0.0 for _ in self.continuous_cols]
        num_mask = [0.0 for _ in self.continuous_cols]
        return (
            np.asarray(cat_indices, dtype=np.int64),
            np.asarray(cat_mask, dtype=np.float32),
            np.asarray(num_values, dtype=np.float32),
            np.asarray(num_mask, dtype=np.float32),
        )

    def transform(self, df: pd.DataFrame) -> List[EncodedSubject]:
        if not self.categorical_vocabs and not self.continuous_stats:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        sequences: List[EncodedSubject] = []

        for _, group in df.groupby(self.subject_id_col):
            group_sorted = (
                group.sort_values(self.sequence_col)
                if self.sequence_col and self.sequence_col in group.columns
                else group
            )

            (
                encoder_cat,
                encoder_cat_mask,
                encoder_num,
                encoder_num_mask,
            ) = self._encode_row(group_sorted.iloc[0])

            target_rows = group_sorted.iloc[1:]

            if len(target_rows) == 0:
                target_len = 1
                target_cat = np.full(
                    (target_len, len(self.categorical_cols)),
                    self.SPECIAL_TOKENS["[PAD]"],
                    dtype=np.int64,
                )
                target_cat_mask = np.zeros_like(target_cat, dtype=np.float32)
                target_num = np.zeros(
                    (target_len, len(self.continuous_cols)), dtype=np.float32
                )
                target_num_mask = np.zeros_like(target_num, dtype=np.float32)
                stop_targets = np.ones((target_len,), dtype=np.int64)
            else:
                encoded_cat_rows: List[np.ndarray] = []
                encoded_cat_masks: List[np.ndarray] = []
                encoded_num_rows: List[np.ndarray] = []
                encoded_num_masks: List[np.ndarray] = []

                for _, row in target_rows.iterrows():
                    cat_vals, cat_masks, num_vals, num_masks = self._encode_row(row)
                    encoded_cat_rows.append(cat_vals)
                    encoded_cat_masks.append(cat_masks)
                    encoded_num_rows.append(num_vals)
                    encoded_num_masks.append(num_masks)

                target_cat = np.stack(encoded_cat_rows, axis=0)
                target_cat_mask = np.stack(encoded_cat_masks, axis=0)
                target_num = np.stack(encoded_num_rows, axis=0)
                target_num_mask = np.stack(encoded_num_masks, axis=0)

                stop_targets = np.zeros((len(target_rows),), dtype=np.int64)
                stop_targets[-1] = 1
                target_len = len(target_rows)

            (
                start_cat,
                start_cat_mask,
                start_num,
                start_num_mask,
            ) = self._start_tokens()

            decoder_input_cat = np.concatenate(
                [start_cat[None, :], target_cat[:-1] if len(target_cat) > 1 else np.empty((0, len(self.categorical_cols)), dtype=np.int64)],
                axis=0,
            )
            decoder_input_cat_mask = np.concatenate(
                [start_cat_mask[None, :], target_cat_mask[:-1] if len(target_cat_mask) > 1 else np.empty((0, len(self.categorical_cols)), dtype=np.float32)],
                axis=0,
            )
            decoder_input_num = np.concatenate(
                [start_num[None, :], target_num[:-1] if len(target_num) > 1 else np.empty((0, len(self.continuous_cols)), dtype=np.float32)],
                axis=0,
            )
            decoder_input_num_mask = np.concatenate(
                [start_num_mask[None, :], target_num_mask[:-1] if len(target_num_mask) > 1 else np.empty((0, len(self.continuous_cols)), dtype=np.float32)],
                axis=0,
            )

            sequences.append(
                EncodedSubject(
                    encoder_cat=encoder_cat,
                    encoder_cat_mask=encoder_cat_mask,
                    encoder_num=encoder_num,
                    encoder_num_mask=encoder_num_mask,
                    decoder_input_cat=decoder_input_cat,
                    decoder_input_cat_mask=decoder_input_cat_mask,
                    decoder_input_num=decoder_input_num,
                    decoder_input_num_mask=decoder_input_num_mask,
                    target_cat=target_cat,
                    target_cat_mask=target_cat_mask,
                    target_num=target_num,
                    target_num_mask=target_num_mask,
                    stop_targets=stop_targets,
                    target_length=target_len,
                )
            )

        return sequences

    # ------------------------------------------------------------------
    # Decoding utilities for synthesis
    # ------------------------------------------------------------------
    def categorical_vocab_size(self, col: str) -> int:
        return len(self.categorical_vocabs[col])

    def continuous_dim(self) -> int:
        return len(self.continuous_cols)

    def categorical_dim(self) -> int:
        return len(self.categorical_cols)

    def decode_categorical(self, col: str, index: int) -> str:
        reverse_vocab = self.categorical_reverse_vocabs[col]
        return reverse_vocab.get(index, "")

    def denormalize_continuous(self, col: str, value: float) -> float:
        mean, std = self.continuous_stats[col]
        return value * std + mean

    def sample_initial_row(self) -> pd.Series:
        if not self.initial_rows:
            raise RuntimeError("Preprocessor has no initial rows to sample from.")
        return self.initial_rows[random.randrange(len(self.initial_rows))]

    def encode_from_series(
        self, series: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._encode_row(series)


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------


class SDTMDataset(Dataset):
    def __init__(
        self,
        encoded_subjects: List[EncodedSubject],
        max_target_len: int,
        categorical_dim: int,
        continuous_dim: int,
        pad_index: int,
    ) -> None:
        self.encoded_subjects = encoded_subjects
        self.max_target_len = max_target_len
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim
        self.pad_index = pad_index

    def __len__(self) -> int:
        return len(self.encoded_subjects)

    def _pad_sequence(self, array: np.ndarray, pad_value: float) -> np.ndarray:
        if array.size == 0:
            return np.zeros((self.max_target_len, array.shape[-1]), dtype=array.dtype)
        padded = np.full(
            (self.max_target_len, array.shape[-1]),
            pad_value,
            dtype=array.dtype,
        )
        length = min(len(array), self.max_target_len)
        padded[:length] = array[:length]
        return padded

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject = self.encoded_subjects[idx]
        tgt_len = min(subject.target_length, self.max_target_len)

        decoder_input_cat = self._pad_sequence(
            subject.decoder_input_cat, self.pad_index
        )
        decoder_input_cat_mask = self._pad_sequence(
            subject.decoder_input_cat_mask, 0.0
        )
        decoder_input_num = self._pad_sequence(subject.decoder_input_num, 0.0)
        decoder_input_num_mask = self._pad_sequence(
            subject.decoder_input_num_mask, 0.0
        )

        target_cat = self._pad_sequence(subject.target_cat, self.pad_index)
        target_cat_mask = self._pad_sequence(subject.target_cat_mask, 0.0)
        target_num = self._pad_sequence(subject.target_num, 0.0)
        target_num_mask = self._pad_sequence(subject.target_num_mask, 0.0)

        stop_targets = np.full((self.max_target_len,), -1, dtype=np.int64)
        stop_targets[:tgt_len] = subject.stop_targets[:tgt_len]

        target_padding_mask = np.zeros((self.max_target_len,), dtype=np.float32)
        target_padding_mask[:tgt_len] = 1.0

        return {
            "encoder_cat": torch.tensor(subject.encoder_cat, dtype=torch.long).unsqueeze(0),
            "encoder_cat_mask": torch.tensor(subject.encoder_cat_mask, dtype=torch.float32).unsqueeze(0),
            "encoder_num": torch.tensor(subject.encoder_num, dtype=torch.float32).unsqueeze(0),
            "encoder_num_mask": torch.tensor(subject.encoder_num_mask, dtype=torch.float32).unsqueeze(0),
            "decoder_input_cat": torch.tensor(decoder_input_cat, dtype=torch.long),
            "decoder_input_cat_mask": torch.tensor(decoder_input_cat_mask, dtype=torch.float32),
            "decoder_input_num": torch.tensor(decoder_input_num, dtype=torch.float32),
            "decoder_input_num_mask": torch.tensor(decoder_input_num_mask, dtype=torch.float32),
            "target_cat": torch.tensor(target_cat, dtype=torch.long),
            "target_cat_mask": torch.tensor(target_cat_mask, dtype=torch.float32),
            "target_num": torch.tensor(target_num, dtype=torch.float32),
            "target_num_mask": torch.tensor(target_num_mask, dtype=torch.float32),
            "stop_targets": torch.tensor(stop_targets, dtype=torch.long),
            "target_padding_mask": torch.tensor(target_padding_mask, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# 3. Model components
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RowEmbedding(nn.Module):
    def __init__(
        self,
        categorical_vocab_sizes: Dict[str, int],
        continuous_dim: int,
        embedding_dim: int,
        d_model: int,
        categorical_cols: List[str],
        continuous_cols: List[str],
    ) -> None:
        super().__init__()
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.categorical_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(size, embedding_dim)
                for col, size in categorical_vocab_sizes.items()
            }
        )
        total_dim = embedding_dim * len(categorical_vocab_sizes) + continuous_dim
        self.proj = nn.Linear(total_dim, d_model)

    def forward(
        self,
        cat_inputs: torch.Tensor,
        cat_mask: torch.Tensor,
        num_inputs: torch.Tensor,
        num_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Inputs are (batch, seq_len, feature_dim)
        batch_size, seq_len, _ = cat_inputs.shape
        cat_embeddings: List[torch.Tensor] = []
        for idx, col in enumerate(self.categorical_cols):
            emb = self.categorical_embeddings[col](cat_inputs[:, :, idx])
            cat_embeddings.append(emb)
        if cat_embeddings:
            cat_repr = torch.cat(cat_embeddings, dim=-1)
        else:
            cat_repr = torch.zeros(
                batch_size, seq_len, 0, device=cat_inputs.device, dtype=torch.float32
            )

        num_repr = num_inputs * num_mask
        combined = torch.cat([cat_repr, num_repr], dim=-1)
        return self.proj(combined)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        categorical_vocab_sizes: Dict[str, int],
        continuous_dim: int,
        categorical_cols: List[str],
        continuous_cols: List[str],
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        self.row_embedding = RowEmbedding(
            categorical_vocab_sizes,
            continuous_dim,
            embedding_dim,
            d_model,
            categorical_cols,
            continuous_cols,
        )
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols

        self.categorical_heads = nn.ModuleDict(
            {
                col: nn.Linear(d_model, categorical_vocab_sizes[col])
                for col in categorical_cols
            }
        )
        self.continuous_heads = nn.ModuleDict(
            {col: nn.Linear(d_model, 1) for col in continuous_cols}
        )
        self.stop_head = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(sz, sz, device=self.stop_head.weight.device), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        encoder_cat: torch.Tensor,
        encoder_cat_mask: torch.Tensor,
        encoder_num: torch.Tensor,
        encoder_num_mask: torch.Tensor,
        decoder_cat: torch.Tensor,
        decoder_cat_mask: torch.Tensor,
        decoder_num: torch.Tensor,
        decoder_num_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        encoder_emb = self.row_embedding(
            encoder_cat, encoder_cat_mask, encoder_num, encoder_num_mask
        )
        encoder_emb = self.positional_encoding(encoder_emb)
        memory = self.encoder(encoder_emb)

        decoder_emb = self.row_embedding(
            decoder_cat, decoder_cat_mask, decoder_num, decoder_num_mask
        )
        decoder_emb = self.positional_encoding(decoder_emb)

        decoder_output = self.decoder(
            decoder_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        categorical_outputs = {
            col: head(decoder_output) for col, head in self.categorical_heads.items()
        }
        continuous_outputs = {
            col: head(decoder_output).squeeze(-1)
            for col, head in self.continuous_heads.items()
        }
        stop_logits = self.stop_head(decoder_output).squeeze(-1)

        return {
            "categorical": categorical_outputs,
            "continuous": continuous_outputs,
            "stop": stop_logits,
        }


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------


def compute_losses(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    categorical_cols: List[str],
    continuous_cols: List[str],
    pad_indices: Dict[str, int],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: Dict[str, float] = {}
    total_loss = torch.zeros(1, device=device)

    target_cat = batch["target_cat"].to(device)
    target_cat_mask = batch["target_cat_mask"].to(device)
    for idx, col in enumerate(categorical_cols):
        logits = outputs["categorical"][col]
        logits = logits.view(-1, logits.size(-1))
        targets = target_cat[:, :, idx].reshape(-1)
        mask = target_cat_mask[:, :, idx].reshape(-1)

        valid = mask > 0
        if valid.sum() == 0:
            continue
        filtered_logits = logits[valid]
        filtered_targets = targets[valid]
        loss = F.cross_entropy(filtered_logits, filtered_targets)
        losses[f"cat_{col}"] = loss.item()
        total_loss = total_loss + loss

    target_num = batch["target_num"].to(device)
    target_num_mask = batch["target_num_mask"].to(device)
    for idx, col in enumerate(continuous_cols):
        preds = outputs["continuous"][col]
        mask = target_num_mask[:, :, idx]
        if mask.sum() == 0:
            continue
        diff = preds - target_num[:, :, idx]
        loss = (diff ** 2 * mask).sum() / mask.sum()
        losses[f"num_{col}"] = loss.item()
        total_loss = total_loss + loss

    stop_logits = outputs["stop"]
    stop_targets = batch["stop_targets"].to(device)
    stop_mask = stop_targets >= 0
    if stop_mask.sum() > 0:
        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logits[stop_mask].float(), stop_targets[stop_mask].float()
        )
        losses["stop"] = stop_loss.item()
        total_loss = total_loss + stop_loss

    return total_loss, losses


def train_model(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    categorical_cols: List[str],
    continuous_cols: List[str],
    pad_indices: Dict[str, int],
    epochs: int = 10,
) -> None:
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)

            tgt_len = batch["decoder_input_cat"].size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_len)
            tgt_padding_mask = batch["target_padding_mask"] == 0

            outputs = model(
                encoder_cat=batch["encoder_cat"],
                encoder_cat_mask=batch["encoder_cat_mask"],
                encoder_num=batch["encoder_num"],
                encoder_num_mask=batch["encoder_num_mask"],
                decoder_cat=batch["decoder_input_cat"],
                decoder_cat_mask=batch["decoder_input_cat_mask"],
                decoder_num=batch["decoder_input_num"],
                decoder_num_mask=batch["decoder_input_num_mask"],
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            optimizer.zero_grad()
            loss, loss_components = compute_losses(
                outputs,
                batch,
                categorical_cols,
                continuous_cols,
                pad_indices,
                device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")


# ---------------------------------------------------------------------------
# 5. Synthesis
# ---------------------------------------------------------------------------


def synthesize_data(
    model: Seq2SeqTransformer,
    preprocessor: SDTMSeq2SeqPreprocessor,
    device: torch.device,
    num_subjects: int = 5,
    max_steps: Optional[int] = None,
) -> pd.DataFrame:
    model.eval()
    generated_records: List[Dict[str, object]] = []
    max_steps = max_steps or preprocessor.max_target_len

    with torch.no_grad():
        for subject_idx in range(num_subjects):
            first_row = preprocessor.sample_initial_row().copy()
            subject_id = f"SYNTH-{subject_idx + 1:03d}"

            (
                enc_cat,
                enc_cat_mask,
                enc_num,
                enc_num_mask,
            ) = preprocessor.encode_from_series(first_row)

            (
                start_cat,
                start_cat_mask,
                start_num,
                start_num_mask,
            ) = preprocessor._start_tokens()

            decoder_cat = torch.tensor(
                start_cat[None, None, :], dtype=torch.long, device=device
            )
            decoder_cat_mask = torch.tensor(
                start_cat_mask[None, None, :], dtype=torch.float32, device=device
            )
            decoder_num = torch.tensor(
                start_num[None, None, :], dtype=torch.float32, device=device
            )
            decoder_num_mask = torch.tensor(
                start_num_mask[None, None, :], dtype=torch.float32, device=device
            )

            encoder_cat_tensor = torch.tensor(enc_cat[None, None, :], dtype=torch.long, device=device)
            encoder_cat_mask_tensor = torch.tensor(
                enc_cat_mask[None, None, :], dtype=torch.float32, device=device
            )
            encoder_num_tensor = torch.tensor(enc_num[None, None, :], dtype=torch.float32, device=device)
            encoder_num_mask_tensor = torch.tensor(
                enc_num_mask[None, None, :], dtype=torch.float32, device=device
            )

            subject_rows = [first_row.to_dict()]

            for step in range(max_steps):
                tgt_len = decoder_cat.size(1)
                tgt_mask = model.generate_square_subsequent_mask(tgt_len)

                outputs = model(
                    encoder_cat=encoder_cat_tensor,
                    encoder_cat_mask=encoder_cat_mask_tensor,
                    encoder_num=encoder_num_tensor,
                    encoder_num_mask=encoder_num_mask_tensor,
                    decoder_cat=decoder_cat,
                    decoder_cat_mask=decoder_cat_mask,
                    decoder_num=decoder_num,
                    decoder_num_mask=decoder_num_mask,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=None,
                )

                new_row_cat = []
                new_row_num = []
                new_row_num_mask = []
                row_dict: Dict[str, object] = {}

                for idx, col in enumerate(preprocessor.categorical_cols):
                    logits = outputs["categorical"][col][0, -1]
                    logits[preprocessor.SPECIAL_TOKENS["[PAD]"]] = -float("inf")
                    logits[preprocessor.SPECIAL_TOKENS["[START]"]] = -float("inf")
                    probs = torch.softmax(logits, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).item()
                    value = preprocessor.decode_categorical(col, sampled_idx)
                    if value in ("[PAD]", "[START]"):
                        value = "[MISSING]"
                    row_dict[col] = value if value not in ("[MISSING]", "[UNK]") else None
                    new_row_cat.append(sampled_idx)

                for idx, col in enumerate(preprocessor.continuous_cols):
                    pred = outputs["continuous"][col][0, -1].item()
                    value = preprocessor.denormalize_continuous(col, pred)
                    row_dict[col] = value
                    new_row_num.append(pred)
                    new_row_num_mask.append(1.0)

                subject_rows.append(row_dict)

                stop_prob = torch.sigmoid(outputs["stop"][0, -1]).item()

                new_cat_tensor = torch.tensor(
                    [new_row_cat], dtype=torch.long, device=device
                )
                new_cat_mask_tensor = torch.ones_like(new_cat_tensor, dtype=torch.float32)
                new_num_tensor = torch.tensor(
                    [new_row_num], dtype=torch.float32, device=device
                )
                if preprocessor.continuous_cols:
                    new_num_mask_tensor = torch.tensor(
                        [new_row_num_mask], dtype=torch.float32, device=device
                    )
                else:
                    new_num_mask_tensor = torch.zeros_like(new_num_tensor, dtype=torch.float32)

                decoder_cat = torch.cat([decoder_cat, new_cat_tensor], dim=1)
                decoder_cat_mask = torch.cat([decoder_cat_mask, new_cat_mask_tensor], dim=1)
                if preprocessor.continuous_cols:
                    decoder_num = torch.cat([decoder_num, new_num_tensor], dim=1)
                    decoder_num_mask = torch.cat(
                        [decoder_num_mask, new_num_mask_tensor], dim=1
                    )

                if stop_prob > 0.6:
                    break

            for seq_idx, row in enumerate(subject_rows):
                record = {
                    preprocessor.subject_id_col: subject_id,
                    preprocessor.sequence_col or "sequence": seq_idx + 1,
                }
                record.update(row)
                generated_records.append(record)

    return pd.DataFrame(generated_records)


# ---------------------------------------------------------------------------
# 6. Dummy data
# ---------------------------------------------------------------------------


def generate_dummy_data(domain: str = "AE", num_subjects: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng()
    records: List[Dict[str, object]] = []

    if domain == "AE":
        severities = ["MILD", "MODERATE", "SEVERE"]
        terms = ["HEADACHE", "NAUSEA", "FATIGUE", "DIZZINESS"]
        for subj in range(num_subjects):
            usubjid = f"SUBJ-{subj + 1:03d}"
            num_events = rng.integers(1, 5)
            start_day = 1
            for seq in range(1, num_events + 1):
                duration = rng.integers(1, 10)
                records.append(
                    {
                        "usubjid": usubjid,
                        "aeseq": seq,
                        "aeterm": rng.choice(terms),
                        "aesev": rng.choice(severities),
                        "aestdy": start_day,
                        "aeendy": start_day + duration,
                    }
                )
                start_day += duration
    elif domain == "EX":
        doses = [25, 50, 100]
        for subj in range(num_subjects):
            usubjid = f"SUBJ-{subj + 1:03d}"
            num_events = rng.integers(2, 6)
            start_day = 1
            for seq in range(1, num_events + 1):
                duration = rng.integers(5, 20)
                records.append(
                    {
                        "usubjid": usubjid,
                        "exseq": seq,
                        "exdose": rng.choice(doses),
                        "exstdy": start_day,
                        "exendy": start_day + duration,
                    }
                )
                start_day += duration
    else:
        raise ValueError(f"Unsupported domain '{domain}'.")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    DOMAIN = "AE"
    SUBJECT_ID_COL = "usubjid"
    SEQUENCE_COL = "aeseq"
    CONTINUOUS_COLS = ["aestdy", "aeendy"]

    MAX_TARGET_LEN = 8
    BATCH_SIZE = 8
    D_MODEL = 128
    NHEAD = 8
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 256
    EPOCHS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = generate_dummy_data(DOMAIN, num_subjects=120)
    print(f"Generated training data with shape {df.shape}")

    preprocessor = SDTMSeq2SeqPreprocessor(
        subject_id_col=SUBJECT_ID_COL,
        sequence_col=SEQUENCE_COL,
        continuous_cols=CONTINUOUS_COLS,
        auto_detect_continuous=True,
    )
    preprocessor.fit(df)
    encoded_subjects = preprocessor.transform(df)

    max_target_len = min(MAX_TARGET_LEN, preprocessor.max_target_len)

    dataset = SDTMDataset(
        encoded_subjects,
        max_target_len=max_target_len,
        categorical_dim=preprocessor.categorical_dim(),
        continuous_dim=preprocessor.continuous_dim(),
        pad_index=SDTMSeq2SeqPreprocessor.SPECIAL_TOKENS["[PAD]"],
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    categorical_vocab_sizes = {
        col: preprocessor.categorical_vocab_size(col)
        for col in preprocessor.categorical_cols
    }

    model = Seq2SeqTransformer(
        categorical_vocab_sizes=categorical_vocab_sizes,
        continuous_dim=preprocessor.continuous_dim(),
        categorical_cols=preprocessor.categorical_cols,
        continuous_cols=preprocessor.continuous_cols,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_model(
        model,
        dataloader,
        optimizer,
        device,
        categorical_cols=preprocessor.categorical_cols,
        continuous_cols=preprocessor.continuous_cols,
        pad_indices={
            col: SDTMSeq2SeqPreprocessor.SPECIAL_TOKENS["[PAD]"]
            for col in preprocessor.categorical_cols
        },
        epochs=EPOCHS,
    )

    synthetic_df = synthesize_data(
        model,
        preprocessor,
        device,
        num_subjects=3,
        max_steps=max_target_len,
    )

    print("\nSynthetic data sample:")
    print(synthetic_df.head())


if __name__ == "__main__":
    main()

