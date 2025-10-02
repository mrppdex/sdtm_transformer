import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = types.ModuleType("torch")

    class _FakeTensor(list):
        def tolist(self):
            return list(self)

        @property
        def shape(self):
            return (len(self),)

    def tensor(data, dtype=None):  # noqa: ARG001 - dtype kept for API parity
        if isinstance(data, _FakeTensor):
            return _FakeTensor(data)
        return _FakeTensor(list(data))

    torch.tensor = tensor  # type: ignore[attr-defined]
    torch.long = "long"  # type: ignore[attr-defined]
    torch.device = lambda *args, **kwargs: "cpu"  # type: ignore[attr-defined]

    class _NoGrad:  # pragma: no cover - only used when torch missing
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch.no_grad = _NoGrad  # type: ignore[attr-defined]
    torch.nn = types.SimpleNamespace(Module=object, functional=types.SimpleNamespace())  # type: ignore[attr-defined]
    torch.utils = types.SimpleNamespace(  # type: ignore[attr-defined]
        data=types.SimpleNamespace(Dataset=object, DataLoader=object)
    )
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", torch.nn)
    sys.modules.setdefault("torch.utils", torch.utils)
    sys.modules.setdefault("torch.utils.data", torch.utils.data)

MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "sdtm_transformer.py"
spec = importlib.util.spec_from_file_location("sdtm_transformer_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

SDTMPreprocessor = module.SDTMPreprocessor
SDTMDataset = module.SDTMDataset
generate_dummy_data = module.generate_dummy_data


@pytest.fixture
def example_dataframe():
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        [
            {"usubjid": "SUBJ-001", "exseq": 1, "exdose": 50, "exstdy": 1, "arm": "A"},
            {"usubjid": "SUBJ-001", "exseq": 2, "exdose": 75, "exstdy": 5, "arm": "A"},
            {"usubjid": "SUBJ-002", "exseq": 1, "exdose": 50, "exstdy": 2, "arm": "B"},
        ]
    )
    df["exdose"] = df["exdose"].astype(float) + rng.normal(0, 0.1, size=len(df))
    return df


def test_preprocessor_builds_vocab_and_sequences(example_dataframe):
    preprocessor = SDTMPreprocessor(
        subject_id_col="usubjid",
        sequence_col="exseq",
    )

    preprocessor.fit(example_dataframe)
    sequences = preprocessor.transform(example_dataframe)

    assert preprocessor.vocab["[SOS]"] == 1
    assert "arm__A" in preprocessor.vocab
    assert any(key.startswith("exdose__") for key in preprocessor.vocab)
    assert preprocessor.column_types["exdose"] == "continuous"
    assert preprocessor.column_types["arm"] == "categorical"

    assert len(sequences) == 2
    first_sequence = sequences[0]
    assert first_sequence[0] == preprocessor.vocab["[SOS]"]
    assert first_sequence[-1] == preprocessor.vocab["[EOS]"]
    assert preprocessor.vocab["[EOR]"] in first_sequence


def test_dataset_padding_and_target_alignment():
    sequences = [[1, 2, 3, 4], [1, 2]]
    dataset = SDTMDataset(sequences, max_seq_len=5)

    inputs, targets = dataset[1]
    assert inputs.tolist() == [1, 2, 0, 0]
    assert targets.tolist() == [2, 0, 0, 0]
    assert inputs.shape == targets.shape == (4,)


def test_generate_dummy_data_supported_domains():
    ex_df = generate_dummy_data(domain="EX", num_subjects=5)
    ae_df = generate_dummy_data(domain="AE", num_subjects=5)
    ds_df = generate_dummy_data(domain="DS", num_subjects=5)

    assert set(["usubjid", "exseq", "exdose", "exstdy", "exendy"]).issubset(ex_df.columns)
    assert set(["usubjid", "aeseq", "aeterm", "aesev", "aestdy", "aeendy"]).issubset(ae_df.columns)
    assert set(["usubjid", "dsseq", "dsterm", "dsstdy"]).issubset(ds_df.columns)
    assert ex_df["usubjid"].nunique() == 5
    assert ae_df["usubjid"].nunique() == 5
    assert ds_df["usubjid"].nunique() == 5


def test_generate_dummy_data_invalid_domain_raises():
    with pytest.raises(ValueError):
        generate_dummy_data(domain="UNKNOWN", num_subjects=1)
