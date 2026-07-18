from datetime import datetime

import datasets
import pyarrow as pa

from feast.infra.offline_stores.contrib.hf_datasets_offline_store.hf_datasets import (
    HFDatasetsOfflineStore,
)
from feast.infra.offline_stores.contrib.hf_datasets_offline_store.hf_datasets_source import (
    HFDatasetSource,
)
from feast.value_type import ValueType


class _FakeDataset:
    def __init__(self, features):
        self.features = features


def test_load_dataset_uses_constructor_fields(monkeypatch):
    captured = {}

    def fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return _FakeDataset({"text": datasets.Value("string")})

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    source = HFDatasetSource(
        path="dummy",
        config_name="config",
        split="train",
        revision="rev",
        data_dir="/tmp/data",
        data_files={"train": "train.parquet"},
        streaming=True,
        storage_options={"anon": True},
    )

    list(source.get_table_column_names_and_types(config=None))

    assert captured["path"] == "dummy"
    assert captured["name"] == "config"
    assert captured["split"] == "train"
    assert captured["revision"] == "rev"
    assert captured["data_dir"] == "/tmp/data"
    assert captured["data_files"] == {"train": "train.parquet"}
    assert captured["streaming"] is True
    assert captured["storage_options"] == {"anon": True}


def test_get_table_column_names_and_types_returns_raw_hf_types(monkeypatch):
    source = HFDatasetSource(path="dummy", split="train")

    features = {
        "text": datasets.Value("string"),
        "score": datasets.Value("float32"),
    }

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda **kwargs: _FakeDataset(features),
    )

    column_types = list(source.get_table_column_names_and_types(config=None))

    assert column_types == [("text", "string"), ("score", "float32")]


def test_infer_features_returns_feast_value_types(monkeypatch):
    source = HFDatasetSource(path="dummy", split="train")

    features = {
        "text": datasets.Value("string"),
        "score": datasets.Value("float32"),
    }

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda **kwargs: _FakeDataset(features),
    )

    inferred = source.infer_features()

    assert inferred == {"text": ValueType.STRING, "score": ValueType.FLOAT}


def test_pull_all_from_table_or_query_allows_missing_bounds(monkeypatch):
    source = HFDatasetSource(path="dummy", split="train")
    table = pa.table(
        {
            "event_timestamp": [datetime(2024, 1, 1, 12, 0, 0)],
            "entity_id": [1],
            "feature": [1.5],
        }
    )

    monkeypatch.setattr(HFDatasetsOfflineStore, "_load_dataset", lambda _source: object())
    monkeypatch.setattr(HFDatasetsOfflineStore, "_load_arrow_table", lambda _dataset: table)

    retrieval_job = HFDatasetsOfflineStore.pull_all_from_table_or_query(
        config=None,
        data_source=source,
        join_key_columns=["entity_id"],
        feature_name_columns=["feature"],
        timestamp_field="event_timestamp",
        start_date=None,
        end_date=None,
    )

    assert retrieval_job.to_arrow().to_pydict() == {
        "entity_id": [1],
        "feature": [1.5],
        "event_timestamp": [datetime(2024, 1, 1, 12, 0, 0)],
    }
