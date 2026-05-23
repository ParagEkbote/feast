from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from feast.data_source import DataSource
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import RepoConfig
from feast.type_map import (
    feast_value_type_to_python_type,
)

from google.protobuf.duration_pb2 import Duration


@dataclass
class HFDatasetSource(DataSource):
    """
    Hugging Face Datasets-backed Feast DataSource.

    This datasource acts as a declarative wrapper around:
        datasets.load_dataset(...)

    It stores dataset-loading configuration and registry metadata,
    but does NOT perform query execution itself.

    Execution logic belongs in:
        - HFDatasetsOfflineStore
        - HFDatasetsRetrievalJob
    """

    # ------------------------------------------------------------------
    # Core HF Dataset Configuration
    # ------------------------------------------------------------------

    path: str = ""
    """
    HF Hub dataset name OR local dataset path.

    Examples:
        "imdb"
        "HuggingFaceH4/ultrachat_200k"
        "/data/my_dataset"
        "parquet"
    """

    split: str = "train"
    """
    Dataset split.

    Examples:
        "train"
        "validation"
        "test"
    """

    revision: Optional[str] = None
    """
    HF Hub revision / branch / commit hash.
    Useful for reproducibility.
    """

    streaming: bool = False
    """
    Whether to use IterableDataset streaming mode.
    MVP implementations may ignore this initially.
    """

    data_files: Optional[Dict[str, Any]] = None
    """
    Optional data_files argument passed to load_dataset().

    Example:
        {
            "train": "train.parquet",
            "test": "test.parquet"
        }
    """

    cache_dir: Optional[str] = None
    """
    Optional Hugging Face datasets cache directory.
    """

    token: Optional[str] = None
    """
    HF auth token for private/gated datasets.
    """

    load_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Additional kwargs forwarded directly to:
        datasets.load_dataset(...)

    Example:
        {
            "trust_remote_code": True
        }
    """

    # ------------------------------------------------------------------
    # Feast-Specific Metadata
    # ------------------------------------------------------------------

    timestamp_field: Optional[str] = None
    """
    Event timestamp column used for point-in-time joins.
    """

    created_timestamp_column: Optional[str] = None
    """
    Optional created timestamp column used for deduplication.
    """

    field_mapping: Dict[str, str] = field(default_factory=dict)
    """
    Raw column name -> Feast feature name mapping.
    """

    # ------------------------------------------------------------------
    # Feast Required Metadata
    # ------------------------------------------------------------------

    description: str = ""
    owner: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, config: RepoConfig):
        """
        Validate datasource configuration.
        """

        if not self.path:
            raise ValueError("HFDatasetSource.path must be provided.")

        if not isinstance(self.split, str):
            raise TypeError("split must be a string.")

        if self.data_files is not None and not isinstance(
            self.data_files, dict
        ):
            raise TypeError("data_files must be a dictionary.")

        if self.field_mapping and not isinstance(
            self.field_mapping, dict
        ):
            raise TypeError("field_mapping must be a dictionary.")

        if self.streaming and self.created_timestamp_column:
            # Streaming + dedup semantics may become tricky.
            # Better to explicitly guard early.
            raise ValueError(
                "created_timestamp_column is not currently supported "
                "with streaming datasets."
            )

    # ------------------------------------------------------------------
    # Feast Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def from_proto(data_source: DataSourceProto) -> "HFDatasetSource":
        """
        Deserialize from Feast registry protobuf.
        """

        custom_options = data_source.custom_options.configuration

        return HFDatasetSource(
            name=data_source.name,
            path=custom_options.get("path", ""),
            split=custom_options.get("split", "train"),
            revision=custom_options.get("revision"),
            streaming=custom_options.get("streaming", "False") == "True",
            cache_dir=custom_options.get("cache_dir"),
            token=custom_options.get("token"),
            timestamp_field=custom_options.get("timestamp_field"),
            created_timestamp_column=custom_options.get(
                "created_timestamp_column"
            ),
            description=data_source.description,
            owner=data_source.owner,
            tags=dict(data_source.tags),
        )

    def to_proto(self) -> DataSourceProto:
        """
        Serialize datasource into Feast registry protobuf.
        """

        data_source_proto = DataSourceProto(
            type="hf_datasets",
            name=self.name,
            description=self.description,
            owner=self.owner,
            timestamp_field=self.timestamp_field or "",
            created_timestamp_column=(
                self.created_timestamp_column or ""
            ),
            field_mapping=self.field_mapping,
            tags=self.tags,
        )

        custom_options = {
            "path": self.path,
            "split": self.split,
            "streaming": str(self.streaming),
        }

        if self.revision:
            custom_options["revision"] = self.revision

        if self.cache_dir:
            custom_options["cache_dir"] = self.cache_dir

        if self.token:
            custom_options["token"] = self.token

        if self.timestamp_field:
            custom_options["timestamp_field"] = self.timestamp_field

        if self.created_timestamp_column:
            custom_options[
                "created_timestamp_column"
            ] = self.created_timestamp_column

        # Store arbitrary HF kwargs as strings
        for k, v in self.load_dataset_kwargs.items():
            custom_options[f"hf_kwarg_{k}"] = str(v)

        data_source_proto.custom_options.configuration.update(
            custom_options
        )

        return data_source_proto

    # ------------------------------------------------------------------
    # Equality / Representation
    # ------------------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, HFDatasetSource):
            return False

        return (
            self.path == other.path
            and self.split == other.split
            and self.revision == other.revision
            and self.streaming == other.streaming
            and self.data_files == other.data_files
        )

    # ------------------------------------------------------------------
    # Utility Helpers
    # ------------------------------------------------------------------

    def get_table_query_string(self) -> str:
        """
        Human-readable datasource identifier.

        Feast uses similar helpers internally for logging/debugging.
        """

        components = [self.path]

        if self.split:
            components.append(f"split={self.split}")

        if self.revision:
            components.append(f"revision={self.revision}")

        return "::".join(components)

    # ------------------------------------------------------------------
    # Schema Inference (Optional Utility)
    # ------------------------------------------------------------------

    def infer_features(self):
        """
        Infer HF dataset schema using datasets.load_dataset().

        This helper is optional and primarily useful for:
            - debugging
            - schema inspection
            - auto-generating Feast fields
        """

        from datasets import load_dataset

        dataset = load_dataset(
            path=self.path,
            split=self.split,
            revision=self.revision,
            streaming=self.streaming,
            data_files=self.data_files,
            cache_dir=self.cache_dir,
            token=self.token,
            **self.load_dataset_kwargs,
        )

        return dataset.features