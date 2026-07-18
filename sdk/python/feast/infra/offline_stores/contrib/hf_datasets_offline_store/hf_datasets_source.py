import json
from typing import Any, Dict, List, Optional, Union

from datasets import ClassLabel, Sequence, Value

from feast import feature
from feast.data_source import DataSource
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import RepoConfig
from feast.type_map import hf_to_feast_value_type
from feast.value_type import ValueType

# Fully qualified path Feast uses to dynamically resolve and instantiate this
# class from the registry. Must stay in sync with this module's location.
_DATA_SOURCE_CLASS_TYPE = (
    "feast.infra.offline_stores.contrib.hf_datasets_offline_store."
    "hf_datasets_source.HFDatasetSource"
)


class HFDatasetSource(DataSource):
    """Hugging Face Datasets-backed Feast DataSource.

    A declarative wrapper around ``datasets.load_dataset(...)``: it stores
    dataset-loading configuration and registry metadata but does not perform
    query execution itself. Execution logic belongs to
    ``HFDatasetsOfflineStore`` and ``HFDatasetsRetrievalJob``.

    Attributes:
        path: HF Hub dataset name or local dataset path, e.g. ``"imdb"``,
            ``"HuggingFaceH4/ultrachat_200k"``, ``"/data/my_dataset"``, or a
            loader name like ``"parquet"``.
        split: Dataset split to load, e.g. ``"train"``, ``"validation"``,
            ``"test"``.
        revision: HF Hub revision, branch, or commit hash, for
            reproducibility.
        streaming: Whether to load the dataset in ``IterableDataset``
            streaming mode. Not currently supported by the offline store —
            reads will raise ``NotImplementedError``.
        data_files: Optional ``data_files`` argument forwarded to
            ``load_dataset()``, e.g.
            ``{"train": "train.parquet", "test": "test.parquet"}``.
        cache_dir: Optional Hugging Face datasets cache directory.
        token: HF auth token for private/gated datasets. Stored as
            plaintext in the Feast registry when serialized — prefer
            referencing an environment variable at load time over hardcoding
            a real token here in production registries.
        load_dataset_kwargs: Additional kwargs forwarded directly to
            ``load_dataset()``, e.g. ``{"trust_remote_code": True}``.
        timestamp_field: Event timestamp column used for point-in-time
            joins.
        created_timestamp_column: Optional created-timestamp column used
            for deduplication during materialization.
        field_mapping: Mapping from raw column name to Feast feature name.
        description: Human-readable description of the data source.
        owner: Owner of the data source.
        tags: Arbitrary key-value tags.
    """

    def source_type(self) -> DataSourceProto.SourceType.ValueType:
        return DataSourceProto.CUSTOM_SOURCE

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        path: str,
        config_name: Optional[str] = None,
        split: str = "train",
        revision: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str], Dict[str, str]]] = None,
        streaming: bool = False,
        storage_options: Optional[Dict[str, Any]] = None,
        timestamp_field: Optional[str] = None,
        created_timestamp_column: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        owner: str = "",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if name is None:
            name = path
        super().__init__(
            name=name,
            timestamp_field=timestamp_field,
            created_timestamp_column=created_timestamp_column,
            field_mapping=field_mapping,
            description=description,
            tags=tags,
            owner=owner,
        )
        self.path = path
        self.config_name = config_name
        self.split = split
        self.revision = revision
        self.data_dir = data_dir
        self.streaming = streaming
        self.storage_options = storage_options
        self.data_files = data_files
        self.cache_dir = cache_dir
        self.token = token
        self.load_dataset_kwargs = load_dataset_kwargs or {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, config: RepoConfig) -> None:
        """Validate the data source configuration."""
        if not self.path:
            raise ValueError("HFDatasetSource.path must be provided.")

        if not isinstance(self.split, str):
            raise TypeError("split must be a string.")

        if self.data_files is not None and not isinstance(
            self.data_files, (str, list, dict)
        ):
            raise TypeError("data_files must be a string, list, or dictionary.")

        if self.field_mapping and not isinstance(self.field_mapping, dict):
            raise TypeError("field_mapping must be a dictionary.")

        if self.streaming and self.created_timestamp_column:
            # Streaming + dedup semantics may become tricky.
            # Better to explicitly guard early.
            raise ValueError(
                "created_timestamp_column is not currently supported with streaming datasets."
            )

    # ------------------------------------------------------------------
    # Feast Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def from_proto(data_source: DataSourceProto) -> "HFDatasetSource":
        """Deserialize a data source from the Feast registry."""
        raw_configuration = data_source.custom_options.configuration

        if not raw_configuration:
            raise ValueError(
                f"HFDatasetSource '{data_source.name}' has no custom_options.configuration payload."
            )

        try:
            config = json.loads(raw_configuration.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"Failed to deserialize HFDatasetSource custom_options for '{data_source.name}': {e}"
            ) from e

        return HFDatasetSource(
            name=data_source.name,
            path=config.get("path", ""),
            config_name=config.get("config_name"),
            split=config.get("split", "train"),
            revision=config.get("revision"),
            data_dir=config.get("data_dir"),
            data_files=config.get("data_files"),
            streaming=config.get("streaming", False),
            storage_options=config.get("storage_options"),
            cache_dir=config.get("cache_dir"),
            token=config.get("token"),
            load_dataset_kwargs=config.get("load_dataset_kwargs", {}),
            timestamp_field=data_source.timestamp_field or None,
            created_timestamp_column=data_source.created_timestamp_column or None,
            field_mapping=dict(data_source.field_mapping),
            description=data_source.description,
            owner=data_source.owner,
            tags=dict(data_source.tags),
        )

    def _to_proto_impl(self) -> DataSourceProto:
        """Serialize the data source to the Feast registry."""
        config: Dict[str, Any] = {
            "path": self.path,
            "split": self.split,
            "streaming": self.streaming,
        }

        if self.config_name:
            config["config_name"] = self.config_name

        if self.revision:
            config["revision"] = self.revision

        if self.data_dir:
            config["data_dir"] = self.data_dir

        if self.data_files:
            config["data_files"] = self.data_files

        if self.storage_options:
            config["storage_options"] = self.storage_options

        if self.cache_dir:
            config["cache_dir"] = self.cache_dir

        if self.token:
            config["token"] = self.token

        if self.load_dataset_kwargs:
            config["load_dataset_kwargs"] = self.load_dataset_kwargs

        configuration_bytes = json.dumps(config).encode("utf-8")

        data_source_proto = DataSourceProto(
            type=DataSourceProto.SourceType.CUSTOM_SOURCE,
            data_source_class_type=_DATA_SOURCE_CLASS_TYPE,
            name=self.name,
            description=self.description,
            owner=self.owner,
            timestamp_field=self.timestamp_field or "",
            created_timestamp_column=self.created_timestamp_column or "",
            field_mapping=self.field_mapping,
            tags=self.tags,
        )

        data_source_proto.custom_options.configuration = configuration_bytes

        return data_source_proto

    # ------------------------------------------------------------------
    # Equality / Hashing / Representation
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare two data sources by their load configuration."""
        if not isinstance(other, HFDatasetSource):
            return False

        return (
            self.path == other.path
            and self.config_name == other.config_name
            and self.split == other.split
            and self.revision == other.revision
            and self.data_dir == other.data_dir
            and self.streaming == other.streaming
            and self.storage_options == other.storage_options
            and self.data_files == other.data_files
        )

    def __hash__(self):
        return super().__hash__()

    # ------------------------------------------------------------------
    # Utility Helpers
    # ------------------------------------------------------------------


    @staticmethod
    def source_datatype_to_feast_value_type():
        return hf_to_feast_value_type

    @staticmethod
    def _feature_dtype_to_string(feature: Any) -> str:
            
        from datasets import (
            Array2D,
            Array3D,
            Array4D,
            Array5D,
            Audio,
            ClassLabel,
            Image,
            Sequence,
            Translation,
            TranslationVariableLanguages,
            Value,
            Video,
        )
        if isinstance(feature, Value):
            return feature.dtype

        if isinstance(feature, ClassLabel):
            return "int64"

        if isinstance(feature, Sequence):
            return f"array<{HFDatasetSource._feature_dtype_to_string(feature.feature)}>"

        if isinstance(feature, Image):
            return "binary"

        if isinstance(feature, Audio):
            return "binary"

        if isinstance(feature, Video):
            return "binary"

        if isinstance(feature, Translation):
            return "string"

        if isinstance(feature, TranslationVariableLanguages):
            return "string"

        if isinstance(feature, Array2D):
            return f"array<{feature.dtype}>"

        if isinstance(feature, Array3D):
            return f"array<{feature.dtype}>"

        if isinstance(feature, Array4D):
            return f"array<{feature.dtype}>"

        if isinstance(feature, Array5D):
            return f"array<{feature.dtype}>"

        # Struct-like nested Features
        if isinstance(feature, dict):
            return "struct"

        return str(feature)

    def get_table_column_names_and_types(self, config):
        from datasets import load_dataset

        ds = load_dataset(
            path=self.path,
            name=self.config_name,
            split=self.split,
            revision=self.revision,
            streaming=self.streaming,
            data_dir=self.data_dir,
            data_files=self.data_files,
            cache_dir=self.cache_dir,
            storage_options=self.storage_options,
            token=self.token,
            **self.load_dataset_kwargs,
        )

        return (
            (name, self._feature_dtype_to_string(feature))
            for name, feature in ds.features.items()
        )

    def get_table_query_string(self) -> str:
        """Return a concise identifier for this data source."""
        components = [self.path]

        if self.split:
            components.append(f"split={self.split}")

        if self.revision:
            components.append(f"revision={self.revision}")

        return "::".join(components)

    # ------------------------------------------------------------------
    # Schema Inference (Optional Utility)
    # ------------------------------------------------------------------

    def infer_features(self) -> Dict[str, ValueType]:
        """Infer dataset features by loading the dataset and mapping dtypes to Feast types."""
        from datasets import load_dataset

        dataset = load_dataset(
            path=self.path,
            name=self.config_name,
            split=self.split,
            revision=self.revision,
            data_dir=self.data_dir,
            data_files=self.data_files,
            streaming=self.streaming,
            storage_options=self.storage_options,
            cache_dir=self.cache_dir,
            token=self.token,
            **self.load_dataset_kwargs,
        )

        return {
            name: self.source_datatype_to_feast_value_type()(
                self._feature_dtype_to_string(feature)
            )
            for name, feature in dataset.features.items()
        }
