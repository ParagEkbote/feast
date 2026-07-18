from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from feast.data_source import DataSource
from feast.feature_view import FeatureView
from feast.infra.offline_stores.offline_store import OfflineStore, RetrievalJob
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.saved_dataset import SavedDatasetStorage

from .hf_datasets_source import HFDatasetSource

# =============================================================================
# Retrieval Job
# =============================================================================


class HFDatasetsRetrievalJob(RetrievalJob):
    """RetrievalJob implementation backed by Hugging Face Datasets.

    Wraps a lazily-evaluated Arrow table and provides:
        - the standard Feast RetrievalJob interface (``to_df``, ``to_arrow``)
        - HF-native export helpers (``to_hf_dataset``, ``save_to_disk``,
          ``to_parquet``)
        - Hugging Face Hub publishing (``push_to_hub``) with automatic
          dataset card generation
        - remote-storage export for scalable batch materialization
    """

    def __init__(
        self,
        evaluation_function,
        full_feature_names: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a lazy historical retrieval job.

        Args:
            evaluation_function: Zero-argument callable that executes the
                retrieval query and returns a ``pyarrow.Table``. Only invoked
                once, on first access, and cached thereafter.
            full_feature_names: Whether feature columns are prefixed with
                their feature view name (``<view>__<feature>``).
            metadata: Optional lineage metadata (feature refs, project,
                retrieval timestamp, etc.) attached to this job.
        """
        self._evaluation_function = evaluation_function
        self._full_feature_names = full_feature_names
        self._metadata = metadata or {}

        self._arrow_table: Optional[pa.Table] = None

    # -------------------------------------------------------------------------
    # Internal Evaluation
    # -------------------------------------------------------------------------

    def _evaluate_arrow(self) -> pa.Table:
        """Run the evaluation function once and cache the result.

        Returns:
            The retrieved data as a ``pyarrow.Table``.
        """
        if self._arrow_table is None:
            self._arrow_table = self._evaluation_function()

        return self._arrow_table

    # -------------------------------------------------------------------------
    # Feast RetrievalJob API
    # -------------------------------------------------------------------------

    def to_arrow(self) -> pa.Table:
        """Return the retrieval result as a ``pyarrow.Table``.

        Returns:
            The retrieved data as a ``pyarrow.Table``.
        """
        return self._evaluate_arrow()

    def to_df(self) -> pd.DataFrame:
        """Return the retrieval result as a pandas DataFrame.

        Returns:
            The retrieved data as a ``pandas.DataFrame``.
        """
        return self.to_arrow().to_pandas()

    def to_remote_storage(self, path: str, **kwargs: Any) -> List[str]:
        """Write the retrieval result to a remote storage location.

        Supports scalable batch materialization by allowing a custom
        Materialization Engine to distribute the reading and writing of
        offline store records to blob storage. Writes a single Parquet
        file via ``pyarrow.parquet``, which supports any filesystem
        registered with ``fsspec`` (e.g. local paths, ``s3://``, ``gs://``).

        Args:
            path: Destination path or URI to write the Parquet file to.
            **kwargs: Additional keyword arguments forwarded to
                ``pyarrow.parquet.write_table``.

        Returns:
            A list containing the single output path written.

        Note:
            This is a minimal, single-file implementation. For very large
            materializations, consider chunked/partitioned writes instead.
        """
        table = self.to_arrow()
        pq.write_table(table, path, **kwargs)
        return [path]

    # -------------------------------------------------------------------------
    # HF-native Exports
    # -------------------------------------------------------------------------

    def to_hf_dataset(self) -> Dataset:
        """Export the retrieval result as a Hugging Face ``Dataset``.

        Returns:
            The retrieved data as a ``datasets.Dataset``.
        """
        return Dataset(self.to_arrow())

    def save_to_disk(self, path: str) -> None:
        """Save the retrieval result to disk as an HF dataset artifact.

        Args:
            path: Local directory path to save the dataset to.
        """
        dataset = self.to_hf_dataset()
        dataset.save_to_disk(path)

    def to_parquet(self, path: str) -> None:
        """Export the retrieval result to a Parquet file.

        Args:
            path: Destination path for the Parquet file.
        """
        self.to_hf_dataset().to_parquet(path)

    # -------------------------------------------------------------------------
    # Dataset Card Generation
    # -------------------------------------------------------------------------

    def generate_dataset_card(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        license: str = "apache-2.0",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Generate a Hugging Face dataset card (README.md) for this result.

        Args:
            title: Dataset card title. Defaults to a generic Feast title.
            description: Dataset card description. Defaults to a generic
                description referencing Feast historical retrieval.
            license: SPDX license identifier for the dataset card metadata.
            tags: Tags to include in the dataset card metadata. Defaults to
                ``["feast", "feature-store", "datasets"]``.

        Returns:
            The rendered dataset card as a Markdown string.
        """
        dataset = self.to_hf_dataset()

        schema_lines = []

        for column_name, feature in dataset.features.items():
            schema_lines.append(f"| {column_name} | {feature} |")

        schema_markdown = "\n".join(schema_lines)

        feature_refs = self._metadata.get("feature_refs", [])
        retrieval_timestamp = datetime.now(UTC).isoformat()

        card = f"""
---
license: {license}
tags:
{os.linesep.join([f'  - {tag}' for tag in (tags or ['feast', 'feature-store', 'datasets'])])}
---

# {title or "Feast Feature Retrieval Dataset"}

{description or "Generated using Feast historical feature retrieval."}

## Retrieval Metadata

- Retrieval timestamp: {retrieval_timestamp}
- Number of feature refs: {len(feature_refs)}

## Feature References

{os.linesep.join([f"- {ref}" for ref in feature_refs])}

## Dataset Schema

| Column | Type |
|--------|------|
{schema_markdown}

## Generated By

- Feast
- Hugging Face Datasets
- HFDatasetsOfflineStore
"""

        return card.strip()

    # -------------------------------------------------------------------------
    # Hub Publishing
    # -------------------------------------------------------------------------

    def push_to_hub(
        self,
        repo_id: str,
        split: str = "train",
        private: bool = False,
        token: Optional[str] = None,
        generate_card: bool = True,
        card_title: Optional[str] = None,
        card_description: Optional[str] = None,
        card_license: str = "apache-2.0",
        card_tags: Optional[List[str]] = None,
        exist_ok: bool = True,
        **kwargs: Any,
    ) -> None:
        """Push the retrieval result to the Hugging Face Hub.

        Creates the destination dataset repo if it doesn't exist, optionally
        generates and uploads a dataset card, and uploads the dataset
        artifact.

        Args:
            repo_id: Target Hub repo ID, e.g. ``"username/dataset-name"``.
            split: Split name to associate with the uploaded data.
            private: Whether to create the repo as private.
            token: HF auth token. Falls back to the environment/cached
                token if not provided.
            generate_card: Whether to generate and upload a README.md
                dataset card alongside the data.
            card_title: Title for the generated dataset card.
            card_description: Description for the generated dataset card.
            card_license: License identifier for the generated dataset card.
            card_tags: Tags for the generated dataset card.
            exist_ok: Whether to silently reuse an existing repo instead of
                raising an error.
            **kwargs: Reserved for future use.
        """
        dataset = self.to_hf_dataset()

        api = HfApi(token=token)

        # ---------------------------------------------------------------------
        # Create Repository
        # ---------------------------------------------------------------------

        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=exist_ok,
        )

        # ---------------------------------------------------------------------
        # Generate Dataset Card
        # ---------------------------------------------------------------------

        readme_content = None

        if generate_card:
            readme_content = self.generate_dataset_card(
                title=card_title,
                description=card_description,
                license=card_license,
                tags=card_tags,
            )

        # ---------------------------------------------------------------------
        # Temporary Upload Directory
        # ---------------------------------------------------------------------

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "dataset")

            dataset.save_to_disk(dataset_path)

            # Write README.md
            if readme_content is not None:
                readme_path = os.path.join(tmpdir, "README.md")

                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme_content)

            # Upload Dataset Directory
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                repo_type="dataset",
            )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """Lineage metadata attached to this retrieval job.

        Returns:
            A dict of metadata such as feature refs, project name, and
            retrieval timestamp.
        """
        return self._metadata


# =============================================================================
# Offline Store Config
# =============================================================================


class HFDatasetsOfflineStoreConfig(FeastConfigBaseModel):
    """Configuration for the Hugging Face Datasets offline store.

    Referenced from a feature repo's ``feature_store.yaml`` under the
    ``offline_store`` key. Currently has no store-specific settings beyond
    the required ``type`` discriminator; per-dataset configuration (path,
    split, revision, etc.) lives on ``HFDatasetSource`` instead.

    Attributes:
        type: Discriminator identifying this config. Feast's
            ``get_offline_config_from_type`` resolves ``offline_store.type``
            from ``feature_store.yaml`` to find which Config class to
            import, but passes the *original string* through unchanged as
            this field's value — it does not rewrite a short alias to the
            fully qualified path first. So this field must accept whatever
            string is actually written in the yaml:
              - ``"hf_datasets"``: the short alias, resolved via a
                ``"hf_datasets"`` entry registered in
                ``OFFLINE_STORE_CLASS_FOR_TYPE`` (matches the convention
                used by every other contrib store, e.g.
                ``PostgreSQLOfflineStoreConfig.type = Literal["postgres"]``).
              - the fully qualified class path: works standalone, without
                registering anything in core Feast's ``repo_config.py``,
                since ``get_offline_store_type`` treats any string ending in
                ``"OfflineStore"`` as already resolved.
    """

    type: Literal[
        "hf_datasets",
        "feast.infra.offline_stores.contrib.hf_datasets_offline_store.hf_datasets.HFDatasetsOfflineStore",
    ] = "hf_datasets"


# =============================================================================
# Offline Store
# =============================================================================


class HFDatasetsOfflineStore(OfflineStore):
    """Feast OfflineStore implementation backed by Hugging Face Datasets.

    Execution pipeline for historical retrieval and materialization pulls:

        HF Dataset (Arrow-backed) -> pyarrow.Table
            -> DuckDB (date filter / PIT join / dedup)
            -> Arrow Result -> HF Dataset Artifact

    Every read path loads the underlying dataset once via
    ``datasets.load_dataset`` and reads its backing Arrow table directly
    through ``_load_arrow_table``, registering that table with an
    in-process DuckDB connection to perform filtering, joins, and
    deduplication in SQL. This avoids the ``Dataset -> pandas -> Arrow``
    round-trip: ``datasets.Dataset`` is already Arrow-backed, so converting
    to pandas first is a wasted extra copy.
    """

    @staticmethod
    def pull_latest_from_table_or_query(
        config: RepoConfig,
        data_source: DataSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        created_timestamp_column: Optional[str],
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        """Pull the latest feature values per entity for materialization.

        Invoked by ``feast materialize`` / ``feast materialize-incremental``.
        Filters rows to the ``[start_date, end_date]`` window on
        ``timestamp_field``, then deduplicates per entity (per
        ``join_key_columns``) keeping only the most recent row. When
        ``created_timestamp_column`` is present it is used as the primary
        tie-breaker for recency (falling back to ``timestamp_field``);
        otherwise ``timestamp_field`` alone determines recency.

        Args:
            config: The RepoConfig for the current feature store.
            data_source: The data source to pull from. Must be an
                ``HFDatasetSource``.
            join_key_columns: Entity join key column names to deduplicate on.
            feature_name_columns: Feature column names to select.
            timestamp_field: Event timestamp column used for filtering and
                recency ordering.
            created_timestamp_column: Optional created-timestamp column used
                as the primary tie-breaker when multiple rows share the same
                ``timestamp_field`` value.
            start_date: Inclusive lower bound on ``timestamp_field``.
            end_date: Inclusive upper bound on ``timestamp_field``.

        Returns:
            A ``RetrievalJob`` that lazily evaluates to the deduplicated
            Arrow table.

        Raises:
            TypeError: If ``data_source`` is not an ``HFDatasetSource``.
            NotImplementedError: If the data source is configured for
                streaming.
            ValueError: If ``timestamp_field`` is missing from the loaded
                dataset.
        """
        if not isinstance(data_source, HFDatasetSource):
            raise TypeError(f"Expected HFDatasetSource, got {type(data_source)}")

        if data_source.streaming:
            raise NotImplementedError("Streaming datasets are not yet supported.")

        def evaluate() -> pa.Table:
            return HFDatasetsOfflineStore._pull_latest(
                source=data_source,
                join_key_columns=join_key_columns,
                feature_name_columns=feature_name_columns,
                timestamp_field=timestamp_field,
                created_timestamp_column=created_timestamp_column,
                start_date=start_date,
                end_date=end_date,
            )

        return HFDatasetsRetrievalJob(
            evaluation_function=evaluate,
            metadata={
                "source_path": data_source.path,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

    @staticmethod
    def _load_arrow_table(dataset: Dataset) -> pa.Table:
        """Extract the backing ``pyarrow.Table`` from an already-loaded HF dataset.

        Accepts a loaded ``Dataset`` rather than a source, so callers load
        the dataset exactly once and pass it in here, instead of loading it
        twice (once for the dataset object, once for the Arrow table).

        Args:
            dataset: An already-loaded ``datasets.Dataset``.

        Returns:
            The underlying ``pyarrow.Table``.

        Note:
            ``datasets.Dataset`` is Arrow-backed; ``dataset.data`` (an
            ``InMemoryTable``/``MemoryMappedTable`` wrapper) exposes the raw
            table via ``.table`` in current ``datasets`` releases. The
            ``to_pandas()``-based fallback is a correctness safety net for
            older/unexpected internals, not the primary path, since it
            reintroduces the pandas round-trip this helper exists to avoid.
        """
        if hasattr(dataset, "data") and hasattr(dataset.data, "table"):
            return dataset.data.table

        if hasattr(dataset, "_data") and hasattr(dataset._data, "table"):
            return dataset._data.table

        # Fallback only: reintroduces a pandas round-trip, so it should
        # never be hit against a normally-loaded in-memory Dataset.
        return pa.Table.from_pandas(dataset.to_pandas())

    @staticmethod
    def _build_timestamp_predicates(
        timestamp_field: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> tuple[List[str], List[Any]]:
        predicates: List[str] = []
        params: List[Any] = []

        if start_date is not None:
            predicates.append(
                f"CAST({timestamp_field} AS TIMESTAMP) >= CAST(? AS TIMESTAMP)"
            )
            params.append(start_date)

        if end_date is not None:
            predicates.append(
                f"CAST({timestamp_field} AS TIMESTAMP) <= CAST(? AS TIMESTAMP)"
            )
            params.append(end_date)

        return predicates, params

    @staticmethod
    def _pull_latest(
        source: HFDatasetSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        created_timestamp_column: Optional[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pa.Table:
        """Load, filter, and deduplicate a dataset for materialization.

        Args:
            source: The HFDatasetSource to load data from.
            join_key_columns: Entity join key column names to deduplicate on.
            feature_name_columns: Feature column names to select.
            timestamp_field: Event timestamp column used for filtering and
                recency ordering.
            created_timestamp_column: Optional created-timestamp column used
                as the primary tie-breaker for recency.
            start_date: Inclusive lower bound on ``timestamp_field``.
            end_date: Inclusive upper bound on ``timestamp_field``.

        Returns:
            The deduplicated result as a ``pyarrow.Table``.

        Raises:
            ValueError: If ``timestamp_field`` is missing from the loaded
                dataset, or if ``join_key_columns`` is empty.
        """
        dataset = HFDatasetsOfflineStore._load_dataset(source)
        table = HFDatasetsOfflineStore._load_arrow_table(dataset)

        if timestamp_field not in table.column_names:
            raise ValueError(f"Timestamp field '{timestamp_field}' not found in dataset.")

        if not join_key_columns:
            raise ValueError("join_key_columns must be non-empty for pull_latest_from_table_or_query.")

        con = duckdb.connect()
        con.register("source_table", table)

        has_created_ts = bool(created_timestamp_column) and created_timestamp_column in table.column_names

        select_columns = list(dict.fromkeys(join_key_columns + feature_name_columns + [timestamp_field]))
        if has_created_ts:
            select_columns.append(created_timestamp_column)
        select_columns_sql = ", ".join(select_columns)

        order_by_sql = (
            f"{created_timestamp_column} DESC, {timestamp_field} DESC"
            if has_created_ts
            else f"{timestamp_field} DESC"
        )
        partition_by_sql = ", ".join(join_key_columns)

        predicates, params = HFDatasetsOfflineStore._build_timestamp_predicates(
            timestamp_field,
            start_date,
            end_date,
        )
        where_clause = f" WHERE {' AND '.join(predicates)}" if predicates else ""

        query = f"""
        SELECT {select_columns_sql}
        FROM (
            SELECT {select_columns_sql}
            FROM source_table{where_clause}
        )
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY {partition_by_sql}
            ORDER BY {order_by_sql}
        ) = 1
        """

        result_table = con.execute(query, params).to_arrow_table()

        return result_table

    @staticmethod
    def pull_all_from_table_or_query(
        config: RepoConfig,
        data_source: DataSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        created_timestamp_column: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> RetrievalJob:
        """Pull all rows within a date range, without deduplication.

        Used for ``SavedDataset`` creation and data quality monitoring
        validation, where every observed row in the window is needed rather
        than just the latest value per entity.

        Args:
            config: The RepoConfig for the current feature store.
            data_source: The data source to pull from. Must be an
                ``HFDatasetSource``.
            join_key_columns: Entity join key column names to select.
            feature_name_columns: Feature column names to select.
            timestamp_field: Event timestamp column used for filtering.
            created_timestamp_column: Optional created-timestamp column,
                included in the selected output columns when present.
            start_date: Inclusive lower bound on ``timestamp_field``.
            end_date: Inclusive upper bound on ``timestamp_field``.

        Returns:
            A ``RetrievalJob`` that lazily evaluates to the filtered,
            un-deduplicated Arrow table.

        Raises:
            TypeError: If ``data_source`` is not an ``HFDatasetSource``.
            NotImplementedError: If the data source is configured for
                streaming.
            ValueError: If ``timestamp_field`` is missing from the loaded
                dataset, or if ``start_date``/``end_date`` are not provided.
        """
        if not isinstance(data_source, HFDatasetSource):
            raise TypeError(f"Expected HFDatasetSource, got {type(data_source)}")

        if data_source.streaming:
            raise NotImplementedError("Streaming datasets are not yet supported.")

        def evaluate() -> pa.Table:
            return HFDatasetsOfflineStore._pull_all(
                source=data_source,
                join_key_columns=join_key_columns,
                feature_name_columns=feature_name_columns,
                timestamp_field=timestamp_field,
                created_timestamp_column=created_timestamp_column,
                start_date=start_date,
                end_date=end_date,
            )

        return HFDatasetsRetrievalJob(
            evaluation_function=evaluate,
            metadata={
                "source_path": data_source.path,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            },
        )

    @staticmethod
    def _pull_all(
        source: HFDatasetSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        created_timestamp_column: Optional[str] = None,
    ) -> pa.Table:
        """Load and date-filter a dataset without deduplication.

        Args:
            source: The HFDatasetSource to load data from.
            join_key_columns: Entity join key column names to select.
            feature_name_columns: Feature column names to select.
            timestamp_field: Event timestamp column used for filtering.
            start_date: Inclusive lower bound on ``timestamp_field``.
            end_date: Inclusive upper bound on ``timestamp_field``.
            created_timestamp_column: Optional created-timestamp column,
                included in the selected output columns when present.

        Returns:
            The filtered result as a ``pyarrow.Table``.

        Raises:
            ValueError: If ``timestamp_field`` is missing from the loaded
                dataset.
        """
        dataset = HFDatasetsOfflineStore._load_dataset(source)
        table = HFDatasetsOfflineStore._load_arrow_table(dataset)

        if timestamp_field not in table.column_names:
            raise ValueError(f"Timestamp field '{timestamp_field}' not found in dataset.")

        con = duckdb.connect()
        con.register("source_table", table)

        select_columns = list(dict.fromkeys(join_key_columns + feature_name_columns + [timestamp_field]))
        if created_timestamp_column and created_timestamp_column in table.column_names:
            select_columns.append(created_timestamp_column)
        select_columns_sql = ", ".join(select_columns)

        predicates, params = HFDatasetsOfflineStore._build_timestamp_predicates(
            timestamp_field,
            start_date,
            end_date,
        )
        where_clause = f" WHERE {' AND '.join(predicates)}" if predicates else ""

        query = f"""
        SELECT {select_columns_sql}
        FROM source_table{where_clause}
        """

        result_table = con.execute(query, params).to_arrow_table()

        return result_table

    # -------------------------------------------------------------------------
    # Historical Feature Retrieval
    # -------------------------------------------------------------------------

    @staticmethod
    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Union[pd.DataFrame, str],
        registry,
        project: str,
        full_feature_names: bool = False,
    ) -> RetrievalJob:
        """Perform a point-in-time correct join of features onto an entity df.

        This is the main entry point used by
        ``FeatureStore.get_historical_features()`` to retrieve training data.

        Args:
            config: The RepoConfig for the current feature store.
            feature_views: Feature views referenced by ``feature_refs``.
            feature_refs: Feature references to retrieve, in
                ``<feature_view>:<feature>`` form.
            entity_df: Entity dataframe (or path to a Parquet file containing
                one) with entity keys and event timestamps to join features
                onto.
            registry: The Feast registry.
            project: The feature store project name.
            full_feature_names: Whether to prefix feature columns with their
                feature view name in the output.

        Returns:
            A ``RetrievalJob`` that lazily evaluates to the joined Arrow
            table.
        """
        def evaluate() -> pa.Table:
            return HFDatasetsOfflineStore._run_historical_retrieval(
                feature_views=feature_views,
                entity_df=entity_df,
                full_feature_names=full_feature_names,
            )

        return HFDatasetsRetrievalJob(
            evaluation_function=evaluate,
            full_feature_names=full_feature_names,
            metadata={
                "feature_refs": feature_refs,
                "project": project,
                "num_feature_views": len(feature_views),
                "retrieval_timestamp": datetime.now(UTC).isoformat(),
            },
        )

    # -------------------------------------------------------------------------
    # Core Retrieval Engine
    # -------------------------------------------------------------------------

    @staticmethod
    def _run_historical_retrieval(
        feature_views: List[FeatureView],
        entity_df: Union[pd.DataFrame, str],
        full_feature_names: bool = False,
    ) -> pa.Table:
        """Execute the point-in-time join across all feature views using DuckDB.

        The entity dataframe is user-supplied pandas (per the
        ``get_historical_features`` contract) and is registered with DuckDB
        as-is. Each feature view's HF dataset, however, is loaded once and
        registered via its backing Arrow table rather than a
        ``to_pandas()`` conversion, consistent with ``_pull_latest`` /
        ``_pull_all``.

        Args:
            feature_views: Feature views to join onto the entity dataframe.
            entity_df: Entity dataframe, or a path to a Parquet file
                containing one.
            full_feature_names: Whether to prefix feature columns with their
                feature view name in the output.

        Returns:
            The joined result as a ``pyarrow.Table``.

        Raises:
            TypeError: If ``entity_df`` is not a DataFrame or Parquet path,
                or if a feature view's batch source is not an
                ``HFDatasetSource``.
            NotImplementedError: If a feature view's source is configured
                for streaming.
            ValueError: If a source doesn't define ``timestamp_field``, or
                ``timestamp_field`` is missing from the loaded dataset.
        """
        if isinstance(entity_df, str):
            entity_df = pd.read_parquet(entity_df)

        if not isinstance(entity_df, pd.DataFrame):
            raise TypeError("entity_df must be a pandas DataFrame or parquet path.")

        con = duckdb.connect()
        con.register("entity_df", entity_df)

        result_table: Optional[pa.Table] = None

        # ---------------------------------------------------------------------
        # Process Feature Views
        # ---------------------------------------------------------------------

        for feature_view in feature_views:
            source = feature_view.batch_source

            if not isinstance(source, HFDatasetSource):
                raise TypeError(f"Expected HFDatasetSource, got {type(source)}")

            # Streaming not yet supported
            if source.streaming:
                raise NotImplementedError("Streaming datasets are not yet supported.")

            dataset = HFDatasetsOfflineStore._load_dataset(source)
            feature_table = HFDatasetsOfflineStore._load_arrow_table(dataset)

            # -----------------------------------------------------------------
            # Timestamp Validation
            # -----------------------------------------------------------------

            if source.timestamp_field is None:
                raise ValueError(
                    f"HFDatasetSource for FeatureView {feature_view.name} must define timestamp_field."
                )

            timestamp_field = source.timestamp_field

            if timestamp_field not in feature_table.column_names:
                raise ValueError(f"Timestamp field '{timestamp_field}' not found in dataset.")

            # -----------------------------------------------------------------
            # Register Feature Table
            # -----------------------------------------------------------------

            feature_table_name = f"{feature_view.name}_features"

            con.register(feature_table_name, feature_table)

            # -----------------------------------------------------------------
            # Join Keys
            # -----------------------------------------------------------------

            entity_keys = [
                field.name
                for field in feature_view.entity_columns
            ]

            join_conditions = [f"e.{key} = f.{key}" for key in entity_keys]

            join_condition_sql = " AND ".join(join_conditions)

            # -----------------------------------------------------------------
            # Feature Projection
            # -----------------------------------------------------------------

            feature_columns = []

            for field in feature_view.schema:
                if field.name in entity_keys:
                    continue

                if field.name == timestamp_field:
                    continue

                alias = field.name

                if full_feature_names:
                    alias = f"{feature_view.name}__{field.name}"

                feature_columns.append(f"f.{field.name} AS {alias}")

            feature_column_sql = ", ".join(feature_columns)

            # -----------------------------------------------------------------
            # Point-in-Time Join
            # -----------------------------------------------------------------

            query = f"""
            SELECT
                e.*,
                {feature_column_sql}
            FROM entity_df e
            LEFT JOIN {feature_table_name} f
            ON
                {join_condition_sql}
                AND CAST(f.{timestamp_field} AS TIMESTAMP)
                    <=
                    CAST(e.event_timestamp AS TIMESTAMP)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY
                    CAST(e.event_timestamp AS TIMESTAMP),
                    {", ".join([f"e.{k}" for k in entity_keys])}
                ORDER BY CAST(f.{timestamp_field} AS TIMESTAMP) DESC
            ) = 1
            """

            result_table = con.execute(query).to_arrow_table()

            con.unregister("entity_df")
            con.register("entity_df", result_table)

        return result_table

    # -------------------------------------------------------------------------
    # Dataset Loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_dataset(source: HFDatasetSource) -> Dataset:
        """Load a Hugging Face dataset for the given source configuration.

        Args:
            source: The HFDatasetSource describing what to load.

        Returns:
            The loaded ``datasets.Dataset``.
        """
        return load_dataset(
            path=source.path,
            name=source.config_name,
            split=source.split,
            revision=source.revision,
            data_dir=source.data_dir,
            data_files=source.data_files,
            streaming=source.streaming,
            storage_options=source.storage_options,
            cache_dir=source.cache_dir,
            token=source.token,
            **source.load_dataset_kwargs,
        )

    # -------------------------------------------------------------------------
    # Optional Feast Interfaces
    # -------------------------------------------------------------------------

    @staticmethod
    def offline_write_batch(
        config: RepoConfig,
        feature_view: FeatureView,
        table,
        progress,
    ) -> None:
        """Write a pyarrow table directly to this feature view's batch source.

        Supports the Feast push API for offline writes. Not yet implemented.

        Args:
            config: The RepoConfig for the current feature store.
            feature_view: The feature view being written to.
            table: The pyarrow table of feature values to write.
            progress: Optional progress callback.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("offline_write_batch is not implemented yet.")

    @staticmethod
    def write_logged_features(
        config: RepoConfig,
        data,
        source: SavedDatasetStorage,
        logging_config,
        registry,
    ) -> None:
        """Write logged features for a SavedDataset to this offline store.

        Used internally to support feature logging for SavedDatasets. Not
        yet implemented.

        Args:
            config: The RepoConfig for the current feature store.
            data: A pyarrow table, or a path to a Parquet file, containing
                the logged feature data.
            source: The LoggingSource/SavedDatasetStorage destination.
            logging_config: The LoggingConfig describing how logging is set
                up.
            registry: The Feast registry.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("write_logged_features is not implemented yet.")

