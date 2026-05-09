from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_data"

CSV_PATH = INPUT_DIR / "ams_coreconcept_annotations.csv"
DATASETS_DIR = INPUT_DIR / "datasets"
DECISION_TREE_PATH = INPUT_DIR / "decision_tree.txt"

OUTPUT_DIR = BASE_DIR / "V2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = OUTPUT_DIR / "ams_coreconcept_annotations_gpt_output.csv"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "ams_coreconcept_annotations_gpt_output.jsonl"
CONFUSION_MATRIX_GEOMETRY_CSV_PATH = OUTPUT_DIR / "confusion_matrix_geometry.csv"
CONFUSION_MATRIX_ENTITY_CSV_PATH = OUTPUT_DIR / "confusion_matrix_entity.csv"
CONFUSION_MATRICES_JSON_PATH = OUTPUT_DIR / "confusion_matrices.json"
RUN_METRICS_JSON_PATH = OUTPUT_DIR / "run_metrics.json"
DATABASE_SCHEMA_PATH = BASE_DIR / "dashboard" / "db" / "schema.sql"


# -----------------------------
# Config
# -----------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.75"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
MAX_BACKOFF_SECONDS = float(os.getenv("MAX_BACKOFF_SECONDS", "60"))

MAX_SAMPLE_FEATURES = int(os.getenv("MAX_SAMPLE_FEATURES", "5"))
MAX_UNIQUE_VALUES = int(os.getenv("MAX_UNIQUE_VALUES", "10"))
MAX_PROPERTY_STRING_LENGTH = int(os.getenv("MAX_PROPERTY_STRING_LENGTH", "300"))


GEOMETRY_TYPES = [
    "PlainVectorRegion",
    "PointDS",
    "VectorTessellation",
    "LineDS",
]

ENTITY_TYPES = [
    "ObjectDS",
    "AmountDS",
    "ExistenceDS",
    "EventDS",
    "PatchDS",
    "LatticeDS",
    "ContourDS",
    "PointMeasuresDS",
    "CoverageDS",
]


ANNOTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "geometry": {
            "type": "string",
            "enum": GEOMETRY_TYPES,
        },
        "entity": {
            "type": "string",
            "enum": ENTITY_TYPES,
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning_summary": {
            "type": "string",
            "description": (
                "Brief explanation of the classification. "
                "Do not include hidden chain-of-thought."
            ),
        },
    },
    "required": [
        "geometry",
        "entity",
        "confidence",
        "reasoning_summary",
    ],
}


# -----------------------------
# File loading
# -----------------------------
def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path.read_text(encoding="utf-8")


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Loads the annotation CSV.

    Expected columns:
    Title, EnglishTitle, PageLink, MapLink, Kaartlaag, Geometry, Entity

    First tries comma-separated CSV.
    If that produces one column, retries semicolon-separated CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    expected_columns = [
        "Title",
        "EnglishTitle",
        "PageLink",
        "MapLink",
        "Kaartlaag",
        "Geometry",
        "Entity",
    ]

    missing = [col for col in expected_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


def load_dataset_json(kaartlaag: str) -> dict[str, Any] | list[Any]:
    """
    Loads input_data/datasets/{Kaartlaag}.json.
    """
    json_path = DATASETS_DIR / f"{kaartlaag}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found for Kaartlaag '{kaartlaag}': {json_path}"
        )

    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# DataFrame setup
# -----------------------------
def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures output columns exist with compatible dtypes.

    Geometry and Entity are the gold labels from the input CSV and are kept intact.
    GPTGeometry and GPTEntity hold model predictions used by confusion matrices.
    """

    string_columns = [
        "GPTGeometry",
        "GPTEntity",
        "GPTReasoningSummary",
        "GPTError",
    ]

    for column in string_columns:
        if column not in df.columns:
            df[column] = pd.Series([pd.NA] * len(df), dtype="string")
        else:
            df[column] = df[column].astype("string")

    if "GPTConfidence" not in df.columns:
        df["GPTConfidence"] = pd.Series([pd.NA] * len(df), dtype="Float64")
    else:
        df["GPTConfidence"] = (
            pd.to_numeric(df["GPTConfidence"], errors="coerce")
            .astype("Float64")
        )

    if "RunID" not in df.columns:
        df["RunID"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["RunID"] = df["RunID"].astype("string")

    if "RowIndex" not in df.columns:
        df.insert(0, "RowIndex", df.index.astype(int))
    else:
        df["RowIndex"] = pd.to_numeric(df["RowIndex"], errors="coerce").astype("Int64")

    return df


# -----------------------------
# JSON / GeoJSON summarization
# -----------------------------
def compact_value(value: Any, max_string_length: int = MAX_PROPERTY_STRING_LENGTH) -> Any:
    """
    Prevents huge strings, lists, dicts, or nested geometry from entering the prompt.
    """
    if isinstance(value, str):
        if len(value) > max_string_length:
            return value[:max_string_length] + "...[truncated]"
        return value

    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, list):
        return [compact_value(item, max_string_length) for item in value[:5]]

    if isinstance(value, dict):
        blocked_keys = {"coordinates", "geometry", "bbox"}

        compacted = {}
        for key, nested_value in list(value.items())[:15]:
            if str(key) in blocked_keys:
                compacted[str(key)] = "[removed]"
            else:
                compacted[str(key)] = compact_value(nested_value, max_string_length)

        return compacted

    return str(value)[:max_string_length]


def summarize_geojson_for_annotation(
    dataset_json: dict[str, Any],
    max_sample_features: int = MAX_SAMPLE_FEATURES,
    max_unique_values: int = MAX_UNIQUE_VALUES,
) -> dict[str, Any]:
    """
    Summarizes a GeoJSON-like dataset for LLM annotation.

    Keeps:
    - dataset type
    - feature count
    - geometry type counts
    - property keys
    - sample properties
    - numeric/categorical field summaries

    Removes:
    - coordinates
    - full geometry arrays
    - full feature list
    """

    features = dataset_json.get("features", [])

    summary: dict[str, Any] = {
        "dataset_type": dataset_json.get("type"),
        "top_level_keys": list(dataset_json.keys()),
        "feature_count": len(features) if isinstance(features, list) else None,
        "geometry_type_counts": {},
        "property_keys": [],
        "sample_properties": [],
        "field_summaries": {},
    }

    if not isinstance(features, list):
        return summary

    geometry_counter: Counter[str] = Counter()
    property_key_counter: Counter[str] = Counter()

    numeric_values: dict[str, list[float]] = defaultdict(list)
    categorical_values: dict[str, set[str]] = defaultdict(set)

    for feature in features:
        if not isinstance(feature, dict):
            continue

        geometry = feature.get("geometry")
        properties = feature.get("properties")

        if isinstance(geometry, dict):
            geometry_type = geometry.get("type")
            if geometry_type:
                geometry_counter[str(geometry_type)] += 1

        if not isinstance(properties, dict):
            continue

        property_key_counter.update(str(key) for key in properties.keys())

        if len(summary["sample_properties"]) < max_sample_features:
            summary["sample_properties"].append(
                {
                    str(key): compact_value(value)
                    for key, value in properties.items()
                }
            )

        for key, value in properties.items():
            key = str(key)

            if isinstance(value, bool) or value is None:
                continue

            if isinstance(value, (int, float)):
                numeric_values[key].append(float(value))
            else:
                if len(categorical_values[key]) < max_unique_values:
                    categorical_values[key].add(str(value))

    summary["geometry_type_counts"] = dict(geometry_counter)
    summary["property_keys"] = list(property_key_counter.keys())

    field_summaries: dict[str, Any] = {}

    for key, values in numeric_values.items():
        if not values:
            continue

        field_summaries[key] = {
            "kind": "numeric",
            "min": min(values),
            "max": max(values),
            "sample_values": values[:max_unique_values],
        }

    for key, values in categorical_values.items():
        field_summaries[key] = {
            "kind": "categorical",
            "sample_values": sorted(values)[:max_unique_values],
        }

    summary["field_summaries"] = field_summaries

    return summary


def summarize_generic_json_for_annotation(
    dataset_json: dict[str, Any] | list[Any],
    max_items: int = MAX_SAMPLE_FEATURES,
) -> dict[str, Any]:
    """
    Fallback summarizer for non-GeoJSON JSON files.
    """
    if isinstance(dataset_json, dict):
        return {
            "json_type": "dict",
            "top_level_keys": list(dataset_json.keys()),
            "sample": compact_value(dataset_json),
        }

    if isinstance(dataset_json, list):
        return {
            "json_type": "list",
            "item_count": len(dataset_json),
            "sample_items": [compact_value(item) for item in dataset_json[:max_items]],
        }

    return {
        "json_type": type(dataset_json).__name__,
        "value_preview": compact_value(dataset_json),
    }


def summarize_dataset_for_annotation(
    dataset_json: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    """
    Detects GeoJSON-like files and summarizes them.
    """
    if isinstance(dataset_json, dict):
        if dataset_json.get("type") == "FeatureCollection" and isinstance(
            dataset_json.get("features"), list
        ):
            return summarize_geojson_for_annotation(dataset_json)

    return summarize_generic_json_for_annotation(dataset_json)


# -----------------------------
# Prompt construction
# -----------------------------
def build_input_payload(
    row: pd.Series,
    dataset_json: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    """
    Sends all CSV columns except gold/prediction columns, plus a compact dataset summary.
    """
    hidden_columns = [
        "Geometry",
        "Entity",
        "GPTGeometry",
        "GPTEntity",
        "GPTConfidence",
        "GPTReasoningSummary",
        "GPTError",
        "RunID",
        "RowIndex",
    ]
    metadata = row.drop(labels=hidden_columns, errors="ignore").to_dict()

    return {
        "metadata": metadata,
        "dataset_summary": summarize_dataset_for_annotation(dataset_json),
    }


def build_messages(
    decision_tree: str,
    row: pd.Series,
    dataset_json: dict[str, Any] | list[Any],
) -> list[dict[str, str]]:
    payload = build_input_payload(row, dataset_json)

    system_instructions = f"""
You are an expert data annotation assistant.

Your task is to classify an Amsterdam map layer using exactly one Geometry type and exactly one Entity type.

You must follow the decision tree exactly.

Allowed Geometry labels:
{json.dumps(GEOMETRY_TYPES, ensure_ascii=False)}

Allowed Entity labels:
{json.dumps(ENTITY_TYPES, ensure_ascii=False)}

Decision tree:
{decision_tree}

Important interpretation rules:
- Do not classify only from the raw geometry type. First infer what phenomenon the dataset represents.
- The dataset_summary may omit raw coordinates on purpose.
- Geometry should be inferred from geometry_type_counts, metadata, layer name, attributes, and dataset description.

Geometry interpretation:
- Point and MultiPoint usually indicate PointDS.
- LineString and MultiLineString usually indicate LineDS.
- Polygon and MultiPolygon can indicate either PlainVectorRegion or VectorTessellation.
- Use VectorTessellation when polygons form, or are intended to form, a complete or near-complete partition of the study area.
- Typical VectorTessellation examples include neighbourhoods, districts, boroughs, postcode areas, census/statistical units, administrative areas, land-use maps, zoning maps, and area-wide categorical classifications.
- Use PlainVectorRegion only when the polygons are discrete bounded regions, patches, areas of interest, or footprints that do not partition the study area.

Entity interpretation:
- Use ObjectDS for discrete identifiable objects.
- Use AmountDS for counts, quantities, capacities, totals, rates, percentages, scores, or measured numeric values assigned to places/objects.
- Use CoverageDS when the layer represents a continuous spatial field or exhaustive thematic coverage across the study area.
- Use ExistenceDS when the main meaning is Boolean presence/absence of a phenomenon.
- Use EventDS when the main meaning is something that happened, occurs, changes, or unfolds in time.
- Use PointMeasuresDS for measurements of continuous phenomena sampled at point locations.
- Use ContourDS for isolines, interval bands, contour-like classes, or boundaries between value intervals.

Output rules:
- Return only valid JSON matching the required schema.
- Do not invent labels outside the allowed labels.
- Use the dataset summary and metadata as evidence.
- If the evidence is imperfect, choose the most likely valid label and lower the confidence.
- The reasoning_summary should be short and practical.
- In the reasoning_summary, mention the decisive rule used, such as tessellation, network, reporting unit, amount, object, patch, coverage, or event.
""".strip()

    user_input = f"""
Classify the following map layer.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    return [
        {
            "role": "system",
            "content": system_instructions,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]


# -----------------------------
# OpenAI call with retry/backoff
# -----------------------------
def calculate_backoff_seconds(attempt: int) -> float:
    base_delay = 2**attempt
    jitter = random.uniform(0, 1)
    return min(MAX_BACKOFF_SECONDS, base_delay + jitter)


def call_gpt(
    client: OpenAI,
    decision_tree: str,
    row: pd.Series,
    dataset_json: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    messages = build_messages(
        decision_tree=decision_tree,
        row=row,
        dataset_json=dataset_json,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=MODEL,
                input=messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "coreconcept_annotation",
                        "schema": ANNOTATION_SCHEMA,
                        "strict": True,
                    }
                },
            )

            return json.loads(response.output_text)

        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise

            sleep_seconds = calculate_backoff_seconds(attempt)

            print(
                f"Rate limit hit. Retrying in {sleep_seconds:.1f} seconds "
                f"after attempt {attempt + 1}/{MAX_RETRIES}."
            )

            time.sleep(sleep_seconds)

        except (APITimeoutError, APIConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                raise

            sleep_seconds = calculate_backoff_seconds(attempt)

            print(
                f"Transient API error: {type(e).__name__}. "
                f"Retrying in {sleep_seconds:.1f} seconds "
                f"after attempt {attempt + 1}/{MAX_RETRIES}."
            )

            time.sleep(sleep_seconds)

    raise RuntimeError("Unexpected retry loop exit.")


# -----------------------------
# Confusion matrices and metrics
# -----------------------------
def create_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def build_confusion_matrix(
    df: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    labels: list[str],
) -> pd.DataFrame:
    completed = df[df[predicted_column].notna() & df["GPTError"].isna()].copy()
    matrix = pd.crosstab(
        completed[actual_column],
        completed[predicted_column],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )

    observed_labels = sorted(
        set(completed[actual_column].dropna().astype(str))
        | set(completed[predicted_column].dropna().astype(str))
    )
    ordered_labels = labels + [label for label in observed_labels if label not in labels]

    matrix = matrix.reindex(index=ordered_labels, columns=ordered_labels, fill_value=0)
    matrix["__actual_total"] = matrix.sum(axis=1)
    predicted_totals = matrix.sum(axis=0).to_frame().T
    predicted_totals.index = ["__predicted_total"]

    return pd.concat([matrix, predicted_totals])


def dataframe_to_records(matrix: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for actual_label, row in matrix.iterrows():
        for predicted_label, count in row.items():
            records.append(
                {
                    "actual_label": str(actual_label),
                    "predicted_label": str(predicted_label),
                    "count": int(count),
                }
            )
    return records


def calculate_accuracy(
    df: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
) -> float | None:
    completed = df[df[predicted_column].notna() & df["GPTError"].isna()]
    if completed.empty:
        return None

    return float((completed[actual_column] == completed[predicted_column]).mean())



def normalize_database_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def apply_database_schema(cursor: Any) -> None:
    if not DATABASE_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Database schema not found: {DATABASE_SCHEMA_PATH}")

    schema_sql = DATABASE_SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in schema_sql.split(";"):
        statement = statement.strip()
        if statement:
            cursor.execute(statement)


def persist_run_to_database(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    matrices: dict[str, list[dict[str, Any]]],
    database_url: str,
) -> None:
    """
    Persists a completed run directly to PostgreSQL.

    This keeps the dashboard in sync without a separate artifact-import step.
    """
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            apply_database_schema(cursor)
            cursor.execute(
                """
                INSERT INTO annotation_runs (
                    id, model, started_at, completed_at, input_csv_path, output_dir,
                    total_rows, completed_rows, error_rows, geometry_accuracy,
                    entity_accuracy, mean_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    model = EXCLUDED.model,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    input_csv_path = EXCLUDED.input_csv_path,
                    output_dir = EXCLUDED.output_dir,
                    total_rows = EXCLUDED.total_rows,
                    completed_rows = EXCLUDED.completed_rows,
                    error_rows = EXCLUDED.error_rows,
                    geometry_accuracy = EXCLUDED.geometry_accuracy,
                    entity_accuracy = EXCLUDED.entity_accuracy,
                    mean_confidence = EXCLUDED.mean_confidence
                """,
                (
                    metrics["run_id"],
                    metrics["model"],
                    metrics.get("started_at"),
                    metrics.get("completed_at"),
                    metrics.get("input_csv_path"),
                    metrics.get("output_dir"),
                    metrics.get("total_rows", 0),
                    metrics.get("completed_rows", 0),
                    metrics.get("error_rows", 0),
                    metrics.get("geometry_accuracy"),
                    metrics.get("entity_accuracy"),
                    metrics.get("mean_confidence"),
                ),
            )

            cursor.execute(
                "DELETE FROM annotation_results WHERE run_id = %s",
                (metrics["run_id"],),
            )
            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO annotation_results (
                        run_id, row_index, title, english_title, page_link, map_link,
                        kaartlaag, gold_geometry, gold_entity, predicted_geometry,
                        predicted_entity, confidence, reasoning_summary, error
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        metrics["run_id"],
                        int(row["RowIndex"]),
                        normalize_database_value(row.get("Title")),
                        normalize_database_value(row.get("EnglishTitle")),
                        normalize_database_value(row.get("PageLink")),
                        normalize_database_value(row.get("MapLink")),
                        normalize_database_value(row.get("Kaartlaag")),
                        normalize_database_value(row.get("Geometry")),
                        normalize_database_value(row.get("Entity")),
                        normalize_database_value(row.get("GPTGeometry")),
                        normalize_database_value(row.get("GPTEntity")),
                        normalize_database_value(row.get("GPTConfidence")),
                        normalize_database_value(row.get("GPTReasoningSummary")),
                        normalize_database_value(row.get("GPTError")),
                    ),
                )

            cursor.execute(
                "DELETE FROM confusion_matrix_cells WHERE run_id = %s",
                (metrics["run_id"],),
            )
            for matrix_type, cells in matrices.items():
                for cell in cells:
                    cursor.execute(
                        """
                        INSERT INTO confusion_matrix_cells (
                            run_id, matrix_type, actual_label, predicted_label, count
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            metrics["run_id"],
                            matrix_type,
                            cell["actual_label"],
                            cell["predicted_label"],
                            cell["count"],
                        ),
                    )
        connection.commit()

def write_run_artifacts(
    df: pd.DataFrame,
    run_dir: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = run_dir / "annotations.csv"
    output_jsonl_path = run_dir / "annotations.jsonl"
    geometry_matrix_path = run_dir / "confusion_matrix_geometry.csv"
    entity_matrix_path = run_dir / "confusion_matrix_entity.csv"
    matrices_json_path = run_dir / "confusion_matrices.json"
    metrics_json_path = run_dir / "run_metrics.json"

    df.to_csv(output_csv_path, index=False)

    geometry_matrix = build_confusion_matrix(df, "Geometry", "GPTGeometry", GEOMETRY_TYPES)
    entity_matrix = build_confusion_matrix(df, "Entity", "GPTEntity", ENTITY_TYPES)

    geometry_matrix.to_csv(geometry_matrix_path)
    entity_matrix.to_csv(entity_matrix_path)

    total_rows = int(len(df))
    completed_rows = int((df["GPTGeometry"].notna() & df["GPTError"].isna()).sum())
    error_rows = int(df["GPTError"].notna().sum())
    mean_confidence = df["GPTConfidence"].dropna().mean()

    metrics = {
        **run_metadata,
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "error_rows": error_rows,
        "geometry_accuracy": calculate_accuracy(df, "Geometry", "GPTGeometry"),
        "entity_accuracy": calculate_accuracy(df, "Entity", "GPTEntity"),
        "mean_confidence": None if pd.isna(mean_confidence) else float(mean_confidence),
    }

    matrices = {
        "geometry": dataframe_to_records(geometry_matrix),
        "entity": dataframe_to_records(entity_matrix),
    }

    metrics_json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    matrices_json_path.write_text(
        json.dumps(matrices, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    shutil.copyfile(output_csv_path, OUTPUT_CSV_PATH)
    shutil.copyfile(output_jsonl_path, OUTPUT_JSONL_PATH)
    shutil.copyfile(geometry_matrix_path, CONFUSION_MATRIX_GEOMETRY_CSV_PATH)
    shutil.copyfile(entity_matrix_path, CONFUSION_MATRIX_ENTITY_CSV_PATH)
    shutil.copyfile(matrices_json_path, CONFUSION_MATRICES_JSON_PATH)
    shutil.copyfile(metrics_json_path, RUN_METRICS_JSON_PATH)

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        persist_run_to_database(df, metrics, matrices, database_url)
        print("Run saved to PostgreSQL for dashboard comparison.")
    else:
        print("DATABASE_URL is not set; dashboard database was not updated.")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Amsterdam core concept datasets with OpenAI and report performance."
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("RUN_ID"),
        help="Optional run identifier. Defaults to a UTC timestamp plus a random suffix.",
    )
    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    run_id = args.run_id or create_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_jsonl_path = run_dir / "annotations.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()

    client = OpenAI()

    decision_tree = load_text_file(DECISION_TREE_PATH)
    df = load_annotations(CSV_PATH)
    df = ensure_output_columns(df)
    df["RunID"] = run_id

    with run_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for idx, row in df.iterrows():
            kaartlaag = str(row["Kaartlaag"]).strip()

            if not kaartlaag or kaartlaag.lower() == "nan":
                print(f"Skipping row {idx}: empty Kaartlaag")
                df.at[idx, "GPTError"] = "Empty Kaartlaag"
                df.to_csv(run_dir / "annotations.csv", index=False)
                continue

            try:
                dataset_json = load_dataset_json(kaartlaag)

                dataset_summary = summarize_dataset_for_annotation(dataset_json)
                summary_length = len(json.dumps(dataset_summary, ensure_ascii=False))

                print(
                    f"Processing row {idx}: {kaartlaag} "
                    f"(dataset summary length: {summary_length:,} chars)"
                )

                annotation = call_gpt(
                    client=client,
                    decision_tree=decision_tree,
                    row=row,
                    dataset_json=dataset_json,
                )

                geometry = str(annotation["geometry"])
                entity = str(annotation["entity"])
                confidence = float(annotation["confidence"])
                reasoning_summary = str(annotation["reasoning_summary"])

                df.at[idx, "GPTGeometry"] = geometry
                df.at[idx, "GPTEntity"] = entity
                df.at[idx, "GPTConfidence"] = confidence
                df.at[idx, "GPTReasoningSummary"] = reasoning_summary
                df.at[idx, "GPTError"] = pd.NA

                result = {
                    "run_id": run_id,
                    "row_index": int(idx),
                    "kaartlaag": kaartlaag,
                    "status": "success",
                    "gold_annotation": {
                        "geometry": str(row["Geometry"]),
                        "entity": str(row["Entity"]),
                    },
                    "annotation": {
                        "geometry": geometry,
                        "entity": entity,
                        "confidence": confidence,
                        "reasoning_summary": reasoning_summary,
                    },
                    "dataset_summary_length_chars": summary_length,
                }

                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                jsonl_file.flush()

                df.to_csv(run_dir / "annotations.csv", index=False)

                print(
                    f"Processed row {idx}: {kaartlaag} "
                    f"-> Geometry={geometry}, Entity={entity}, Confidence={confidence}"
                )

                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                error_message = str(e)

                error_result = {
                    "run_id": run_id,
                    "row_index": int(idx),
                    "kaartlaag": kaartlaag,
                    "status": "error",
                    "error": error_message,
                }

                jsonl_file.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                jsonl_file.flush()

                df.at[idx, "GPTError"] = str(error_message)
                df.to_csv(run_dir / "annotations.csv", index=False)

                print(f"Error on row {idx} / {kaartlaag}: {error_message}")

    completed_at = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "model": MODEL,
        "started_at": started_at,
        "completed_at": completed_at,
        "input_csv_path": str(CSV_PATH.relative_to(BASE_DIR)),
        "output_dir": str(run_dir.relative_to(BASE_DIR)),
    }
    metrics = write_run_artifacts(df, run_dir, run_metadata)

    print("Done.")
    print(f"Run ID: {run_id}")
    print(f"CSV output written to: {run_dir / 'annotations.csv'}")
    print(f"JSONL log written to: {run_jsonl_path}")
    print(f"Geometry confusion matrix written to: {run_dir / 'confusion_matrix_geometry.csv'}")
    print(f"Entity confusion matrix written to: {run_dir / 'confusion_matrix_entity.csv'}")
    print(
        "Accuracy: "
        f"Geometry={metrics['geometry_accuracy']}, Entity={metrics['entity_accuracy']}"
    )


if __name__ == "__main__":
    main()
