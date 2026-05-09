from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
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

OUTPUT_CSV_PATH = OUTPUT_DIR / "ams_coreconcept_annotations_gpt_output.csv"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "ams_coreconcept_annotations_gpt_output.jsonl"


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

    df = pd.read_csv(csv_path)

    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep=";")

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

    This prevents pandas errors like:
    Invalid value '0.95' for dtype 'str'
    """

    if "GPTConfidence" not in df.columns:
        df["GPTConfidence"] = pd.Series([pd.NA] * len(df), dtype="Float64")
    else:
        df["GPTConfidence"] = (
            pd.to_numeric(df["GPTConfidence"], errors="coerce")
            .astype("Float64")
        )

    if "GPTReasoningSummary" not in df.columns:
        df["GPTReasoningSummary"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["GPTReasoningSummary"] = df["GPTReasoningSummary"].astype("string")

    if "GPTError" not in df.columns:
        df["GPTError"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["GPTError"] = df["GPTError"].astype("string")

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
    Sends all CSV columns except Geometry and Entity,
    plus a compact dataset summary.
    """
    metadata = row.drop(labels=["Geometry", "Entity"], errors="ignore").to_dict()

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
- Use PlainVectorRegion only when the polygons are selected regions, isolated areas, project areas, protected areas, buffers, zones, or patches that do not together partition the study area.

Entity interpretation:
- Use NetworkDS when the dataset represents routes, roads, streets, tram/metro lines, transport links, utility lines, pipelines, cables, waterways, flows, or connected line infrastructure.
- Do not classify network lines as ObjectDS merely because individual line features have names or IDs.
- Use ObjectDS for identifiable real-world objects, places, facilities, assets, or managed entities.
- Use PatchDS for bounded selected areas where the main meaning is an occurrence, patch, zone, or spatial extent of a phenomenon.
- Do not use PatchDS merely because the feature is a polygon or a designated area.
- Use AmountDS when the main meaning is a count, total, capacity, density, percentage, rate, intensity, forecast, volume, yield, or other numeric magnitude.
- Use LatticeDS when the dataset consists of reporting or statistical units used to aggregate attributes, such as neighbourhoods, districts, postcode zones, census areas, or grid cells.
- Use CoverageDS when the dataset provides a full-area thematic classification, such as land use, zoning, function categories, or homogeneous classes across the study area.
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
# Main
# -----------------------------
def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    client = OpenAI()

    decision_tree = load_text_file(DECISION_TREE_PATH)
    df = load_annotations(CSV_PATH)
    df = ensure_output_columns(df)

    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as jsonl_file:
        for idx, row in df.iterrows():
            kaartlaag = str(row["Kaartlaag"]).strip()

            if not kaartlaag or kaartlaag.lower() == "nan":
                print(f"Skipping row {idx}: empty Kaartlaag")
                df.at[idx, "GPTError"] = "Empty Kaartlaag"
                df.to_csv(OUTPUT_CSV_PATH, index=False)
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

                df.at[idx, "Geometry"] = geometry
                df.at[idx, "Entity"] = entity
                df.at[idx, "GPTConfidence"] = confidence
                df.at[idx, "GPTReasoningSummary"] = reasoning_summary
                df.at[idx, "GPTError"] = pd.NA

                result = {
                    "row_index": int(idx),
                    "kaartlaag": kaartlaag,
                    "status": "success",
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

                df.to_csv(OUTPUT_CSV_PATH, index=False)

                print(
                    f"Processed row {idx}: {kaartlaag} "
                    f"-> Geometry={geometry}, Entity={entity}, Confidence={confidence}"
                )

                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                error_message = str(e)

                error_result = {
                    "row_index": int(idx),
                    "kaartlaag": kaartlaag,
                    "status": "error",
                    "error": error_message,
                }

                jsonl_file.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                jsonl_file.flush()

                df.at[idx, "GPTError"] = str(error_message)
                df.to_csv(OUTPUT_CSV_PATH, index=False)

                print(f"Error on row {idx} / {kaartlaag}: {error_message}")

    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Done.")
    print(f"CSV output written to: {OUTPUT_CSV_PATH}")
    print(f"JSONL log written to: {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    main()