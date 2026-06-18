from __future__ import annotations

import argparse
import importlib.util
import json
import os
import hashlib
import math
import random
import shutil
import subprocess
import time
import uuid
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


# -----------------------------
# Paths
# -----------------------------
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parents[2]
INPUT_DIR = BASE_DIR / "input_data"

CSV_PATH = INPUT_DIR / "ams_coreconcept_annotations.csv"
DATASETS_DIR = INPUT_DIR / "datasets"
DECISION_TREE_PATH = SCRIPT_PATH.parent / "decision_tree.txt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = OUTPUT_DIR / "ams_coreconcept_annotations_gpt_output.csv"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "ams_coreconcept_annotations_gpt_output.jsonl"
CONFUSION_MATRIX_GEOMETRY_CSV_PATH = OUTPUT_DIR / "confusion_matrix_geometry.csv"
CONFUSION_MATRIX_ENTITY_CSV_PATH = OUTPUT_DIR / "confusion_matrix_entity.csv"
CONFUSION_MATRICES_JSON_PATH = OUTPUT_DIR / "confusion_matrices.json"
RUN_METRICS_JSON_PATH = OUTPUT_DIR / "run_metrics.json"

load_dotenv(BASE_DIR / ".env")


# -----------------------------
# Config
# -----------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
)

REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.75"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
MAX_BACKOFF_SECONDS = float(os.getenv("MAX_BACKOFF_SECONDS", "60"))

MAX_SAMPLE_FEATURES = int(os.getenv("MAX_SAMPLE_FEATURES", "5"))
MAX_UNIQUE_VALUES = int(os.getenv("MAX_UNIQUE_VALUES", "10"))
MAX_PROPERTY_STRING_LENGTH = int(os.getenv("MAX_PROPERTY_STRING_LENGTH", "300"))

# Entity hint settings.
# These hints are weak evidence for the LLM, not automatic labels.
# They are mainly intended to reduce ObjectDS/PatchDS/AmountDS mistakes.
OBJECT_KEYWORDS = {
    "gebouw", "gebouwen", "building", "buildings",
    "pand", "panden", "bag", "adres", "address", "adressen",
    "monument", "monumenten", "heritage",
    "school", "schools", "scholen", "onderwijs",
    "park", "parken", "tuin", "garden",
    "brug", "bridge", "bruggen", "kade", "quay",
    "asset", "assets", "object", "objects", "objecten",
    "facility", "facilities", "voorziening", "voorzieningen",
    "project", "projects", "projecten", "site", "sites",
    "speeltuin", "playground", "sport", "sporthal", "cemetery", "begraafplaats",
    "station", "halte", "stop", "locatie", "location", "plaats",
}

PATCH_KEYWORDS = {
    "zone", "zones", "gebied", "gebieden", "area", "areas",
    "buffer", "buffers", "restriction", "restricted", "beperking", "beperkingen",
    "verbod", "verboden", "affected", "invloed", "influence",
    "risk", "risico", "risicogebied", "veiligheidszone",
    "policy", "beleid", "beleidsgebied", "bescherming", "beschermd",
    "protection", "protected", "milieuzone", "environmental", "aandachtsgebied",
    "overlast", "nuisance", "zoekgebied", "search area",
    "werkgebied", "werkingsgebied", "werkingsgebieden",
    "contourgebied", "impact", "hinder", "maatregel", "maatregelen",
}

REPORTING_UNIT_KEYWORDS = {
    "buurt", "buurten", "wijk", "wijken", "stadsdeel", "stadsdelen",
    "district", "districts", "postcode", "pc4", "census", "statistisch",
    "statistical", "administrative", "administratief", "grid", "raster", "cell", "cells",
}

COVERAGE_KEYWORDS = {
    "landgebruik", "land use", "zoning", "bestemming", "bestemmingsplan",
    "functie", "function", "functional", "classificatie", "classification",
    "dekking", "coverage", "bodemgebruik", "ground use",
}


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
    "NetworkDS",
]




ENTITY_HIERARCHY = {
    "ObjectDS": ["Entity", "DiscreteEntity", "ObjectDS"],
    "EventDS": ["Entity", "DiscreteEntity", "EventDS"],
    "PatchDS": ["Entity", "RegionEntity", "PatchDS"],
    "LatticeDS": ["Entity", "RegionEntity", "TessellatedRegionEntity", "LatticeDS"],
    "CoverageDS": ["Entity", "RegionEntity", "TessellatedRegionEntity", "CoverageDS"],
    "ContourDS": ["Entity", "RegionEntity", "ValueRegionEntity", "ContourDS"],
    "NetworkDS": ["Entity", "LineEntity", "NetworkDS"],
    "PointMeasuresDS": ["Entity", "FieldObservationEntity", "PointMeasuresDS"],
    "AmountDS": ["Entity", "AttributeEntity", "AmountDS"],
    "ExistenceDS": ["Entity", "AttributeEntity", "ExistenceDS"],
}

GEOMETRY_HIERARCHY = {
    "PointDS": ["Geometry", "PointDS"],
    "LineDS": ["Geometry", "LineDS"],
    "PlainVectorRegion": ["Geometry", "RegionGeometry", "PlainVectorRegion"],
    "VectorTessellation": ["Geometry", "RegionGeometry", "VectorTessellation"],
}

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    run_suffix: str
    model: str


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def get_provider_configs(provider: str) -> list[ProviderConfig]:
    providers = {
        "gpt": ProviderConfig("gpt", "_gpt", OPENAI_MODEL),
        "gemini": ProviderConfig("gemini", "_gemini", GEMINI_MODEL),
    }

    if provider == "both":
        return [providers["gpt"], providers["gemini"]]

    return [providers[provider]]


def provider_run_id(base_run_id: str, provider_config: ProviderConfig) -> str:
    if base_run_id.endswith(provider_config.run_suffix):
        return base_run_id
    return f"{base_run_id}{provider_config.run_suffix}"


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
    Title, EnglishTitle, PageLink, MapLink, Kaartlaag,
    Geometry, Entity, MapDescriptionTitle, MapDescription, MapDescriptionSource
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    except pd.errors.ParserError:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", engine="python")

    expected_columns = [
        "Title",
        "EnglishTitle",
        "PageLink",
        "MapLink",
        "Kaartlaag",
        "Geometry",
        "Entity",
        "MapDescription",
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



def percentile(values: list[float], q: float) -> float | None:
    """Returns a simple percentile without requiring numpy."""
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])

    q = max(0.0, min(1.0, q))
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(clean[int(position)])

    weight = position - lower
    return float(clean[lower] * (1 - weight) + clean[upper] * weight)


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def iter_coordinate_pairs(value: Any):
    """Yields coordinate pairs from a nested GeoJSON coordinate structure."""
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return

    if isinstance(value, list):
        for item in value:
            yield from iter_coordinate_pairs(item)


def ring_area_and_perimeter(ring: Any) -> tuple[float, float]:
    """
    Computes planar area and perimeter for one GeoJSON ring.

    The coordinates are usually lon/lat, so the absolute area is approximate.
    That is acceptable here because the values are used only as weak shape hints.
    """
    points = list(iter_coordinate_pairs(ring))
    if len(points) < 3:
        return 0.0, 0.0

    if points[0] != points[-1]:
        points.append(points[0])

    doubled_area = 0.0
    perimeter = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        doubled_area += x1 * y2 - x2 * y1
        perimeter += math.hypot(x2 - x1, y2 - y1)

    return abs(doubled_area) / 2.0, perimeter


def polygon_area_and_perimeter(coordinates: Any) -> tuple[float, float]:
    """Computes approximate area/perimeter for a GeoJSON Polygon."""
    if not isinstance(coordinates, list) or not coordinates:
        return 0.0, 0.0

    exterior_area, exterior_perimeter = ring_area_and_perimeter(coordinates[0])
    hole_area = 0.0
    hole_perimeter = 0.0
    for hole in coordinates[1:]:
        area, perimeter = ring_area_and_perimeter(hole)
        hole_area += area
        hole_perimeter += perimeter

    return max(0.0, exterior_area - hole_area), exterior_perimeter + hole_perimeter


def geometry_area_and_perimeter(geometry: dict[str, Any]) -> tuple[float, float]:
    """Computes approximate area/perimeter for Polygon and MultiPolygon geometries."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "Polygon":
        return polygon_area_and_perimeter(coordinates)

    if geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        total_area = 0.0
        total_perimeter = 0.0
        for polygon_coordinates in coordinates:
            area, perimeter = polygon_area_and_perimeter(polygon_coordinates)
            total_area += area
            total_perimeter += perimeter
        return total_area, total_perimeter

    return 0.0, 0.0


def summarize_polygon_shape_hints(features: list[Any]) -> dict[str, Any]:
    """
    Builds weak geometry-derived hints for region entity classification.

    These are useful for ObjectDS vs PatchDS but should never override clear metadata.
    """
    areas: list[float] = []
    perimeters: list[float] = []
    compactness_values: list[float] = []
    boundary_complexity_values: list[float] = []
    xs: list[float] = []
    ys: list[float] = []

    polygon_feature_count = 0

    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue

        geometry_type = geometry.get("type")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue

        polygon_feature_count += 1
        area, perimeter = geometry_area_and_perimeter(geometry)
        if area <= 0 or perimeter <= 0:
            continue

        areas.append(area)
        perimeters.append(perimeter)
        compactness_values.append((4.0 * math.pi * area) / (perimeter * perimeter))
        boundary_complexity_values.append(perimeter / math.sqrt(area))

        for x, y in iter_coordinate_pairs(geometry.get("coordinates")):
            xs.append(x)
            ys.append(y)

    if not areas:
        return {
            "available": False,
            "reason": "No Polygon or MultiPolygon geometries with usable coordinates were found.",
        }

    bbox_area = None
    if xs and ys:
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width > 0 and height > 0:
            bbox_area = width * height

    total_area = sum(areas)
    median_area = percentile(areas, 0.5)
    max_area = max(areas)
    median_compactness = percentile(compactness_values, 0.5)
    median_boundary_complexity = percentile(boundary_complexity_values, 0.5)

    max_to_median_area_ratio = None
    if median_area and median_area > 0:
        max_to_median_area_ratio = max_area / median_area

    approx_total_area_to_bbox_area = None
    if bbox_area and bbox_area > 0:
        approx_total_area_to_bbox_area = total_area / bbox_area

    shape_hint = "mixed_or_weak"
    reasons: list[str] = []

    if polygon_feature_count >= 20 and median_compactness is not None and median_compactness >= 0.25:
        shape_hint = "more_object_like"
        reasons.append("many relatively compact polygon features")

    if (
        (max_to_median_area_ratio is not None and max_to_median_area_ratio >= 10)
        or (median_compactness is not None and median_compactness < 0.15)
        or (median_boundary_complexity is not None and median_boundary_complexity >= 12)
    ):
        shape_hint = "more_patch_like"
        reasons.append("large area spread, low compactness, or high boundary complexity")

    if not reasons:
        reasons.append("shape metrics are not decisive")

    return {
        "available": True,
        "note": (
            "Weak geometry-derived hint only. Use to support ObjectDS vs PatchDS, "
            "but do not override clear metadata or decision-tree rules."
        ),
        "polygon_feature_count": polygon_feature_count,
        "area_min": round_or_none(min(areas)),
        "area_median": round_or_none(median_area),
        "area_max": round_or_none(max_area),
        "area_max_to_median_ratio": round_or_none(max_to_median_area_ratio),
        "compactness_median": round_or_none(median_compactness),
        "boundary_complexity_median": round_or_none(median_boundary_complexity),
        "approx_total_area_to_bbox_area": round_or_none(approx_total_area_to_bbox_area),
        "shape_hint": shape_hint,
        "shape_hint_reasons": reasons,
    }


def flatten_text_values(value: Any, max_items: int = 200) -> list[str]:
    """Collects textual values from nested structures for keyword hinting."""
    texts: list[str] = []

    def visit(item: Any) -> None:
        if len(texts) >= max_items:
            return
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            for key, nested_value in item.items():
                texts.append(str(key))
                visit(nested_value)
        elif isinstance(item, list):
            for nested_value in item:
                visit(nested_value)
        elif item is not None and not isinstance(item, (int, float, bool)):
            texts.append(str(item))

    visit(value)
    return texts[:max_items]


def keywords_found(text: str, keywords: set[str]) -> list[str]:
    normalized = text.lower()
    return sorted(keyword for keyword in keywords if keyword.lower() in normalized)


def build_entity_hints(metadata: dict[str, Any], dataset_summary: dict[str, Any]) -> dict[str, Any]:
    """
    Produces weak, inspectable hints for entity classification.

    The LLM still has to follow the decision tree. These hints are intended to make
    ambiguous ObjectDS/PatchDS/AmountDS cases easier to reason about.
    """
    text_parts: list[str] = []
    text_parts.extend(flatten_text_values(metadata))
    text_parts.extend(flatten_text_values(dataset_summary.get("property_keys", [])))
    text_parts.extend(flatten_text_values(dataset_summary.get("sample_properties", [])))
    text_parts.extend(flatten_text_values(dataset_summary.get("field_summaries", {})))
    corpus = " | ".join(text_parts).lower()

    object_matches = keywords_found(corpus, OBJECT_KEYWORDS)
    patch_matches = keywords_found(corpus, PATCH_KEYWORDS)
    reporting_unit_matches = keywords_found(corpus, REPORTING_UNIT_KEYWORDS)
    coverage_matches = keywords_found(corpus, COVERAGE_KEYWORDS)

    polygon_shape_hints = dataset_summary.get("polygon_shape_hints", {})
    shape_hint = None
    if isinstance(polygon_shape_hints, dict):
        shape_hint = polygon_shape_hints.get("shape_hint")

    object_score = len(object_matches)
    patch_score = len(patch_matches)

    if shape_hint == "more_object_like":
        object_score += 1
    elif shape_hint == "more_patch_like":
        patch_score += 1

    object_patch_hint = "mixed_or_weak"
    if object_score >= patch_score + 2:
        object_patch_hint = "more_object_like"
    elif patch_score >= object_score + 2:
        object_patch_hint = "more_patch_like"

    warning = (
        "Hints are weak evidence. Do not classify from hints alone. "
        "Clear metadata and the decision-tree branches take precedence."
    )

    if reporting_unit_matches:
        warning += " Reporting-unit terms may indicate LatticeDS before ObjectDS/PatchDS."
    if coverage_matches:
        warning += " Coverage/classification terms may indicate CoverageDS before ObjectDS/PatchDS."

    return {
        "object_patch_hint": object_patch_hint,
        "object_keyword_score": object_score,
        "patch_keyword_score": patch_score,
        "object_keywords_found": object_matches[:20],
        "patch_keywords_found": patch_matches[:20],
        "reporting_unit_keywords_found": reporting_unit_matches[:20],
        "coverage_keywords_found": coverage_matches[:20],
        "interpretation": (
            "ObjectDS means the feature is the mapped thing; "
            "PatchDS means the feature is an area where a condition, policy, restriction, "
            "or phenomenon applies."
        ),
        "warning": warning,
    }


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
    summary["polygon_shape_hints"] = summarize_polygon_shape_hints(features)

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

    dataset_summary = summarize_dataset_for_annotation(dataset_json)
    if isinstance(dataset_summary, dict):
        dataset_summary["entity_hints"] = build_entity_hints(metadata, dataset_summary)

    return {
        "metadata": metadata,
        "dataset_summary": dataset_summary,
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
- Geometry should be inferred from geometry_type_counts, metadata, layer name, MapDescription, attributes, and dataset description.
- MapDescription is a human-readable description of the map layer. Use it as important evidence for what phenomenon the map represents, but still follow the decision tree and dataset summary.

Geometry interpretation:
- Point and MultiPoint usually indicate PointDS.
- LineString and MultiLineString usually indicate LineDS.
- Polygon and MultiPolygon can indicate either PlainVectorRegion or VectorTessellation.
- Use VectorTessellation when polygons form, or are intended to form, a complete or near-complete partition of the study area.
- Typical VectorTessellation examples include neighbourhoods, districts, boroughs, postcode areas, census/statistical units, administrative areas, land-use maps, zoning maps, and area-wide categorical classifications.
- Use PlainVectorRegion only when the polygons are discrete bounded regions, patches, areas of interest, or footprints that do not partition the study area.

Entity interpretation:
- Use ObjectDS for discrete identifiable objects, places, facilities, assets, projects, or named managed entities.
- Use PatchDS for selected bounded zones, affected areas, restriction areas, policy areas, or phenomenon extents that do not represent discrete real-world objects and do not form a full-area classification.
- ObjectDS vs PatchDS distinction: ObjectDS means the feature is the mapped thing; PatchDS means the feature is an area where a condition, policy, phenomenon, or restriction applies.
- The dataset_summary may contain entity_hints and polygon_shape_hints. Treat these as weak supporting evidence only; never let them override clear metadata or the decision tree.
- Use AmountDS for counts, quantities, capacities, totals, rates, percentages, scores, or measured numeric values only when the numeric magnitude itself is the main mapped phenomenon.
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
    model: str,
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
                model=model,
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


def gemini_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": ANNOTATION_SCHEMA["properties"],
        "required": ANNOTATION_SCHEMA["required"],
    }


def parse_gemini_response(response_body: dict[str, Any]) -> dict[str, Any]:
    candidates = response_body.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response_body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise RuntimeError(f"Gemini returned an empty response: {response_body}")

    return json.loads(text)


def call_gemini(
    api_key: str,
    model: str,
    decision_tree: str,
    row: pd.Series,
    dataset_json: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    messages = build_messages(
        decision_tree=decision_tree,
        row=row,
        dataset_json=dataset_json,
    )
    system_message = next(message["content"] for message in messages if message["role"] == "system")
    user_message = next(message["content"] for message in messages if message["role"] == "user")

    request_body = {
        "systemInstruction": {
            "parts": [{"text": system_message}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": gemini_schema(),
        },
    }
    url = GEMINI_API_URL_TEMPLATE.format(model=model, api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            return parse_gemini_response(response_body)

        except urllib.error.HTTPError as e:
            retryable = e.code == 429 or 500 <= e.code < 600
            error_body = e.read().decode("utf-8", errors="replace")
            if attempt == MAX_RETRIES - 1 or not retryable:
                raise RuntimeError(f"Gemini API error {e.code}: {error_body}") from e

            sleep_seconds = calculate_backoff_seconds(attempt)
            print(
                f"Gemini API error {e.code}. Retrying in {sleep_seconds:.1f} seconds "
                f"after attempt {attempt + 1}/{MAX_RETRIES}."
            )
            time.sleep(sleep_seconds)

        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise

            sleep_seconds = calculate_backoff_seconds(attempt)
            print(
                f"Transient Gemini API error: {type(e).__name__}. "
                f"Retrying in {sleep_seconds:.1f} seconds "
                f"after attempt {attempt + 1}/{MAX_RETRIES}."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Unexpected Gemini retry loop exit.")


def call_model(
    provider_config: ProviderConfig,
    openai_client: OpenAI | None,
    gemini_api_key: str | None,
    decision_tree: str,
    row: pd.Series,
    dataset_json: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    if provider_config.name == "gpt":
        if openai_client is None:
            raise RuntimeError("OpenAI client is not configured.")
        return call_gpt(
            openai_client,
            provider_config.model,
            decision_tree,
            row,
            dataset_json,
        )

    if provider_config.name == "gemini":
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is missing. Add it to your .env file.")
        return call_gemini(
            gemini_api_key,
            provider_config.model,
            decision_tree,
            row,
            dataset_json,
        )

    raise ValueError(f"Unsupported provider: {provider_config.name}")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def get_git_metadata() -> dict[str, Any]:
    status = run_git_command(["status", "--porcelain"])
    return {
        "git_commit": run_git_command(["rev-parse", "HEAD"]),
        "git_branch": run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


def build_run_provenance(input_csv_path: Path) -> dict[str, Any]:
    return {
        **get_git_metadata(),
        "script_path": str(SCRIPT_PATH.relative_to(BASE_DIR)),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "decision_tree_path": str(DECISION_TREE_PATH.relative_to(BASE_DIR)),
        "decision_tree_sha256": file_sha256(DECISION_TREE_PATH),
        "input_csv_sha256": file_sha256(input_csv_path),
        "runtime_config": {
            "openai_model": OPENAI_MODEL,
            "gemini_model": GEMINI_MODEL,
            "request_delay_seconds": REQUEST_DELAY_SECONDS,
            "max_retries": MAX_RETRIES,
            "max_backoff_seconds": MAX_BACKOFF_SECONDS,
            "max_sample_features": MAX_SAMPLE_FEATURES,
            "max_unique_values": MAX_UNIQUE_VALUES,
            "max_property_string_length": MAX_PROPERTY_STRING_LENGTH,
        },
    }

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
    completed["__actual_for_scoring"] = completed.apply(
        lambda row: label_for_scoring(row[actual_column], row[predicted_column]),
        axis=1,
    )
    completed["__predicted_for_scoring"] = completed[predicted_column].apply(
        lambda value: "" if value is None or pd.isna(value) else str(value).strip()
    )

    matrix = pd.crosstab(
        completed["__actual_for_scoring"],
        completed["__predicted_for_scoring"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )

    observed_labels = sorted(
        set(completed["__actual_for_scoring"].dropna().astype(str))
        | set(completed["__predicted_for_scoring"].dropna().astype(str))
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


def completed_rows_for_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["GPTGeometry"].notna() & df["GPTEntity"].notna() & df["GPTError"].isna()]


def split_gold_labels(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def is_prediction_match(gold_value: Any, predicted_value: Any) -> bool:
    predicted = "" if predicted_value is None or pd.isna(predicted_value) else str(predicted_value).strip()
    if not predicted:
        return False
    gold_labels = split_gold_labels(gold_value)
    if not gold_labels:
        return False
    return predicted in gold_labels


def label_for_scoring(gold_value: Any, predicted_value: Any) -> str:
    gold_labels = split_gold_labels(gold_value)
    predicted = "" if predicted_value is None or pd.isna(predicted_value) else str(predicted_value).strip()
    if predicted and predicted in gold_labels:
        return predicted
    return gold_labels[0] if gold_labels else ""


def hierarchy_for_column(actual_column: str, predicted_column: str) -> dict[str, list[str]]:
    column_names = {actual_column, predicted_column}
    if column_names <= {"Entity", "GPTEntity"}:
        return ENTITY_HIERARCHY
    if column_names <= {"Geometry", "GPTGeometry"}:
        return GEOMETRY_HIERARCHY
    raise ValueError(f"No hierarchy configured for {actual_column!r}/{predicted_column!r}")


def best_gold_path_for_prediction(
    gold_value: Any,
    predicted_label: str,
    hierarchy: dict[str, list[str]],
) -> list[str]:
    gold_paths = [hierarchy[label] for label in split_gold_labels(gold_value) if label in hierarchy]
    if not gold_paths:
        return []

    predicted_path = set(hierarchy.get(predicted_label, []))
    if not predicted_path:
        return gold_paths[0]

    return max(gold_paths, key=lambda gold_path: len(set(gold_path) & predicted_path))


def hierarchical_f1_for_column(df: pd.DataFrame, actual_column: str, predicted_column: str) -> float | None:
    completed = df[df[predicted_column].notna() & df["GPTError"].isna()]
    if completed.empty:
        return None

    hierarchy = hierarchy_for_column(actual_column, predicted_column)
    true_positive_nodes = 0
    predicted_nodes = 0
    gold_nodes = 0

    for actual, predicted in zip(completed[actual_column], completed[predicted_column]):
        predicted_label = str(predicted).strip()
        predicted_path = set(hierarchy.get(predicted_label, []))
        gold_path = set(best_gold_path_for_prediction(actual, predicted_label, hierarchy))
        if not predicted_path or not gold_path:
            continue

        true_positive_nodes += len(gold_path & predicted_path)
        predicted_nodes += len(predicted_path)
        gold_nodes += len(gold_path)

    if predicted_nodes == 0 or gold_nodes == 0 or true_positive_nodes == 0:
        return 0.0

    hierarchical_precision = true_positive_nodes / predicted_nodes
    hierarchical_recall = true_positive_nodes / gold_nodes
    return float(
        2
        * hierarchical_precision
        * hierarchical_recall
        / (hierarchical_precision + hierarchical_recall)
    )


def calculate_accuracy(
    df: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
) -> float | None:
    completed = df[df[predicted_column].notna() & df["GPTError"].isna()]
    if completed.empty:
        return None

    return float(completed.apply(lambda row: is_prediction_match(row[actual_column], row[predicted_column]), axis=1).mean())


def calculate_joint_accuracy(df: pd.DataFrame) -> float | None:
    completed = completed_rows_for_metrics(df)
    if completed.empty:
        return None

    correct = completed.apply(
        lambda row: is_prediction_match(row["Geometry"], row["GPTGeometry"])
        and is_prediction_match(row["Entity"], row["GPTEntity"]),
        axis=1,
    )
    return float(correct.mean())


def calculate_exact_mismatch_count(df: pd.DataFrame) -> int:
    completed = completed_rows_for_metrics(df)
    if completed.empty:
        return 0

    mismatch = completed.apply(
        lambda row: (not is_prediction_match(row["Geometry"], row["GPTGeometry"]))
        or (not is_prediction_match(row["Entity"], row["GPTEntity"])),
        axis=1,
    )
    return int(mismatch.sum())


def calculate_per_label_metrics(
    df: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    labels: list[str],
) -> dict[str, Any]:
    completed = df[df[predicted_column].notna() & df["GPTError"].isna()]
    actual_for_scoring = completed.apply(
        lambda row: label_for_scoring(row[actual_column], row[predicted_column]),
        axis=1,
    )
    predicted_for_scoring = completed[predicted_column].apply(
        lambda value: "" if value is None or pd.isna(value) else str(value).strip()
    )
    observed_labels = sorted(
        set(actual_for_scoring.dropna().astype(str))
        | set(predicted_for_scoring.dropna().astype(str))
        | set(labels)
    )

    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []

    for label in observed_labels:
        actual_is_label = actual_for_scoring == label
        predicted_is_label = predicted_for_scoring == label
        true_positive = int((actual_is_label & predicted_is_label).sum())
        false_positive = int((~actual_is_label & predicted_is_label).sum())
        false_negative = int((actual_is_label & ~predicted_is_label).sum())
        support = int(actual_is_label.sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
        if support > 0:
            f1_values.append(float(f1))

    return {
        "macro_f1": None if not f1_values else float(sum(f1_values) / len(f1_values)),
        "labels": per_label,
    }


def build_evaluation_metrics(df: pd.DataFrame) -> dict[str, Any]:
    geometry_label_metrics = calculate_per_label_metrics(
        df, "Geometry", "GPTGeometry", GEOMETRY_TYPES
    )
    entity_label_metrics = calculate_per_label_metrics(
        df, "Entity", "GPTEntity", ENTITY_TYPES
    )

    return {
        "geometry_accuracy": calculate_accuracy(df, "Geometry", "GPTGeometry"),
        "entity_accuracy": calculate_accuracy(df, "Entity", "GPTEntity"),
        "joint_accuracy": calculate_joint_accuracy(df),
        "exact_mismatch_count": calculate_exact_mismatch_count(df),
        "geometry_macro_f1": geometry_label_metrics["macro_f1"],
        "entity_macro_f1": entity_label_metrics["macro_f1"],
        "geometry_hier_f1": hierarchical_f1_for_column(df, "Geometry", "GPTGeometry"),
        "entity_hier_f1": hierarchical_f1_for_column(df, "Entity", "GPTEntity"),
        "per_label_metrics": {
            "geometry": geometry_label_metrics["labels"],
            "entity": entity_label_metrics["labels"],
        },
    }


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
        **build_evaluation_metrics(df),
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

    return metrics



def apply_database_schema(database_url: str) -> None:
    import psycopg

    schema_sql = (BASE_DIR / "dashboard" / "db" / "schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()


def load_import_run_callable() -> Any:
    """Loads dashboard/scripts/import_run_to_db.py regardless of current working directory."""
    module_name = "dashboard.scripts.import_run_to_db"
    try:
        module = __import__(module_name, fromlist=["import_run"])
        return module.import_run
    except ModuleNotFoundError:
        module_path = BASE_DIR / "dashboard" / "scripts" / "import_run_to_db.py"
        if not module_path.exists():
            raise

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.import_run


def save_run_to_database(run_dir: Path, database_url: str) -> None:
    import psycopg

    apply_database_schema(database_url)
    import_run = load_import_run_callable()
    import_run(run_dir, database_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Amsterdam core concept datasets with OpenAI and report performance."
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("RUN_ID"),
        help="Optional run identifier. Defaults to a UTC timestamp plus a random suffix.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=CSV_PATH,
        help="Gold-label input CSV to evaluate against.",
    )
    parser.add_argument(
        "--provider",
        choices=["both", "gpt", "gemini"],
        default=os.getenv("ANNOTATION_PROVIDER", "both"),
        help="Which model provider to run. Defaults to both GPT and Gemini.",
    )
    return parser.parse_args()


# -----------------------------
# Run status and execution
# -----------------------------
def write_run_status(
    run_dir: Path,
    provider_config: ProviderConfig,
    run_id: str,
    status: str,
    total_rows: int,
    completed_rows: int,
    error_rows: int,
    current_row: int | None = None,
    current_kaartlaag: str | None = None,
    message: str | None = None,
) -> None:
    status_payload = {
        "run_id": run_id,
        "provider": provider_config.name,
        "model": provider_config.model,
        "status": status,
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "error_rows": error_rows,
        "current_row": current_row,
        "current_kaartlaag": current_kaartlaag,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "status.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_annotation_for_provider(
    provider_config: ProviderConfig,
    run_id: str,
    input_csv_path: Path,
    decision_tree: str,
) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_jsonl_path = run_dir / "annotations.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()

    openai_client = OpenAI() if provider_config.name == "gpt" else None
    gemini_api_key = get_gemini_api_key() if provider_config.name == "gemini" else None

    df = load_annotations(input_csv_path)
    df = ensure_output_columns(df)
    df["RunID"] = run_id

    total_rows = int(len(df))
    completed_count = 0
    error_count = 0
    write_run_status(
        run_dir,
        provider_config,
        run_id,
        "running",
        total_rows,
        completed_count,
        error_count,
        message="Run started.",
    )

    with run_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for idx, row in df.iterrows():
            kaartlaag = str(row["Kaartlaag"]).strip()
            write_run_status(
                run_dir,
                provider_config,
                run_id,
                "running",
                total_rows,
                completed_count,
                error_count,
                current_row=int(idx),
                current_kaartlaag=kaartlaag,
                message=f"Processing {kaartlaag or 'empty Kaartlaag'}.",
            )

            if not kaartlaag or kaartlaag.lower() == "nan":
                print(f"[{provider_config.name}] Skipping row {idx}: empty Kaartlaag")
                df.at[idx, "GPTError"] = "Empty Kaartlaag"
                error_count += 1
                df.to_csv(run_dir / "annotations.csv", index=False)
                continue

            try:
                dataset_json = load_dataset_json(kaartlaag)

                dataset_summary = summarize_dataset_for_annotation(dataset_json)
                summary_length = len(json.dumps(dataset_summary, ensure_ascii=False))

                print(
                    f"[{provider_config.name}] Processing row {idx}: {kaartlaag} "
                    f"(dataset summary length: {summary_length:,} chars)"
                )

                annotation = call_model(
                    provider_config=provider_config,
                    openai_client=openai_client,
                    gemini_api_key=gemini_api_key,
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
                completed_count += 1

                result = {
                    "run_id": run_id,
                    "provider": provider_config.name,
                    "model": provider_config.model,
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
                    f"[{provider_config.name}] Processed row {idx}: {kaartlaag} "
                    f"-> Geometry={geometry}, Entity={entity}, Confidence={confidence}"
                )
                write_run_status(
                    run_dir,
                    provider_config,
                    run_id,
                    "running",
                    total_rows,
                    completed_count,
                    error_count,
                    current_row=int(idx),
                    current_kaartlaag=kaartlaag,
                    message=f"Completed {kaartlaag}.",
                )

                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                error_message = str(e)
                error_count += 1

                error_result = {
                    "run_id": run_id,
                    "provider": provider_config.name,
                    "model": provider_config.model,
                    "row_index": int(idx),
                    "kaartlaag": kaartlaag,
                    "status": "error",
                    "error": error_message,
                }

                jsonl_file.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                jsonl_file.flush()

                df.at[idx, "GPTError"] = str(error_message)
                df.to_csv(run_dir / "annotations.csv", index=False)

                print(f"[{provider_config.name}] Error on row {idx} / {kaartlaag}: {error_message}")
                write_run_status(
                    run_dir,
                    provider_config,
                    run_id,
                    "running",
                    total_rows,
                    completed_count,
                    error_count,
                    current_row=int(idx),
                    current_kaartlaag=kaartlaag,
                    message=f"Error on {kaartlaag}: {error_message}",
                )

    completed_at = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "provider": provider_config.name,
        "model": provider_config.model,
        "started_at": started_at,
        "completed_at": completed_at,
        "input_csv_path": str(input_csv_path.relative_to(BASE_DIR)),
        "output_dir": str(run_dir.relative_to(BASE_DIR)),
        "provenance": build_run_provenance(input_csv_path),
    }
    metrics = write_run_artifacts(df, run_dir, run_metadata)
    write_run_status(
        run_dir,
        provider_config,
        run_id,
        "completed",
        total_rows,
        int(metrics["completed_rows"]),
        int(metrics["error_rows"]),
        message="Run completed and artifacts were written.",
    )

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        save_run_to_database(run_dir, database_url)

    print("Done.")
    print(f"Provider: {provider_config.name}")
    print(f"Model: {provider_config.model}")
    print(f"Run ID: {run_id}")
    print(f"CSV output written to: {run_dir / 'annotations.csv'}")
    print(f"JSONL log written to: {run_jsonl_path}")
    print(f"Geometry confusion matrix written to: {run_dir / 'confusion_matrix_geometry.csv'}")
    print(f"Entity confusion matrix written to: {run_dir / 'confusion_matrix_entity.csv'}")
    print(
        "Accuracy: "
        f"Geometry={metrics['geometry_accuracy']}, "
        f"Entity={metrics['entity_accuracy']}, "
        f"Joint={metrics['joint_accuracy']}"
    )
    return metrics


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    provider_configs = get_provider_configs(args.provider)
    if any(provider.name == "gpt" for provider in provider_configs) and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )
    if any(provider.name == "gemini" for provider in provider_configs) and not get_gemini_api_key():
        raise EnvironmentError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    base_run_id = args.run_id or create_run_id()
    input_csv_path = args.input_csv if args.input_csv.is_absolute() else BASE_DIR / args.input_csv
    decision_tree = load_text_file(DECISION_TREE_PATH)

    for provider_config in provider_configs:
        run_id = provider_run_id(base_run_id, provider_config) if args.provider == "both" else base_run_id
        run_annotation_for_provider(provider_config, run_id, input_csv_path, decision_tree)


if __name__ == "__main__":
    main()
