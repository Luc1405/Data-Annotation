from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def none_if_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def import_run(run_dir: Path, database_url: str) -> None:
    metrics = load_json(run_dir / "run_metrics.json")
    matrices = load_json(run_dir / "confusion_matrices.json")
    annotations_path = run_dir / "annotations.csv"

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO annotation_runs (
                    id, model, started_at, completed_at, input_csv_path, output_dir,
                    total_rows, completed_rows, error_rows, geometry_accuracy,
                    entity_accuracy, joint_accuracy, exact_mismatch_count,
                    geometry_macro_f1, entity_macro_f1, mean_confidence,
                    provenance, per_label_metrics
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    joint_accuracy = EXCLUDED.joint_accuracy,
                    exact_mismatch_count = EXCLUDED.exact_mismatch_count,
                    geometry_macro_f1 = EXCLUDED.geometry_macro_f1,
                    entity_macro_f1 = EXCLUDED.entity_macro_f1,
                    mean_confidence = EXCLUDED.mean_confidence,
                    provenance = EXCLUDED.provenance,
                    per_label_metrics = EXCLUDED.per_label_metrics
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
                    metrics.get("joint_accuracy"),
                    metrics.get("exact_mismatch_count"),
                    metrics.get("geometry_macro_f1"),
                    metrics.get("entity_macro_f1"),
                    metrics.get("mean_confidence"),
                    Jsonb(metrics.get("provenance", {})),
                    Jsonb(metrics.get("per_label_metrics", {})),
                ),
            )

            cur.execute("DELETE FROM annotation_results WHERE run_id = %s", (metrics["run_id"],))
            with annotations_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    cur.execute(
                        """
                        INSERT INTO annotation_results (
                            run_id, row_index, title, english_title, page_link, map_link,
                            kaartlaag, gold_geometry, gold_entity, predicted_geometry,
                            predicted_entity, confidence, reasoning_summary, error
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            metrics["run_id"],
                            int(row.get("RowIndex") or reader.line_num - 2),
                            none_if_empty(row.get("Title")),
                            none_if_empty(row.get("EnglishTitle")),
                            none_if_empty(row.get("PageLink")),
                            none_if_empty(row.get("MapLink")),
                            row["Kaartlaag"],
                            none_if_empty(row.get("Geometry")),
                            none_if_empty(row.get("Entity")),
                            none_if_empty(row.get("GPTGeometry")),
                            none_if_empty(row.get("GPTEntity")),
                            float(row["GPTConfidence"]) if row.get("GPTConfidence") else None,
                            none_if_empty(row.get("GPTReasoningSummary")),
                            none_if_empty(row.get("GPTError")),
                        ),
                    )

            cur.execute("DELETE FROM confusion_matrix_cells WHERE run_id = %s", (metrics["run_id"],))
            for matrix_type in ("geometry", "entity"):
                for cell in matrices.get(matrix_type, []):
                    cur.execute(
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
        conn.commit()

    print(f"Imported run {metrics['run_id']} from {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import annotation run artifacts into PostgreSQL.")
    parser.add_argument("run_dir", type=Path, help="Path like ../V2/runs/<run-id>")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required. Set it or pass --database-url.")
    import_run(args.run_dir, args.database_url)
