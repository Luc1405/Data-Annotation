import { rm } from "fs/promises";
import path from "path";
import { Pool } from "pg";

export type RunSummary = {
  id: string;
  model: string;
  started_at: string | null;
  completed_at: string | null;
  total_rows: number;
  completed_rows: number;
  error_rows: number;
  geometry_accuracy: number | null;
  entity_accuracy: number | null;
  mean_confidence: number | null;
};

export type AnnotationResult = {
  row_index: number;
  title: string | null;
  english_title: string | null;
  kaartlaag: string;
  gold_geometry: string | null;
  gold_entity: string | null;
  predicted_geometry: string | null;
  predicted_entity: string | null;
  confidence: number | null;
  reasoning_summary: string | null;
  error: string | null;
};

export type ConfusionCell = {
  matrix_type: "geometry" | "entity";
  actual_label: string;
  predicted_label: string;
  count: number;
};

export type RunComparisonRow = {
  row_index: number;
  title: string | null;
  english_title: string | null;
  kaartlaag: string;
  gold_geometry: string | null;
  gold_entity: string | null;
  baseline_predicted_geometry: string | null;
  baseline_predicted_entity: string | null;
  baseline_confidence: number | null;
  baseline_reasoning_summary: string | null;
  baseline_error: string | null;
  candidate_predicted_geometry: string | null;
  candidate_predicted_entity: string | null;
  candidate_confidence: number | null;
  candidate_reasoning_summary: string | null;
  candidate_error: string | null;
};

let pool: Pool | null = null;

function getPool() {
  if (!process.env.DATABASE_URL) {
    return null;
  }

  if (!pool) {
    pool = new Pool({ connectionString: process.env.DATABASE_URL });
  }

  return pool;
}

export async function getRuns(): Promise<RunSummary[]> {
  const db = getPool();
  if (!db) return [];

  const result = await db.query<RunSummary>(
    `SELECT id, model, started_at, completed_at, total_rows, completed_rows,
            error_rows, geometry_accuracy, entity_accuracy, mean_confidence
       FROM annotation_runs
      ORDER BY COALESCE(started_at, created_at) DESC`
  );
  return result.rows;
}

export async function getRun(id: string): Promise<RunSummary | null> {
  const db = getPool();
  if (!db) return null;

  const result = await db.query<RunSummary>(
    `SELECT id, model, started_at, completed_at, total_rows, completed_rows,
            error_rows, geometry_accuracy, entity_accuracy, mean_confidence
       FROM annotation_runs
      WHERE id = $1`,
    [id]
  );
  return result.rows[0] ?? null;
}

export async function getResults(runId: string): Promise<AnnotationResult[]> {
  const db = getPool();
  if (!db) return [];

  const result = await db.query<AnnotationResult>(
    `SELECT row_index, title, english_title, kaartlaag, gold_geometry, gold_entity,
            predicted_geometry, predicted_entity, confidence, reasoning_summary, error
       FROM annotation_results
      WHERE run_id = $1
      ORDER BY row_index ASC`,
    [runId]
  );
  return result.rows;
}


export async function getRunComparisonRows(
  baselineRunId: string,
  candidateRunId: string
): Promise<RunComparisonRow[]> {
  const db = getPool();
  if (!db) return [];

  const result = await db.query<RunComparisonRow>(
    `SELECT
        COALESCE(baseline.row_index, candidate.row_index) AS row_index,
        COALESCE(baseline.title, candidate.title) AS title,
        COALESCE(baseline.english_title, candidate.english_title) AS english_title,
        COALESCE(baseline.kaartlaag, candidate.kaartlaag) AS kaartlaag,
        COALESCE(baseline.gold_geometry, candidate.gold_geometry) AS gold_geometry,
        COALESCE(baseline.gold_entity, candidate.gold_entity) AS gold_entity,
        baseline.predicted_geometry AS baseline_predicted_geometry,
        baseline.predicted_entity AS baseline_predicted_entity,
        baseline.confidence AS baseline_confidence,
        baseline.reasoning_summary AS baseline_reasoning_summary,
        baseline.error AS baseline_error,
        candidate.predicted_geometry AS candidate_predicted_geometry,
        candidate.predicted_entity AS candidate_predicted_entity,
        candidate.confidence AS candidate_confidence,
        candidate.reasoning_summary AS candidate_reasoning_summary,
        candidate.error AS candidate_error
       FROM (
        SELECT *
          FROM annotation_results
         WHERE run_id = $1
       ) baseline
       FULL OUTER JOIN (
        SELECT *
          FROM annotation_results
         WHERE run_id = $2
       ) candidate
         ON candidate.row_index = baseline.row_index
      ORDER BY row_index ASC`,
    [baselineRunId, candidateRunId]
  );
  return result.rows;
}

export async function getConfusionCells(runId: string): Promise<ConfusionCell[]> {
  const db = getPool();
  if (!db) return [];

  const result = await db.query<ConfusionCell>(
    `SELECT matrix_type, actual_label, predicted_label, count
       FROM confusion_matrix_cells
      WHERE run_id = $1
      ORDER BY matrix_type, actual_label, predicted_label`,
    [runId]
  );
  return result.rows;
}

export function formatPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value: number | null) {
  if (value === null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}


export async function deleteRunById(runId: string) {
  const db = getPool();
  if (!db) return;

  const result = await db.query<{ output_dir: string | null }>(
    "SELECT output_dir FROM annotation_runs WHERE id = $1",
    [runId]
  );
  const outputDir = result.rows[0]?.output_dir;

  await db.query("DELETE FROM annotation_runs WHERE id = $1", [runId]);

  if (outputDir && process.env.DELETE_RUN_ARTIFACTS !== "false") {
    const repositoryRoot = path.resolve(process.cwd(), "..");
    const runsRoot = path.resolve(repositoryRoot, "V2", "runs");
    const artifactPath = path.resolve(repositoryRoot, outputDir);

    if (artifactPath.startsWith(`${runsRoot}${path.sep}`)) {
      await rm(artifactPath, { recursive: true, force: true });
    }
  }
}
