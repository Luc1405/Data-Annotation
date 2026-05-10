CREATE TABLE IF NOT EXISTS annotation_runs (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    input_csv_path TEXT,
    output_dir TEXT,
    total_rows INTEGER NOT NULL DEFAULT 0,
    completed_rows INTEGER NOT NULL DEFAULT 0,
    error_rows INTEGER NOT NULL DEFAULT 0,
    geometry_accuracy DOUBLE PRECISION,
    entity_accuracy DOUBLE PRECISION,
    joint_accuracy DOUBLE PRECISION,
    exact_mismatch_count INTEGER,
    geometry_macro_f1 DOUBLE PRECISION,
    entity_macro_f1 DOUBLE PRECISION,
    mean_confidence DOUBLE PRECISION,
    provenance JSONB,
    per_label_metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS annotation_results (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES annotation_runs(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,
    title TEXT,
    english_title TEXT,
    page_link TEXT,
    map_link TEXT,
    kaartlaag TEXT NOT NULL,
    gold_geometry TEXT,
    gold_entity TEXT,
    predicted_geometry TEXT,
    predicted_entity TEXT,
    confidence DOUBLE PRECISION,
    reasoning_summary TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, row_index)
);

CREATE TABLE IF NOT EXISTS confusion_matrix_cells (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES annotation_runs(id) ON DELETE CASCADE,
    matrix_type TEXT NOT NULL CHECK (matrix_type IN ('geometry', 'entity')),
    actual_label TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (run_id, matrix_type, actual_label, predicted_label)
);

CREATE INDEX IF NOT EXISTS idx_annotation_results_run_id ON annotation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_annotation_results_errors ON annotation_results(run_id, error);
CREATE INDEX IF NOT EXISTS idx_confusion_matrix_cells_run_type ON confusion_matrix_cells(run_id, matrix_type);

ALTER TABLE IF EXISTS annotation_runs
    ADD COLUMN IF NOT EXISTS joint_accuracy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS exact_mismatch_count INTEGER,
    ADD COLUMN IF NOT EXISTS geometry_macro_f1 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS entity_macro_f1 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS provenance JSONB,
    ADD COLUMN IF NOT EXISTS per_label_metrics JSONB;
