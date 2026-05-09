# Data Annotation & Enrichment

This repository contains a Python script for annotating Amsterdam map layers with the OpenAI API, evaluating the model against gold labels, and exporting artifacts that can be viewed in a local Next.js dashboard.

## Python annotation script

### Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env  # or create .env manually
```

Your `.env` file should include:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4-mini
```

Optional runtime settings:

```bash
REQUEST_DELAY_SECONDS=0.75
MAX_RETRIES=8
MAX_BACKOFF_SECONDS=60
RUN_ID=my-experiment-name
```

### Run annotation

```bash
python layer_annotation.py
```

You can also supply a readable run id:

```bash
python layer_annotation.py --run-id prompt-v3-gpt-5-4-mini
```

Each run is written to `V2/runs/<run-id>/` with:

- `annotations.csv` — gold labels plus model predictions in `GPTGeometry` and `GPTEntity`.
- `annotations.jsonl` — row-level processing log.
- `confusion_matrix_geometry.csv` — actual-vs-predicted geometry matrix.
- `confusion_matrix_entity.csv` — actual-vs-predicted entity matrix.
- `confusion_matrices.json` — normalized matrix cells for database import.
- `run_metrics.json` — accuracy, confidence, row counts, model, and timestamps.

For backward compatibility, the latest run is also copied to the top-level `V2/` output files.

## Local dashboard

The dashboard lives in `dashboard/` and uses Next.js with PostgreSQL. PgAdmin can create and inspect the database.

### 1. Create the database schema

Create a PostgreSQL database, then run the schema in PgAdmin or with `psql`:

```bash
psql "$DATABASE_URL" -f dashboard/db/schema.sql
```

The schema creates:

- `annotation_runs`
- `annotation_results`
- `confusion_matrix_cells`

### 2. Import a run

After running `layer_annotation.py`, import its artifacts:

```bash
cd dashboard
python -m pip install 'psycopg[binary]'
DATABASE_URL=postgres://postgres:postgres@localhost:5432/data_annotation \
  python scripts/import_run_to_db.py ../V2/runs/<run-id>
```

### 3. Start the dashboard

```bash
cd dashboard
npm install
cp .env.example .env.local
npm run dev
```

Open <http://localhost:3000> to see all runs, compare headline metrics, and inspect confusion matrices and row-level mismatches.
