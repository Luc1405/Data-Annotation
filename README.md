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

To make completed runs appear in the dashboard automatically, set `DATABASE_URL` in this same root `.env` file. The script applies `dashboard/db/schema.sql` automatically when it saves a run, so you only need to create the empty PostgreSQL database first.

```bash
DATABASE_URL=postgres://postgres:postgres@localhost:5432/data_annotation
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
python annotation_versions/Baseline/layer_annotation.py
```

You can also supply a readable run id:

```bash
python annotation_versions/Baseline/layer_annotation.py --run-id prompt-v3-gpt-5-4-mini
```

Each run is written to `V2/runs/<run-id>/` with:

- `annotations.csv` — gold labels plus model predictions in `GPTGeometry` and `GPTEntity`.
- `annotations.jsonl` — row-level processing log.
- `confusion_matrix_geometry.csv` — actual-vs-predicted geometry matrix.
- `confusion_matrix_entity.csv` — actual-vs-predicted entity matrix.
- `confusion_matrices.json` — normalized matrix cells for database import.
- `run_metrics.json` — accuracy, joint accuracy, mismatch counts, macro F1, per-label metrics, confidence, row counts, model, timestamps, and provenance hashes.

For backward compatibility, the latest run is also copied to the top-level `V2/` output files.


### Comparing script versions

Best practice is to keep all runnable scripts in `annotation_versions/<version>/` and use git commits as the authoritative history for old versions. Each run records the current git commit, dirty state, script hash, decision-tree hash, input CSV hash, model, and runtime settings in `run_metrics.json`, so important benchmark runs should be made from a committed state.

Each version folder can include its own `decision_tree.txt` alongside the script. The dashboard's **Run script** page recursively offers every Python file in `annotation_versions/` as a selectable script. Use readable run IDs such as `prompt-v4-gpt-5-4-mini` so comparisons remain understandable later.

## Local dashboard

The dashboard lives in `dashboard/` and uses Next.js with PostgreSQL. PgAdmin can create and inspect the database. The dashboard reads environment variables from the project root `.env`, so separate `dashboard/.env.local` is not required.

### 1. Create the database

Create an empty PostgreSQL database in PgAdmin, for example `data_annotation`. You do not have to manually import run artifacts after every script run: if `DATABASE_URL` is set when your selected script runs, it saves the run directly to PostgreSQL and the dashboard will list it automatically.

The schema file is still available at `dashboard/db/schema.sql` if you want to inspect or apply it manually. It creates:
### 1. Create the database schema

Create a PostgreSQL database, then run the schema in PgAdmin or with `psql`:

```bash
psql "$DATABASE_URL" -f dashboard/db/schema.sql
```

The schema creates:

- `annotation_runs`
- `annotation_results`
- `confusion_matrix_cells`

### 2. Start the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open <http://localhost:3000> to see all saved runs, start new script runs, compare headline and joint metrics, inspect confusion matrices and row-level mismatches, and delete runs. By default, deleting a run removes the database rows and that run's `V2/runs/<run-id>` artifact folder. Set `DELETE_RUN_ARTIFACTS=false` in the project root `.env` if you want dashboard deletes to keep files on disk.

### Backfilling older runs

“Importing artifacts” means loading an existing `V2/runs/<run-id>/` folder into PostgreSQL. You only need this for older runs that were created before `DATABASE_URL` was configured, or for runs copied in from another machine. New runs do not need a separate import step.

```bash
cd dashboard
DATABASE_URL=postgres://postgres:postgres@localhost:5432/data_annotation \
  python scripts/import_run_to_db.py ../V2/runs/<run-id>
```
### 2. Import a run

After running your annotation script, import its artifacts:

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
npm run dev
```

Open <http://localhost:3000> to see all runs, compare headline metrics, and inspect confusion matrices and row-level mismatches.
