import Link from "next/link";
import { readFile, readdir } from "fs/promises";
import path from "path";
import { startAnnotationRunAction } from "../actions";

type RunStatus = {
  run_id: string;
  provider: string;
  model: string;
  status: string;
  total_rows: number;
  completed_rows: number;
  error_rows: number;
  current_row: number | null;
  current_kaartlaag: string | null;
  message: string | null;
  updated_at: string;
};

type RunTracker = {
  runId: string;
  provider: "gpt" | "gemini";
  label: string;
  model: string;
  status: string;
  totalRows: number;
  completedRows: number;
  errorRows: number;
  message: string;
  updatedAt: string | null;
};


async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

async function getRunTrackers(baseRunId?: string): Promise<RunTracker[]> {
  if (!baseRunId) return [];

  const repositoryRoot = path.resolve(process.cwd(), "..");
  const runsRoot = path.join(repositoryRoot, "output", "runs");
  const providers = [
    { provider: "gpt" as const, label: "GPT run", runId: `${baseRunId}_gpt` },
    { provider: "gemini" as const, label: "Gemini run", runId: `${baseRunId}_gemini` },
  ];

  return Promise.all(providers.map(async (providerInfo) => {
    const runDir = path.join(runsRoot, providerInfo.runId);
    const status = await readJsonFile<RunStatus>(path.join(runDir, "status.json"));
    const metrics = await readJsonFile<{ model?: string; total_rows?: number; completed_rows?: number; error_rows?: number; completed_at?: string }>(path.join(runDir, "run_metrics.json"));

    return {
      runId: providerInfo.runId,
      provider: providerInfo.provider,
      label: providerInfo.label,
      model: status?.model ?? metrics?.model ?? (providerInfo.provider === "gpt" ? (process.env.OPENAI_MODEL ?? "gpt-5.4-mini") : (process.env.GEMINI_MODEL ?? "gemini-3.1-flash-lite-preview")),
      status: status?.status ?? (metrics?.completed_at ? "completed" : "waiting"),
      totalRows: status?.total_rows ?? metrics?.total_rows ?? 0,
      completedRows: status?.completed_rows ?? metrics?.completed_rows ?? 0,
      errorRows: status?.error_rows ?? metrics?.error_rows ?? 0,
      message: status?.message ?? (metrics?.completed_at ? "Run artifacts are available." : "Waiting for the script to create status output."),
      updatedAt: status?.updated_at ?? metrics?.completed_at ?? null,
    };
  }));
}

function statusPercent(tracker: RunTracker) {
  if (!tracker.totalRows) return 0;
  return Math.min(100, Math.round(((tracker.completedRows + tracker.errorRows) / tracker.totalRows) * 100));
}

async function getAvailableScripts() {
  const repositoryRoot = path.resolve(process.cwd(), "..");
  const scripts = [{ label: "Current script: layer_annotation.py", value: "layer_annotation.py" }];
  const versionsDir = path.join(repositoryRoot, "annotation_versions");

  try {
    const entries = await readdir(versionsDir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith(".py")) {
        scripts.push({ label: `Versioned script: annotation_versions/${entry.name}`, value: `annotation_versions/${entry.name}` });
      }
    }
  } catch {
    // The versions folder is optional; the page still works with the current script.
  }

  return scripts;
}

export default async function RunScriptPage({ searchParams }: { searchParams: Promise<{ started?: string; error?: string }> }) {
  const scripts = await getAvailableScripts();
  const params = await searchParams;
  const watchedRunId = params.started;
  const trackers = await getRunTrackers(watchedRunId);

  return (
    <main className="shell">
      <section className="header">
        <div>
          <div className="eyebrow">Run annotation</div>
          <h1>Start an annotation script</h1>
          <p>Choose a script version, set a readable base run ID, and start paired GPT and Gemini background runs from the dashboard.</p>
        </div>
        <Link className="button" href="/">All runs</Link>
      </section>

      {params.started ? (
        <div className="notice" style={{ marginBottom: 20 }}>
          Started paired runs for base ID <strong>{params.started}</strong>. Use the live tracker below to check each run; click refresh to update status.
        </div>
      ) : null}
      {params.error ? (
        <div className="notice error" style={{ marginBottom: 20 }}>
          <strong>Could not start the run:</strong> {params.error}
        </div>
      ) : null}

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>Run settings</h2>
        <form className="actions run-form" action={startAnnotationRunAction}>
          <label className="field">
            <span>Base run ID</span>
            <input className="input" name="runId" placeholder="prompt-v4" required />
            <small>Runs will be saved as <code>&lt;base&gt;_gpt</code> and <code>&lt;base&gt;_gemini</code>.</small>
          </label>
          <label className="field">
            <span>Script</span>
            <select className="select" name="scriptPath" defaultValue="layer_annotation.py">
              {scripts.map((script) => <option key={script.value} value={script.value}>{script.label}</option>)}
            </select>
          </label>
          <label className="field">
            <span>GPT model override</span>
            <input className="input" name="model" placeholder="Leave blank to use OPENAI_MODEL" />
          </label>
          <label className="field">
            <span>Gemini model</span>
            <input className="input" name="geminiModel" defaultValue="gemini-3.1-flash-lite-preview" />
          </label>
          <div className="model-pair" aria-label="Models that will run">
            <div><strong>GPT</strong><span>Creates <code>_gpt</code> run</span></div>
            <div><strong>Gemini</strong><span>Creates <code>_gemini</code> run</span></div>
          </div>
          <button className="button" type="submit">Start paired run</button>
        </form>
      </section>

      {trackers.length ? (
        <section className="card" style={{ marginBottom: 20 }}>
          <div className="tracker-header">
            <div>
              <h2>Live run tracker</h2>
              <p>Progress is read from each run folder under <code>output/runs</code>.</p>
            </div>
            <Link className="button" href={`/run?started=${encodeURIComponent(watchedRunId ?? "")}`}>Refresh status</Link>
          </div>
          <div className="tracker-grid">
            {trackers.map((tracker) => {
              const percent = statusPercent(tracker);
              return (
                <div className={`tracker-card ${tracker.provider}`} key={tracker.runId}>
                  <div className="tracker-title">
                    <span className="badge">{tracker.label}</span>
                    <span className={`badge ${tracker.status === "completed" ? "success" : tracker.errorRows ? "warning" : ""}`}>{tracker.status}</span>
                  </div>
                  <h3>{tracker.runId}</h3>
                  <p>{tracker.model}</p>
                  <div className="progress-bar" aria-label={`${tracker.runId} progress`}>
                    <div style={{ width: `${percent}%` }} />
                  </div>
                  <div className="tracker-stats">
                    <span>{percent}%</span>
                    <span>{tracker.completedRows} completed</span>
                    <span>{tracker.errorRows} errors</span>
                    <span>{tracker.totalRows || "—"} total</span>
                  </div>
                  <p className="tracker-message">{tracker.message}</p>
                  {tracker.updatedAt ? <small>Updated {new Date(tracker.updatedAt).toLocaleString()}</small> : null}
                </div>
              );
            })}
          </div>
        </section>
      ) : null}

      <section className="card">
        <h2>Recommended workflow for script versions</h2>
        <p>Use git as the source of truth for old versions. Each provider run records the git commit, dirty state, script hash, decision-tree hash, input CSV hash, model, and runtime settings in its provenance.</p>
        <ul>
          <li>Edit <code>layer_annotation.py</code> for the next experiment and run it with a descriptive base run ID.</li>
          <li>Commit meaningful script or prompt changes before important runs so the run provenance points back to an immutable git commit.</li>
          <li>If you want a long-lived alternate implementation, copy it into <code>annotation_versions/&lt;name&gt;.py</code>. This page will offer those files in the script selector.</li>
          <li>Avoid editing old files after using them for benchmark runs; create a new versioned file or use a new git commit instead.</li>
        </ul>
      </section>
    </main>
  );
}
