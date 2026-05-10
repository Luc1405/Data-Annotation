import Link from "next/link";
import { readdir } from "fs/promises";
import path from "path";
import { startAnnotationRunAction } from "../actions";

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

  return (
    <main className="shell">
      <section className="header">
        <div>
          <div className="eyebrow">Run annotation</div>
          <h1>Start an annotation script</h1>
          <p>Choose a script version, set a readable run ID, and start a background annotation run from the dashboard.</p>
        </div>
        <Link className="button" href="/">All runs</Link>
      </section>

      {params.started ? (
        <div className="notice" style={{ marginBottom: 20 }}>
          Started run <strong>{params.started}</strong>. Refresh the dashboard after the script finishes to see imported metrics.
        </div>
      ) : null}
      {params.error ? (
        <div className="notice error" style={{ marginBottom: 20 }}>
          <strong>Could not start the run:</strong> {params.error}
        </div>
      ) : null}

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>Run settings</h2>
        <form className="actions" action={startAnnotationRunAction}>
          <label className="field">
            <span>Run ID</span>
            <input className="input" name="runId" placeholder="prompt-v4-gpt-5-4-mini" required />
          </label>
          <label className="field">
            <span>Script</span>
            <select className="select" name="scriptPath" defaultValue="layer_annotation.py">
              {scripts.map((script) => <option key={script.value} value={script.value}>{script.label}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Model override</span>
            <input className="input" name="model" placeholder="Leave blank to use .env" />
          </label>
          <button className="button" type="submit">Start run</button>
        </form>
      </section>

      <section className="card">
        <h2>Recommended workflow for script versions</h2>
        <p>Use git as the source of truth for old versions. Each run now records the git commit, dirty state, script hash, decision-tree hash, input CSV hash, model, and runtime settings in its provenance.</p>
        <ul>
          <li>Edit <code>layer_annotation.py</code> for the next experiment and run it with a descriptive run ID.</li>
          <li>Commit meaningful script or prompt changes before important runs so the run provenance points back to an immutable git commit.</li>
          <li>If you want a long-lived alternate implementation, copy it into <code>annotation_versions/&lt;name&gt;.py</code>. This page will offer those files in the script selector.</li>
          <li>Avoid editing old files after using them for benchmark runs; create a new versioned file or use a new git commit instead.</li>
        </ul>
      </section>
    </main>
  );
}
