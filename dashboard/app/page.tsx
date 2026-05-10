import Link from "next/link";
import { formatNumber, formatPercent, getRuns } from "../lib/db";

export default async function Home({ searchParams }: { searchParams: Promise<{ candidate?: string; compare?: string }> }) {
  const runs = await getRuns();
  const params = await searchParams;
  const compareId = params.candidate ?? params.compare;
  const baseRun = runs[0];
  const compareRun = runs.find((run) => run.id === compareId) ?? runs[1];

  return (
    <main className="shell">
      <section className="header">
        <div>
          <div className="eyebrow">OpenAI Annotation Evaluation</div>
          <h1>Run dashboard</h1>
          <p>View saved annotation runs, accuracy metrics, and confusion matrices from your local PostgreSQL database.</p>
        </div>
        <div className="actions">
          <Link className="button" href="/compare">Compare runs</Link>
          <Link className="button" href={baseRun ? `/runs/${baseRun.id}` : "#"}>Latest run</Link>
        </div>
      </section>

      {!process.env.DATABASE_URL ? (
        <div className="notice">
          <strong>Database is not configured.</strong> Set <code>DATABASE_URL</code>, apply <code>db/schema.sql</code>, import a run, and restart the dashboard.
        </div>
      ) : null}

      <section className="grid metrics">
        <div className="metric"><div className="label">Runs</div><div className="value">{runs.length}</div></div>
        <div className="metric"><div className="label">Latest geometry accuracy</div><div className="value">{formatPercent(baseRun?.geometry_accuracy ?? null)}</div></div>
        <div className="metric"><div className="label">Latest entity accuracy</div><div className="value">{formatPercent(baseRun?.entity_accuracy ?? null)}</div></div>
        <div className="metric"><div className="label">Mean confidence</div><div className="value">{formatNumber(baseRun?.mean_confidence ?? null)}</div></div>
      </section>

      {baseRun && compareRun ? (
        <section className="card" style={{ marginBottom: 20 }}>
          <h2>Quick comparison</h2>
          <form className="actions" action="/compare">
            <span>Compare latest run with</span>
            <input type="hidden" name="baseline" value={baseRun.id} />
            <select className="select" name="candidate" defaultValue={compareRun.id}>
              {runs.filter((run) => run.id !== baseRun.id).map((run) => (
                <option key={run.id} value={run.id}>{run.id}</option>
              ))}
            </select>
            <button className="button" type="submit">Open comparison</button>
          </form>
          <div className="table-wrap" style={{ marginTop: 16 }}>
            <table>
              <thead><tr><th>Metric</th><th>{baseRun.id}</th><th>{compareRun.id}</th><th>Delta</th></tr></thead>
              <tbody>
                <tr><td>Geometry accuracy</td><td>{formatPercent(baseRun.geometry_accuracy)}</td><td>{formatPercent(compareRun.geometry_accuracy)}</td><td>{formatPercent((compareRun.geometry_accuracy ?? 0) - (baseRun.geometry_accuracy ?? 0))}</td></tr>
                <tr><td>Entity accuracy</td><td>{formatPercent(baseRun.entity_accuracy)}</td><td>{formatPercent(compareRun.entity_accuracy)}</td><td>{formatPercent((compareRun.entity_accuracy ?? 0) - (baseRun.entity_accuracy ?? 0))}</td></tr>
                <tr><td>Mean confidence</td><td>{formatNumber(baseRun.mean_confidence)}</td><td>{formatNumber(compareRun.mean_confidence)}</td><td>{formatNumber((compareRun.mean_confidence ?? 0) - (baseRun.mean_confidence ?? 0))}</td></tr>
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="card">
        <h2>Runs</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Run</th><th>Model</th><th>Started</th><th>Rows</th><th>Errors</th><th>Geometry</th><th>Entity</th><th></th></tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id}>
                  <td><span className="badge">{run.id}</span></td>
                  <td>{run.model}</td>
                  <td>{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</td>
                  <td>{run.completed_rows}/{run.total_rows}</td>
                  <td>{run.error_rows}</td>
                  <td>{formatPercent(run.geometry_accuracy)}</td>
                  <td>{formatPercent(run.entity_accuracy)}</td>
                  <td><Link href={`/runs/${run.id}`}>Open</Link></td>
                </tr>
              ))}
              {runs.length === 0 ? <tr><td colSpan={8}>No runs found yet.</td></tr> : null}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
