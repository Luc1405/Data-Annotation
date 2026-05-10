import Link from "next/link";
import { formatNumber, formatPercent, getRunComparisonRows, getRuns, type RunComparisonRow, type RunSummary } from "../../lib/db";

function metricDelta(candidateValue: number | null, baselineValue: number | null, formatter: (value: number | null) => string) {
  if (candidateValue === null || baselineValue === null) return "—";
  const delta = candidateValue - baselineValue;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${formatter(delta)}`;
}

function isGeometryCorrect(row: Pick<RunComparisonRow, "gold_geometry" | "baseline_predicted_geometry" | "candidate_predicted_geometry">, side: "baseline" | "candidate") {
  const predicted = side === "baseline" ? row.baseline_predicted_geometry : row.candidate_predicted_geometry;
  return row.gold_geometry !== null && row.gold_geometry === predicted;
}

function isEntityCorrect(row: Pick<RunComparisonRow, "gold_entity" | "baseline_predicted_entity" | "candidate_predicted_entity">, side: "baseline" | "candidate") {
  const predicted = side === "baseline" ? row.baseline_predicted_entity : row.candidate_predicted_entity;
  return row.gold_entity !== null && row.gold_entity === predicted;
}

function isJointCorrect(row: RunComparisonRow, side: "baseline" | "candidate") {
  return isGeometryCorrect(row, side) && isEntityCorrect(row, side);
}

function statusBadge(row: RunComparisonRow) {
  const baselineCorrect = isJointCorrect(row, "baseline");
  const candidateCorrect = isJointCorrect(row, "candidate");

  if (baselineCorrect && candidateCorrect) return <span className="badge success">Both correct</span>;
  if (!baselineCorrect && candidateCorrect) return <span className="badge success">Candidate fixed</span>;
  if (baselineCorrect && !candidateCorrect) return <span className="badge error">Candidate regressed</span>;
  return <span className="badge warning">Both wrong</span>;
}

function SummaryCards({ rows }: { rows: RunComparisonRow[] }) {
  const bothCorrect = rows.filter((row) => isJointCorrect(row, "baseline") && isJointCorrect(row, "candidate")).length;
  const candidateFixed = rows.filter((row) => !isJointCorrect(row, "baseline") && isJointCorrect(row, "candidate")).length;
  const candidateRegressed = rows.filter((row) => isJointCorrect(row, "baseline") && !isJointCorrect(row, "candidate")).length;
  const bothWrong = rows.filter((row) => !isJointCorrect(row, "baseline") && !isJointCorrect(row, "candidate")).length;

  return (
    <section className="grid metrics">
      <div className="metric"><div className="label">Compared rows</div><div className="value">{rows.length}</div></div>
      <div className="metric"><div className="label">Both correct</div><div className="value">{bothCorrect}</div></div>
      <div className="metric"><div className="label">Candidate fixed</div><div className="value">{candidateFixed}</div></div>
      <div className="metric"><div className="label">Candidate regressed</div><div className="value">{candidateRegressed}</div></div>
      <div className="metric"><div className="label">Both wrong</div><div className="value">{bothWrong}</div></div>
    </section>
  );
}

function RunMetricTable({ baselineRun, candidateRun }: { baselineRun: RunSummary; candidateRun: RunSummary }) {
  return (
    <section className="card" style={{ marginBottom: 20 }}>
      <h2>Headline metrics</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Candidate delta</th></tr>
          </thead>
          <tbody>
            <tr><td>Run ID</td><td><span className="badge">{baselineRun.id}</span></td><td><span className="badge">{candidateRun.id}</span></td><td>—</td></tr>
            <tr><td>Model</td><td>{baselineRun.model}</td><td>{candidateRun.model}</td><td>—</td></tr>
            <tr><td>Completed rows</td><td>{baselineRun.completed_rows}/{baselineRun.total_rows}</td><td>{candidateRun.completed_rows}/{candidateRun.total_rows}</td><td>{candidateRun.completed_rows - baselineRun.completed_rows}</td></tr>
            <tr><td>Errors</td><td>{baselineRun.error_rows}</td><td>{candidateRun.error_rows}</td><td>{candidateRun.error_rows - baselineRun.error_rows}</td></tr>
            <tr><td>Geometry accuracy</td><td>{formatPercent(baselineRun.geometry_accuracy)}</td><td>{formatPercent(candidateRun.geometry_accuracy)}</td><td>{metricDelta(candidateRun.geometry_accuracy, baselineRun.geometry_accuracy, formatPercent)}</td></tr>
            <tr><td>Entity accuracy</td><td>{formatPercent(baselineRun.entity_accuracy)}</td><td>{formatPercent(candidateRun.entity_accuracy)}</td><td>{metricDelta(candidateRun.entity_accuracy, baselineRun.entity_accuracy, formatPercent)}</td></tr>
            <tr><td>Mean confidence</td><td>{formatNumber(baselineRun.mean_confidence)}</td><td>{formatNumber(candidateRun.mean_confidence)}</td><td>{metricDelta(candidateRun.mean_confidence, baselineRun.mean_confidence, formatNumber)}</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default async function ComparePage({ searchParams }: { searchParams: Promise<{ baseline?: string; candidate?: string }> }) {
  const runs = await getRuns();
  const params = await searchParams;
  const baselineRun = runs.find((run) => run.id === params.baseline) ?? runs[0];
  const candidateRun = runs.find((run) => run.id === params.candidate && run.id !== baselineRun?.id) ?? runs.find((run) => run.id !== baselineRun?.id);
  const rows = baselineRun && candidateRun ? await getRunComparisonRows(baselineRun.id, candidateRun.id) : [];

  return (
    <main className="shell">
      <section className="header">
        <div>
          <div className="eyebrow">Run comparison</div>
          <h1>Compare annotation runs</h1>
          <p>Select a baseline run and a candidate run to compare headline accuracy, paired outcomes, and row-level prediction changes.</p>
        </div>
        <Link className="button" href="/">All runs</Link>
      </section>

      {!process.env.DATABASE_URL ? (
        <div className="notice">
          <strong>Database is not configured.</strong> Set <code>DATABASE_URL</code>, apply <code>db/schema.sql</code>, import runs, and restart the dashboard.
        </div>
      ) : null}

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>Select runs</h2>
        <form className="actions" action="/compare">
          <label className="field">
            <span>Baseline</span>
            <select className="select" name="baseline" defaultValue={baselineRun?.id ?? ""}>
              {runs.map((run) => <option key={run.id} value={run.id}>{run.id}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Candidate</span>
            <select className="select" name="candidate" defaultValue={candidateRun?.id ?? ""}>
              {runs.map((run) => <option key={run.id} value={run.id}>{run.id}</option>)}
            </select>
          </label>
          <button className="button" type="submit" disabled={runs.length < 2}>Compare</button>
        </form>
        {runs.length < 2 ? <p>Add at least two runs before comparing versions.</p> : null}
      </section>

      {baselineRun && candidateRun ? (
        <>
          <RunMetricTable baselineRun={baselineRun} candidateRun={candidateRun} />
          <SummaryCards rows={rows} />

          <section className="card">
            <h2>Row-level comparison</h2>
            <p>Rows are joined by their saved row index, so the table highlights what the candidate run fixed or regressed against the baseline.</p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th><th>Layer</th><th>Status</th><th>Gold</th><th>Baseline prediction</th><th>Candidate prediction</th><th>Confidence</th><th>Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.row_index}>
                      <td>{row.row_index}</td>
                      <td><strong>{row.kaartlaag}</strong><br />{row.english_title ?? row.title}</td>
                      <td>{statusBadge(row)}</td>
                      <td>{row.gold_geometry ?? "—"}<br />{row.gold_entity ?? "—"}</td>
                      <td>{row.baseline_predicted_geometry ?? "—"}<br />{row.baseline_predicted_entity ?? "—"}{row.baseline_error ? <p><span className="badge error">{row.baseline_error}</span></p> : null}</td>
                      <td>{row.candidate_predicted_geometry ?? "—"}<br />{row.candidate_predicted_entity ?? "—"}{row.candidate_error ? <p><span className="badge error">{row.candidate_error}</span></p> : null}</td>
                      <td>{row.baseline_confidence?.toFixed(2) ?? "—"} → {row.candidate_confidence?.toFixed(2) ?? "—"}</td>
                      <td><p><strong>Baseline:</strong> {row.baseline_reasoning_summary ?? "—"}</p><p><strong>Candidate:</strong> {row.candidate_reasoning_summary ?? "—"}</p></td>
                    </tr>
                  ))}
                  {rows.length === 0 ? <tr><td colSpan={8}>No overlapping rows found for the selected runs.</td></tr> : null}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </main>
  );
}
