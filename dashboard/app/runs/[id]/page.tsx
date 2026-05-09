import Link from "next/link";
import { notFound } from "next/navigation";
import { formatNumber, formatPercent, getConfusionCells, getResults, getRun, type ConfusionCell } from "../../../lib/db";

function ConfusionMatrix({ title, cells }: { title: string; cells: ConfusionCell[] }) {
  const labels = Array.from(new Set(cells.flatMap((cell) => [cell.actual_label, cell.predicted_label])))
    .filter((label) => !label.startsWith("__"))
    .sort();
  const countByPair = new Map(cells.map((cell) => [`${cell.actual_label}|${cell.predicted_label}`, cell.count]));

  return (
    <div className="card">
      <h2>{title}</h2>
      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr><th>Actual \ Predicted</th>{labels.map((label) => <th key={label}>{label}</th>)}<th>Total</th></tr>
          </thead>
          <tbody>
            {labels.map((actual) => (
              <tr key={actual}>
                <th>{actual}</th>
                {labels.map((predicted) => {
                  const value = countByPair.get(`${actual}|${predicted}`) ?? 0;
                  return <td className={actual === predicted ? "diagonal" : value > 0 ? "off-diagonal" : ""} key={predicted}>{value}</td>;
                })}
                <td>{countByPair.get(`${actual}|__actual_total`) ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default async function RunDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const run = await getRun(id);
  if (!run && process.env.DATABASE_URL) notFound();

  const results = run ? await getResults(run.id) : [];
  const cells = run ? await getConfusionCells(run.id) : [];
  const geometryCells = cells.filter((cell) => cell.matrix_type === "geometry");
  const entityCells = cells.filter((cell) => cell.matrix_type === "entity");

  return (
    <main className="shell">
      <section className="header">
        <div>
          <div className="eyebrow">Run detail</div>
          <h1>{run?.id ?? id}</h1>
          <p>Inspect row-level predictions against gold labels and identify common label confusions.</p>
        </div>
        <Link className="button" href="/">All runs</Link>
      </section>

      {!run ? (
        <div className="notice">Run data is unavailable. Check <code>DATABASE_URL</code> and import the run artifacts.</div>
      ) : (
        <>
          <section className="grid metrics">
            <div className="metric"><div className="label">Model</div><div className="value" style={{ fontSize: 18 }}>{run.model}</div></div>
            <div className="metric"><div className="label">Completed rows</div><div className="value">{run.completed_rows}/{run.total_rows}</div></div>
            <div className="metric"><div className="label">Geometry accuracy</div><div className="value">{formatPercent(run.geometry_accuracy)}</div></div>
            <div className="metric"><div className="label">Entity accuracy</div><div className="value">{formatPercent(run.entity_accuracy)}</div></div>
            <div className="metric"><div className="label">Mean confidence</div><div className="value">{formatNumber(run.mean_confidence)}</div></div>
          </section>

          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", marginBottom: 20 }}>
            <ConfusionMatrix title="Geometry confusion matrix" cells={geometryCells} />
            <ConfusionMatrix title="Entity confusion matrix" cells={entityCells} />
          </div>

          <section className="card">
            <h2>Predictions</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th><th>Layer</th><th>Gold Geometry</th><th>Predicted Geometry</th><th>Gold Entity</th><th>Predicted Entity</th><th>Confidence</th><th>Status / reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result) => {
                    const hasMismatch = result.gold_geometry !== result.predicted_geometry || result.gold_entity !== result.predicted_entity;
                    return (
                      <tr key={result.row_index}>
                        <td>{result.row_index}</td>
                        <td><strong>{result.kaartlaag}</strong><br />{result.english_title ?? result.title}</td>
                        <td>{result.gold_geometry ?? "—"}</td>
                        <td>{result.predicted_geometry ?? "—"}</td>
                        <td>{result.gold_entity ?? "—"}</td>
                        <td>{result.predicted_entity ?? "—"}</td>
                        <td>{result.confidence?.toFixed(2) ?? "—"}</td>
                        <td>{result.error ? <span className="badge error">{result.error}</span> : <><span className={hasMismatch ? "badge error" : "badge"}>{hasMismatch ? "Mismatch" : "Match"}</span><p>{result.reasoning_summary}</p></>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
