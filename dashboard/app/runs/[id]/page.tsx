import Link from "next/link";
import { notFound } from "next/navigation";
import { formatNumber, formatPercent, getConfusionCells, getResults, getRun, type ConfusionCell, type RunSummary } from "../../../lib/db";
import { isPredictionMatch } from "../../../lib/scoring";

type LabelMetric = {
  label: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number | null;
  true_positive: number | null;
  false_positive: number | null;
  false_negative: number | null;
};

function provenanceText(provenance: Record<string, unknown> | null, key: string) {
  const value = provenance?.[key];
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return "—";
}

function shortHash(value: string) {
  return value.length > 12 ? value.slice(0, 12) : value;
}

function toFiniteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function getPerLabelMetrics(run: RunSummary, metricType: "geometry" | "entity") {
  const metrics = run.per_label_metrics?.[metricType];
  if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) return [];

  return Object.entries(metrics as Record<string, Record<string, unknown>>)
    .map(([label, values]) => ({
      label,
      precision: toFiniteNumber(values.precision),
      recall: toFiniteNumber(values.recall),
      f1: toFiniteNumber(values.f1),
      support: toFiniteNumber(values.support),
      true_positive: toFiniteNumber(values.true_positive),
      false_positive: toFiniteNumber(values.false_positive),
      false_negative: toFiniteNumber(values.false_negative),
    }))
    .sort((a, b) => (b.support ?? 0) - (a.support ?? 0) || a.label.localeCompare(b.label));
}

function barWidth(value: number | null) {
  return `${Math.max(0, Math.min(1, value ?? 0)) * 100}%`;
}

function ScoreBar({ value }: { value: number | null }) {
  return (
    <div className="score-cell">
      <span>{formatNumber(value)}</span>
      <div className="mini-bar" aria-hidden="true"><div style={{ width: barWidth(value) }} /></div>
    </div>
  );
}

function PerformanceHero({ run }: { run: RunSummary }) {
  const metricGroups = [
    { title: "Geometry", accuracy: run.geometry_accuracy, macroF1: run.geometry_macro_f1, hierarchicalF1: run.geometry_hier_f1 },
    { title: "Entity", accuracy: run.entity_accuracy, macroF1: run.entity_macro_f1, hierarchicalF1: run.entity_hier_f1 },
  ];

  return (
    <section className="performance-grid">
      {metricGroups.map((group) => (
        <div className="performance-card" key={group.title}>
          <div className="performance-card-header">
            <div>
              <div className="label">{group.title} evaluation</div>
              <div className="hero-value">{formatNumber(group.hierarchicalF1)}</div>
            </div>
            <span className="badge success">Hierarchical F1</span>
          </div>
          <div className="metric-bars">
            <div><span>Accuracy</span><strong>{formatPercent(group.accuracy)}</strong><div className="progress-bar"><div style={{ width: barWidth(group.accuracy) }} /></div></div>
            <div><span>Macro F1</span><strong>{formatNumber(group.macroF1)}</strong><div className="progress-bar"><div style={{ width: barWidth(group.macroF1) }} /></div></div>
            <div><span>Hierarchical F1</span><strong>{formatNumber(group.hierarchicalF1)}</strong><div className="progress-bar"><div style={{ width: barWidth(group.hierarchicalF1) }} /></div></div>
          </div>
        </div>
      ))}
    </section>
  );
}

function PerLabelMetricsTable({ title, metrics }: { title: string; metrics: LabelMetric[] }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      <p>Precision, recall, and F1 by label, sorted by support so high-volume classes are easiest to audit.</p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Label</th><th>F1</th><th>Precision</th><th>Recall</th><th>Support</th><th>TP / FP / FN</th></tr></thead>
          <tbody>
            {metrics.map((metric) => (
              <tr key={metric.label}>
                <td><strong>{metric.label}</strong></td>
                <td><ScoreBar value={metric.f1} /></td>
                <td><ScoreBar value={metric.precision} /></td>
                <td><ScoreBar value={metric.recall} /></td>
                <td>{metric.support ?? "—"}</td>
                <td>{metric.true_positive ?? "—"} / {metric.false_positive ?? "—"} / {metric.false_negative ?? "—"}</td>
              </tr>
            ))}
            {metrics.length === 0 ? <tr><td colSpan={6}>No per-label metrics were saved for this run.</td></tr> : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

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
                  return <td className={isPredictionMatch(actual, predicted) ? "diagonal" : value > 0 ? "off-diagonal" : ""} key={predicted}>{value}</td>;
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
  const geometryMetrics = run ? getPerLabelMetrics(run, "geometry") : [];
  const entityMetrics = run ? getPerLabelMetrics(run, "entity") : [];

  return (
    <main className="shell">
      <section className="header">
        <div>
          <div className="eyebrow">Run detail</div>
          <h1>{run?.id ?? id}</h1>
          <p>Inspect headline performance, per-label precision/recall/F1, row-level predictions, and common label confusions.</p>
        </div>
        <Link className="button" href="/">All runs</Link>
      </section>

      {!run ? (
        <div className="notice">Run data is unavailable. Check <code>DATABASE_URL</code> and import the run artifacts.</div>
      ) : (
        <>
          <PerformanceHero run={run} />

          <section className="grid metrics">
            <div className="metric"><div className="label">Model</div><div className="value" style={{ fontSize: 18 }}>{run.model}</div></div>
            <div className="metric"><div className="label">Completed rows</div><div className="value">{run.completed_rows}/{run.total_rows}</div></div>
            <div className="metric"><div className="label">Joint accuracy</div><div className="value">{formatPercent(run.joint_accuracy)}</div></div>
            <div className="metric"><div className="label">Exact mismatches</div><div className="value">{run.exact_mismatch_count ?? "—"}</div></div>
            <div className="metric"><div className="label">Mean confidence</div><div className="value">{formatNumber(run.mean_confidence)}</div></div>
          </section>

          <div className="grid metrics-detail">
            <PerLabelMetricsTable title="Geometry label performance" metrics={geometryMetrics} />
            <PerLabelMetricsTable title="Entity label performance" metrics={entityMetrics} />
          </div>

          <section className="card" style={{ marginBottom: 20 }}>
            <h2>Run provenance</h2>
            <div className="table-wrap">
              <table>
                <tbody>
                  <tr><th>Git commit</th><td>{shortHash(provenanceText(run.provenance, "git_commit"))}</td><th>Git branch</th><td>{provenanceText(run.provenance, "git_branch")}</td></tr>
                  <tr><th>Git dirty</th><td>{provenanceText(run.provenance, "git_dirty")}</td><th>Script</th><td>{provenanceText(run.provenance, "script_path")}</td></tr>
                  <tr><th>Script SHA-256</th><td>{shortHash(provenanceText(run.provenance, "script_sha256"))}</td><th>Decision tree SHA-256</th><td>{shortHash(provenanceText(run.provenance, "decision_tree_sha256"))}</td></tr>
                  <tr><th>Input CSV SHA-256</th><td>{shortHash(provenanceText(run.provenance, "input_csv_sha256"))}</td><th>Normal F1 (Macro)</th><td>Geometry {formatNumber(run.geometry_macro_f1)} / Entity {formatNumber(run.entity_macro_f1)}</td></tr>
                  <tr><th>Hierarchical F1</th><td>Geometry {formatNumber(run.geometry_hier_f1)}</td><th></th><td>Entity {formatNumber(run.entity_hier_f1)}</td></tr>
                </tbody>
              </table>
            </div>
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
                    const hasMismatch = !isPredictionMatch(result.gold_geometry, result.predicted_geometry) || !isPredictionMatch(result.gold_entity, result.predicted_entity);
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
