import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

const API = "/api";

async function jget(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export default function App() {
  const [dataset, setDataset] = useState("nsl_kdd");
  const [featuresText, setFeaturesText] = useState("{\n  \"count\": 210,\n  \"serror_rate\": 0.92,\n  \"src_bytes\": 40\n}");
  const [pred, setPred] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [comparison, setComparison] = useState([]);
  const [importance, setImportance] = useState(null);
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState(null);
  const [err, setErr] = useState("");

  const load = async () => {
    try {
      setErr("");
      const [m, c, i, h, s] = await Promise.all([
        jget("/metrics"),
        jget("/comparison"),
        jget("/feature-importance?dataset=" + dataset),
        jget("/history?limit=50"),
        jget("/analytics/summary"),
      ]);
      setMetrics(m);
      setComparison(c);
      setImportance(i);
      setHistory(h);
      setSummary(s);
    } catch (e) {
      setErr(String(e.message || e));
    }
  };

  useEffect(() => { load(); }, [dataset]);

  const submit = async () => {
    try {
      setErr("");
      const r = await fetch(API + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, features: JSON.parse(featuresText) }),
      });
      if (!r.ok) throw new Error(await r.text());
      setPred(await r.json());
      await load();
    } catch (e) {
      setErr(String(e.message || e));
    }
  };

  const uploadCsv = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(API + "/predict/csv?dataset=" + dataset, { method: "POST", body: fd });
    if (!r.ok) { setErr(await r.text()); return; }
    await load();
  };

  const dist = summary?.categories || {};
  const cm = useMemo(() => {
    const row = metrics.find((x) => x.dataset === dataset && x.model_name === "proposed_binary");
    return row?.extra?.confusion_matrix || [];
  }, [metrics, dataset]);

  const perfPlot = comparison.filter((d) => d.dataset === dataset);
  const accTrace = perfPlot.length
    ? [
        { x: ["Baseline RF", "Proposed RF"], y: [perfPlot[0].baseline_binary.accuracy, perfPlot[0].proposed_binary.accuracy], name: "Accuracy", type: "bar" },
        { x: ["Baseline RF", "Proposed RF"], y: [perfPlot[0].baseline_binary.f1_score, perfPlot[0].proposed_binary.f1_score], name: "F1", type: "bar" },
      ]
    : [];
  const costTrace = perfPlot.length
    ? [
        { x: ["Baseline", "Proposed"], y: [perfPlot[0].baseline_binary.train_seconds, perfPlot[0].proposed_binary.train_seconds], name: "Train s", type: "bar" },
        { x: ["Baseline", "Proposed"], y: [perfPlot[0].baseline_binary.predict_seconds, perfPlot[0].proposed_binary.predict_seconds], name: "Predict s", type: "bar" },
      ]
    : [];
  const imp = importance
    ? Object.entries(importance.ranks?.random_forest || {}).slice(0, 15)
    : [];

  return (
    <>
      <header>
        <h1>IoMT Intrusion Detection & Threat Analytics</h1>
        <div style={{ width: 220 }}>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="nsl_kdd">NSL-KDD</option>
            <option value="ciciomt2024">CICIoMT2024</option>
            <option value="wustl_ehms">WUSTL-EHMS-2020</option>
          </select>
        </div>
      </header>
      {err && <div className="card" style={{ margin: "12px 24px", color: "#ff5d73" }}>{err}</div>}
      <div className="grid">
        <div className="card kpi wide">
          <div><small>Predictions</small><div>{summary?.total_predictions ?? 0}</div></div>
          <div><small>Malicious</small><div className="mal">{summary?.malicious ?? 0}</div></div>
          <div><small>Normal</small><div className="ok">{summary?.normal ?? 0}</div></div>
          <div><small>Avg latency ms</small><div>{(summary?.avg_execution_ms || 0).toFixed(2)}</div></div>
        </div>
        <div className="card">
          <h3>Predict traffic</h3>
          <label>Feature JSON</label>
          <textarea rows={10} value={featuresText} onChange={(e) => setFeaturesText(e.target.value)} />
          <button onClick={submit}>Run prediction</button>
          <label>CSV upload</label>
          <input type="file" accept=".csv" onChange={(e) => e.target.files[0] && uploadCsv(e.target.files[0])} />
          {pred && (
            <p>
              Result: <b className={pred.binary_label === "malicious" ? "mal" : "ok"}>{pred.binary_label}</b> / {pred.attack_category}
              <br />confidence {(pred.confidence * 100).toFixed(1)}% · {pred.execution_ms.toFixed(2)} ms
            </p>
          )}
        </div>
        <div className="card">
          <h3>Explainable features</h3>
          <Plot
            data={[{ type: "bar", orientation: "h", y: (pred?.explanation || imp.map(([k]) => ({ feature: k })) ).map((x) => x.feature || x[0]), x: (pred?.explanation || imp.map(([k, v]) => ({ importance: v }))).map((x) => x.importance || x[1]) }]}
            layout={{ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#e8eefc" }, margin: { l: 180, r: 10, t: 10, b: 30 }, height: 360 }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </div>
        <div className="card">
          <h3>Attack distribution</h3>
          <Plot data={[{ type: "pie", labels: Object.keys(dist), values: Object.values(dist) }]} layout={{ paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#e8eefc" }, height: 320, margin: { t: 10 } }} style={{ width: "100%" }} useResizeHandler />
        </div>
        <div className="card">
          <h3>Detection performance</h3>
          <Plot data={accTrace} layout={{ barmode: "group", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#e8eefc" }, height: 320 }} style={{ width: "100%" }} useResizeHandler />
        </div>
        <div className="card">
          <h3>Computational cost</h3>
          <Plot data={costTrace} layout={{ barmode: "group", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#e8eefc" }, height: 320 }} style={{ width: "100%" }} useResizeHandler />
        </div>
        <div className="card">
          <h3>Confusion matrix (proposed binary)</h3>
          <Plot data={[{ z: cm, type: "heatmap", colorscale: "Blues" }]} layout={{ paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#e8eefc" }, height: 300 }} style={{ width: "100%" }} useResizeHandler />
        </div>
        <div className="card">
          <h3>Model metrics</h3>
          <table>
            <thead><tr><th>Model</th><th>Acc</th><th>P</th><th>R</th><th>F1</th><th>Feat</th><th>Train s</th></tr></thead>
            <tbody>
              {metrics.filter((m) => m.dataset === dataset).slice(0, 8).map((m, i) => (
                <tr key={i}><td>{m.model_name}</td><td>{m.accuracy.toFixed(3)}</td><td>{m.precision.toFixed(3)}</td><td>{m.recall.toFixed(3)}</td><td>{m.f1_score.toFixed(3)}</td><td>{m.n_features}</td><td>{m.train_seconds.toFixed(2)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card wide">
          <h3>Historical predictions</h3>
          <table>
            <thead><tr><th>Time</th><th>Label</th><th>Category</th><th>Conf</th><th>ms</th></tr></thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.id}>
                  <td>{h.created_at}</td>
                  <td className={h.binary_label === "malicious" ? "mal" : "ok"}>{h.binary_label}</td>
                  <td>{h.attack_category}</td>
                  <td>{(h.confidence * 100).toFixed(1)}%</td>
                  <td>{h.execution_ms.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
