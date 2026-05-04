import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { ChimeraAPI } from "@/lib/api";
import type { EvaluateResponse } from "@/lib/types";

export function EvaluatePage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [limit, setLimit] = useState("20");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluateResponse | null>(null);

  useEffect(() => {
    ChimeraAPI.getConfig().then((c) => {
      const ds = Object.keys((c.config as any).datasets ?? {});
      setDatasets(ds);
      if (ds.length) setSelected(ds[0]);
    });
  }, []);

  const run = async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      const r = await ChimeraAPI.evaluate({
        dataset: selected,
        limit: limit ? Number(limit) : undefined,
      });
      setResult(r);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <h1 className="text-xl font-semibold">Evaluate</h1>

      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col">
            <label className="mb-1 text-xs text-[var(--color-text-muted)]">Dataset</label>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-3 py-2 text-sm"
            >
              {datasets.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col">
            <label className="mb-1 text-xs text-[var(--color-text-muted)]">Limit</label>
            <Input
              type="number"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              className="w-28"
            />
          </div>
          <Button onClick={run} disabled={loading || !selected}>
            {loading ? "Running..." : "Run evaluation"}
          </Button>
        </div>
      </Card>

      {error && (
        <Card>
          <div className="text-sm text-red-400">Error: {error}</div>
        </Card>
      )}

      {result && (
        <>
          <Card title={`Result — ${result.config_name}`}>
            <div className="flex flex-wrap gap-3">
              {Object.entries(result.metrics).map(([k, v]) => (
                <div
                  key={k}
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-3 py-2"
                >
                  <div className="text-xs uppercase text-[var(--color-text-muted)]">{k}</div>
                  <div className="text-lg font-semibold text-[var(--color-accent)]">
                    {v.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title={`Per-example (${result.per_example.length})`}>
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-[var(--color-text-muted)]">
                  <tr>
                    <th className="px-2 py-1">qid</th>
                    <th className="px-2 py-1">question</th>
                    <th className="px-2 py-1">reference</th>
                    <th className="px-2 py-1">prediction</th>
                    <th className="px-2 py-1">EM</th>
                    <th className="px-2 py-1">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_example.map((row: any, i: number) => (
                    <tr key={i} className="border-t border-[var(--color-border)]">
                      <td className="px-2 py-1"><code>{row.qid}</code></td>
                      <td className="px-2 py-1 max-w-[200px] truncate">{row.question}</td>
                      <td className="px-2 py-1 max-w-[180px] truncate">{row.reference}</td>
                      <td className="px-2 py-1 max-w-[180px] truncate">{row.prediction}</td>
                      <td className="px-2 py-1">
                        <Badge tone={row.em === 1 ? "success" : "danger"}>
                          {row.em?.toFixed(2)}
                        </Badge>
                      </td>
                      <td className="px-2 py-1">
                        <Badge tone="accent">{row.f1?.toFixed(2)}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
