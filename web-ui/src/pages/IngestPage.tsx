import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Textarea } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { ChimeraAPI } from "@/lib/api";
import type { IngestResponse } from "@/lib/types";

export function IngestPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selected, setSelected] = useState("sample");
  const [rawDocs, setRawDocs] = useState("");
  const [docId, setDocId] = useState("manual-1");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    ChimeraAPI.getConfig()
      .then((c) => {
        const ds = (c.config as any).datasets as Record<string, unknown> | undefined;
        setDatasets(Object.keys(ds ?? {}));
      })
      .catch(() => {});
  }, []);

  const runDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await ChimeraAPI.ingest({ dataset: selected });
      setResult(r);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  };

  const runInline = async () => {
    if (!rawDocs.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await ChimeraAPI.ingest({
        documents: [{ doc_id: docId, content: rawDocs }],
      });
      setResult(r);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      <h1 className="text-xl font-semibold">Ingest</h1>

      <Card title="From configured dataset">
        <div className="flex gap-2">
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-3 py-2 text-sm"
          >
            {datasets.length === 0 && <option value="sample">sample</option>}
            {datasets.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <Button onClick={runDataset} disabled={loading}>
            {loading ? "Ingesting..." : "Ingest dataset"}
          </Button>
        </div>
      </Card>

      <Card title="From inline text">
        <div className="flex flex-col gap-3">
          <Input value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="doc_id" />
          <Textarea
            rows={6}
            value={rawDocs}
            onChange={(e) => setRawDocs(e.target.value)}
            placeholder="Paste document content here..."
          />
          <Button onClick={runInline} disabled={loading || !rawDocs.trim()}>
            Ingest inline
          </Button>
        </div>
      </Card>

      {error && (
        <Card>
          <div className="text-sm text-red-400">Error: {error}</div>
        </Card>
      )}

      {result && (
        <Card title="Result">
          <div className="flex flex-wrap gap-2">
            <Badge tone="accent">docs: {result.stats.documents}</Badge>
            <Badge>chunks: {result.stats.chunks}</Badge>
            <Badge>triples raw: {result.stats.triples_raw}</Badge>
            <Badge tone="success">triples pruned: {result.stats.triples_pruned}</Badge>
          </div>
          <div className="mt-3 text-sm text-[var(--color-text-muted)]">
            State totals — chunks: {result.state.chunks}, triples: {result.state.triples}
          </div>
        </Card>
      )}
    </div>
  );
}
