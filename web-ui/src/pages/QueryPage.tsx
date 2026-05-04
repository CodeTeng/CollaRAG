import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { ChimeraAPI } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";
import { useAppStore } from "@/store";

export function QueryPage() {
  const [text, setText] = useState("Who invented dynamite?");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const addTrace = useAppStore((s) => s.addTrace);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await ChimeraAPI.query({ text });
      setResult(r);
      addTrace(text, r);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <h1 className="text-xl font-semibold">Query</h1>

      <Card>
        <div className="flex gap-2">
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Ask anything about the ingested knowledge graph..."
          />
          <Button onClick={run} disabled={loading || !text.trim()}>
            {loading ? "Querying..." : "Ask"}
          </Button>
        </div>
      </Card>

      {error && (
        <Card>
          <div className="text-sm text-red-400">Error: {error}</div>
        </Card>
      )}

      {result && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <Card title="Answer" className="md:col-span-2">
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              {result.answer}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {result.intent && <Badge tone="accent">intent: {result.intent}</Badge>}
              {result.strategy && <Badge tone="success">strategy: {result.strategy}</Badge>}
              <Badge>confidence: {result.confidence.toFixed(2)}</Badge>
              <Badge>evidence: {result.evidence_chunk_ids.length}</Badge>
            </div>
          </Card>

          <Card title="Retrieval metadata">
            <pre className="max-h-48 overflow-auto text-xs text-[var(--color-text-muted)]">
              {JSON.stringify(result.trace?.retrieval_metadata ?? {}, null, 2)}
            </pre>
          </Card>

          <Card title={`Evidence chunks (${result.chunks.length})`} className="md:col-span-3">
            <div className="flex flex-col gap-3">
              {result.chunks.map((c) => (
                <div
                  key={c.chunk_id}
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] p-3 text-sm"
                >
                  <div className="mb-1 flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
                    <code>{c.chunk_id}</code>
                    <span>doc={c.doc_id}</span>
                    <span>#{c.index}</span>
                  </div>
                  <div>{c.text}</div>
                </div>
              ))}
              {!result.chunks.length && (
                <div className="text-sm text-[var(--color-text-muted)]">
                  No chunks retrieved.
                </div>
              )}
            </div>
          </Card>

          {result.triples.length > 0 && (
            <Card title={`Triples (${result.triples.length})`} className="md:col-span-3">
              <div className="flex flex-wrap gap-2">
                {result.triples.map((t, i) => (
                  <span
                    key={i}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-2 py-1 text-xs"
                  >
                    <span className="text-[var(--color-accent)]">{t.subject}</span>{" "}
                    — {t.predicate} →{" "}
                    <span className="text-[var(--color-accent)]">{t.object}</span>
                    <span className="ml-2 text-[var(--color-text-muted)]">[{t.layer}]</span>
                  </span>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
