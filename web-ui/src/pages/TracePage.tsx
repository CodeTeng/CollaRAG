import { useAppStore } from "@/store";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export function TracePage() {
  const traces = useAppStore((s) => s.traces);
  const clearTraces = useAppStore((s) => s.clearTraces);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Trace</h1>
        <Button variant="secondary" onClick={clearTraces} disabled={!traces.length}>
          Clear
        </Button>
      </div>

      {!traces.length && (
        <Card>
          <div className="text-sm text-[var(--color-text-muted)]">
            No queries yet. Go to <code>/query</code> and ask something.
          </div>
        </Card>
      )}

      {traces.map((t) => (
        <Card
          key={t.id}
          title={<span className="truncate">{t.question}</span>}
          actions={
            <span className="text-xs text-[var(--color-text-muted)]">
              {new Date(t.at).toLocaleTimeString()}
            </span>
          }
        >
          <div className="mb-2 flex flex-wrap gap-2">
            {t.response.intent && <Badge tone="accent">intent: {t.response.intent}</Badge>}
            {t.response.strategy && (
              <Badge tone="success">strategy: {t.response.strategy}</Badge>
            )}
            <Badge>{t.response.chunks.length} chunks</Badge>
            <Badge>{t.response.triples.length} triples</Badge>
          </div>
          <div className="mb-2 text-sm">
            <span className="text-[var(--color-text-muted)]">answer: </span>
            {t.response.answer}
          </div>
          <details className="text-xs">
            <summary className="cursor-pointer text-[var(--color-text-muted)]">
              raw trace
            </summary>
            <pre className="mt-2 max-h-64 overflow-auto">
              {JSON.stringify(t.response.trace, null, 2)}
            </pre>
          </details>
        </Card>
      ))}
    </div>
  );
}
