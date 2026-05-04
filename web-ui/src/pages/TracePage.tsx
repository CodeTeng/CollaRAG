import { History, Trash2, ChevronDown } from "lucide-react";
import { useAppStore } from "@/store";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { PageHeader } from "@/components/PageHeader";
import { IntentBadge } from "@/components/IntentBadge";

export function TracePage() {
  const traces = useAppStore((s) => s.traces);
  const clearTraces = useAppStore((s) => s.clearTraces);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<History className="h-5 w-5" />}
        title="历史追踪"
        subtitle={`本次会话共查询 ${traces.length} 次 · 数据仅保存在浏览器端`}
        actions={
          <Button
            variant="secondary"
            onClick={clearTraces}
            disabled={!traces.length}
          >
            <Trash2 className="h-4 w-4" /> 清空记录
          </Button>
        }
      />

      {!traces.length && (
        <Card>
          <div className="text-sm text-[var(--color-text-muted)]">
            还没有查询记录。去{" "}
            <code className="font-mono text-[var(--color-accent)]">智能问答</code>{" "}
            页面提问试试。
          </div>
        </Card>
      )}

      <div className="flex flex-col gap-3">
        {traces.map((t) => (
          <Card
            key={t.id}
            title={
              <span className="line-clamp-1 max-w-2xl">{t.question}</span>
            }
            actions={
              <span className="text-xs text-[var(--color-text-muted)]">
                {new Date(t.at).toLocaleString("zh-CN", { hour12: false })}
              </span>
            }
          >
            <div className="mb-3 flex flex-wrap gap-2">
              {t.response.intent && (
                <IntentBadge kind="intent" value={t.response.intent} />
              )}
              {t.response.strategy && (
                <IntentBadge kind="strategy" value={t.response.strategy} />
              )}
              <Badge tone="neutral">证据 {t.response.chunks.length} 条</Badge>
              <Badge tone="neutral">三元组 {t.response.triples.length} 条</Badge>
              <Badge tone="neutral">
                置信度 {(t.response.confidence * 100).toFixed(0)}%
              </Badge>
            </div>
            <div className="mb-3 rounded-md bg-[var(--color-bg-soft)] p-3 text-sm leading-relaxed">
              <span className="text-xs uppercase tracking-wider text-[var(--color-text-subtle)]">
                回答
              </span>
              <div className="mt-1" style={{ fontFamily: "var(--font-serif)" }}>
                {t.response.answer}
              </div>
            </div>
            <details className="group text-xs">
              <summary className="flex cursor-pointer items-center gap-1 text-[var(--color-text-muted)] hover:text-[var(--color-text)]">
                <ChevronDown className="h-3 w-3 transition group-open:rotate-180" />
                查看原始 trace
              </summary>
              <pre className="mt-2 max-h-64 overflow-auto rounded bg-[var(--color-bg-soft)] p-2 font-mono text-[0.7rem] text-[var(--color-text-muted)]">
                {JSON.stringify(t.response.trace, null, 2)}
              </pre>
            </details>
          </Card>
        ))}
      </div>
    </div>
  );
}
