import { History, Trash2, ChevronDown, Bot, Wrench, RotateCcw, Gauge } from "lucide-react";
import { useAppStore } from "@/store";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { PageHeader } from "@/components/PageHeader";
import { IntentBadge } from "@/components/IntentBadge";

const AGENT_LABELS: Record<string, string> = {
  preprocessor: "预处理",
  single_hop: "单跳",
  multi_hop: "多跳",
  summarization: "摘要",
  other: "兜底",
  greeting: "问候",
  cache: "缓存",
};

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
        {traces.map((t) => {
          const at = t.response.agent_trace;
          return (
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
              {/* 标签行 */}
              <div className="mb-3 flex flex-wrap gap-2">
                {/* 智能体标签 */}
                {at?.agent_type && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-[var(--color-intent)]/40 bg-[var(--color-intent-soft)] px-2.5 py-0.5 text-xs font-medium text-[var(--color-intent)]">
                    <Bot className="h-3 w-3" />
                    {AGENT_LABELS[at.agent_type] || at.agent_type}
                  </span>
                )}
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

              {/* 智能体调度详情 */}
              {at && (at.tool_call_log?.length > 0 || at.reflect_count > 0 || at.quality != null) && (
                <div className="mb-3 flex flex-wrap items-center gap-3 rounded-md bg-[var(--color-bg-soft)] px-3 py-2 text-xs text-[var(--color-text-muted)]">
                  {at.tool_call_log?.length > 0 && (
                    <span className="flex items-center gap-1">
                      <Wrench className="h-3 w-3 text-[var(--color-intent)]" />
                      工具调用 {at.tool_call_log.length} 次
                    </span>
                  )}
                  {at.reflect_count > 0 && (
                    <span className="flex items-center gap-1">
                      <RotateCcw className="h-3 w-3 text-[var(--color-warning)]" />
                      反思 {at.reflect_count} 轮
                    </span>
                  )}
                  {at.quality != null && (
                    <span className="flex items-center gap-1">
                      <Gauge className="h-3 w-3 text-[var(--color-accent)]" />
                      质量 {Math.round(at.quality * 100)}%
                    </span>
                  )}
                  {at.map_count != null && (
                    <span className="text-[var(--color-text-subtle)]">
                      Map 摘要 {at.map_count} 条
                    </span>
                  )}
                </div>
              )}

              {/* 工具调用序列 */}
              {at?.tool_call_log && at.tool_call_log.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-1">
                  {at.tool_call_log.map((tool, i) => (
                    <code
                      key={i}
                      className="rounded border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-1.5 py-0.5 font-mono text-[0.6rem] text-[var(--color-intent)]"
                    >
                      {tool}
                    </code>
                  ))}
                </div>
              )}

              {/* 回答 */}
              <div className="mb-3 rounded-md bg-[var(--color-bg-soft)] p-3 text-sm leading-relaxed">
                <span className="text-xs uppercase tracking-wider text-[var(--color-text-subtle)]">
                  回答
                </span>
                <div className="mt-1" style={{ fontFamily: "var(--font-serif)" }}>
                  {t.response.answer}
                </div>
              </div>

              {/* 原始 trace */}
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
          );
        })}
      </div>
    </div>
  );
}
