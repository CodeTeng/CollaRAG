import { useState } from "react";
import {
  Send,
  MessageSquareText,
  Compass,
  Quote,
  Network,
  Activity,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { PageHeader } from "@/components/PageHeader";
import { IntentBadge } from "@/components/IntentBadge";
import { ChimeraAPI } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";
import { useAppStore } from "@/store";

const PRESET_QUESTIONS = [
  "Ada Verdant 在哪家公司工作？",
  "收购 Ada Verdant 所在公司的公司是哪家？",
  "比较 Ada Verdant 和 Bruno Falk 的研究方向。",
  "介绍一下 Cara Solis 在 Nimbus Labs 的角色。",
];

export function QueryPage() {
  const [text, setText] = useState("Ada Verdant 在哪家公司工作？");
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
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<MessageSquareText className="h-5 w-5" />}
        title="智能问答"
        subtitle="基于知识图谱 + IntentAgent 意图路由 · 6 类意图 → 6 种检索策略"
      />

      {/* 查询输入 */}
      <Card>
        <div className="flex flex-col gap-3">
          <div className="flex gap-2">
            <Input
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && run()}
              placeholder="请输入你的问题，按 Enter 提交…"
              className="flex-1"
            />
            <Button onClick={run} disabled={loading || !text.trim()}>
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> 正在思考
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" /> 提问
                </>
              )}
            </Button>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-[var(--color-text-muted)]">
            <span className="mr-1">示例：</span>
            {PRESET_QUESTIONS.map((q) => (
              <button
                key={q}
                className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-3 py-1 hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
                onClick={() => setText(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* 错误提示 */}
      {error && (
        <Card accent="intent">
          <div className="flex items-start gap-2 text-sm text-[var(--color-danger)]">
            <AlertTriangle className="mt-0.5 h-4 w-4" />
            <div>
              <div className="font-semibold">请求失败</div>
              <div className="text-[var(--color-text-muted)]">{error}</div>
            </div>
          </div>
        </Card>
      )}

      {/* 结果区 */}
      {result && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {/* 答案 */}
          <Card
            title="模型回答"
            titleIcon={<Quote className="h-4 w-4" />}
            className="md:col-span-2"
            accent={result.intent ? "intent" : undefined}
          >
            <p
              className="whitespace-pre-wrap text-base leading-relaxed text-[var(--color-text)]"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              {result.answer}
            </p>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              {result.intent && <IntentBadge kind="intent" value={result.intent} />}
              {result.strategy && (
                <IntentBadge kind="strategy" value={result.strategy} />
              )}
              <Badge tone="neutral">
                置信度 {(result.confidence * 100).toFixed(0)}%
              </Badge>
              <Badge tone="neutral">证据 {result.evidence_chunk_ids.length} 条</Badge>
            </div>
          </Card>

          {/* 检索元数据 */}
          <Card title="检索元数据" titleIcon={<Activity className="h-4 w-4" />}>
            <pre className="max-h-48 overflow-auto font-mono text-[0.7rem] leading-relaxed text-[var(--color-text-muted)]">
              {JSON.stringify(result.trace?.retrieval_metadata ?? {}, null, 2)}
            </pre>
          </Card>

          {/* 证据片段 */}
          <Card
            title={
              <>
                证据片段
                <span className="ml-2 text-xs font-normal text-[var(--color-text-muted)]">
                  {result.chunks.length} 条
                </span>
              </>
            }
            titleIcon={<Compass className="h-4 w-4" />}
            className="md:col-span-3"
          >
            <div className="flex flex-col gap-3">
              {result.chunks.map((c, i) => (
                <div
                  key={c.chunk_id}
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] p-3 text-sm"
                >
                  <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[0.7rem] text-[var(--color-text-subtle)]">
                    <span className="rounded bg-[var(--color-bg-soft)] px-1.5 py-0.5 font-mono">
                      #{i + 1}
                    </span>
                    <code className="rounded bg-[var(--color-bg-soft)] px-1.5 py-0.5 font-mono">
                      {c.chunk_id}
                    </code>
                    <span>文档: {c.doc_id}</span>
                    <span>·</span>
                    <span>序号 {c.index}</span>
                  </div>
                  <div className="leading-relaxed">{c.text}</div>
                </div>
              ))}
              {!result.chunks.length && (
                <div className="text-sm text-[var(--color-text-muted)]">
                  未检索到任何片段。
                </div>
              )}
            </div>
          </Card>

          {/* 三元组 */}
          {result.triples.length > 0 && (
            <Card
              title={
                <>
                  图谱三元组
                  <span className="ml-2 text-xs font-normal text-[var(--color-text-muted)]">
                    {result.triples.length} 条
                  </span>
                </>
              }
              titleIcon={<Network className="h-4 w-4" />}
              className="md:col-span-3"
              accent="adagraph"
            >
              <div className="flex flex-wrap gap-2">
                {result.triples.map((t, i) => (
                  <span
                    key={i}
                    className="group inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-2.5 py-1 text-xs"
                  >
                    <span className="font-medium text-[var(--color-adagraph)]">
                      {t.subject}
                    </span>
                    <span className="text-[var(--color-text-subtle)]">
                      —{t.predicate}→
                    </span>
                    <span className="font-medium text-[var(--color-adagraph)]">
                      {t.object}
                    </span>
                    <span className="ml-1 rounded bg-[var(--color-bg-soft)] px-1 font-mono text-[0.65rem] text-[var(--color-text-subtle)]">
                      {t.layer}
                    </span>
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
