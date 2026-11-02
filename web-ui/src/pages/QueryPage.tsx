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
  Bot,
  GitBranch,
  RotateCcw,
  Wrench,
  CheckCircle2,
  ChevronDown,
  Gauge,
  Pen,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { PageHeader } from "@/components/PageHeader";
import { IntentBadge } from "@/components/IntentBadge";
import { ChimeraAPI } from "@/lib/api";
import { extractErrorMessage, type QueryResponse } from "@/lib/types";
import { useAppStore } from "@/store";

const PRESET_QUESTIONS = [
  "Ada Verdant 在哪家公司工作？",
  "收购 Ada Verdant 所在公司的公司是哪家？",
  "比较 Ada Verdant 和 Bruno Falk 的研究方向。",
  "介绍一下 Cara Solis 在 Nimbus Labs 的角色。",
];

const AGENT_LABELS: Record<string, string> = {
  preprocessor: "预处理智能体",
  single_hop: "单跳智能体",
  multi_hop: "多跳智能体",
  summarization: "摘要智能体",
  other: "兜底智能体",
  greeting: "问候",
  cache: "缓存命中",
};

const STRATEGY_LABELS: Record<string, string> = {
  native_rag: "单轮检索",
  plan_execute: "Plan-Execute-Reflect",
  map_reduce: "Map-Reduce 摘要",
  web_search: "Web + 本地混合",
  greeting: "直接回复",
  cache: "QA 缓存",
};

function QualityGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80
      ? "text-emerald-400"
      : pct >= 60
        ? "text-amber-400"
        : "text-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-2 w-20 overflow-hidden rounded-full bg-[var(--color-surface-hi)]">
        <div
          className={`absolute left-0 top-0 h-full rounded-full ${
            pct >= 80
              ? "bg-emerald-500"
              : pct >= 60
                ? "bg-amber-500"
                : "bg-red-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs font-semibold tabular-nums ${color}`}>
        {pct}%
      </span>
    </div>
  );
}

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
    } catch (e: unknown) {
      setError(extractErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const at = result?.agent_trace;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<MessageSquareText className="h-5 w-5" />}
        title="智能问答"
        subtitle="CollaRAG 多智能体协作 · 5 智能体 × 31 工具 · 意图路由 → 专长执行 → 反思验证"
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

          {/* 智能体调度卡 */}
          <Card
            title="智能体调度"
            titleIcon={<Bot className="h-4 w-4" />}
            accent="intent"
          >
            <div className="flex flex-col gap-3">
              {/* 派发的智能体 */}
              {at?.agent_type && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--color-text-subtle)]">
                    派发至
                  </span>
                  <span className="rounded-md border border-[var(--color-intent)]/40 bg-[var(--color-intent-soft)] px-2 py-0.5 text-xs font-semibold text-[var(--color-intent)]">
                    {AGENT_LABELS[at.agent_type] || at.agent_type}
                  </span>
                </div>
              )}

              {/* 策略 */}
              {result.strategy && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--color-text-subtle)]">
                    策略
                  </span>
                  <span className="text-xs font-medium text-[var(--color-text)]">
                    {STRATEGY_LABELS[result.strategy] || result.strategy}
                  </span>
                </div>
              )}

              {/* 质量评分 */}
              {at?.quality != null && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--color-text-subtle)]">
                    质量
                  </span>
                  <QualityGauge value={at.quality} />
                </div>
              )}

              {/* 反思轮数 */}
              {at != null && at.reflect_count > 0 && (
                <div className="flex items-center gap-2">
                  <RotateCcw className="h-3 w-3 text-[var(--color-warning)]" />
                  <span className="text-xs text-[var(--color-text-muted)]">
                    反思 {at.reflect_count} 轮
                  </span>
                </div>
              )}

              {/* 迭代次数 */}
              {at?.iterations != null && (
                <div className="flex items-center gap-2">
                  <GitBranch className="h-3 w-3 text-[var(--color-text-subtle)]" />
                  <span className="text-xs text-[var(--color-text-muted)]">
                    工具迭代 {at.iterations} 次
                  </span>
                </div>
              )}

              {/* Map 数量 */}
              {at?.map_count != null && (
                <div className="flex items-center gap-2">
                  <Gauge className="h-3 w-3 text-[var(--color-text-subtle)]" />
                  <span className="text-xs text-[var(--color-text-muted)]">
                    Map 摘要 {at.map_count} 条
                  </span>
                </div>
              )}
            </div>
          </Card>

          {/* 查询改写（如果发生了改写） */}
          {at?.rewritten_query &&
            at.original_query &&
            at.rewritten_query !== at.original_query && (
              <Card
                title="查询改写"
                titleIcon={<Pen className="h-4 w-4" />}
                className="md:col-span-3"
              >
                <div className="flex flex-col gap-2 text-sm">
                  <div className="flex items-start gap-2">
                    <span className="shrink-0 rounded bg-[var(--color-surface-hi)] px-1.5 py-0.5 text-[0.65rem] text-[var(--color-text-subtle)]">
                      原始
                    </span>
                    <span className="text-[var(--color-text-muted)]">
                      {at.original_query}
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="shrink-0 rounded bg-[var(--color-intent-soft)] px-1.5 py-0.5 text-[0.65rem] text-[var(--color-intent)]">
                      改写
                    </span>
                    <span className="font-medium text-[var(--color-text)]">
                      {at.rewritten_query}
                    </span>
                  </div>
                </div>
              </Card>
            )}

          {/* 工具调用时间线 */}
          {at?.tool_call_log && at.tool_call_log.length > 0 && (
            <Card
              title={
                <>
                  工具调用时间线
                  <span className="ml-2 text-xs font-normal text-[var(--color-text-muted)]">
                    {at.tool_call_log.length} 次调用
                  </span>
                </>
              }
              titleIcon={<Wrench className="h-4 w-4" />}
              className="md:col-span-3"
              accent="intent"
            >
              <div className="flex flex-wrap gap-1">
                {at.tool_call_log.map((tool, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-2 py-1 text-xs"
                  >
                    <span className="rounded bg-[var(--color-bg-soft)] px-1 py-px font-mono text-[0.6rem] text-[var(--color-text-subtle)]">
                      {i + 1}
                    </span>
                    <code className="font-mono text-[var(--color-intent)]">
                      {tool}
                    </code>
                  </span>
                ))}
              </div>
            </Card>
          )}

          {/* Plan 步骤（多跳智能体） */}
          {at?.plan && at.plan.length > 0 && (
            <Card
              title="执行计划"
              titleIcon={<GitBranch className="h-4 w-4" />}
              className="md:col-span-3"
              accent="intent"
            >
              <div className="flex flex-col gap-2">
                {at.plan.map((step, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 rounded-md border border-[var(--color-border)]/60 bg-[var(--color-bg-soft)] p-3"
                  >
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--color-intent)]/20 text-xs font-bold text-[var(--color-intent)]">
                      {i + 1}
                    </span>
                    <div className="flex-1 text-sm">
                      <div className="font-medium text-[var(--color-text)]">
                        {(step.step as string) || JSON.stringify(step)}
                      </div>
                      {Array.isArray(step.tools_hint) &&
                        (step.tools_hint as string[]).length > 0 && (
                          <div className="mt-1 flex gap-1">
                            {(step.tools_hint as string[]).map((t) => (
                              <code
                                key={t}
                                className="rounded bg-[var(--color-surface-hi)] px-1 py-px font-mono text-[0.6rem] text-[var(--color-text-subtle)]"
                              >
                                {t}
                              </code>
                            ))}
                          </div>
                        )}
                    </div>
                    {/* step result */}
                    {at.step_results[i] && (
                      <details className="group text-xs">
                        <summary className="cursor-pointer text-[var(--color-text-subtle)] hover:text-[var(--color-text)]">
                          <ChevronDown className="inline h-3 w-3 transition group-open:rotate-180" />{" "}
                          结果
                        </summary>
                        <div className="mt-1 max-h-32 overflow-auto rounded bg-[var(--color-surface)] p-2 text-[var(--color-text-muted)]">
                          {at.step_results[i]}
                        </div>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

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

          {/* 检索元数据 */}
          <Card
            title="检索元数据"
            titleIcon={<Activity className="h-4 w-4" />}
            className="md:col-span-3"
          >
            <details className="group">
              <summary className="flex cursor-pointer items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]">
                <ChevronDown className="h-3 w-3 transition group-open:rotate-180" />
                展开原始 trace JSON
              </summary>
              <pre className="mt-2 max-h-48 overflow-auto font-mono text-[0.7rem] leading-relaxed text-[var(--color-text-muted)]">
                {JSON.stringify(result.trace?.retrieval_metadata ?? {}, null, 2)}
              </pre>
            </details>
          </Card>
        </div>
      )}
    </div>
  );
}
