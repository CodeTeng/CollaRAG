import { useEffect, useState } from "react";
import {
  Bot,
  Sparkles,
  Wrench,
  CheckCircle2,
  XCircle,
  Brain,
  Cpu,
  Zap,
  GitBranch,
  FileText,
  Globe,
  Filter,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { PageHeader } from "@/components/PageHeader";
import { StatCard } from "@/components/StatCard";
import { ChimeraAPI } from "@/lib/api";
import type {
  AgentsOverviewResponse,
  ToolMatrixResponse,
} from "@/lib/types";

const AGENT_ICONS: Record<string, typeof Bot> = {
  filter: Filter,
  zap: Zap,
  "git-branch": GitBranch,
  "file-text": FileText,
  globe: Globe,
};

const AGENT_CN_SHORT: Record<string, string> = {
  preprocessor: "预处理",
  single_hop: "单跳",
  multi_hop: "多跳",
  summarization: "摘要",
  other: "兜底",
};

export function AgentsPage() {
  const [overview, setOverview] = useState<AgentsOverviewResponse | null>(null);
  const [matrix, setMatrix] = useState<ToolMatrixResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([ChimeraAPI.agentsOverview(), ChimeraAPI.toolMatrix()])
      .then(([o, m]) => {
        if (!cancelled) {
          setOverview(o);
          setMatrix(m);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-[var(--color-text-muted)]">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> 加载多智能体架构…
      </div>
    );
  }

  if (!overview || !matrix) return null;

  const allToolNames = new Set(
    overview.tool_categories.flatMap((c) => c.tools.map((t) => t.name)),
  );

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<Bot className="h-5 w-5" />}
        title="多智能体架构"
        subtitle="创新点二 · CollaRAG — 5 个专长智能体 × 31 个工具 × 10 个功能类别"
      />

      {/* 概览统计 */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <StatCard
          label="智能体"
          value={overview.agent_count}
          icon={<Bot className="h-3 w-3" />}
          tone="intent"
        />
        <StatCard
          label="工具总数"
          value={overview.tool_count}
          icon={<Wrench className="h-3 w-3" />}
          tone="intent"
        />
        <StatCard
          label="LLM 工具"
          value={overview.llm_tool_count}
          icon={<Brain className="h-3 w-3" />}
          tone="accent"
        />
        <StatCard
          label="确定性工具"
          value={overview.deterministic_tool_count}
          icon={<Cpu className="h-3 w-3" />}
          tone="accent"
        />
        <StatCard
          label="CollaRAG"
          value={overview.enabled ? "已启用" : "未启用"}
          icon={
            overview.enabled ? (
              <CheckCircle2 className="h-3 w-3" />
            ) : (
              <XCircle className="h-3 w-3" />
            )
          }
          tone={overview.enabled ? "intent" : "default"}
        />
      </div>

      {/* 多智能体流水线 */}
      <Card
        title="多智能体协作流水线"
        titleIcon={<Sparkles className="h-4 w-4" />}
        accent="intent"
      >
        <div className="flex flex-col gap-4">
          {/* 流程图 */}
          <div className="flex items-center justify-center gap-1 overflow-x-auto py-4">
            {overview.agents.map((agent, i) => {
              const Icon = AGENT_ICONS[agent.icon] || Bot;
              return (
                <div key={agent.agent_type} className="flex items-center gap-1">
                  <div className="flex flex-col items-center gap-2">
                    <div
                      className={`flex h-14 w-14 items-center justify-center rounded-xl border-2 ${
                        agent.color === "intent"
                          ? "border-[var(--color-intent)]/50 bg-[var(--color-intent-soft)]"
                          : "border-[var(--color-accent)]/50 bg-[var(--color-accent)]/10"
                      }`}
                    >
                      <Icon
                        className={`h-6 w-6 ${
                          agent.color === "intent"
                            ? "text-[var(--color-intent)]"
                            : "text-[var(--color-accent)]"
                        }`}
                      />
                    </div>
                    <div className="text-center">
                      <div className="text-xs font-semibold text-[var(--color-text)]">
                        {agent.cn_name}
                      </div>
                      <div className="text-[0.65rem] text-[var(--color-text-subtle)]">
                        {agent.tool_count} 工具
                      </div>
                    </div>
                  </div>
                  {i < overview.agents.length - 1 && (
                    <div className="mx-2 flex flex-col items-center">
                      <div className="h-px w-8 bg-[var(--color-border-hi)]" />
                      <svg
                        viewBox="0 0 8 8"
                        className="h-2 w-2 -translate-x-[1px] text-[var(--color-border-hi)]"
                      >
                        <path d="M0 0 L8 4 L0 8 Z" fill="currentColor" />
                      </svg>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="rounded-md bg-[var(--color-bg-soft)] p-3 text-xs leading-relaxed text-[var(--color-text-muted)]">
            <strong>查询 → 预处理</strong>（改写/共指消解）→{" "}
            <strong>意图分类</strong>（TreeIntentClassifier）→ 路由至{" "}
            <strong>专长智能体</strong>（单跳 / 多跳 / 摘要 / 兜底）→{" "}
            <strong>工具调用</strong>（31 工具按权限矩阵分配）→ 反思校验 → 答案合成
          </div>
        </div>
      </Card>

      {/* 5 个智能体详情 */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {overview.agents.map((agent) => {
          const Icon = AGENT_ICONS[agent.icon] || Bot;
          return (
            <Card
              key={agent.agent_type}
              accent={agent.color === "intent" ? "intent" : "accent"}
              title={
                <div className="flex items-center gap-2">
                  <Icon
                    className={`h-4 w-4 ${
                      agent.color === "intent"
                        ? "text-[var(--color-intent)]"
                        : "text-[var(--color-accent)]"
                    }`}
                  />
                  <span>{agent.cn_name}</span>
                  <Badge tone="neutral">{agent.tool_count} 工具</Badge>
                </div>
              }
            >
              <p className="mb-3 text-sm text-[var(--color-text-muted)]">
                {agent.description}
              </p>
              <div className="flex flex-wrap gap-1">
                {agent.tool_names.map((name) => (
                  <span
                    key={name}
                    className="rounded border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-1.5 py-0.5 font-mono text-[0.65rem] text-[var(--color-text-subtle)]"
                  >
                    {name}
                  </span>
                ))}
              </div>
            </Card>
          );
        })}
      </div>

      {/* 工具权限矩阵 */}
      <Card
        title="工具 × 智能体 权限矩阵"
        titleIcon={<Wrench className="h-4 w-4" />}
        accent="intent"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                <th className="sticky left-0 z-10 bg-[var(--color-surface)] px-3 py-2 text-left font-medium text-[var(--color-text-muted)]">
                  工具
                </th>
                {matrix.agent_types.map((at) => (
                  <th
                    key={at}
                    className="px-3 py-2 text-center font-medium text-[var(--color-text-muted)]"
                  >
                    {AGENT_CN_SHORT[at] || at}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.tool_categories.map((cat) => (
                <>
                  <tr key={`cat-${cat.category}`}>
                    <td
                      colSpan={matrix.agent_types.length + 1}
                      className="border-t border-[var(--color-border)] bg-[var(--color-bg-soft)] px-3 py-1.5 text-[0.7rem] font-semibold uppercase tracking-wider text-[var(--color-text-subtle)]"
                    >
                      {cat.cn_name}
                    </td>
                  </tr>
                  {cat.tools.map((tool) => (
                    <tr
                      key={tool.name}
                      className="border-t border-[var(--color-border)]/40 hover:bg-[var(--color-surface-hi)]/40"
                    >
                      <td className="sticky left-0 z-10 bg-[var(--color-surface)] px-3 py-1.5">
                        <div className="flex items-center gap-2">
                          <code className="font-mono text-[var(--color-text)]">
                            {tool.name}
                          </code>
                          {tool.uses_llm && (
                            <span className="rounded bg-[var(--color-intent-soft)] px-1 py-px text-[0.6rem] text-[var(--color-intent)]">
                              LLM
                            </span>
                          )}
                        </div>
                        <div className="text-[0.6rem] text-[var(--color-text-subtle)]">
                          {tool.cn_name}
                        </div>
                      </td>
                      {matrix.agent_types.map((at) => {
                        const has = matrix.matrix[at]?.includes(tool.name);
                        return (
                          <td key={at} className="px-3 py-1.5 text-center">
                            {has ? (
                              <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-intent)]/20">
                                <CheckCircle2 className="h-3 w-3 text-[var(--color-intent)]" />
                              </span>
                            ) : (
                              <span className="text-[var(--color-border)]">—</span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </>
              ))}
              {/* 合计行 */}
              <tr className="border-t-2 border-[var(--color-border)]">
                <td className="sticky left-0 z-10 bg-[var(--color-surface)] px-3 py-2 font-semibold text-[var(--color-text)]">
                  合计
                </td>
                {matrix.agent_types.map((at) => (
                  <td
                    key={at}
                    className="px-3 py-2 text-center font-semibold text-[var(--color-intent)]"
                  >
                    {matrix.tool_totals[at]}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* 工具分类概览 */}
      <Card
        title="工具分类一览"
        titleIcon={<Cpu className="h-4 w-4" />}
      >
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {overview.tool_categories.map((cat) => (
            <div
              key={cat.category}
              className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-3"
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-semibold text-[var(--color-text)]">
                  {cat.cn_name}
                </span>
                <Badge tone="neutral">{cat.tools.length}</Badge>
              </div>
              <div className="flex flex-col gap-1">
                {cat.tools.map((t) => (
                  <div
                    key={t.name}
                    className="flex items-center justify-between text-xs"
                  >
                    <span className="flex items-center gap-1.5">
                      <code className="font-mono text-[var(--color-text-muted)]">
                        {t.name}
                      </code>
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="text-[var(--color-text-subtle)]">
                        {t.cn_name}
                      </span>
                      {t.uses_llm ? (
                        <Brain className="h-3 w-3 text-[var(--color-intent)]" />
                      ) : (
                        <Cpu className="h-3 w-3 text-[var(--color-accent)]" />
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
