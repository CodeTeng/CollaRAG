import { useEffect, useState } from "react";
import {
  Gauge,
  Play,
  Loader2,
  AlertTriangle,
  Target,
  Clock,
  Cpu,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { PageHeader } from "@/components/PageHeader";
import { StatCard } from "@/components/StatCard";
import { ChimeraAPI } from "@/lib/api";
import { extractErrorMessage, type EvaluateResponse } from "@/lib/types";

const METRIC_LABEL: Record<string, string> = {
  em: "精确匹配 EM",
  f1: "F1",
  rouge_l: "ROUGE-L",
  avg_latency_s: "平均延迟 (秒)",
};

export function EvaluatePage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [limit, setLimit] = useState("100");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluateResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    ChimeraAPI.getConfig()
      .then((c) => {
        if (cancelled) return;
        const ds = Object.keys(c.config.datasets ?? {});
        setDatasets(ds);
        if (ds.length) setSelected(ds[0]);
      })
      .catch(() => {
        /* 静默：评测页对 config 获取失败不阻断 */
      });
    return () => {
      cancelled = true;
    };
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
    } catch (e: unknown) {
      setError(extractErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<Gauge className="h-5 w-5" />}
        title="数据集评测"
        subtitle="在标注 QA 集上跑批量查询，统计 EM / F1 / ROUGE-L 与平均延迟"
      />

      {/* 运行参数 */}
      <Card title="运行参数" titleIcon={<Play className="h-4 w-4" />}>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col">
            <label className="mb-1 text-xs text-[var(--color-text-muted)]">
              数据集
            </label>
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
            <label className="mb-1 text-xs text-[var(--color-text-muted)]">
              最多样本数
            </label>
            <Input
              type="number"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              className="w-32"
            />
          </div>
          <Button onClick={run} disabled={loading || !selected}>
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> 正在评测
              </>
            ) : (
              <>
                <Play className="h-4 w-4" /> 开始评测
              </>
            )}
          </Button>
          <div className="ml-auto text-xs text-[var(--color-text-muted)]">
            提示：100 条真实 DeepSeek 调用约需 5–10 分钟
          </div>
        </div>
      </Card>

      {error && (
        <Card accent="intent">
          <div className="flex items-start gap-2 text-sm text-[var(--color-danger)]">
            <AlertTriangle className="mt-0.5 h-4 w-4" />
            <div>
              <div className="font-semibold">评测失败</div>
              <div className="text-[var(--color-text-muted)]">{error}</div>
            </div>
          </div>
        </Card>
      )}

      {result && (
        <>
          {/* 核心指标 */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {Object.entries(result.metrics).map(([k, v]) => (
              <StatCard
                key={k}
                label={METRIC_LABEL[k] ?? k}
                value={
                  k === "avg_latency_s" ? v.toFixed(2) : (v as number).toFixed(4)
                }
                tone={k === "em" || k === "f1" ? "accent" : "default"}
                icon={
                  k === "avg_latency_s" ? (
                    <Clock className="h-3 w-3" />
                  ) : (
                    <Target className="h-3 w-3" />
                  )
                }
              />
            ))}
          </div>

          {/* 详细表格 */}
          <Card
            title={
              <>
                每条样本详情
                <span className="ml-2 text-xs font-normal text-[var(--color-text-muted)]">
                  {result.per_example.length} 条 · 配置：
                  <code className="ml-1 font-mono">{result.config_name}</code>
                </span>
              </>
            }
            titleIcon={<Cpu className="h-4 w-4" />}
          >
            <div className="max-h-[28rem] overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-[var(--color-surface)] text-[var(--color-text-muted)]">
                  <tr>
                    <th className="px-2 py-1.5">编号</th>
                    <th className="px-2 py-1.5">问题</th>
                    <th className="px-2 py-1.5">参考答案</th>
                    <th className="px-2 py-1.5">模型回答</th>
                    <th className="px-2 py-1.5 text-right">EM</th>
                    <th className="px-2 py-1.5 text-right">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_example.map((row, i) => (
                    <tr
                      key={i}
                      className="border-t border-[var(--color-border)]/60 hover:bg-[var(--color-surface-hi)]/60"
                    >
                      <td className="px-2 py-1.5 font-mono text-[0.7rem] text-[var(--color-text-subtle)]">
                        {row.qid}
                      </td>
                      <td className="max-w-[220px] truncate px-2 py-1.5">
                        {row.question}
                      </td>
                      <td className="max-w-[180px] truncate px-2 py-1.5 text-[var(--color-text-muted)]">
                        {row.reference}
                      </td>
                      <td className="max-w-[200px] truncate px-2 py-1.5">
                        {row.prediction}
                      </td>
                      <td className="px-2 py-1.5 text-right">
                        <Badge tone={row.em === 1 ? "success" : "danger"}>
                          {row.em?.toFixed(2)}
                        </Badge>
                      </td>
                      <td className="px-2 py-1.5 text-right">
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
