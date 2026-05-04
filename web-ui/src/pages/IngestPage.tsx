import { useEffect, useState } from "react";
import {
  Upload,
  FolderOpen,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Sparkles,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input, Textarea } from "@/components/ui/Input";
import { PageHeader } from "@/components/PageHeader";
import { StatCard } from "@/components/StatCard";
import { ChimeraAPI } from "@/lib/api";
import { extractErrorMessage, type IngestResponse } from "@/lib/types";

export function IngestPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selected, setSelected] = useState("sample");
  const [rawDocs, setRawDocs] = useState("");
  const [docId, setDocId] = useState("manual-1");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    ChimeraAPI.getConfig()
      .then((c) => {
        if (cancelled) return;
        const ds = c.config.datasets as Record<string, unknown> | undefined;
        setDatasets(Object.keys(ds ?? {}));
      })
      .catch(() => {
        /* 静默 */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const runDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await ChimeraAPI.ingest({ dataset: selected });
      setResult(r);
    } catch (e: unknown) {
      setError(extractErrorMessage(e));
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
    } catch (e: unknown) {
      setError(extractErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<Upload className="h-5 w-5" />}
        title="文档摄取"
        subtitle="将原始文本切块、抽取三元组、入图；构建知识图谱的第一步"
      />

      {/* 结果统计（放最显眼位置） */}
      {result && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatCard
            label="文档数"
            value={result.stats.documents}
            icon={<FileText className="h-3 w-3" />}
          />
          <StatCard
            label="分块数"
            value={result.stats.chunks}
            tone="adagraph"
            icon={<Sparkles className="h-3 w-3" />}
          />
          <StatCard
            label="抽取三元组（原始）"
            value={result.stats.triples_raw}
            tone="adagraph"
          />
          <StatCard
            label="剪枝后保留"
            value={result.stats.triples_pruned}
            unit={
              result.stats.triples_raw > 0
                ? `${Math.round(
                    (result.stats.triples_pruned / result.stats.triples_raw) * 100,
                  )}%`
                : undefined
            }
            tone="adagraph"
          />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 从数据集摄取 */}
        <Card
          title="从已注册数据集摄取"
          titleIcon={<FolderOpen className="h-4 w-4" />}
          accent="adagraph"
        >
          <p className="mb-3 text-xs text-[var(--color-text-muted)]">
            数据集在 <code className="font-mono">config.datasets</code> 段注册，
            支持 plain_text / hotpotqa / mock_wiki 等 loader。
          </p>
          <div className="flex gap-2">
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="flex-1 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-3 py-2 text-sm"
            >
              {datasets.length === 0 && <option value="sample">sample</option>}
              {datasets.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            <Button onClick={runDataset} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> 摄取中
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" /> 开始摄取
                </>
              )}
            </Button>
          </div>
        </Card>

        {/* 手动粘贴 */}
        <Card title="粘贴单篇文档" titleIcon={<FileText className="h-4 w-4" />}>
          <div className="flex flex-col gap-3">
            <Input
              value={docId}
              onChange={(e) => setDocId(e.target.value)}
              placeholder="文档 ID，例如 manual-1"
            />
            <Textarea
              rows={6}
              value={rawDocs}
              onChange={(e) => setRawDocs(e.target.value)}
              placeholder="直接把文档内容粘贴到这里…"
            />
            <Button onClick={runInline} disabled={loading || !rawDocs.trim()}>
              摄取此文档
            </Button>
          </div>
        </Card>
      </div>

      {error && (
        <Card accent="intent">
          <div className="flex items-start gap-2 text-sm text-[var(--color-danger)]">
            <AlertTriangle className="mt-0.5 h-4 w-4" />
            <div>
              <div className="font-semibold">摄取失败</div>
              <div className="text-[var(--color-text-muted)]">{error}</div>
            </div>
          </div>
        </Card>
      )}

      {result && (
        <Card title="运行摘要" titleIcon={<CheckCircle2 className="h-4 w-4" />}>
          <div className="text-sm text-[var(--color-text-muted)]">
            累计入图统计 — 分块 {result.state.chunks} 条，三元组{" "}
            {result.state.triples} 条。
          </div>
        </Card>
      )}
    </div>
  );
}
