import { useEffect, useState } from "react";
import {
  ToggleRight,
  ToggleLeft,
  Sparkles,
  Power,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { PageHeader } from "@/components/PageHeader";
import { ChimeraAPI } from "@/lib/api";
import type { PluginsResponse } from "@/lib/types";

const PLUGIN_META: Record<
  string,
  { cn: string; tagline: string; accent: "adagraph" | "intent" }
> = {
  adagraph: {
    cn: "AdaGraph · 自适应图谱构建",
    tagline:
      "创新点一 — 动态粒度分块 + 分层三元组抽取 + 双重冗余剪枝，让图谱构建按语义密度自适应伸缩。",
    accent: "adagraph",
  },
  colla_rag: {
    cn: "CollaRAG · 多智能体协作检索",
    tagline:
      "创新点二 — 5 个专长智能体 + 31 个工具 + 10 类功能，Plan-Execute-Reflect 多跳推理 + 意图路由 + 工具权限矩阵。",
    accent: "intent",
  },
};

export function PluginsPage() {
  const [data, setData] = useState<PluginsResponse | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = async () => {
    setData(await ChimeraAPI.plugins());
  };

  useEffect(() => {
    let cancelled = false;
    ChimeraAPI.plugins().then((r) => {
      if (!cancelled) setData(r);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const toggle = async (name: string, next: boolean) => {
    setBusy(name);
    try {
      await ChimeraAPI.togglePlugin(name, next);
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        icon={<ToggleRight className="h-5 w-5" />}
        title="插件开关"
        subtitle="热切换两大研究创新点 · 切换后会重建 Pipeline，但保留已构建的图谱与向量索引"
      />

      {data && (
        <div className="grid gap-4 md:grid-cols-2">
          {data.plugins.map((p) => {
            const meta = PLUGIN_META[p.name] ?? {
              cn: p.name,
              tagline: "",
              accent: "accent" as const,
            };
            return (
              <Card
                key={p.name}
                accent={meta.accent}
                title={
                  <div className="flex items-center gap-2">
                    <Sparkles
                      className={
                        meta.accent === "adagraph"
                          ? "h-4 w-4 text-[var(--color-adagraph)]"
                          : "h-4 w-4 text-[var(--color-intent)]"
                      }
                    />
                    <span>{meta.cn}</span>
                    {p.enabled ? (
                      <span className="ml-2 inline-flex items-center gap-1 text-xs text-[var(--color-success)]">
                        <CheckCircle2 className="h-3 w-3" /> 已启用
                      </span>
                    ) : (
                      <span className="ml-2 inline-flex items-center gap-1 text-xs text-[var(--color-text-subtle)]">
                        <XCircle className="h-3 w-3" /> 已关闭
                      </span>
                    )}
                  </div>
                }
                actions={
                  <Button
                    variant={p.enabled ? "secondary" : "primary"}
                    disabled={busy === p.name}
                    onClick={() => toggle(p.name, !p.enabled)}
                  >
                    {busy === p.name ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> 切换中
                      </>
                    ) : p.enabled ? (
                      <>
                        <ToggleLeft className="h-4 w-4" /> 关闭
                      </>
                    ) : (
                      <>
                        <ToggleRight className="h-4 w-4" /> 启用
                      </>
                    )}
                  </Button>
                }
              >
                <p className="mb-4 text-sm leading-relaxed text-[var(--color-text-muted)]">
                  {meta.tagline}
                </p>
                <div className="mb-2 text-xs uppercase tracking-[0.15em] text-[var(--color-text-subtle)]">
                  当前激活的插槽实现
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(p.active_slots).map(([slot, impl]) => (
                    <span
                      key={slot}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-2.5 py-1 text-xs"
                    >
                      <span className="text-[var(--color-text-subtle)]">{slot}</span>
                      <span className="mx-1 text-[var(--color-text-subtle)]">:</span>
                      <span className="font-mono text-[var(--color-accent)]">{impl}</span>
                    </span>
                  ))}
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {data && (
        <Card title="全局激活实现（调试视图）" titleIcon={<Power className="h-4 w-4" />}>
          <pre className="max-h-56 overflow-auto font-mono text-[0.7rem] leading-relaxed text-[var(--color-text-muted)]">
            {JSON.stringify(data.active_implementations, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );
}
