import { cn } from "@/lib/utils";

const INTENT_LABELS: Record<string, string> = {
  factual: "事实型",
  analytical: "分析型",
  comparative: "比较型",
  multi_hop: "多跳型",
  exploratory: "探索型",
  follow_up: "追问型",
};

const STRATEGY_LABELS: Record<string, string> = {
  direct: "直接检索",
  multi_hop: "多跳扩展",
  parallel: "并行比较",
  iterative: "迭代分解",
  expansion: "邻居扩展",
  context_aware: "上下文感知",
};

interface Props {
  kind: "intent" | "strategy";
  value: string;
  className?: string;
}

/**
 * 6 类意图 / 6 种策略的中文胶囊徽章。用一致的色彩编码辅助辨识。
 */
export function IntentBadge({ kind, value, className }: Props) {
  const display =
    kind === "intent"
      ? INTENT_LABELS[value] ?? value
      : STRATEGY_LABELS[value] ?? value;

  const palette = kind === "intent"
    ? "border-[var(--color-intent)]/40 bg-[var(--color-intent-soft)] text-[var(--color-intent)]"
    : "border-[var(--color-info)]/40 bg-[color:oklch(0.74_0.12_220/0.12)] text-[var(--color-info)]";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        palette,
        className,
      )}
    >
      <span className="text-[0.6rem] opacity-60">
        {kind === "intent" ? "意图" : "策略"}
      </span>
      {display}
    </span>
  );
}
