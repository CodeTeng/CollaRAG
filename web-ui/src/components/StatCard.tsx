import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  label: ReactNode;
  value: ReactNode;
  /** 右上角的单位或状态标签 */
  unit?: ReactNode;
  /** 前置图标 */
  icon?: ReactNode;
  /** 给某些关键指标加一抹色彩强调 */
  tone?: "default" | "adagraph" | "intent" | "accent";
  className?: string;
}

/**
 * 仪表盘式指标卡。
 * 用于显示文档数 / 节点数 / Token 消耗 / EM / F1 等关键数字。
 */
export function StatCard({ label, value, unit, icon, tone = "default", className }: Props) {
  const toneRing = {
    default: "border-[var(--color-border)]",
    adagraph: "border-[var(--color-adagraph)]/30 shadow-[0_0_0_1px_inset_var(--color-adagraph-soft)]",
    intent: "border-[var(--color-intent)]/30 shadow-[0_0_0_1px_inset_var(--color-intent-soft)]",
    accent: "border-[var(--color-accent)]/30",
  }[tone];

  const toneIcon = {
    default: "text-[var(--color-text-muted)]",
    adagraph: "text-[var(--color-adagraph)]",
    intent: "text-[var(--color-intent)]",
    accent: "text-[var(--color-accent)]",
  }[tone];

  return (
    <div
      className={cn(
        "ticker-in relative flex flex-col gap-1 rounded-lg border bg-[var(--color-surface)] p-4",
        toneRing,
        className,
      )}
    >
      <div className="flex items-center justify-between text-[0.7rem] uppercase tracking-[0.18em] text-[var(--color-text-subtle)]">
        <span className="flex items-center gap-1.5">
          {icon && <span className={toneIcon}>{icon}</span>}
          {label}
        </span>
        {unit && <span className="normal-case tracking-normal">{unit}</span>}
      </div>
      <div
        className="text-2xl font-semibold tabular-nums text-[var(--color-text)]"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        {value}
      </div>
    </div>
  );
}
