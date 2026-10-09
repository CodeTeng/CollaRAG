import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type Tone = "accent" | "neutral" | "success" | "warning" | "danger";

interface Props {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}

const tones: Record<Tone, string> = {
  accent: "bg-[var(--color-accent)] text-[var(--color-accent-fg)]",
  neutral: "bg-[var(--color-surface-hi)] text-[var(--color-text-muted)]",
  success: "bg-emerald-500/20 text-emerald-300",
  warning: "bg-amber-500/20 text-amber-300",
  danger: "bg-red-500/20 text-red-300",
};

export function Badge({ children, tone = "neutral", className }: Props) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
