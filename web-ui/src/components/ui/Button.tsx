import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

const styles: Record<Variant, string> = {
  primary: "bg-[var(--color-accent)] text-[var(--color-accent-fg)] hover:brightness-110",
  secondary:
    "bg-[var(--color-surface-hi)] text-[var(--color-text)] border border-[var(--color-border)] hover:bg-[var(--color-border)]",
  ghost: "bg-transparent text-[var(--color-text)] hover:bg-[var(--color-surface-hi)]",
  danger: "bg-red-600 text-white hover:bg-red-700",
};

export function Button({ variant = "primary", className, children, ...rest }: Props) {
  return (
    <button
      {...rest}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60",
        styles[variant],
        className,
      )}
    >
      {children}
    </button>
  );
}
