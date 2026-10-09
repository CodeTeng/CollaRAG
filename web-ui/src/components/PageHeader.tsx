import type { ReactNode } from "react";

interface Props {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
}

/**
 * 页面主标题：印章式金色下划线 + 副标题 + 右侧动作区。
 * 统一所有页面的顶部观感。
 */
export function PageHeader({ title, subtitle, icon, actions }: Props) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="flex items-start gap-3">
        {icon && (
          <div className="mt-1 flex h-11 w-11 items-center justify-center rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-accent)]">
            {icon}
          </div>
        )}
        <div>
          <h1
            className="title-seal text-2xl font-bold tracking-tight"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            {title}
          </h1>
          {subtitle && (
            <p className="mt-2 text-sm text-[var(--color-text-muted)]">{subtitle}</p>
          )}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}
