import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  title?: ReactNode;
  /** 标题前的小图标（与 title 配套） */
  titleIcon?: ReactNode;
  actions?: ReactNode;
  className?: string;
  /** 内容区（body）的额外 class；需要让内容撑满卡片高度时传 "h-full" 等 */
  bodyClassName?: string;
  /** 让卡片根容器以 flex 纵向布局，配合 bodyClassName="flex-1" 撑满剩余高度 */
  fill?: boolean;
  /** 高亮色带：左侧 2px 竖条；用于强调某些卡片隶属的创新点 */
  accent?: "adagraph" | "intent" | "accent";
  children: ReactNode;
}

export function Card({
  title,
  titleIcon,
  actions,
  className,
  bodyClassName,
  fill,
  accent,
  children,
}: Props) {
  const accentBar =
    accent === "adagraph"
      ? "before:bg-[var(--color-adagraph)]"
      : accent === "intent"
      ? "before:bg-[var(--color-intent)]"
      : accent === "accent"
      ? "before:bg-[var(--color-accent)]"
      : "";

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]",
        fill && "flex flex-col",
        accent &&
          "before:absolute before:left-0 before:top-0 before:h-full before:w-[2px] before:content-['']",
        accentBar,
        className,
      )}
    >
      {(title || actions) && (
        <div className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] px-4 py-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-[var(--color-text)]">
            {titleIcon && (
              <span className="text-[var(--color-accent)]">{titleIcon}</span>
            )}
            {title}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </div>
  );
}
