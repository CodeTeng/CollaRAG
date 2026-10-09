import { NavLink, Outlet } from "react-router-dom";
import {
  Upload,
  MessageSquareText,
  Share2,
  Gauge,
  ToggleRight,
  FileCog,
  History,
  Sparkles,
  Bot,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ChimeraLogo } from "@/components/ChimeraLogo";

const nav = [
  { to: "/ingest",   label: "文档摄取",    icon: Upload,           hint: "构建知识图谱" },
  { to: "/query",    label: "智能问答",    icon: MessageSquareText, hint: "多智能体协作回答" },
  { to: "/agents",   label: "智能体架构",  icon: Bot,               hint: "5 智能体 × 31 工具" },
  { to: "/graph",    label: "图谱可视",    icon: Share2,            hint: "节点与关系" },
  { to: "/evaluate", label: "数据集评测",  icon: Gauge,             hint: "EM / F1 / 延迟" },
  { to: "/plugins",  label: "插件开关",    icon: ToggleRight,       hint: "两创新点热切换" },
  { to: "/config",   label: "在线配置",    icon: FileCog,           hint: "编辑 YAML" },
  { to: "/trace",    label: "历史追踪",    icon: History,           hint: "每次查询详情" },
];

export function Layout() {
  return (
    <div className="flex h-full">
      {/* 左侧导航 */}
      <aside className="relative flex w-64 shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-bg-soft)]/80 p-5 backdrop-blur">
        {/* 品牌 */}
        <div className="mb-8 flex items-start gap-3">
          <ChimeraLogo className="h-10 w-10 text-[var(--color-accent-soft)]" />
          <div className="leading-tight">
            <div
              className="text-[1.35rem] font-bold tracking-tight text-[var(--color-text)]"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Chimera<span className="text-[var(--color-accent)]">·RAG</span>
            </div>
            <div className="mt-0.5 text-[0.7rem] font-medium tracking-wide text-[var(--color-text-subtle)]">
              嵌合式 · 知识图谱问答
            </div>
          </div>
        </div>

        {/* 两创新点徽记 */}
        <div className="mb-6 flex flex-col gap-2">
          <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[var(--color-text-subtle)]">
            研究创新点
          </div>
          <div className="flex items-center gap-2 rounded-md border border-[var(--color-adagraph)]/30 bg-[var(--color-adagraph-soft)] px-3 py-2 text-xs">
            <Sparkles className="h-3.5 w-3.5 text-[var(--color-adagraph)]" />
            <div className="flex-1 leading-tight">
              <div className="font-semibold text-[var(--color-text)]">AdaGraph</div>
              <div className="text-[0.65rem] text-[var(--color-text-subtle)]">
                自适应图谱构建
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-md border border-[var(--color-intent)]/30 bg-[var(--color-intent-soft)] px-3 py-2 text-xs">
            <Bot className="h-3.5 w-3.5 text-[var(--color-intent)]" />
            <div className="flex-1 leading-tight">
              <div className="font-semibold text-[var(--color-text)]">CollaRAG</div>
              <div className="text-[0.65rem] text-[var(--color-text-subtle)]">
                多智能体协作检索
              </div>
            </div>
          </div>
        </div>

        {/* 主导航 */}
        <nav className="flex flex-1 flex-col gap-1">
          <div className="mb-2 px-2 text-[0.65rem] uppercase tracking-[0.2em] text-[var(--color-text-subtle)]">
            功能
          </div>
          {nav.map((n) => {
            const Icon = n.icon;
            return (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  cn(
                    "group flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition",
                    isActive
                      ? "bg-[var(--color-accent)]/15 text-[var(--color-accent)] ring-1 ring-[var(--color-accent)]/30"
                      : "text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hi)] hover:text-[var(--color-text)]",
                  )
                }
              >
                <Icon className="h-4 w-4 shrink-0" />
                <div className="flex-1 leading-tight">
                  <div className="font-medium">{n.label}</div>
                  <div className="text-[0.65rem] text-[var(--color-text-subtle)] group-hover:text-[var(--color-text-muted)]">
                    {n.hint}
                  </div>
                </div>
              </NavLink>
            );
          })}
        </nav>

        {/* 底部版本 */}
        <div className="mt-4 flex items-center justify-between border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-text-muted)]">
          <span>v0.1.0</span>
          <code className="rounded bg-[var(--color-surface-hi)] px-1.5 py-0.5 font-mono text-[0.7rem]">
            chimera-rag
          </code>
        </div>
      </aside>

      {/* 主内容 */}
      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
