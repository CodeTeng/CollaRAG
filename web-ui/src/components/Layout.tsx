import { NavLink, Outlet } from "react-router-dom";
import {
  Database,
  MessageSquare,
  Network,
  BarChart3,
  Puzzle,
  FileCog,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { to: "/ingest",   label: "Ingest",    icon: Database },
  { to: "/query",    label: "Query",     icon: MessageSquare },
  { to: "/graph",    label: "Graph",     icon: Network },
  { to: "/evaluate", label: "Evaluate",  icon: BarChart3 },
  { to: "/plugins",  label: "Plugins",   icon: Puzzle },
  { to: "/config",   label: "Config",    icon: FileCog },
  { to: "/trace",    label: "Trace",     icon: Activity },
];

export function Layout() {
  return (
    <div className="flex h-full">
      <aside className="flex w-56 shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <div className="mb-6">
          <div className="text-lg font-bold text-[var(--color-accent)]">Chimera-RAG</div>
          <div className="text-xs text-[var(--color-text-muted)]">
            AdaGraph × IntentAgent
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1">
          {nav.map((n) => {
            const Icon = n.icon;
            return (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition",
                    isActive
                      ? "bg-[var(--color-accent)] text-[var(--color-accent-fg)] font-medium"
                      : "text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hi)] hover:text-[var(--color-text)]",
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {n.label}
              </NavLink>
            );
          })}
        </nav>
        <div className="mt-auto border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-text-muted)]">
          <code className="rounded bg-[var(--color-surface-hi)] px-1.5 py-0.5">v0.1.0</code>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
