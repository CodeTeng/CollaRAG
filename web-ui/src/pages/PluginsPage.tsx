import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ChimeraAPI } from "@/lib/api";
import type { PluginsResponse } from "@/lib/types";

export function PluginsPage() {
  const [data, setData] = useState<PluginsResponse | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    setData(await ChimeraAPI.plugins());
  };

  useEffect(() => {
    void refresh();
  }, []);

  const toggle = async (name: string, next: boolean) => {
    setBusy(true);
    try {
      await ChimeraAPI.togglePlugin(name, next);
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      <h1 className="text-xl font-semibold">Plugins</h1>
      <p className="text-sm text-[var(--color-text-muted)]">
        Hot-toggle the two research innovations. Graph &amp; vector state
        are preserved across toggles.
      </p>

      {data && (
        <div className="grid gap-4 md:grid-cols-2">
          {data.plugins.map((p) => (
            <Card
              key={p.name}
              title={
                <div className="flex items-center gap-2">
                  <span className="uppercase tracking-wide">{p.name}</span>
                  {p.enabled ? (
                    <Badge tone="success">enabled</Badge>
                  ) : (
                    <Badge tone="neutral">disabled</Badge>
                  )}
                </div>
              }
              actions={
                <Button
                  variant={p.enabled ? "secondary" : "primary"}
                  disabled={busy}
                  onClick={() => toggle(p.name, !p.enabled)}
                >
                  {p.enabled ? "Disable" : "Enable"}
                </Button>
              }
            >
              <div className="text-xs text-[var(--color-text-muted)] mb-2">
                Active slots:
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(p.active_slots).map(([slot, impl]) => (
                  <span
                    key={slot}
                    className="rounded border border-[var(--color-border)] bg-[var(--color-surface-hi)] px-2 py-0.5 text-xs"
                  >
                    <span className="text-[var(--color-text-muted)]">{slot}:</span>{" "}
                    <span className="text-[var(--color-accent)]">{impl}</span>
                  </span>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}

      {data && (
        <Card title="Active implementations (global view)">
          <pre className="text-xs text-[var(--color-text-muted)]">
            {JSON.stringify(data.active_implementations, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );
}
