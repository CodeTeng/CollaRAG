import { useEffect, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import cytoscape from "cytoscape";
import { Share2, Search, RotateCcw, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { PageHeader } from "@/components/PageHeader";
import { ChimeraAPI } from "@/lib/api";
import type { GraphResponse } from "@/lib/types";

export function GraphPage() {
  const [data, setData] = useState<GraphResponse | null>(null);
  const [entity, setEntity] = useState("");
  const [loading, setLoading] = useState(true);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const refresh = async (e?: string) => {
    setLoading(true);
    try {
      const r = await ChimeraAPI.graph(e ? { entity: e, hops: 2 } : undefined);
      setData(r);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    ChimeraAPI.graph()
      .then((r) => {
        if (!cancelled) setData(r);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Defensive: cytoscape throws "invalid string ID" on empty ids, so drop any
  // node/edge whose endpoints are blank before handing data to the renderer.
  const safeNodes = data ? data.nodes.filter((n) => n.id.trim()) : [];
  const safeEdges = data
    ? data.edges.filter((e) => e.source.trim() && e.target.trim())
    : [];

  const elements = data
    ? [
        ...safeNodes.map((n) => ({ data: { id: n.id, label: n.label } })),
        ...safeEdges.map((e, i) => ({
          data: { id: `e${i}`, source: e.source, target: e.target, label: e.label },
        })),
      ]
    : [];

  // The cytoscape container starts at height 0 (flex layout settles a tick
  // later), so the initial cose layout runs against a 0-sized viewport and
  // collapses every node into the top-left corner. Once data + a real size
  // are available, re-run the layout and fit the graph into view.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || elements.length === 0) return;
    const id = window.setTimeout(() => {
      cy.resize();
      cy.layout({ name: "cose", animate: false }).run();
      cy.fit(undefined, 30);
    }, 50);
    return () => window.clearTimeout(id);
  }, [data]);

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col gap-4">
      <PageHeader
        icon={<Share2 className="h-5 w-5" />}
        title="图谱可视化"
        subtitle={
          data
            ? `共 ${data.nodes.length} 个节点 / ${data.edges.length} 条边`
            : "加载中…"
        }
      />

      <Card>
        <div className="flex flex-wrap gap-2">
          <div className="relative flex-1 min-w-[240px]">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-subtle)]" />
            <Input
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
              placeholder="按实体名过滤子图（留空加载全图）"
              className="pl-8"
            />
          </div>
          <Button onClick={() => refresh(entity || undefined)} disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> 加载中
              </>
            ) : (
              <>
                <Search className="h-4 w-4" /> 应用筛选
              </>
            )}
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              setEntity("");
              void refresh();
            }}
          >
            <RotateCcw className="h-4 w-4" /> 重置
          </Button>
        </div>
      </Card>

      <Card className="flex-1 min-h-0" fill bodyClassName="flex-1 min-h-0">
        <div className="h-full w-full">
          <CytoscapeComponent
            elements={elements}
            cy={(c) => (cyRef.current = c)}
            style={{ width: "100%", height: "100%" }}
            layout={{ name: "cose", animate: false }}
            stylesheet={[
              {
                selector: "node",
                style: {
                  label: "data(label)",
                  "background-color": "#d4af37",
                  color: "#f3e9cb",
                  "font-size": 10,
                  "font-family": "Noto Sans SC, sans-serif",
                  "text-valign": "center",
                  "text-halign": "center",
                  "text-outline-color": "#14181f",
                  "text-outline-width": 2,
                  "border-width": 1,
                  "border-color": "#c89a3a",
                },
              },
              {
                selector: "edge",
                style: {
                  label: "data(label)",
                  width: 1.2,
                  "line-color": "#4a4f5f",
                  "target-arrow-color": "#4a4f5f",
                  "target-arrow-shape": "triangle",
                  "curve-style": "bezier",
                  "font-size": 8,
                  color: "#8a8f9d",
                },
              },
            ]}
          />
        </div>
      </Card>
    </div>
  );
}
