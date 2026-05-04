import { useEffect, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import cytoscape from "cytoscape";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { ChimeraAPI } from "@/lib/api";
import type { GraphResponse } from "@/lib/types";

export function GraphPage() {
  const [data, setData] = useState<GraphResponse | null>(null);
  const [entity, setEntity] = useState("");
  const [loading, setLoading] = useState(false);
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
    void refresh();
  }, []);

  const elements = data
    ? [
        ...data.nodes.map((n) => ({
          data: { id: n.id, label: n.label },
        })),
        ...data.edges.map((e, i) => ({
          data: {
            id: `e${i}`,
            source: e.source,
            target: e.target,
            label: e.label,
          },
        })),
      ]
    : [];

  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col gap-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">Graph</h1>
        {data && (
          <>
            <Badge tone="accent">{data.nodes.length} nodes</Badge>
            <Badge tone="neutral">{data.edges.length} edges</Badge>
          </>
        )}
      </div>

      <Card>
        <div className="flex gap-2">
          <Input
            value={entity}
            onChange={(e) => setEntity(e.target.value)}
            placeholder="Filter by entity (leave empty for full graph)..."
          />
          <Button onClick={() => refresh(entity || undefined)} disabled={loading}>
            {loading ? "Loading..." : "Apply"}
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              setEntity("");
              void refresh();
            }}
          >
            Reset
          </Button>
        </div>
      </Card>

      <Card className="flex-1">
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
                  "background-color": "#22d3ee",
                  color: "#eef2ff",
                  "font-size": 10,
                  "text-valign": "center",
                  "text-halign": "center",
                  "text-outline-color": "#0f172a",
                  "text-outline-width": 2,
                },
              },
              {
                selector: "edge",
                style: {
                  label: "data(label)",
                  width: 1.5,
                  "line-color": "#475569",
                  "target-arrow-color": "#475569",
                  "target-arrow-shape": "triangle",
                  "curve-style": "bezier",
                  "font-size": 8,
                  color: "#94a3b8",
                },
              },
            ]}
          />
        </div>
      </Card>
    </div>
  );
}
