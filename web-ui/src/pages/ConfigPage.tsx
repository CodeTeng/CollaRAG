import { useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ChimeraAPI } from "@/lib/api";

export function ConfigPage() {
  const [text, setText] = useState("");
  const [original, setOriginal] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    const r = await ChimeraAPI.getConfig();
    const pretty = JSON.stringify(r.config, null, 2);
    setText(pretty);
    setOriginal(pretty);
  };

  useEffect(() => {
    void load();
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const parsed = JSON.parse(text);
      const r = await ChimeraAPI.putConfig(parsed);
      setMessage(`Saved. Active: ${JSON.stringify(r.active_implementations)}`);
      setOriginal(text);
    } catch (e: any) {
      setError(e?.response?.data?.detail ? JSON.stringify(e.response.data.detail, null, 2) : e.message);
    } finally {
      setSaving(false);
    }
  };

  const dirty = text !== original;

  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col gap-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">Config</h1>
        {dirty && <Badge tone="warning">unsaved</Badge>}
      </div>

      <Card
        className="flex-1"
        actions={
          <>
            <Button variant="secondary" onClick={() => setText(original)} disabled={!dirty}>
              Reset
            </Button>
            <Button onClick={save} disabled={!dirty || saving}>
              {saving ? "Saving..." : "Save"}
            </Button>
          </>
        }
        title="Edit config (JSON representation of config.yaml)"
      >
        <div className="h-[60vh]">
          <Editor
            height="100%"
            defaultLanguage="json"
            theme="vs-dark"
            value={text}
            onChange={(v) => setText(v ?? "")}
            options={{
              minimap: { enabled: false },
              fontSize: 12,
              tabSize: 2,
              scrollBeyondLastLine: false,
            }}
          />
        </div>
      </Card>

      {error && (
        <Card>
          <pre className="whitespace-pre-wrap text-xs text-red-400">{error}</pre>
        </Card>
      )}
      {message && (
        <Card>
          <pre className="whitespace-pre-wrap text-xs text-emerald-300">{message}</pre>
        </Card>
      )}
    </div>
  );
}
