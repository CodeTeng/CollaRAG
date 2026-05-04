import { useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import {
  FileCog,
  Save,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { PageHeader } from "@/components/PageHeader";
import { ChimeraAPI } from "@/lib/api";
import { extractErrorMessage } from "@/lib/types";

export function ConfigPage() {
  const [text, setText] = useState("");
  const [original, setOriginal] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    ChimeraAPI.getConfig()
      .then((r) => {
        if (cancelled) return;
        const pretty = JSON.stringify(r.config, null, 2);
        setText(pretty);
        setOriginal(pretty);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(extractErrorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const parsed = JSON.parse(text);
      const r = await ChimeraAPI.putConfig(parsed);
      setMessage(
        `保存成功。当前激活的实现：${Object.entries(r.active_implementations)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ")}`,
      );
      setOriginal(text);
    } catch (e: unknown) {
      setError(extractErrorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const dirty = text !== original;

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col gap-4">
      <PageHeader
        icon={<FileCog className="h-5 w-5" />}
        title="在线配置"
        subtitle="直接修改运行时配置的 JSON 视图；保存后会热重建 Pipeline"
        actions={dirty && <Badge tone="warning">未保存改动</Badge>}
      />

      <Card
        className="flex-1"
        title="配置编辑器"
        titleIcon={<FileCog className="h-4 w-4" />}
        actions={
          <>
            <Button
              variant="secondary"
              onClick={() => setText(original)}
              disabled={!dirty}
            >
              <RotateCcw className="h-4 w-4" /> 重置
            </Button>
            <Button onClick={save} disabled={!dirty || saving}>
              {saving ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> 保存中
                </>
              ) : (
                <>
                  <Save className="h-4 w-4" /> 保存
                </>
              )}
            </Button>
          </>
        }
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
              fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, monospace",
              scrollBeyondLastLine: false,
            }}
          />
        </div>
      </Card>

      {error && (
        <Card accent="intent">
          <div className="flex items-start gap-2 text-sm">
            <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--color-danger)]" />
            <pre className="whitespace-pre-wrap font-mono text-xs text-[var(--color-danger)]">
              {error}
            </pre>
          </div>
        </Card>
      )}
      {message && (
        <Card accent="accent">
          <div className="flex items-start gap-2 text-sm text-[var(--color-success)]">
            <CheckCircle2 className="mt-0.5 h-4 w-4" />
            <div>{message}</div>
          </div>
        </Card>
      )}
    </div>
  );
}
