import { useEffect, useState } from "react";
import { Cpu, Database, RefreshCw, Search, ScrollText } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import { MemoryGraphCard } from "@/panes/GraphPane";
import { Markdown } from "@/components/Markdown";
import type { AppConfig } from "@/types";

function ConfigRow({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <Icon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
      <span className="w-16 shrink-0 text-muted-foreground">{label}</span>
      <span className="break-all font-mono">{value}</span>
    </div>
  );
}

function ConfigCard() {
  const [config, setConfig] = useState<AppConfig | null>(null);

  useEffect(() => {
    api.getConfig().then(setConfig).catch((e) => console.error("config", e));
  }, []);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {config ? (
          <>
            <ConfigRow icon={Cpu} label="Model" value={config.model} />
            <ConfigRow icon={Database} label="ArcadeDB" value={config.arcade_url} />
            <ConfigRow icon={Search} label="Search" value={config.searxng_url} />
            <ConfigRow icon={ScrollText} label="Logs" value={config.log_level} />
          </>
        ) : (
          <div className="space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-3/4" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SummaryCard({ refreshKey }: { refreshKey: number }) {
  const { activeId, userId } = useApp();
  const [summary, setSummary] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  useEffect(() => {
    if (!activeId) {
      setSummary("");
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .getSummary(activeId, userId)
      .then((r) => !cancelled && setSummary(r.summary))
      .catch(() => !cancelled && setSummary(""))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [activeId, userId, refreshKey]);

  // Force the LLM to regenerate the summary now, regardless of the message-count threshold.
  function regenerate() {
    if (!activeId || regenerating) return;
    setRegenerating(true);
    api
      .refreshSummary(activeId, userId)
      .then((r) => setSummary(r.summary))
      .catch(() => undefined)
      .finally(() => setRegenerating(false));
  }

  const busy = loading || regenerating;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm">Summary</CardTitle>
        <Button
          variant="ghost"
          size="icon"
          className="size-6 text-muted-foreground"
          onClick={regenerate}
          disabled={!activeId || busy}
          title="Regenerate summary now"
        >
          <RefreshCw className={`size-3.5 ${busy ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent>
        {loading && !summary ? (
          <div className="space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
          </div>
        ) : summary ? (
          <Markdown className="text-xs text-muted-foreground">{summary}</Markdown>
        ) : (
          <div className="text-xs text-muted-foreground">
            No summary yet — it updates every few turns.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** The right "Knowledge & Artifacts" pane: read-only config + a live conversation
 *  digest. Future modes extend this (Research → sources, Council → consensus doc). */
export function ContextPane({ refreshKey }: { refreshKey: number }) {
  return (
    <aside className="h-full border-l border-border bg-card">
      <ScrollArea className="h-full">
        <div className="space-y-3 p-3">
          <ConfigCard />
          <SummaryCard refreshKey={refreshKey} />
          <MemoryGraphCard refreshKey={refreshKey} />
        </div>
      </ScrollArea>
    </aside>
  );
}
