import { useEffect, useState } from "react";
import { Cpu, Database, FileText, LayoutPanelLeft, RefreshCw, Search, ScrollText, Zap } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import { MemoryGraphCard } from "@/panes/GraphPane";
import { DocumentsCard } from "@/panes/DocumentsPane";
import { Markdown } from "@/components/Markdown";

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

/** A config row whose value is an editable dropdown. The active value falls back to the server
 *  default and is guarded so a stored choice no longer offered still shows (vs. snapping to 0). */
function SelectRow({
  icon: Icon,
  label,
  value,
  fallback,
  options,
  onChange,
  selectClassName,
  title,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
  fallback: string;
  options: string[];
  onChange: (value: string) => void;
  selectClassName?: string;
  title: string;
}) {
  const current = value || fallback;
  const all = current && !options.includes(current) ? [current, ...options] : options;

  return (
    <div className="flex items-start gap-2 text-xs">
      <Icon className="mt-1.5 size-3.5 shrink-0 text-muted-foreground" />
      <span className="w-16 shrink-0 pt-1.5 text-muted-foreground">{label}</span>
      <Select
        value={current}
        onChange={(e) => onChange(e.target.value)}
        className={selectClassName}
        title={title}
      >
        {all.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </Select>
    </div>
  );
}

function ConfigCard() {
  const { config, model, setModel, effort, setEffort } = useApp();

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {config ? (
          <>
            <SelectRow
              icon={Cpu}
              label="Model"
              value={model}
              fallback={config.model}
              options={config.models?.length ? config.models : [config.model]}
              onChange={setModel}
              selectClassName="font-mono"
              title="Model used for new messages"
            />
            <SelectRow
              icon={Zap}
              label="Effort"
              value={effort}
              fallback={config.effort ?? ""}
              options={config.efforts ?? []}
              onChange={setEffort}
              selectClassName="capitalize"
              title="Thinking effort for new messages"
            />
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

/** The right "Knowledge & Artifacts" pane, split into tabs: Context (config + summary +
 *  memory graph) and Documents (agent-authored artifacts, user-editable when text-based).
 *  When a document is featured (agent created one, or the user clicked a document card in
 *  the chat), the pane flips to the Documents tab automatically.
 *  Future modes extend this (Research → sources, Council → consensus doc). */
export function ContextPane({ refreshKey }: { refreshKey: number }) {
  const { featuredDoc } = useApp();
  const [tab, setTab] = useState("context");

  useEffect(() => {
    if (featuredDoc) setTab("documents");
  }, [featuredDoc]);

  return (
    <aside className="h-full border-l border-border bg-card">
      <Tabs defaultValue="context" value={tab} onValueChange={setTab} className="flex h-full flex-col">
        <div className="border-b border-border p-2">
          <TabsList className="w-full">
            <TabsTrigger value="context">
              <LayoutPanelLeft className="size-3.5" />
              Context
            </TabsTrigger>
            <TabsTrigger value="documents">
              <FileText className="size-3.5" />
              Documents
            </TabsTrigger>
          </TabsList>
        </div>
        {/* The tab panel itself is the scroll container (native overflow on a min-h-0 flex
            child). A nested Radix ScrollArea with h-full broke here: the percentage height
            didn't resolve through the flex chain, so content spilled past the pane and was
            clipped by the grid's overflow-hidden instead of scrolling. */}
        <TabsContent value="context" className="min-h-0 flex-1 overflow-y-auto">
          <div className="space-y-3 p-3">
            <ConfigCard />
            <SummaryCard refreshKey={refreshKey} />
            <MemoryGraphCard refreshKey={refreshKey} />
          </div>
        </TabsContent>
        <TabsContent value="documents" className="min-h-0 flex-1 overflow-y-auto">
          <div className="p-3">
            <DocumentsCard refreshKey={refreshKey} />
          </div>
        </TabsContent>
      </Tabs>
    </aside>
  );
}
