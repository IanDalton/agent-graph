import { useEffect, useState } from "react";
import { Bot, Brain, Cpu, Database, FileText, Gauge, Layers, LayoutPanelLeft, Network, RefreshCw, Search, ScrollText, Sparkles, UserRound, X, Zap } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/api/client";
import { useApp } from "@/state/appcontext";
import { MemoryGraphCard } from "@/panes/graphpane";
import { DocumentsCard } from "@/panes/documentspane";
import { AgentRoster } from "@/panes/agentroster";
import { ProjectCard } from "@/panes/projectcard";
import { FactsCard } from "@/panes/factspane";
import { SwarmFlowCard } from "@/swarm/swarmflowcard";
import { Markdown } from "@/components/markdown";
import type { ContextUsage } from "@/types";

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

/** Per-conversation custom system prompt. Edits live in local state while typing and are saved
 *  (PATCH) on blur, so it's not a request per keystroke. Appended to the base prompt server-side. */
function SystemPromptRow() {
  const { config, conversations, activeId, setConversationSystemPrompt } = useApp();
  const stored = conversations.find((c) => c.conversation_id === activeId)?.system_prompt ?? "";
  const [draft, setDraft] = useState(stored);

  // Re-seed the draft when the active conversation (or its stored prompt) changes, so switching
  // conversations shows that conversation's prompt rather than the previous draft.
  useEffect(() => {
    setDraft(stored);
  }, [activeId, stored]);

  const save = () => {
    if (activeId && draft !== stored) setConversationSystemPrompt(activeId, draft);
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-xs">
        <Bot className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="text-muted-foreground">System prompt</span>
      </div>
      {activeId ? (
        <>
          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={save}
            placeholder="Extra instructions for this conversation, appended to the base prompt…"
            className="max-h-48 min-h-[72px] resize-y text-xs"
            title="Appended to the base system prompt; applies to this conversation from the next turn"
          />
          {config?.base_system_prompt && (
            <details className="text-[10px] text-muted-foreground/70">
              <summary className="cursor-pointer select-none hover:text-muted-foreground">
                View base prompt
              </summary>
              <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[10px] text-muted-foreground/70">
                {config.base_system_prompt}
              </pre>
            </details>
          )}
        </>
      ) : (
        <p className="text-xs text-muted-foreground">Select a conversation to set its prompt.</p>
      )}
    </div>
  );
}

function rangeOptions([lo, hi]: [number, number]): string[] {
  const out: string[] = [];
  for (let i = lo; i <= hi; i++) out.push(String(i));
  return out;
}

/** Per-conversation swarm bounds (swarm mode only). Each dropdown saves immediately (one PATCH per
 *  change) and takes effect on the next turn; an unset value shows the server's config default. */
function SwarmSettingsRows() {
  const { config, conversations, activeId, setConversationSwarmSettings } = useApp();
  const swarm = config?.swarm;
  if (!swarm || !activeId) return null;
  const conv = conversations.find((c) => c.conversation_id === activeId);
  const parallel = conv?.swarm_max_parallel ?? swarm.max_parallel;
  const depth = conv?.swarm_max_depth ?? swarm.max_depth;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Network className="size-3.5 shrink-0" />
        <span>Swarm limits</span>
      </div>
      <SelectRow
        icon={Network}
        label="Parallel"
        value={String(parallel)}
        fallback={String(swarm.max_parallel)}
        options={rangeOptions(swarm.max_parallel_range)}
        onChange={(v) => setConversationSwarmSettings(activeId, { swarm_max_parallel: Number(v) })}
        title="Max sub-agents that run at once per fan-out batch (send_messages)"
      />
      <SelectRow
        icon={Layers}
        label="Depth"
        value={String(depth)}
        fallback={String(swarm.max_depth)}
        options={rangeOptions(swarm.max_depth_range)}
        onChange={(v) => setConversationSwarmSettings(activeId, { swarm_max_depth: Number(v) })}
        title="Max orchestration layers delegation may nest (orchestrator → sub-orchestrator → …)"
      />
    </div>
  );
}

/** The account skill library (active in every chat). Shows installed/authored skills as removable
 *  chips; "Browse" opens the Skill Marketplace dialog to add or author more. */
function SkillsRow() {
  const { skills, removeSkill, openSkillMarketplace } = useApp();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Sparkles className="size-3.5 shrink-0" />
          <span>Skills</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 gap-1 px-2 text-[11px] text-muted-foreground"
          onClick={openSkillMarketplace}
          title="Browse the skill marketplace"
        >
          <Sparkles className="size-3" />
          Browse
        </Button>
      </div>
      {skills.length === 0 ? (
        <p className="text-[11px] text-muted-foreground">
          No skills yet — Browse the marketplace to install or author some. Installed skills are
          active in every chat.
        </p>
      ) : (
        <div className="flex flex-wrap gap-1">
          {skills.map((s) => (
            <span
              key={s.name}
              title={s.description}
              className="inline-flex items-center gap-1 rounded-full border border-border bg-background/60 px-2 py-0.5 text-[11px]"
            >
              {s.name}
              <button
                type="button"
                onClick={() => removeSkill(s.name)}
                title="Remove from your library"
                className="text-muted-foreground transition-colors hover:text-foreground"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ConfigCard() {
  const { config, model, setModel, effort, setEffort, conversations, activeId } = useApp();
  const isSwarm =
    conversations.find((c) => c.conversation_id === activeId)?.mode === "swarm";

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
            <div className="border-t border-border/50 pt-2">
              <SystemPromptRow />
            </div>
            {!isSwarm && (
              <div className="border-t border-border/50 pt-2">
                <SkillsRow />
              </div>
            )}
            {isSwarm && (
              <div className="border-t border-border/50 pt-2">
                <SwarmSettingsRows />
              </div>
            )}
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

// The three context components, each with a fixed colour so the bar segments and rows line up.
const CONTEXT_PARTS = [
  { key: "system_prompt", label: "System prompt", bar: "bg-sky-500", dot: "bg-sky-500" },
  { key: "tools", label: "Tool defs", bar: "bg-amber-500", dot: "bg-amber-500" },
  { key: "messages", label: "Messages", bar: "bg-violet-500", dot: "bg-violet-500" },
] as const;

/** Estimated context-window usage for the active conversation, broken into system prompt / tool
 *  definitions / message history. Refetches on conversation switch, model change, and after each
 *  completed turn (the shared `refreshKey` bump). Mirrors SummaryCard's fetch/cancel pattern. */
function ContextWindowCard({ refreshKey }: { refreshKey: number }) {
  const { activeId, userId, model, config, conversations } = useApp();
  const mode = conversations.find((c) => c.conversation_id === activeId)?.mode ?? "regular";
  // The model that will actually be used for the next turn (selection overrides the server default).
  const effectiveModel = model || config?.model || "";
  const [usage, setUsage] = useState<ContextUsage | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeId) {
      setUsage(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .getContextUsage(activeId, userId, effectiveModel, mode)
      .then((r) => !cancelled && setUsage(r))
      .catch(() => !cancelled && setUsage(null))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [activeId, userId, effectiveModel, mode, refreshKey]);

  const fmt = (n: number) => n.toLocaleString();
  const pct = (n: number) =>
    usage && usage.context_window ? Math.min(100, (n / usage.context_window) * 100) : 0;
  const estimated = usage ? !usage.counter.startsWith("tiktoken") : false;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Gauge className="size-3.5 text-muted-foreground" />
          Context window
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {loading && !usage ? (
          <div className="space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-2 w-full" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        ) : !activeId ? (
          <p className="text-xs text-muted-foreground">Select a conversation to see usage.</p>
        ) : usage ? (
          <>
            <div className="flex items-baseline justify-between text-xs">
              <span className="truncate font-mono text-muted-foreground" title={usage.model}>
                {usage.model}
              </span>
              <span className="shrink-0 font-medium">
                {usage.percent}% of {fmt(usage.context_window)}
              </span>
            </div>

            {/* Stacked usage bar: system → tools → messages, then the remaining free space. */}
            <div className="flex h-2 w-full overflow-hidden rounded-full bg-muted">
              {CONTEXT_PARTS.map((p) => (
                <div
                  key={p.key}
                  className={p.bar}
                  style={{ width: `${pct(usage.components[p.key])}%` }}
                  title={`${p.label}: ${fmt(usage.components[p.key])}`}
                />
              ))}
            </div>

            <div className="space-y-1 pt-1">
              {CONTEXT_PARTS.map((p) => (
                <div key={p.key} className="flex items-center gap-2 text-xs">
                  <span className={`size-2 shrink-0 rounded-full ${p.dot}`} />
                  <span className="flex-1 text-muted-foreground">{p.label}</span>
                  <span className="font-mono">{fmt(usage.components[p.key])}</span>
                </div>
              ))}
              <div className="flex items-center gap-2 border-t border-border/50 pt-1 text-xs">
                <span className="size-2 shrink-0 rounded-full bg-muted-foreground/30" />
                <span className="flex-1 text-muted-foreground">Free</span>
                <span className="font-mono">{fmt(usage.free)}</span>
              </div>
            </div>

            <p className="text-[10px] text-muted-foreground/70">
              {usage.counter === "unavailable"
                ? "Counts unavailable."
                : estimated
                  ? "Estimated tokens (system prompt = base + custom; excludes per-turn facts)."
                  : "Token counts (system prompt = base + custom; excludes per-turn facts)."}
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">Usage unavailable.</p>
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

/** The durable, curator-maintained user profile (cross-conversation context the agent injects each
 *  turn). Read-only here — it's rewritten automatically by the background memory curator every few
 *  turns. Per-user, so it refetches on userId and the shared refreshKey bump (after each turn). */
function UserProfileCard({ refreshKey }: { refreshKey: number }) {
  const { userId } = useApp();
  const [profile, setProfile] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .getUserProfile(userId)
      .then((r) => !cancelled && setProfile(r.profile))
      .catch(() => !cancelled && setProfile(""))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [userId, refreshKey]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <UserRound className="size-3.5 text-muted-foreground" />
          User profile
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading && !profile ? (
          <div className="space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-4/6" />
          </div>
        ) : profile ? (
          <Markdown className="text-xs text-muted-foreground">{profile}</Markdown>
        ) : (
          <div className="text-xs text-muted-foreground">
            No profile yet — it builds up as you chat.
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
  const { featuredDoc, conversations, activeId } = useApp();
  const [tab, setTab] = useState("context");
  const isSwarm =
    conversations.find((c) => c.conversation_id === activeId)?.mode === "swarm";

  useEffect(() => {
    if (featuredDoc) setTab("documents");
  }, [featuredDoc]);

  // The Agents tab only exists in swarm mode; fall back to Context if the mode changes while open.
  useEffect(() => {
    if (!isSwarm && tab === "agents") setTab("context");
  }, [isSwarm, tab]);

  return (
    <aside className="h-full border-l border-border bg-card">
      <Tabs defaultValue="context" value={tab} onValueChange={setTab} className="flex h-full flex-col">
        <div className="border-b border-border p-2">
          <TabsList className="w-full">
            <TabsTrigger value="context">
              <LayoutPanelLeft className="size-3.5" />
              Context
            </TabsTrigger>
            <TabsTrigger value="facts">
              <Brain className="size-3.5" />
              Facts
            </TabsTrigger>
            <TabsTrigger value="documents">
              <FileText className="size-3.5" />
              Documents
            </TabsTrigger>
            {isSwarm && (
              <TabsTrigger value="agents">
                <Bot className="size-3.5" />
                Agents
              </TabsTrigger>
            )}
          </TabsList>
        </div>
        {/* The tab panel itself is the scroll container (native overflow on a min-h-0 flex
            child). A nested Radix ScrollArea with h-full broke here: the percentage height
            didn't resolve through the flex chain, so content spilled past the pane and was
            clipped by the grid's overflow-hidden instead of scrolling. */}
        <TabsContent value="context" className="min-h-0 flex-1 overflow-y-auto">
          <div className="space-y-3 p-3">
            <ProjectCard />
            <ConfigCard />
            <ContextWindowCard refreshKey={refreshKey} />
            {isSwarm && <SwarmFlowCard />}
            <UserProfileCard refreshKey={refreshKey} />
            <SummaryCard refreshKey={refreshKey} />
            <MemoryGraphCard refreshKey={refreshKey} />
          </div>
        </TabsContent>
        <TabsContent value="facts" className="min-h-0 flex-1 overflow-y-auto">
          <div className="p-3">
            <FactsCard refreshKey={refreshKey} />
          </div>
        </TabsContent>
        <TabsContent value="documents" className="min-h-0 flex-1 overflow-y-auto">
          <div className="p-3">
            <DocumentsCard refreshKey={refreshKey} />
          </div>
        </TabsContent>
        {isSwarm && (
          <TabsContent value="agents" className="min-h-0 flex-1 overflow-y-auto">
            <div className="p-3">
              <AgentRoster />
            </div>
          </TabsContent>
        )}
      </Tabs>
    </aside>
  );
}
