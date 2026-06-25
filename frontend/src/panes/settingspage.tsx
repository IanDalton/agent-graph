import { useEffect, useState } from "react";
import { Boxes, Cpu, Database, ScrollText, Search, Server, SlidersHorizontal, Sparkles } from "lucide-react";

import { Dialog } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useApp } from "@/state/appcontext";
import type { SettingsTab } from "@/types";
import { ModelsPanel } from "@/panes/modelspage";
import { SkillMarketplacePanel } from "@/panes/skillmarketplace";

/** One read-only key/value row in the Config tab. */
function ConfigRow({ icon: Icon, label, value }: { icon: typeof Cpu; label: string; value: string }) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <Icon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
      <span className="w-28 shrink-0 text-muted-foreground">{label}</span>
      <span className="break-all font-mono">{value}</span>
    </div>
  );
}

/** Read-only view of the runtime configuration (the same facts the Context pane shows, gathered in
 *  one place) + the fixed base system prompt. Nothing here is editable — it's the "what is this
 *  instance wired to" reference. */
function ConfigPanel() {
  const { config } = useApp();
  if (!config) {
    return <div className="p-4 text-sm text-muted-foreground">Loading configuration…</div>;
  }
  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
      <div className="space-y-2 rounded-lg border border-border bg-background/40 p-3">
        <h3 className="text-sm font-medium">Runtime</h3>
        <ConfigRow icon={Server} label="llama-server" value={config.llamacpp_base_url ?? "—"} />
        <ConfigRow icon={Cpu} label="Provider" value={config.provider ?? "—"} />
        <ConfigRow icon={Cpu} label="Model source" value={config.model_source ?? "—"} />
        <ConfigRow icon={Database} label="ArcadeDB" value={config.arcade_url} />
        <ConfigRow icon={Search} label="Search" value={config.searxng_url} />
        <ConfigRow icon={ScrollText} label="Logs" value={config.log_level} />
        <ConfigRow
          icon={Cpu}
          label="Embeddings"
          value={config.embeddings ? config.embed_model || "on" : "off (substring search)"}
        />
      </div>
      {config.base_system_prompt && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Base system prompt</h3>
          <p className="text-[11px] text-muted-foreground">
            The fixed identity + behaviour prompt on every chat. A conversation's custom prompt is
            appended to it (set per-conversation in the Context pane).
          </p>
          <pre className="max-h-72 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-background/40 p-3 font-mono text-[10px] text-muted-foreground">
            {config.base_system_prompt}
          </pre>
        </div>
      )}
    </div>
  );
}

/** The unified Settings page (opened by the sidebar gear): the advanced configuration that doesn't
 *  need frequent attention — the Model Manager, the Skill Marketplace, and the read-only runtime
 *  config — consolidated off the message bar into one dialog. */
export function SettingsPage() {
  const { settingsOpen, settingsInitialTab, closeSettings } = useApp();
  const [tab, setTab] = useState<SettingsTab>(settingsInitialTab);

  // Jump to the section the opener asked for (the gear → models, "Browse" skills → skills).
  useEffect(() => {
    if (settingsOpen) setTab(settingsInitialTab);
  }, [settingsOpen, settingsInitialTab]);

  return (
    <Dialog
      open={settingsOpen}
      onClose={closeSettings}
      title="Settings"
      className="h-[85vh] w-[min(1100px,94vw)]"
    >
      <div className="flex h-full flex-col">
        <header className="shrink-0 border-b border-border p-4 pr-12">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <SlidersHorizontal className="size-4 text-primary" />
            Settings
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Advanced configuration — local models, skills, and the runtime your chats are wired to.
          </p>
        </header>

        <Tabs
          defaultValue="models"
          value={tab}
          onValueChange={(v) => setTab(v as SettingsTab)}
          className="flex min-h-0 flex-1 flex-col"
        >
          <div className="px-4 pt-3">
            <TabsList className="w-full">
              <TabsTrigger value="models">
                <Boxes className="size-3.5" /> Models
              </TabsTrigger>
              <TabsTrigger value="skills">
                <Sparkles className="size-3.5" /> Skills
              </TabsTrigger>
              <TabsTrigger value="config">
                <SlidersHorizontal className="size-3.5" /> Config
              </TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="models" className="flex min-h-0 flex-1 flex-col">
            <ModelsPanel />
          </TabsContent>
          <TabsContent value="skills" className="flex min-h-0 flex-1 flex-col">
            <SkillMarketplacePanel />
          </TabsContent>
          <TabsContent value="config" className="flex min-h-0 flex-1 flex-col">
            <ConfigPanel />
          </TabsContent>
        </Tabs>
      </div>
    </Dialog>
  );
}
