import { useMemo, useState } from "react";
import { Check, Loader2, Plus, RefreshCw, Search, Sparkles, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { useApp } from "@/state/AppContext";
import type { CatalogSkill } from "@/types";

/** One skill card in the marketplace grid. The action reflects three states: enabled on this chat
 *  (→ Added, removable), or not (→ Add to chat). A per-card spinner shows while installing. */
function SkillCard({
  skill,
  enabled,
  busy,
  canAdd,
  onAdd,
  onRemove,
}: {
  skill: CatalogSkill;
  enabled: boolean;
  busy: boolean;
  canAdd: boolean;
  onAdd: () => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-background/40 p-3">
      <div className="flex items-start gap-2">
        <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{skill.name}</span>
            {skill.installed && !enabled && (
              <Badge variant="secondary" className="shrink-0 text-[10px]">
                In library
              </Badge>
            )}
          </div>
          <p className="mt-0.5 line-clamp-3 text-xs text-muted-foreground">
            {skill.description || "No description provided."}
          </p>
        </div>
      </div>
      <div className="mt-auto flex justify-end">
        {enabled ? (
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={onRemove}
            title="Remove this skill from the current chat"
          >
            <Check className="size-3.5 text-primary" />
            Added
            <X className="size-3 text-muted-foreground" />
          </Button>
        ) : (
          <Button
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={onAdd}
            disabled={busy || !canAdd}
            title={canAdd ? "Install and enable for this chat" : "Open or start a chat first"}
          >
            {busy ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Plus className="size-3.5" />
            )}
            {busy ? "Adding…" : "Add to chat"}
          </Button>
        )}
      </div>
    </div>
  );
}

/** The Skill Marketplace dialog: browse Claude's live Agent Skills catalog and load skills onto the
 *  active conversation. Mounted once at the shell level; reads its open state from AppContext. */
export function SkillMarketplace() {
  const {
    skillMarketplaceOpen,
    closeSkillMarketplace,
    catalog,
    catalogLoading,
    refreshCatalog,
    addSkillToChat,
    setConversationSkills,
    conversations,
    activeId,
  } = useApp();
  const [query, setQuery] = useState("");
  // Names currently being installed/enabled, for per-card spinners.
  const [busy, setBusy] = useState<Set<string>>(new Set());

  const enabledSet = useMemo(() => {
    const conv = conversations.find((c) => c.conversation_id === activeId);
    return new Set(conv?.enabled_skills ?? []);
  }, [conversations, activeId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return catalog;
    return catalog.filter(
      (s) =>
        s.name.toLowerCase().includes(q) || s.description.toLowerCase().includes(q)
    );
  }, [catalog, query]);

  const withBusy = async (name: string, fn: () => Promise<void>) => {
    setBusy((prev) => new Set(prev).add(name));
    try {
      await fn();
    } finally {
      setBusy((prev) => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    }
  };

  const add = (s: CatalogSkill) => withBusy(s.name, () => addSkillToChat(s.name, s.installed));

  const remove = (name: string) => {
    if (!activeId) return;
    const next = Array.from(enabledSet).filter((n) => n !== name);
    void setConversationSkills(activeId, next);
  };

  return (
    <Dialog
      open={skillMarketplaceOpen}
      onClose={closeSkillMarketplace}
      title="Skill Marketplace"
      className="h-[80vh] w-[min(900px,92vw)]"
    >
      <div className="flex h-full flex-col">
        <header className="shrink-0 border-b border-border p-4 pr-12">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <Sparkles className="size-4 text-primary" />
            Skill Marketplace
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Browse Claude's Agent Skills and load them onto this chat. Each skill adds focused
            instructions (and runnable scripts) the assistant can use.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search skills…"
                className="h-8 w-full rounded-md border border-input bg-background pl-7 pr-2 text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 gap-1 px-2 text-xs text-muted-foreground"
              onClick={() => refreshCatalog()}
              disabled={catalogLoading}
              title="Refresh the catalog from the marketplace"
            >
              <RefreshCw className={`size-3.5 ${catalogLoading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
          {!activeId && (
            <p className="mt-2 text-[11px] text-amber-500/90">
              Open or start a chat to add skills.
            </p>
          )}
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {catalogLoading && catalog.length === 0 ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
              <Sparkles className="size-6 opacity-40" />
              {catalog.length === 0
                ? "The marketplace is unavailable right now — try Refresh."
                : "No skills match your search."}
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {filtered.map((s) => (
                <SkillCard
                  key={s.name}
                  skill={s}
                  enabled={enabledSet.has(s.name)}
                  busy={busy.has(s.name)}
                  canAdd={!!activeId}
                  onAdd={() => add(s)}
                  onRemove={() => remove(s.name)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </Dialog>
  );
}
