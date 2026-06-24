import { useMemo, useState } from "react";
import { Check, Loader2, Pencil, Plus, RefreshCw, Search, Sparkles, Trash2, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { useApp } from "@/state/appcontext";

/** Unified card model: a marketplace skill (installable) or a user-authored skill (editable). */
type Entry =
  | { kind: "market"; name: string; description: string; installed: boolean }
  | { kind: "user"; name: string; description: string };

function SkillCard({
  entry,
  busy,
  onInstall,
  onRemove,
  onEdit,
}: {
  entry: Entry;
  busy: boolean;
  onInstall: () => void;
  onRemove: () => void;
  onEdit: () => void;
}) {
  const installed = entry.kind === "user" || entry.installed;
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-background/40 p-3">
      <div className="flex items-start gap-2">
        <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{entry.name}</span>
            {entry.kind === "user" && (
              <Badge variant="secondary" className="shrink-0 text-[10px]">
                Authored
              </Badge>
            )}
          </div>
          <p className="mt-0.5 line-clamp-3 text-xs text-muted-foreground">
            {entry.description || "No description provided."}
          </p>
        </div>
      </div>
      <div className="mt-auto flex items-center justify-end gap-1">
        {entry.kind === "user" && (
          <Button variant="ghost" size="sm" className="h-7 gap-1 text-xs" onClick={onEdit} title="Edit this skill">
            <Pencil className="size-3.5" />
            Edit
          </Button>
        )}
        {installed ? (
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={onRemove}
            disabled={busy}
            title="Remove from your library"
          >
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Check className="size-3.5 text-primary" />}
            Installed
            <Trash2 className="size-3 text-muted-foreground" />
          </Button>
        ) : (
          <Button
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={onInstall}
            disabled={busy}
            title="Install into your library (active in all chats)"
          >
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Plus className="size-3.5" />}
            {busy ? "Installing…" : "Install"}
          </Button>
        )}
      </div>
    </div>
  );
}

type Draft = { name: string; description: string; body: string };

/** The create/edit form for a user-authored skill. */
function SkillEditor({
  initial,
  isEdit,
  onCancel,
  onSaved,
}: {
  initial: Draft;
  isEdit: boolean;
  onCancel: () => void;
  onSaved: () => void;
}) {
  const { saveSkill, skills } = useApp();
  const [draft, setDraft] = useState<Draft>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nameValid = /^[a-z][a-z0-9-]{0,63}$/.test(draft.name);
  // Warn when a new authored skill shadows an existing library skill (upsert-by-name overwrites).
  const collides = !isEdit && skills.some((s) => s.name === draft.name);

  const save = async () => {
    if (!nameValid || !draft.body.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await saveSkill(draft);
      onSaved();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3 p-4">
      <div className="space-y-1">
        <label className="text-xs font-medium text-muted-foreground">Name</label>
        <input
          value={draft.name}
          disabled={isEdit}
          onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value.toLowerCase() }))}
          placeholder="my-skill"
          className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-60"
        />
        {draft.name && !nameValid && (
          <p className="text-[11px] text-destructive">
            Kebab-case only: lowercase letters, digits, hyphens (e.g. my-skill).
          </p>
        )}
        {collides && (
          <p className="text-[11px] text-amber-500/90">
            A skill named “{draft.name}” already exists — saving will overwrite it.
          </p>
        )}
      </div>
      <div className="space-y-1">
        <label className="text-xs font-medium text-muted-foreground">Description</label>
        <input
          value={draft.description}
          onChange={(e) => setDraft((d) => ({ ...d, description: e.target.value }))}
          placeholder="One line: what it does and when to use it"
          className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>
      <div className="space-y-1">
        <label className="text-xs font-medium text-muted-foreground">Instructions</label>
        <Textarea
          value={draft.body}
          onChange={(e) => setDraft((d) => ({ ...d, body: e.target.value }))}
          placeholder="The skill's full instructions (markdown). The assistant loads these when the skill is relevant."
          className="min-h-[260px] resize-y text-xs"
        />
      </div>
      {error && <p className="text-[11px] text-destructive">{error}</p>}
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel} disabled={saving}>
          Cancel
        </Button>
        <Button size="sm" onClick={save} disabled={saving || !nameValid || !draft.body.trim()}>
          {saving ? <Loader2 className="mr-1 size-3.5 animate-spin" /> : null}
          {isEdit ? "Save changes" : "Create skill"}
        </Button>
      </div>
    </div>
  );
}

/** The Skill Marketplace dialog: browse Claude's catalog + your authored skills, install/remove, and
 *  author your own. Skills live in an account-wide library and are active in every regular chat. */
export function SkillMarketplace() {
  const {
    skillMarketplaceOpen,
    closeSkillMarketplace,
    catalog,
    catalogLoading,
    refreshCatalog,
    installSkill,
    removeSkill,
    skills,
    getSkillContent,
  } = useApp();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [view, setView] = useState<"browse" | "editor">("browse");
  const [editorInitial, setEditorInitial] = useState<Draft>({ name: "", description: "", body: "" });
  const [editing, setEditing] = useState(false);

  // Merge the marketplace catalog with the user's authored skills (which aren't on GitHub).
  const entries = useMemo<Entry[]>(() => {
    const catalogNames = new Set(catalog.map((c) => c.name));
    const market: Entry[] = catalog.map((c) => ({
      kind: "market",
      name: c.name,
      description: c.description,
      installed: c.installed,
    }));
    const authored: Entry[] = skills
      .filter((s) => s.source === "user" && !catalogNames.has(s.name))
      .map((s) => ({ kind: "user", name: s.name, description: s.description }));
    return [...authored, ...market];
  }, [catalog, skills]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(
      (s) => s.name.toLowerCase().includes(q) || s.description.toLowerCase().includes(q)
    );
  }, [entries, query]);

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

  const openCreate = () => {
    setEditorInitial({ name: "", description: "", body: "" });
    setEditing(false);
    setView("editor");
  };

  const openEdit = async (name: string) => {
    try {
      const full = await getSkillContent(name);
      setEditorInitial({ name: full.name, description: full.description, body: full.body });
    } catch {
      setEditorInitial({ name, description: "", body: "" });
    }
    setEditing(true);
    setView("editor");
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
            Install Claude's Agent Skills or author your own. Skills go into your library and are
            available in every chat — and assignable to swarm specialists.
          </p>
          {view === "browse" ? (
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
                size="sm"
                className="h-8 gap-1 px-2 text-xs"
                onClick={openCreate}
                title="Author your own skill"
              >
                <Plus className="size-3.5" />
                Create skill
              </Button>
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
          ) : (
            <div className="mt-3">
              <Button
                variant="ghost"
                size="sm"
                className="h-8 gap-1 px-2 text-xs text-muted-foreground"
                onClick={() => setView("browse")}
              >
                <X className="size-3.5" />
                Back to catalog
              </Button>
            </div>
          )}
        </header>

        {view === "editor" ? (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <SkillEditor
              initial={editorInitial}
              isEdit={editing}
              onCancel={() => setView("browse")}
              onSaved={() => setView("browse")}
            />
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {catalogLoading && entries.length === 0 ? (
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-24 w-full" />
                ))}
              </div>
            ) : filtered.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
                <Sparkles className="size-6 opacity-40" />
                {entries.length === 0
                  ? "The marketplace is unavailable right now — try Refresh, or Create a skill."
                  : "No skills match your search."}
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {filtered.map((s) => (
                  <SkillCard
                    key={`${s.kind}:${s.name}`}
                    entry={s}
                    busy={busy.has(s.name)}
                    onInstall={() => withBusy(s.name, () => installSkill(s.name))}
                    onRemove={() => withBusy(s.name, () => removeSkill(s.name))}
                    onEdit={() => openEdit(s.name)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </Dialog>
  );
}
