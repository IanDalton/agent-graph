import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, ReactNode } from "react";
import {
  Archive,
  ArchiveRestore,
  ChevronDown,
  ChevronRight,
  FolderPlus,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Plus,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { useApp } from "@/state/AppContext";
import { ModeIcon } from "@/components/ModeIcon";
import type { Conversation, Project } from "@/types";

/** dataTransfer key for a conversation being dragged onto a project (or the Ungrouped zone). */
const DRAG_MIME = "application/x-conversation-id";

function conversationLabel(c: Conversation): string {
  if (c.title) return c.title;
  return `Chat ${c.conversation_id.slice(0, 6)}`;
}

/** A tiny downward-anchored dropdown menu (the shared Popover anchors upward for the composer). */
function RowMenu({ children }: { children: (close: () => void) => ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        title="More"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-accent group-hover:opacity-100 data-[open=true]:opacity-100"
        data-open={open}
      >
        <MoreHorizontal className="size-4" />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-1 min-w-[11rem] overflow-hidden rounded-lg border border-border bg-card p-1 text-sm shadow-xl"
        >
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}

function MenuItem({
  icon: Icon,
  label,
  onClick,
  danger,
}: {
  icon: typeof Pin;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={cn(
        "flex w-full items-center gap-2 rounded px-2 py-1.5 text-left transition-colors hover:bg-accent",
        danger && "text-red-500 hover:bg-red-500/10"
      )}
    >
      <Icon className="size-3.5 shrink-0" />
      <span className="truncate">{label}</span>
    </button>
  );
}

function ConversationRow({ c }: { c: Conversation }) {
  const {
    activeId,
    selectConversation,
    projects,
    setConversationProject,
    setConversationPinned,
    setConversationArchived,
    deleteConversation,
  } = useApp();
  const active = c.conversation_id === activeId;

  return (
    <button
      type="button"
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_MIME, c.conversation_id);
        e.dataTransfer.effectAllowed = "move";
      }}
      onClick={() => selectConversation(c.conversation_id)}
      className={cn(
        "group flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors",
        active ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
      )}
    >
      <ModeIcon mode={c.mode} className="size-4 shrink-0 text-muted-foreground" />
      <span className="flex-1 truncate">{conversationLabel(c)}</span>
      {c.pinned && <Pin className="size-3 shrink-0 text-muted-foreground" />}
      <RowMenu>
        {(close) => (
          <>
            <MenuItem
              icon={c.pinned ? PinOff : Pin}
              label={c.pinned ? "Unpin" : "Pin to top"}
              onClick={() => {
                setConversationPinned(c.conversation_id, !c.pinned);
                close();
              }}
            />
            <MenuItem
              icon={c.archived ? ArchiveRestore : Archive}
              label={c.archived ? "Unarchive" : "Archive"}
              onClick={() => {
                setConversationArchived(c.conversation_id, !c.archived);
                close();
              }}
            />
            <div className="my-1 border-t border-border" />
            <div className="px-2 py-1 text-xs text-muted-foreground">Move to…</div>
            {(c.project_id ?? null) !== null && (
              <MenuItem
                icon={ChevronRight}
                label="Ungrouped"
                onClick={() => {
                  setConversationProject(c.conversation_id, null);
                  close();
                }}
              />
            )}
            {projects
              .filter((p) => p.project_id !== c.project_id)
              .map((p) => (
                <MenuItem
                  key={p.project_id}
                  icon={ChevronRight}
                  label={p.title || "Untitled project"}
                  onClick={() => {
                    setConversationProject(c.conversation_id, p.project_id);
                    close();
                  }}
                />
              ))}
            <div className="my-1 border-t border-border" />
            <MenuItem
              icon={Trash2}
              label="Delete"
              danger
              onClick={() => {
                if (window.confirm("Delete this conversation permanently?")) {
                  deleteConversation(c.conversation_id);
                }
                close();
              }}
            />
          </>
        )}
      </RowMenu>
    </button>
  );
}

function ProjectGroup({
  project,
  conversations,
}: {
  project: Project;
  conversations: Conversation[];
}) {
  const {
    newConversation,
    setProjectTitle,
    deleteProject,
    setConversationProject,
    activeProjectId,
    selectProject,
    activeId,
  } = useApp();
  const selected = activeProjectId === project.project_id;
  // "Am I chatting inside this project?" — the active conversation belongs to it, or it's selected.
  const hasActiveChat = conversations.some((c) => c.conversation_id === activeId);
  const expandedByActivity = selected || hasActiveChat;
  // Projects start collapsed unless the user is currently inside them.
  const [collapsed, setCollapsed] = useState(!expandedByActivity);
  const [renaming, setRenaming] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [draftTitle, setDraftTitle] = useState(project.title || "");
  const [dragOver, setDragOver] = useState(false);

  // Auto-expand when this project becomes the one being worked in (selected or holds the active
  // chat); leaves manual collapse/expand alone otherwise.
  useEffect(() => {
    if (expandedByActivity) setCollapsed(false);
  }, [expandedByActivity]);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const convId = e.dataTransfer.getData(DRAG_MIME);
    if (convId) setConversationProject(convId, project.project_id);
  };

  return (
    <li
      onDragOver={(e) => {
        // Accept conversation drags; show the drop affordance.
        if (e.dataTransfer.types.includes(DRAG_MIME)) {
          e.preventDefault();
          setDragOver(true);
        }
      }}
      onDragLeave={(e) => {
        // Only clear when the pointer actually leaves the group (not when entering a child).
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false);
      }}
      onDrop={onDrop}
      className={cn(
        "rounded-md",
        dragOver && "bg-primary/10 ring-1 ring-primary/40"
      )}
    >
      <div
        className={cn(
          "group flex items-center gap-1 rounded-md px-1 py-1 text-xs font-medium text-muted-foreground hover:bg-accent/40",
          selected && "bg-accent/60 text-accent-foreground"
        )}
      >
        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          className="shrink-0"
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? (
            <ChevronRight className="size-3.5" />
          ) : (
            <ChevronDown className="size-3.5" />
          )}
        </button>
        <button
          type="button"
          onClick={() => {
            // Select the project (so "New Chat" lands in it) and reveal its chats.
            selectProject(project.project_id);
            setCollapsed(false);
          }}
          className="flex flex-1 items-center gap-1 truncate text-left"
        >
          <span className="truncate uppercase tracking-wide">
            {project.title || "Untitled project"}
          </span>
          <span className="shrink-0 text-[10px] opacity-60">{conversations.length}</span>
        </button>
        <button
          type="button"
          title="New chat in project"
          onClick={() => newConversation("regular", project.project_id)}
          className="rounded p-1 opacity-0 transition-opacity hover:bg-accent group-hover:opacity-100"
        >
          <Plus className="size-3.5" />
        </button>
        <RowMenu>
          {(close) => (
            <>
              <MenuItem
                icon={Pencil}
                label="Rename project"
                onClick={() => {
                  setDraftTitle(project.title || "");
                  setRenaming(true);
                  close();
                }}
              />
              <MenuItem
                icon={Trash2}
                label="Delete project"
                danger
                onClick={() => {
                  // Empty projects carry no data to lose — delete straight away. Only prompt for
                  // confirmation when chats/documents would be cascaded.
                  if (conversations.length === 0) deleteProject(project.project_id);
                  else setConfirmDelete(true);
                  close();
                }}
              />
            </>
          )}
        </RowMenu>
      </div>

      {!collapsed && (
        <ul className="ml-3 space-y-1 border-l border-border pl-1">
          {conversations.map((c) => (
            <li key={c.conversation_id}>
              <ConversationRow c={c} />
            </li>
          ))}
          {conversations.length === 0 && (
            <li className="px-2 py-1.5 text-center text-[11px] text-muted-foreground">
              No chats yet — use +
            </li>
          )}
        </ul>
      )}

      <Dialog
        open={renaming}
        onClose={() => setRenaming(false)}
        title="Rename project"
        className="w-[22rem] p-4"
      >
        <h2 className="mb-3 text-sm font-semibold">Rename project</h2>
        <input
          autoFocus
          value={draftTitle}
          onChange={(e) => setDraftTitle(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              setProjectTitle(project.project_id, draftTitle.trim());
              setRenaming(false);
            }
          }}
          className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring"
          placeholder="Project name"
        />
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={() => setRenaming(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={() => {
              setProjectTitle(project.project_id, draftTitle.trim());
              setRenaming(false);
            }}
          >
            Save
          </Button>
        </div>
      </Dialog>

      <Dialog
        open={confirmDelete}
        onClose={() => setConfirmDelete(false)}
        title="Delete project"
        className="w-[24rem] p-4"
      >
        <h2 className="mb-2 text-sm font-semibold text-red-500">Delete project?</h2>
        <p className="text-sm text-muted-foreground">
          This permanently deletes <strong>{project.title || "this project"}</strong>, its{" "}
          {conversations.length} conversation{conversations.length === 1 ? "" : "s"}, and its
          uploaded documents. Documents you marked <strong>Global</strong> are kept. This cannot be
          undone.
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => {
              deleteProject(project.project_id);
              setConfirmDelete(false);
            }}
          >
            Delete
          </Button>
        </div>
      </Dialog>
    </li>
  );
}

export function Sidebar() {
  const {
    conversations,
    openNewChatPicker,
    projects,
    newProject,
    showArchived,
    setShowArchived,
    activeProjectId,
    selectProject,
    setConversationProject,
  } = useApp();

  const [creating, setCreating] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [dragOverUngrouped, setDragOverUngrouped] = useState(false);

  const activeProject = projects.find((p) => p.project_id === activeProjectId) ?? null;

  // Group conversations by project; pinned-first ordering is already applied by the server.
  const ungrouped = useMemo(
    () => conversations.filter((c) => (c.project_id ?? null) === null),
    [conversations]
  );
  const byProject = useMemo(() => {
    const map = new Map<string, Conversation[]>();
    for (const c of conversations) {
      if (c.project_id) {
        const list = map.get(c.project_id) ?? [];
        list.push(c);
        map.set(c.project_id, list);
      }
    }
    return map;
  }, [conversations]);

  const createProject = async () => {
    // Require a name so the sidebar never fills up with "Untitled project" rows.
    const title = newTitle.trim();
    if (!title) return;
    await newProject(title);
    setNewTitle("");
    setCreating(false);
  };

  return (
    <aside className="flex h-full flex-col border-r border-border bg-card">
      <div className="flex items-center justify-between px-4 py-3">
        <span className="text-sm font-semibold tracking-tight">Mission Control</span>
      </div>

      <div className="flex gap-2 px-3 pb-2">
        <Button
          variant="secondary"
          className="flex-1 justify-start overflow-hidden"
          onClick={openNewChatPicker}
          title={
            activeProject
              ? `New chat in ${activeProject.title || "this project"}`
              : "New chat"
          }
        >
          <Plus className="shrink-0" />
          <span className="truncate">
            New Chat
            {activeProject && (
              <span className="text-muted-foreground">
                {" "}
                · {activeProject.title || "project"}
              </span>
            )}
          </span>
        </Button>
        <Button
          variant="outline"
          size="icon"
          title="New project"
          onClick={() => setCreating(true)}
        >
          <FolderPlus className="size-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1 px-2">
        <ul className="space-y-1 py-1">
          {projects.map((p) => (
            <ProjectGroup
              key={p.project_id}
              project={p}
              conversations={byProject.get(p.project_id) ?? []}
            />
          ))}

          {projects.length > 0 ? (
            <li
              onDragOver={(e) => {
                if (e.dataTransfer.types.includes(DRAG_MIME)) {
                  e.preventDefault();
                  setDragOverUngrouped(true);
                }
              }}
              onDragLeave={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget as Node))
                  setDragOverUngrouped(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setDragOverUngrouped(false);
                const convId = e.dataTransfer.getData(DRAG_MIME);
                if (convId) setConversationProject(convId, null);
              }}
              className={cn(
                "rounded-md",
                dragOverUngrouped && "bg-primary/10 ring-1 ring-primary/40"
              )}
            >
              <button
                type="button"
                onClick={() => selectProject(null)}
                className={cn(
                  "w-full rounded px-2 pt-2 text-left text-[10px] font-medium uppercase tracking-wide text-muted-foreground hover:text-foreground",
                  activeProjectId === null && "text-foreground"
                )}
                title="Ungrouped — drop a chat here to remove it from its project"
              >
                Ungrouped
              </button>
              <ul className="space-y-1 pt-1">
                {ungrouped.map((c) => (
                  <li key={c.conversation_id}>
                    <ConversationRow c={c} />
                  </li>
                ))}
                {ungrouped.length === 0 && (
                  <li className="px-2 py-1.5 text-center text-[11px] text-muted-foreground">
                    Drop a chat here to ungroup it.
                  </li>
                )}
              </ul>
            </li>
          ) : (
            ungrouped.map((c) => (
              <li key={c.conversation_id}>
                <ConversationRow c={c} />
              </li>
            ))
          )}

          {conversations.length === 0 && projects.length === 0 && (
            <li className="px-2 py-4 text-center text-xs text-muted-foreground">
              No conversations yet.
            </li>
          )}
        </ul>
      </ScrollArea>

      <div className="border-t border-border px-3 py-2">
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(e) => setShowArchived(e.target.checked)}
            className="size-3.5 accent-primary"
          />
          Show archived
        </label>
      </div>

      <Dialog
        open={creating}
        onClose={() => setCreating(false)}
        title="New project"
        className="w-[22rem] p-4"
      >
        <h2 className="mb-3 text-sm font-semibold">New project</h2>
        <input
          autoFocus
          value={newTitle}
          onChange={(e) => setNewTitle(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && createProject()}
          className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring"
          placeholder="Project name"
        />
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={() => setCreating(false)}>
            Cancel
          </Button>
          <Button size="sm" onClick={createProject} disabled={!newTitle.trim()}>
            Create
          </Button>
        </div>
      </Dialog>
    </aside>
  );
}
