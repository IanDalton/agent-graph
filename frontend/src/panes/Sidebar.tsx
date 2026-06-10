import { Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { useApp } from "@/state/AppContext";
import { ModeIcon } from "@/components/ModeIcon";
import type { Conversation } from "@/types";

function conversationLabel(c: Conversation): string {
  if (c.title) return c.title;
  return `Chat ${c.conversation_id.slice(0, 6)}`;
}

export function Sidebar() {
  const { conversations, activeId, selectConversation, newConversation } = useApp();

  return (
    <aside className="flex h-full flex-col border-r border-border bg-card">
      <div className="flex items-center justify-between px-4 py-3">
        <span className="text-sm font-semibold tracking-tight">Mission Control</span>
      </div>

      <div className="px-3 pb-2">
        <Button
          variant="secondary"
          className="w-full justify-start"
          onClick={() => void newConversation()}
        >
          <Plus />
          New Chat
        </Button>
      </div>

      <ScrollArea className="flex-1 px-2">
        <ul className="space-y-1 py-1">
          {conversations.map((c) => {
            const active = c.conversation_id === activeId;
            return (
              <li key={c.conversation_id}>
                <button
                  type="button"
                  onClick={() => selectConversation(c.conversation_id)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors",
                    active
                      ? "bg-accent text-accent-foreground"
                      : "hover:bg-accent/50"
                  )}
                >
                  <ModeIcon
                    mode={c.mode}
                    className="size-4 shrink-0 text-muted-foreground"
                  />
                  <span className="flex-1 truncate">{conversationLabel(c)}</span>
                  {/* Status slot — empty for Regular; future modes fill it
                      (e.g. "Searching…", "3/5 agents"). */}
                  <span className="shrink-0 text-xs text-muted-foreground" />
                </button>
              </li>
            );
          })}
          {conversations.length === 0 && (
            <li className="px-2 py-4 text-center text-xs text-muted-foreground">
              No conversations yet.
            </li>
          )}
        </ul>
      </ScrollArea>
    </aside>
  );
}
