import { useCallback, useEffect, useState } from "react";
import { CheckSquare, RefreshCw, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import type { Fact } from "@/types";

/** One fact row: a toggle (checked = included in the agent's context) plus the fact text. */
function FactRow({ fact, onToggle }: { fact: Fact; onToggle: (important: boolean) => void }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-border p-2">
      <button
        type="button"
        onClick={() => onToggle(!fact.important)}
        className="mt-0.5 shrink-0 text-muted-foreground transition-colors hover:text-foreground"
        title={fact.important ? "Included in the assistant's context" : "Excluded from context"}
        aria-pressed={fact.important}
      >
        {fact.important ? (
          <CheckSquare className="size-4 text-primary" />
        ) : (
          <Square className="size-4" />
        )}
      </button>
      <span
        className={`min-w-0 flex-1 text-xs ${
          fact.important ? "" : "text-muted-foreground line-through"
        }`}
      >
        {fact.text}
      </span>
    </div>
  );
}

/** The Facts tab of the right pane: every durable fact the agent has stored about the user, each
 *  with a toggle controlling whether it is always loaded into the agent's context. All facts are
 *  included by default. Re-fetches whenever `refreshKey` bumps (after each completed turn). */
export function FactsCard({ refreshKey }: { refreshKey: number }) {
  const { userId } = useApp();
  const [facts, setFacts] = useState<Fact[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    api
      .listFacts(userId)
      .then(setFacts)
      .catch(() => setFacts([]))
      .finally(() => setLoading(false));
  }, [userId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  function toggle(factId: string, important: boolean) {
    // Optimistic: flip locally, then persist; revert on failure.
    setFacts((prev) =>
      prev.map((f) => (f.fact_id === factId ? { ...f, important } : f))
    );
    api.setFactImportance(factId, userId, important).catch(() => {
      setFacts((prev) =>
        prev.map((f) => (f.fact_id === factId ? { ...f, important: !important } : f))
      );
    });
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm">Facts</CardTitle>
        <Button
          variant="ghost"
          size="icon"
          className="size-6 text-muted-foreground"
          onClick={load}
          disabled={loading}
          title="Refresh facts"
        >
          <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent>
        {loading && facts.length === 0 ? (
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ) : facts.length > 0 ? (
          <div className="space-y-2">
            <p className="text-[10px] text-muted-foreground">
              Checked facts are loaded into the assistant's context every turn. Uncheck a fact to
              keep it stored but leave it out.
            </p>
            {facts.map((f) => (
              <FactRow
                key={f.fact_id}
                fact={f}
                onToggle={(important) => toggle(f.fact_id, important)}
              />
            ))}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No facts yet — as you chat, the assistant remembers durable facts about you here, and
            you can choose which ones it keeps front-of-mind.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
