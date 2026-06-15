import { useState } from "react";
import { Brain, ChevronRight, FileText } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Step } from "@/types";
import { ToolChip } from "@/components/ToolChip";
import { useApp } from "@/state/AppContext";
import { SwarmTaskBoard } from "./SwarmTaskBoard";
import { RunAgentCard } from "./RunAgentCard";
import { DeepResearchPanel } from "./DeepResearchPanel";
import { RosterChip } from "./RosterChip";

/** A collapsible reasoning run — one segment of the agent's thinking between tool calls. */
function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/40 text-xs text-muted-foreground">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-xl px-2.5 py-1.5 text-left transition-colors hover:bg-white/5"
      >
        <ChevronRight
          className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")}
        />
        <Brain className="size-3 shrink-0" />
        <span className="font-medium">Reasoning</span>
      </button>
      {open && (
        <pre className="whitespace-pre-wrap break-words border-t border-white/10 px-2.5 py-2 font-sans">
          {text}
        </pre>
      )}
    </div>
  );
}

/** The artifact card: a big tap target dropped into the chat when the agent creates
 *  (or revises) a document. Clicking it spotlights the document in the side panel. */
function DocumentCard({
  step,
}: {
  step: Extract<Step, { kind: "document" }>;
}) {
  const { featureDocument } = useApp();
  return (
    <button
      type="button"
      onClick={() => featureDocument(step.documentId)}
      title="Open in the Documents panel"
      className="flex w-full items-center gap-3 rounded-xl border border-primary/30 bg-primary/10 px-3 py-3 text-left transition-colors hover:border-primary/60 hover:bg-primary/20"
    >
      <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/20">
        <FileText className="size-5 text-primary" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold">
          {step.title || "Document"}
        </span>
        <span className="block truncate text-xs text-muted-foreground">
          {step.action === "created" ? "Document created" : "Document updated"}
          {step.mimeType ? ` · ${step.mimeType}` : ""} · click to open
        </span>
      </span>
      <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
    </button>
  );
}

/** Routes a Step to the correct swarm-aware renderer. */
export function SwarmStepItem({ step }: { step: Step }) {
  if (step.kind === "thinking") {
    if (!step.text.trim()) return null;
    return <ThinkingBlock text={step.text} />;
  }
  if (step.kind === "document") {
    return <DocumentCard step={step} />;
  }

  // Tool steps: route based on tool name
  const toolName = step.tool.toolName || "";

  switch (toolName) {
    case "run_swarm":
      return <SwarmTaskBoard tool={step.tool} />;
    case "run_agent":
      return <RunAgentCard tool={step.tool} />;
    case "deep_research":
      return <DeepResearchPanel tool={step.tool} />;
    case "create_agent":
    case "update_agent":
    case "delete_agent":
    case "list_agents":
      return <RosterChip tool={step.tool} />;
    default:
      return <ToolChip tool={step.tool} />;
  }
}
