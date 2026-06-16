import { useState } from "react";
import { AlertCircle, CheckCircle2, ChevronRight, FileText, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import { Markdown } from "@/components/Markdown";
import { colorForAgent } from "./agentColors";
import type { AgentRunReport, SendMessageArgs } from "./swarmTypes";

export function AgentTaskCard({
  message,
  report,
  index,
}: {
  message: SendMessageArgs;
  report: AgentRunReport | null;
  index: number;
}) {
  const [outputOpen, setOutputOpen] = useState(false);
  const isLoading = report === null;
  const isError = report?.error;
  const color = colorForAgent(report?.agent_id || message.recipient);

  return (
    <div className="rounded-lg border border-white/10 bg-slate-900/40 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground">
            Message {index + 1}
          </span>
          <span className={cn("inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-mono", color.text)}>
            <span className={cn("inline-block size-2 rounded-full", color.dot)} />
            {report?.name || message.recipient}
          </span>
        </div>
        <div className="shrink-0">
          {isLoading ? (
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          ) : isError ? (
            <AlertCircle className="size-4 text-destructive" />
          ) : (
            <CheckCircle2 className="size-4 text-emerald-500" />
          )}
        </div>
      </div>

      {/* Message / assignment text */}
      <div className="px-3 py-1 text-xs text-muted-foreground border-t border-white/5">
        {message.message}
      </div>

      {/* Output section (when report available) */}
      {report && (
        <>
          {report.output.trim() || report.documents.length > 0 ? (
            <div className="border-t border-white/5">
              {report.output.trim() && (
                <button
                  type="button"
                  onClick={() => setOutputOpen((o) => !o)}
                  className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left transition-colors hover:bg-white/5"
                >
                  <ChevronRight
                    className={cn(
                      "size-3 shrink-0 transition-transform",
                      outputOpen && "rotate-90"
                    )}
                  />
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground font-mono">
                    Output
                  </span>
                </button>
              )}
              {outputOpen && report.output.trim() && (
                <div className="px-3 py-2 border-t border-white/5 max-h-32 overflow-y-auto">
                  <div className="text-xs text-foreground prose prose-invert prose-sm max-w-none">
                    <Markdown>{report.output}</Markdown>
                  </div>
                </div>
              )}

              {/* Documents */}
              {report.documents.length > 0 && (
                <div className="border-t border-white/5 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground font-mono mb-1.5">
                    Documents
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {report.documents.map((doc, i) => (
                      <div
                        key={i}
                        className="inline-flex items-center gap-1 rounded px-2 py-1 bg-white/5 text-[10px] font-mono text-muted-foreground"
                      >
                        <FileText className="size-3" />
                        {doc.title || "document"}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : null}

          {/* Error */}
          {isError && (
            <div className="border-t border-white/5 px-3 py-2">
              <div className="rounded border border-destructive/50 bg-destructive/10 px-2 py-1 text-xs text-destructive font-mono">
                {report.error}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
