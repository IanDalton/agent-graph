import { CheckCircle2, FileText, Loader2, Microscope } from "lucide-react";

import { ToolChip } from "@/components/ToolChip";
import { Markdown } from "@/components/Markdown";
import { Skeleton } from "@/components/ui/skeleton";
import type { ToolEvent } from "@/types";
import { parseDeepResearchArgs, parseDeepResearchResult } from "./parseSwarm";

export function DeepResearchPanel({ tool }: { tool: ToolEvent }) {
  const args = parseDeepResearchArgs(tool.args);
  if (!args) {
    return <ToolChip tool={tool} />;
  }

  const result = parseDeepResearchResult(tool.result);
  const isLoading = !result;
  const isError = result?.error;

  return (
    <div className="rounded-xl border border-violet-500/20 bg-violet-950/20 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-violet-500/20 px-3 py-2">
        <div className="flex items-center gap-2">
          <Microscope className="size-4 text-violet-400" />
          <span className="text-sm font-medium">Deep Research</span>
        </div>
        <div>
          {isLoading ? (
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          ) : isError ? (
            <span className="text-xs text-destructive font-mono">error</span>
          ) : (
            <CheckCircle2 className="size-4 text-emerald-500" />
          )}
        </div>
      </div>

      {/* Loading state */}
      {isLoading && <Skeleton className="h-24 w-full m-3 rounded" />}

      {/* Content */}
      {result && (
        <>
          {result.report && (
            <div className="px-3 py-2 max-h-80 overflow-y-auto border-b border-violet-500/20">
              <div className="text-sm prose prose-invert prose-sm max-w-none">
                <Markdown>{result.report}</Markdown>
              </div>
            </div>
          )}

          {/* Documents */}
          {result.documents.length > 0 && (
            <div className="border-b border-violet-500/20 px-3 py-2">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground font-mono mb-1.5">
                Documents
              </div>
              <div className="flex flex-wrap gap-1.5">
                {result.documents.map((doc, i) => (
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

          {/* Error */}
          {isError && (
            <div className="px-3 py-2">
              <div className="rounded border border-destructive/50 bg-destructive/10 px-2 py-1 text-xs text-destructive font-mono">
                {result.error}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
