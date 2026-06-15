import type {
  AgentRunReport,
  AgentTask,
  DeepResearchArgs,
  DeepResearchResult,
  RunAgentArgs,
  SwarmArgs,
  SwarmDocumentInfo,
  SwarmResult,
} from "./swarmTypes";

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function parseDocuments(raw: unknown): SwarmDocumentInfo[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((d) => d != null && typeof d === "object")
    .map((d) => {
      const o = d as Record<string, unknown>;
      return {
        document_id: str(o.document_id) || undefined,
        title: str(o.title) || undefined,
        mime_type: str(o.mime_type) || undefined,
      };
    });
}

function parseReport(raw: unknown): AgentRunReport | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.agent_id !== "string") return null;
  return {
    agent_id: o.agent_id,
    name: str(o.name) || o.agent_id,
    task: str(o.task),
    output: str(o.output),
    documents: parseDocuments(o.documents),
    error: typeof o.error === "string" ? o.error : null,
  };
}

export function parseSwarmArgs(raw: unknown): SwarmArgs | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.tasks)) return null;
  const tasks: AgentTask[] = o.tasks
    .filter((t) => t != null && typeof t === "object")
    .map((t) => {
      const item = t as Record<string, unknown>;
      if (typeof item.agent !== "string" || typeof item.task !== "string") return null as AgentTask | null;
      return {
        agent: item.agent,
        task: item.task,
        context: typeof item.context === "string" ? item.context : undefined,
      } as AgentTask;
    })
    .filter((t): t is AgentTask => t !== null);
  return { tasks };
}

export function parseSwarmResult(raw: unknown): SwarmResult | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.reports)) return null;
  const reports: AgentRunReport[] = o.reports
    .map(parseReport)
    .filter((r): r is AgentRunReport => r !== null);
  return { reports };
}

export function parseRunAgentArgs(raw: unknown): RunAgentArgs | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.agent !== "string" || typeof o.task !== "string") return null;
  return {
    agent: o.agent,
    task: o.task,
    context: typeof o.context === "string" ? o.context : undefined,
  };
}

export function parseRunAgentResult(raw: unknown): AgentRunReport | null {
  return parseReport(raw);
}

export function parseDeepResearchArgs(raw: unknown): DeepResearchArgs | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.question !== "string") return null;
  return { question: o.question };
}

export function parseDeepResearchResult(raw: unknown): DeepResearchResult | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.question !== "string") return null;
  return {
    question: o.question,
    report: str(o.report),
    documents: parseDocuments(o.documents),
    error: typeof o.error === "string" ? o.error : null,
  };
}
