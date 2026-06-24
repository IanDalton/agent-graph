import type {
  AgentRunReport,
  DeepResearchArgs,
  DeepResearchResult,
  SendMessageArgs,
  SendMessagesArgs,
  SwarmDocumentInfo,
  SwarmResult,
} from "./swarmtypes";

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

export function parseSendMessagesArgs(raw: unknown): SendMessagesArgs | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.messages)) return null;
  const messages: SendMessageArgs[] = o.messages
    .filter((m) => m != null && typeof m === "object")
    .map((m) => {
      const item = m as Record<string, unknown>;
      if (typeof item.recipient !== "string" || typeof item.message !== "string")
        return null as SendMessageArgs | null;
      return {
        recipient: item.recipient,
        message: item.message,
        context: typeof item.context === "string" ? item.context : undefined,
      } as SendMessageArgs;
    })
    .filter((m): m is SendMessageArgs => m !== null);
  return { messages };
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

export function parseSendMessageArgs(raw: unknown): SendMessageArgs | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.recipient !== "string" || typeof o.message !== "string") return null;
  return {
    recipient: o.recipient,
    message: o.message,
    context: typeof o.context === "string" ? o.context : undefined,
  };
}

export function parseSendMessageResult(raw: unknown): AgentRunReport | null {
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
