export interface SendMessageArgs {
  recipient: string;
  message: string;
  context?: string;
}

export interface SwarmDocumentInfo {
  document_id?: string;
  title?: string;
  mime_type?: string;
}

export interface AgentRunReport {
  agent_id: string;
  name: string;
  // The assignment text the agent was sent (the backend keeps the field name `task`).
  task: string;
  output: string;
  documents: SwarmDocumentInfo[];
  error: string | null;
}

export interface SendMessagesArgs {
  messages: SendMessageArgs[];
}

export interface SwarmResult {
  reports: AgentRunReport[];
}

export interface DeepResearchArgs {
  question: string;
}

export interface DeepResearchResult {
  question: string;
  report: string;
  documents: SwarmDocumentInfo[];
  error: string | null;
}
