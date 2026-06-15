export interface AgentTask {
  agent: string;
  task: string;
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
  task: string;
  output: string;
  documents: SwarmDocumentInfo[];
  error: string | null;
}

export interface SwarmArgs {
  tasks: AgentTask[];
}

export interface SwarmResult {
  reports: AgentRunReport[];
}

export interface RunAgentArgs {
  agent: string;
  task: string;
  context?: string;
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
