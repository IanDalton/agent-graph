import { useEffect, useState } from "react";
import { Bot, ChevronDown, ChevronRight, Network, Plus, Sparkles, Trash2, Wrench } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useApp } from "@/state/AppContext";
import type { AgentSpec } from "@/types";

// Tolerant of null/undefined inputs: a spec persisted before a list column existed comes back with
// that field as null (e.g. seeded agents with skills=null), and comparing it must not throw.
const sameSet = (a: string[] | null | undefined, b: string[] | null | undefined) => {
  const x = a ?? [];
  const y = b ?? [];
  return x.length === y.length && [...x].sort().join() === [...y].sort().join();
};

function toggle(list: string[], name: string): string[] {
  return list.includes(name) ? list.filter((n) => n !== name) : [...list, name];
}

/** A labelled group of checkboxes (tools / skills / recipients). */
function CheckGroup({
  icon: Icon,
  label,
  options,
  selected,
  onToggle,
  describe,
  empty,
}: {
  icon: typeof Wrench;
  label: string;
  options: string[];
  selected: string[];
  onToggle: (name: string) => void;
  describe?: (name: string) => string;
  empty?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
        <Icon className="size-3 shrink-0" />
        {label}
      </div>
      {options.length === 0 ? (
        <p className="text-[10px] text-muted-foreground/70">{empty ?? "None available."}</p>
      ) : (
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {options.map((name) => (
            <label
              key={name}
              className="flex cursor-pointer items-center gap-1.5 text-[11px]"
              title={describe?.(name)}
            >
              <input
                type="checkbox"
                className="size-3 accent-primary"
                checked={selected.includes(name)}
                onChange={() => onToggle(name)}
              />
              {name}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function AgentCard({ agent }: { agent: AgentSpec }) {
  const { config, skills, agents, updateAgent, deleteAgent } = useApp();
  const toolGroups = Object.keys(config?.tool_groups ?? {});
  const libraryNames = skills.map((s) => s.name);
  const otherAgents = agents.filter((a) => a.agent_id !== agent.agent_id).map((a) => a.name);

  const [open, setOpen] = useState(false);
  const [role, setRole] = useState(agent.role);
  const [instructions, setInstructions] = useState(agent.instructions);
  // Normalize on read: a null list field (legacy/seeded specs) would otherwise break the checkbox
  // groups' `.includes` and the dirty comparison below.
  const [tools, setTools] = useState<string[]>(agent.tools ?? []);
  const [skl, setSkl] = useState<string[]>(agent.skills ?? []);
  const [recipients, setRecipients] = useState<string[]>(agent.recipients ?? []);

  const dirty =
    role !== agent.role ||
    instructions !== agent.instructions ||
    !sameSet(tools, agent.tools) ||
    !sameSet(skl, agent.skills) ||
    !sameSet(recipients, agent.recipients);

  const save = () =>
    updateAgent(agent.agent_id, { role, instructions, tools, skills: skl, recipients });

  return (
    <div className="rounded-lg border border-border bg-background/40 p-2">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="text-muted-foreground hover:text-foreground"
        >
          {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        </button>
        <Bot className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium">{agent.name}</span>
          {!open && agent.role && (
            <span className="block truncate text-[10px] text-muted-foreground">{agent.role}</span>
          )}
        </span>
        {(agent.skills?.length ?? 0) > 0 && (
          <span className="inline-flex items-center gap-0.5 text-[10px] text-primary">
            <Sparkles className="size-3" />
            {agent.skills.length}
          </span>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="size-6 text-muted-foreground"
          onClick={() => deleteAgent(agent.agent_id)}
          title="Delete agent"
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>

      {open && (
        <div className="mt-2 space-y-2 pl-5">
          <input
            value={role}
            onChange={(e) => setRole(e.target.value)}
            placeholder="One-line role"
            className="h-7 w-full rounded-md border border-input bg-background px-2 text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <CheckGroup
            icon={Wrench}
            label="Tools"
            options={toolGroups}
            selected={tools}
            onToggle={(n) => setTools((t) => toggle(t, n))}
            describe={(n) => config?.tool_groups?.[n] ?? ""}
          />
          <CheckGroup
            icon={Sparkles}
            label="Skills"
            options={libraryNames}
            selected={skl}
            onToggle={(n) => setSkl((s) => toggle(s, n))}
            describe={(n) => skills.find((s) => s.name === n)?.description ?? ""}
            empty="No skills in your library — install some from the marketplace."
          />
          <CheckGroup
            icon={Network}
            label="Can message"
            options={otherAgents}
            selected={recipients}
            onToggle={(n) => setRecipients((r) => toggle(r, n))}
            empty="No other agents to message."
          />
          <Textarea
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="System prompt for this specialist…"
            className="max-h-40 min-h-[64px] resize-y text-[11px]"
          />
          <div className="flex justify-end">
            <Button size="sm" className="h-7 text-xs" disabled={!dirty} onClick={save}>
              Save
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function NewAgentForm() {
  const { createAgent } = useApp();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [instructions, setInstructions] = useState("");
  const [saving, setSaving] = useState(false);

  const nameValid = /^[a-z][a-z0-9-]{1,39}$/.test(name);
  const reset = () => {
    setName("");
    setRole("");
    setInstructions("");
    setOpen(false);
  };
  const create = async () => {
    if (!nameValid || !role.trim() || !instructions.trim()) return;
    setSaving(true);
    try {
      await createAgent({
        name,
        role,
        instructions,
        tools: ["web", "documents"],
        skills: [],
        recipients: [],
      });
      reset();
    } catch {
      // error already logged in AppContext; keep the form open
    } finally {
      setSaving(false);
    }
  };

  if (!open) {
    return (
      <Button
        variant="outline"
        size="sm"
        className="w-full gap-1 text-xs"
        onClick={() => setOpen(true)}
      >
        <Plus className="size-3.5" />
        New agent
      </Button>
    );
  }
  return (
    <div className="space-y-2 rounded-lg border border-dashed border-border p-2">
      <input
        value={name}
        onChange={(e) => setName(e.target.value.toLowerCase())}
        placeholder="kebab-name (e.g. market-researcher)"
        className="h-7 w-full rounded-md border border-input bg-background px-2 text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
      />
      <input
        value={role}
        onChange={(e) => setRole(e.target.value)}
        placeholder="One-line role"
        className="h-7 w-full rounded-md border border-input bg-background px-2 text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
      />
      <Textarea
        value={instructions}
        onChange={(e) => setInstructions(e.target.value)}
        placeholder="System prompt…"
        className="max-h-40 min-h-[56px] resize-y text-[11px]"
      />
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={reset} disabled={saving}>
          Cancel
        </Button>
        <Button
          size="sm"
          className="h-7 text-xs"
          onClick={create}
          disabled={saving || !nameValid || !role.trim() || !instructions.trim()}
        >
          Create
        </Button>
      </div>
    </div>
  );
}

/** The swarm roster editor: view and edit each specialist's tools, skills and chart edges.
 *  Tools/skills set here are what the orchestrator dispatches the agent with. */
export function AgentRoster() {
  const { agents, refreshAgents } = useApp();
  useEffect(() => {
    refreshAgents();
  }, [refreshAgents]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Bot className="size-3.5 text-muted-foreground" />
          Agent roster
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-[10px] text-muted-foreground">
          The specialists the orchestrator dispatches. Grant each the tools and skills its job needs
          (a skill that ships scripts also needs the sandbox tool).
        </p>
        {agents.map((a) => (
          <AgentCard key={a.agent_id} agent={a} />
        ))}
        {agents.length === 0 && (
          <p className="text-[11px] text-muted-foreground">
            No agents yet — they're seeded on your first swarm turn, or add one below.
          </p>
        )}
        <NewAgentForm />
      </CardContent>
    </Card>
  );
}
