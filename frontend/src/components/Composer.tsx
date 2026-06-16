import { useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { Brain, Check, SendHorizonal, Square, Zap } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Popover } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { ModeIcon } from "@/components/ModeIcon";
import { useApp } from "@/state/AppContext";
import type { Mode } from "@/types";

/** The agent modes a conversation can be switched to (mirrors Canvas's new-chat picker). */
const MODE_OPTIONS: { mode: Mode; label: string }[] = [
  { mode: "regular", label: "Regular chat" },
  { mode: "research", label: "Deep research" },
  { mode: "swarm", label: "Agent swarm" },
];

/** A mode-switch chip: shows the active conversation's mode and switches it on select.
 *  Each option carries its own icon, so it can't reuse the single-icon ControlChip. */
function ModeChip({ value, onSelect }: { value: Mode; onSelect: (mode: Mode) => void }) {
  const current = MODE_OPTIONS.find((o) => o.mode === value) ?? MODE_OPTIONS[0];
  return (
    <Popover
      align="start"
      trigger={({ open, toggle }) => (
        <button
          type="button"
          onClick={toggle}
          aria-haspopup="menu"
          aria-expanded={open}
          title="Mode"
          className={cn(
            "inline-flex max-w-[12rem] items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground",
            open && "bg-white/5 text-foreground"
          )}
        >
          <ModeIcon mode={current.mode} className="size-3.5 shrink-0" />
          <span className="truncate font-medium">{current.label}</span>
        </button>
      )}
    >
      {({ close }) => (
        <>
          <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground/70">
            Mode
          </div>
          {MODE_OPTIONS.map((opt) => {
            const selected = opt.mode === value;
            return (
              <button
                key={opt.mode}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                onClick={() => {
                  onSelect(opt.mode);
                  close();
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-white/5",
                  selected ? "text-foreground" : "text-muted-foreground"
                )}
              >
                <ModeIcon mode={opt.mode} className="size-3.5 shrink-0" />
                <span className="flex-1 truncate">{opt.label}</span>
                {selected && <Check className="size-3.5 shrink-0 text-primary" />}
              </button>
            );
          })}
        </>
      )}
    </Popover>
  );
}

/** Strip the provider/path noise from a model label so the indicator stays short:
 *  "ollama/google/gemma-4-26b-a4b-qat" → "gemma-4-26b-a4b-qat", "openai:gpt-5.2" → "gpt-5.2". */
function shortModel(model: string): string {
  const tail = model.split("/").pop() ?? model;
  return tail.includes(":") ? tail.split(":").pop() ?? tail : tail;
}

/** A single embedded control: a low-profile icon+label chip that opens a popover of options. */
function ControlChip({
  icon: Icon,
  label,
  title,
  options,
  current,
  onSelect,
  align = "start",
  textClassName,
}: {
  icon: typeof Brain;
  label: string;
  title: string;
  options: string[];
  current: string;
  onSelect: (value: string) => void;
  align?: "start" | "end";
  textClassName?: string;
}) {
  return (
    <Popover
      align={align}
      trigger={({ open, toggle }) => (
        <button
          type="button"
          onClick={toggle}
          aria-haspopup="menu"
          aria-expanded={open}
          title={title}
          className={cn(
            "inline-flex max-w-[12rem] items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground",
            open && "bg-white/5 text-foreground"
          )}
        >
          <Icon className="size-3.5 shrink-0" />
          <span className={cn("truncate font-medium", textClassName)}>{label}</span>
        </button>
      )}
    >
      {({ close }) => (
        <>
          <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground/70">
            {title}
          </div>
          {options.map((opt) => {
            const selected = opt === current;
            return (
              <button
                key={opt}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                title={opt}
                onClick={() => {
                  onSelect(opt);
                  close();
                }}
                className={cn(
                  "flex w-full items-center justify-between gap-3 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-white/5",
                  selected ? "text-foreground" : "text-muted-foreground"
                )}
              >
                <span className={cn("truncate", textClassName)}>{opt}</span>
                {selected && <Check className="size-3.5 shrink-0 text-primary" />}
              </button>
            );
          })}
        </>
      )}
    </Popover>
  );
}

export function Composer({
  disabled,
  onSend,
  sending,
  onStop,
}: {
  disabled: boolean;
  onSend: (text: string) => void;
  /** While true the action button becomes a Stop/interrupt control. */
  sending: boolean;
  onStop: () => void;
}) {
  const { config, model, setModel, effort, setEffort, conversations, activeId, setConversationMode } =
    useApp();
  const activeMode = conversations.find((c) => c.conversation_id === activeId)?.mode ?? "regular";
  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow the textarea to fit its content, capped by `max-h-40` (CSS clamps the
  // rendered height and `overflow-y-auto` scrolls past the cap). Reset to "auto" first
  // so deleting text shrinks it back; runs after submit clears `text` too.
  useLayoutEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [text]);

  // Active selections, falling back to the server defaults. Guard against a stored value that's no
  // longer offered (e.g. AGENT_MODELS changed) so it still shows rather than snapping to option 0.
  const models = config?.models?.length ? config.models : config ? [config.model] : [];
  const currentModel = model || config?.model || "";
  const modelOptions = models.includes(currentModel) ? models : [currentModel, ...models];

  const efforts = config?.efforts?.length ? config.efforts : [];
  const currentEffort = effort || config?.effort || "";
  const effortOptions =
    currentEffort && !efforts.includes(currentEffort) ? [currentEffort, ...efforts] : efforts;

  const submit = () => {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="p-3">
      {/* The whole input is one rounded slab; the textarea sits flush and the controls live in a
          footer inside the same border, so the model/effort chips read as part of the field. */}
      <div className="flex flex-col gap-1 rounded-2xl border border-white/10 bg-slate-900/40 p-2 shadow-sm transition-colors focus-within:border-white/20">
        <Textarea
          ref={taRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Send a message…"
          className="max-h-40 min-h-[44px] resize-none overflow-y-auto border-0 bg-transparent px-2 py-1.5 text-sm shadow-none focus-visible:ring-0"
          rows={1}
        />
        <div className="flex items-center gap-0.5">
          {/* Only when a conversation is active — the new-chat picker handles mode pre-creation. */}
          {activeId && (
            <>
              <ModeChip
                value={activeMode}
                onSelect={(mode) => setConversationMode(activeId, mode)}
              />
              <span aria-hidden className="select-none text-muted-foreground/30">
                •
              </span>
            </>
          )}
          {modelOptions.length > 0 && (
            <ControlChip
              icon={Brain}
              label={shortModel(currentModel)}
              title="Model"
              options={modelOptions}
              current={currentModel}
              onSelect={setModel}
              textClassName="font-mono"
            />
          )}
          {effortOptions.length > 0 && (
            <>
              <span aria-hidden className="select-none text-muted-foreground/30">
                •
              </span>
              <ControlChip
                icon={Zap}
                label={currentEffort}
                title="Thinking effort"
                options={effortOptions}
                current={currentEffort}
                onSelect={setEffort}
                textClassName="capitalize"
              />
            </>
          )}
          <div className="flex-1" />
          {sending ? (
            <Button
              type="button"
              size="icon"
              variant="secondary"
              onClick={onStop}
              aria-label="Stop"
              className="size-8 rounded-xl"
            >
              <Square className="size-3.5 fill-current" />
            </Button>
          ) : (
            <Button
              type="button"
              size="icon"
              onClick={submit}
              disabled={!text.trim()}
              aria-label="Send"
              className="size-8 rounded-xl"
            >
              <SendHorizonal className="size-4" />
            </Button>
          )}
        </div>
      </div>
      <p className="mt-1.5 px-2 text-[10px] text-muted-foreground/60">
        Enter to send · Shift+Enter for newline
      </p>
    </div>
  );
}
