import type { StreamEvent } from "@/types";

/**
 * Stream one agent turn from the SSE endpoint.
 *
 * `EventSource` only supports GET, but our chat endpoint is a POST (it carries a JSON body),
 * so we read the `text/event-stream` response manually: a `fetch` whose body is a
 * `ReadableStream`, split into `data:` frames and parsed into {@link StreamEvent}s. `onEvent`
 * fires for each frame as it arrives, giving the live token-by-token UI.
 */
export async function streamChat(
  body: {
    user_id: string;
    conversation_id: string;
    prompt: string;
    model?: string;
    effort?: string;
    attachments?: { filename: string; mime_type: string; data: string }[];
  },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flush = (frame: string) => {
    // An SSE frame is one-or-more lines; we only emit `data:` payloads.
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) return;
    try {
      onEvent(JSON.parse(data) as StreamEvent);
    } catch {
      // Ignore unparseable frames (e.g. keep-alive comments).
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    // Frames are separated by a blank line ("\n\n").
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      flush(buffer.slice(0, idx));
      buffer = buffer.slice(idx + 2);
    }
  }
  if (buffer.trim()) flush(buffer);
}
