import type { DownloadEvent, StreamEvent } from "@/types";

/**
 * Read a `text/event-stream` response body manually and fire `onEvent` for each `data:` frame.
 *
 * `EventSource` only supports GET, but our streaming endpoints are POSTs (they carry a JSON body),
 * so we read the response as a `ReadableStream`: split it into `data:` frames (separated by a blank
 * line) and parse each into the caller's event type. Shared by {@link streamChat} (chat turns) and
 * {@link streamDownload} (model download progress) so the frame-reading logic lives in one place.
 */
async function readSseStream<T>(res: Response, onEvent: (event: T) => void): Promise<void> {
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flush = (frame: string) => {
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) return;
    try {
      onEvent(JSON.parse(data) as T);
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

/** Stream one agent turn from the chat SSE endpoint, firing `onEvent` token-by-token. */
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
  await readSseStream<StreamEvent>(res, onEvent);
}

/** Stream a GGUF download from HuggingFace, firing `onEvent` for each progress/done/error frame.
 *  Aborting via `signal` cancels the request; the backend leaves a `.part` file so it can resume. */
export async function streamDownload(
  body: { repo_id: string; file_path: string; revision?: string; quant?: string },
  onEvent: (event: DownloadEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch("/api/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  await readSseStream<DownloadEvent>(res, onEvent);
}
