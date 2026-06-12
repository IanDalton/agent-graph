import { useCallback, useEffect, useState } from "react";
import {
  ArrowLeft,
  Code,
  Download,
  FileText,
  Pencil,
  Play,
  RefreshCw,
  Save,
  Trash2,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { Markdown } from "@/components/Markdown";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import type { DocumentFull, DocumentMeta } from "@/types";

/** Text-encoded, text-shaped documents are editable in place; binary artifacts are not. */
function isEditable(doc: DocumentFull): boolean {
  if (doc.encoding === "base64") return false;
  return (
    doc.mime_type.startsWith("text/") ||
    doc.mime_type === "application/json" ||
    doc.mime_type === "image/svg+xml"
  );
}

/** Trigger a browser download of the document (decoding base64 artifacts to real bytes). */
function downloadDocument(doc: DocumentFull) {
  const bytes =
    doc.encoding === "base64"
      ? Uint8Array.from(atob(doc.content), (c) => c.charCodeAt(0))
      : new TextEncoder().encode(doc.content);
  const blob = new Blob([bytes], { type: doc.mime_type || "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = doc.title?.includes(".") ? doc.title : doc.title || "document";
  a.click();
  URL.revokeObjectURL(url);
}

/** The read-mode body for one document, by media type: live HTML app, PDF, image,
 *  rendered markdown, or a monospace block for everything else. */
function DocumentBody({ doc, showSource }: { doc: DocumentFull; showSource: boolean }) {
  if (doc.encoding === "base64") {
    if (doc.mime_type === "application/pdf") {
      return (
        <iframe
          title={doc.title}
          src={`data:application/pdf;base64,${doc.content}`}
          className="h-[65vh] w-full rounded-md border border-border"
        />
      );
    }
    if (doc.mime_type.startsWith("image/")) {
      return (
        <img
          src={`data:${doc.mime_type};base64,${doc.content}`}
          alt={doc.title}
          className="max-w-full rounded-md border border-border"
        />
      );
    }
    return (
      <div className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
        Binary file ({doc.mime_type}) — use the download button to save it.
      </div>
    );
  }
  if (doc.mime_type === "text/html" && !showSource) {
    // The interactive-artifact path: a sandboxed iframe (scripts allowed, same-origin NOT,
    // so the embedded app can't touch our cookies/API) running the self-contained page.
    return (
      <iframe
        title={doc.title}
        srcDoc={doc.content}
        sandbox="allow-scripts allow-forms allow-popups"
        className="h-[65vh] w-full rounded-md border border-border bg-white"
      />
    );
  }
  if (doc.mime_type === "text/markdown") {
    return <Markdown className="text-xs">{doc.content}</Markdown>;
  }
  return (
    <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded-md bg-muted p-2 font-mono text-[11px]">
      {doc.content}
    </pre>
  );
}

function DocumentRow({
  doc,
  onOpen,
  onDelete,
}: {
  doc: DocumentMeta;
  onOpen: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="group flex items-center gap-2 rounded-md border border-border p-2">
      <button
        type="button"
        onClick={onOpen}
        className="flex min-w-0 flex-1 items-center gap-2 text-left"
        title="Open document"
      >
        <FileText className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium">
            {doc.title || "Untitled"}
          </span>
          <span className="block truncate text-[10px] text-muted-foreground">
            {doc.mime_type}
            {doc.updated_at ? ` · ${new Date(doc.updated_at).toLocaleString()}` : ""}
          </span>
        </span>
      </button>
      <Button
        variant="ghost"
        size="icon"
        className="size-6 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
        onClick={onDelete}
        title="Delete document"
      >
        <Trash2 className="size-3.5" />
      </Button>
    </div>
  );
}

/** Viewer/editor for one opened document. Markdown renders rich; other text shows in a
 *  monospace block. Text documents flip into a textarea on Edit and save via PUT. */
function DocumentView({
  documentId,
  onBack,
}: {
  documentId: string;
  onBack: () => void;
}) {
  const { userId } = useApp();
  const [doc, setDoc] = useState<DocumentFull | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [showSource, setShowSource] = useState(false);
  const [draftTitle, setDraftTitle] = useState("");
  const [draftContent, setDraftContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getDocument(documentId, userId)
      .then((d) => {
        if (cancelled) return;
        setDoc(d);
        setDraftTitle(d.title);
        setDraftContent(d.content);
      })
      .catch(() => !cancelled && setError("Failed to load document."))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [documentId, userId]);

  function save() {
    if (!doc || saving) return;
    setSaving(true);
    setError(null);
    api
      .updateDocument(doc.document_id, userId, {
        title: draftTitle,
        content: draftContent,
      })
      .then((updated) => {
        setDoc(updated);
        setEditing(false);
      })
      .catch(() => setError("Failed to save changes."))
      .finally(() => setSaving(false));
  }

  function cancelEdit() {
    if (doc) {
      setDraftTitle(doc.title);
      setDraftContent(doc.content);
    }
    setEditing(false);
  }

  if (loading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-5/6" />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="size-6 shrink-0 text-muted-foreground"
          onClick={onBack}
          title="Back to document list"
        >
          <ArrowLeft className="size-3.5" />
        </Button>
        {editing ? (
          <input
            value={draftTitle}
            onChange={(e) => setDraftTitle(e.target.value)}
            className="min-w-0 flex-1 rounded-md border border-input bg-transparent px-2 py-1 text-xs font-medium focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            placeholder="Document title"
          />
        ) : (
          <span className="min-w-0 flex-1 truncate text-xs font-medium">
            {doc?.title || "Untitled"}
          </span>
        )}
        {doc && !editing && doc.mime_type === "text/html" && doc.encoding !== "base64" && (
          <Button
            variant="ghost"
            size="icon"
            className="size-6 shrink-0 text-muted-foreground"
            onClick={() => setShowSource((s) => !s)}
            title={showSource ? "Show live preview" : "Show HTML source"}
          >
            {showSource ? <Play className="size-3.5" /> : <Code className="size-3.5" />}
          </Button>
        )}
        {doc && !editing && (
          <Button
            variant="ghost"
            size="icon"
            className="size-6 shrink-0 text-muted-foreground"
            onClick={() => downloadDocument(doc)}
            title="Download"
          >
            <Download className="size-3.5" />
          </Button>
        )}
        {doc && !editing && isEditable(doc) && (
          <Button
            variant="ghost"
            size="icon"
            className="size-6 shrink-0 text-muted-foreground"
            onClick={() => setEditing(true)}
            title="Edit document"
          >
            <Pencil className="size-3.5" />
          </Button>
        )}
        {editing && (
          <>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 shrink-0 text-muted-foreground"
              onClick={cancelEdit}
              disabled={saving}
              title="Discard changes"
            >
              <X className="size-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 shrink-0"
              onClick={save}
              disabled={saving}
              title="Save changes"
            >
              <Save className={`size-3.5 ${saving ? "animate-pulse" : ""}`} />
            </Button>
          </>
        )}
      </div>
      {doc && (
        <Badge variant="secondary" className="text-[10px]">
          {doc.mime_type}
        </Badge>
      )}
      {error && <div className="text-xs text-destructive">{error}</div>}
      {editing ? (
        <Textarea
          value={draftContent}
          onChange={(e) => setDraftContent(e.target.value)}
          className="min-h-[40vh] font-mono text-xs"
          placeholder="Document content"
        />
      ) : doc ? (
        <DocumentBody doc={doc} showSource={showSource} />
      ) : null}
    </div>
  );
}

/** The Documents tab of the right pane: the active conversation's agent-authored documents.
 *  Re-fetches whenever `refreshKey` bumps (after each completed turn), so documents the agent
 *  just created show up without a manual reload. */
export function DocumentsCard({ refreshKey }: { refreshKey: number }) {
  const { activeId, userId, featuredDoc } = useApp();
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);

  const load = useCallback(() => {
    if (!activeId) {
      setDocs([]);
      return;
    }
    setLoading(true);
    api
      .listDocuments(activeId, userId)
      .then(setDocs)
      .catch(() => setDocs([]))
      .finally(() => setLoading(false));
  }, [activeId, userId]);

  // Close any open document only when the conversation changes — NOT on every refresh,
  // or the post-turn refresh would kick the user out of the document they're reading.
  useEffect(() => setOpenId(null), [activeId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  // Spotlight the featured document (the agent just created one, or the user clicked a
  // document card in the chat): open it and refresh the list so it's there on "back".
  useEffect(() => {
    if (featuredDoc) {
      setOpenId(featuredDoc.id);
      load();
    }
  }, [featuredDoc, load]);

  function remove(documentId: string) {
    api
      .deleteDocument(documentId, userId)
      .then(() => {
        if (openId === documentId) setOpenId(null);
        load();
      })
      .catch(() => undefined);
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm">Documents</CardTitle>
        <Button
          variant="ghost"
          size="icon"
          className="size-6 text-muted-foreground"
          onClick={load}
          disabled={!activeId || loading}
          title="Refresh documents"
        >
          <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent>
        {openId ? (
          <DocumentView documentId={openId} onBack={() => setOpenId(null)} />
        ) : loading && docs.length === 0 ? (
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ) : docs.length > 0 ? (
          <div className="space-y-2">
            {docs.map((d) => (
              <DocumentRow
                key={d.document_id}
                doc={d}
                onOpen={() => setOpenId(d.document_id)}
                onDelete={() => remove(d.document_id)}
              />
            ))}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No documents yet — ask the agent to write a report, plan, or note and it will
            appear here. Text documents can be edited in place.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
