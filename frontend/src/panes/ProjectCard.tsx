import { useEffect, useRef, useState } from "react";
import { BookOpen, FileText, FolderOpen, Globe, RefreshCw, Trash2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import { DocumentView } from "@/panes/DocumentsPane";
import type { DocumentMeta, ProjectKb } from "@/types";

/** Document vertex types that are generated knowledge-base pages (vs. uploaded sources). */
const KB_PAGE_KINDS = new Set([
  "KbPage",
  "KbSummary",
  "KbConcept",
  "KbEntity",
  "KbExploration",
  "KbIndex",
]);

const isKbPage = (d: DocumentMeta) => !!d.kind && KB_PAGE_KINDS.has(d.kind);

/** Human label for a KB page kind, e.g. "KbConcept" -> "Concept". */
const kbKindLabel = (kind?: string) =>
  (kind || "Page").replace(/^Kb/, "") || "Page";

/** Read a File into the {filename, mime_type, data(base64)} shape the upload endpoint expects. */
function readAsUpload(
  file: File
): Promise<{ filename: string; mime_type: string; data: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const result = String(reader.result);
      // strip the "data:<mime>;base64," prefix → raw base64 payload
      const data = result.slice(result.indexOf(",") + 1);
      resolve({
        filename: file.name,
        mime_type: file.type || "application/octet-stream",
        data,
      });
    };
    reader.readAsDataURL(file);
  });
}

function ProjectSystemPromptRow({ projectId }: { projectId: string }) {
  const { projects, setProjectSystemPrompt } = useApp();
  const stored = projects.find((p) => p.project_id === projectId)?.system_prompt ?? "";
  const [draft, setDraft] = useState(stored);

  useEffect(() => setDraft(stored), [projectId, stored]);

  const save = () => {
    if (draft !== stored) setProjectSystemPrompt(projectId, draft);
  };

  return (
    <div className="space-y-1.5">
      <span className="text-xs text-muted-foreground">Project system prompt</span>
      <Textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        placeholder="Instructions shared by every conversation in this project…"
        className="max-h-48 min-h-[64px] resize-y text-xs"
        title="Layered between the base prompt and each conversation's own prompt"
      />
    </div>
  );
}

/** Shown in the Context tab when the active conversation belongs to a project: the project's
 *  system prompt and its reference documents (upload, open, mark global, delete). */
export function ProjectCard() {
  const { activeId, conversations, projects, userId } = useApp();
  const conv = conversations.find((c) => c.conversation_id === activeId);
  const projectId = conv?.project_id ?? null;
  const project = projects.find((p) => p.project_id === projectId) ?? null;

  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [kb, setKb] = useState<ProjectKb | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = () => {
    if (!projectId) {
      setDocs([]);
      return;
    }
    api
      .listProjectDocuments(projectId, userId)
      .then(setDocs)
      .catch(() => setDocs([]));
  };

  const loadKb = () => {
    if (!projectId) {
      setKb(null);
      return;
    }
    api
      .getProjectKb(projectId, userId)
      .then(setKb)
      .catch(() => setKb(null));
  };

  useEffect(() => {
    setOpenId(null);
    load();
    loadKb();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, userId]);

  // While a compile is running, poll the KB so the status + new pages appear without a manual refresh.
  useEffect(() => {
    if (!projectId || kb?.status !== "compiling") return;
    const t = setInterval(loadKb, 4000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, userId, kb?.status]);

  const rebuildKb = async () => {
    if (!projectId) return;
    try {
      await api.rebuildProjectKb(projectId, userId);
      setKb((prev) => ({ status: "compiling", compiled_at: prev?.compiled_at ?? null, pages: prev?.pages ?? [] }));
    } catch (err) {
      console.error("rebuild knowledge base failed", err);
    }
  };

  if (!projectId) return null;

  // The KB pages now live in the same project document set; show only the uploaded sources here.
  const sources = docs.filter((d) => !isKbPage(d));
  const kbPages = kb?.pages ?? [];

  const onPick = async (files: FileList | null) => {
    if (!files || files.length === 0 || !projectId) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const upload = await readAsUpload(file);
        await api.uploadProjectDocument(projectId, userId, upload);
      }
      load();
      loadKb();
    } catch (err) {
      console.error("project upload failed", err);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const toggleGlobal = async (doc: DocumentMeta) => {
    try {
      await api.setDocumentGlobal(doc.document_id, userId, !doc.is_global);
      setDocs((prev) =>
        prev.map((d) =>
          d.document_id === doc.document_id ? { ...d, is_global: !doc.is_global } : d
        )
      );
    } catch (err) {
      console.error("toggle global failed", err);
    }
  };

  const remove = async (documentId: string) => {
    try {
      await api.deleteDocument(documentId, userId);
      if (openId === documentId) setOpenId(null);
      load();
    } catch (err) {
      console.error("delete document failed", err);
    }
  };

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <FolderOpen className="size-3.5 text-muted-foreground" />
          {project?.title || "Project"}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <ProjectSystemPromptRow projectId={projectId} />

        {openId && (
          <div className="border-t border-border/50 pt-2">
            <DocumentView documentId={openId} onBack={() => setOpenId(null)} />
          </div>
        )}

        <div className={`space-y-2 border-t border-border/50 pt-2${openId ? " hidden" : ""}`}>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Sources</span>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 text-muted-foreground"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
              title="Upload a source document"
            >
              <Upload className={`size-3.5 ${uploading ? "animate-pulse" : ""}`} />
            </Button>
            <input
              ref={fileInput}
              type="file"
              multiple
              hidden
              onChange={(e) => onPick(e.target.files)}
            />
          </div>

          {sources.length > 0 ? (
            <div className="space-y-2">
              {sources.map((d) => (
                <div
                  key={d.document_id}
                  className="group flex items-center gap-2 rounded-md border border-border p-2"
                >
                  <button
                    type="button"
                    onClick={() => setOpenId(d.document_id)}
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    title="Open document"
                  >
                    <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs font-medium">
                        {d.title || "Untitled"}
                        {d.is_global && (
                          <Globe className="ml-1 inline size-3 text-sky-500" />
                        )}
                      </span>
                      <span className="block truncate text-[10px] text-muted-foreground">
                        {d.mime_type}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => toggleGlobal(d)}
                    title={
                      d.is_global
                        ? "Global — kept when the project is deleted. Click to unmark."
                        : "Mark global (available everywhere, kept on project delete)"
                    }
                    className={`shrink-0 rounded p-1 transition-colors hover:bg-accent ${
                      d.is_global ? "text-sky-500" : "text-muted-foreground opacity-0 group-hover:opacity-100"
                    }`}
                  >
                    <Globe className="size-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => remove(d.document_id)}
                    title="Delete document"
                    className="shrink-0 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-accent group-hover:opacity-100"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              No source documents yet. Upload files and they're auto-compiled into the knowledge
              base below.
            </p>
          )}
        </div>

        <div className={`space-y-2 border-t border-border/50 pt-2${openId ? " hidden" : ""}`}>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <BookOpen className="size-3.5" />
              Knowledge base
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 text-muted-foreground"
              onClick={rebuildKb}
              disabled={kb?.status === "compiling"}
              title="Rebuild the knowledge base from the current sources"
            >
              <RefreshCw
                className={`size-3.5 ${kb?.status === "compiling" ? "animate-spin" : ""}`}
              />
            </Button>
          </div>

          {kbPages.length > 0 ? (
            <div className="space-y-2">
              {kbPages.map((d) => (
                <button
                  key={d.document_id}
                  type="button"
                  onClick={() => setOpenId(d.document_id)}
                  className="group flex w-full items-center gap-2 rounded-md border border-border p-2 text-left"
                  title="Open knowledge-base page"
                >
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate text-xs font-medium">
                    {d.title || "Untitled"}
                  </span>
                  <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                    {kbKindLabel(d.kind)}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {kb?.status === "compiling"
                ? "Compiling the knowledge base from your sources…"
                : kb?.status === "error"
                  ? "The last compile failed — try Rebuild, or check the server logs."
                  : "No knowledge-base pages yet. Upload sources (or click Rebuild) to compile summaries, concepts and entities."}
            </p>
          )}

          {kb?.compiled_at && kb.status !== "compiling" && (
            <p className="text-[10px] text-muted-foreground">
              Last built {new Date(kb.compiled_at).toLocaleString()}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
