import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Boxes,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Cpu,
  Download,
  HardDrive,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Server,
  Settings2,
  Trash2,
  X,
} from "lucide-react";

import { api } from "@/api/client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useApp } from "@/state/appcontext";
import type { Fit, GpuInfo, HardwareProfile, HfFile, HfModel, LoadModelResult, LocalModel } from "@/types";

const KV_TYPES = ["f16", "q8_0", "q4_0"];

const FIT_META: Record<Fit, { label: string; cls: string }> = {
  gpu: { label: "Fits on GPU", cls: "border-emerald-500/30 bg-emerald-500/15 text-emerald-400" },
  partial: { label: "Partial offload", cls: "border-amber-500/30 bg-amber-500/15 text-amber-400" },
  cpu: { label: "CPU + RAM", cls: "border-sky-500/30 bg-sky-500/15 text-sky-400" },
  too_big: { label: "Too big", cls: "border-rose-500/30 bg-rose-500/15 text-rose-400" },
};

function formatBytes(n: number): string {
  if (!n) return "—";
  const gb = n / 1e9;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  return `${(n / 1e6).toFixed(0)} MB`;
}

/** A MiB count (as produced by the recommender's est_*_mb fields) as a human string. */
function formatMb(mb: number): string {
  if (!mb) return "—";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

/** A transfer rate (bytes/sec) as a human string, e.g. "12.3 MB/s". */
function formatSpeed(bps?: number): string {
  if (!bps || bps <= 0) return "— MB/s";
  const mb = bps / 1e6;
  if (mb >= 1) return `${mb.toFixed(1)} MB/s`;
  return `${(bps / 1e3).toFixed(0)} KB/s`;
}

/** Estimated time left (seconds) as "2m 14s" / "45s" / "1h 3m"; "—" when unknown. */
function formatEta(seconds?: number | null): string {
  if (seconds == null || !isFinite(seconds) || seconds < 0) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function FitBadge({ fit }: { fit?: Fit }) {
  if (!fit) return null;
  const meta = FIT_META[fit];
  return <span className={cn("rounded-md border px-1.5 py-0.5 text-[10px] font-medium", meta.cls)}>{meta.label}</span>;
}

/** A small "copy to clipboard" button with a transient check. */
function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable (insecure context) — no-op
    }
  };
  return (
    <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" onClick={copy} title="Copy to clipboard">
      {copied ? <Check className="size-3.5 text-primary" /> : <Copy className="size-3.5" />}
      {copied ? "Copied" : label}
    </Button>
  );
}

/** A small text input matching the inline style used elsewhere (e.g. SkillMarketplace). */
function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "h-8 w-full rounded-md border border-input bg-background px-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring",
        props.className
      )}
    />
  );
}

type ConfigTarget = { repo_id: string; filename: string; size_bytes: number; quant: string };

/** A "Load" button that confirms before recreating the llama-server onto ``filename`` (it interrupts
 *  any in-flight reply), then reports the result to ``onResult``. Shared by the Library cards and the
 *  advanced-config panel (so the slider-tuned context can be the one that gets served). */
function ConfirmLoadButton({
  filename,
  requestedContext,
  kvType,
  disabled,
  label = "Load",
  onResult,
}: {
  filename: string;
  requestedContext?: number;
  kvType?: string;
  disabled?: boolean;
  label?: string;
  onResult?: (res: LoadModelResult) => void;
}) {
  const { loadModel, loadingModel } = useApp();
  const [confirm, setConfirm] = useState(false);
  const busy = loadingModel === filename;
  const otherLoading = loadingModel !== null && !busy;

  const doLoad = async () => {
    setConfirm(false);
    const res = await loadModel(filename, {
      requested_context: requestedContext,
      kv_cache_type: kvType,
    });
    onResult?.(res);
  };

  return (
    <>
      <Button
        size="sm"
        className="h-7 gap-1 text-xs"
        onClick={() => setConfirm(true)}
        disabled={disabled || busy || otherLoading}
        title="Serve this model on the local llama-server"
      >
        {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
        {busy ? "Loading…" : label}
      </Button>
      {confirm && (
        <Dialog open onClose={() => setConfirm(false)} title="Load model" className="w-[min(440px,92vw)]">
          <div className="space-y-3 p-4">
            <h3 className="flex items-center gap-2 text-sm font-semibold">
              <Play className="size-4 text-primary" />
              Load this model?
            </h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              This restarts the local llama-server to serve{" "}
              <code className="rounded bg-background px-1 font-mono">{filename}</code> — it interrupts
              any in-flight reply and takes a few seconds while the model loads into VRAM.
            </p>
            <div className="flex justify-end gap-2">
              <Button variant="outline" size="sm" onClick={() => setConfirm(false)}>
                Cancel
              </Button>
              <Button size="sm" onClick={doLoad}>
                Load model
              </Button>
            </div>
          </div>
        </Dialog>
      )}
    </>
  );
}

/** A code block with a copy button. */
function CommandBlock({ command, label }: { command: string; label: string }) {
  if (!command) return null;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
        <CopyButton text={command} />
      </div>
      <pre className="overflow-x-auto rounded-md border border-border bg-background/60 p-2 text-[11px] leading-relaxed text-foreground">
        {command}
      </pre>
    </div>
  );
}

/** Advanced settings: recompute the recommendation as the user pins context / KV-cache type, and
 *  show the exact llama-server launch command to copy. */
function AdvancedConfig({ target, onBack }: { target: ConfigTarget; onBack: () => void }) {
  const { localModels } = useApp();
  const [contextLen, setContextLen] = useState<number | "">("");
  const [kvType, setKvType] = useState<string>("");
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.recommendModel>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadResult, setLoadResult] = useState<LoadModelResult | null>(null);
  const initialized = useRef(false);
  const downloaded = localModels.some((m) => m.filename === target.filename);

  const recompute = useCallback(
    async (ctx?: number, kv?: string) => {
      setLoading(true);
      try {
        const res = await api.recommendModel({
          size_bytes: target.size_bytes,
          quant: target.quant,
          repo_id: target.repo_id,
          filename: target.filename,
          requested_context: ctx,
          kv_cache_type: kv,
        });
        setResult(res);
        if (!initialized.current) {
          initialized.current = true;
          setContextLen(res.recommendation.context_length);
          setKvType(res.recommendation.kv_cache_type);
        }
        return res;
      } finally {
        setLoading(false);
      }
    },
    [target]
  );

  // Initial recommendation (no overrides) — seeds the controls.
  useEffect(() => {
    recompute().catch(() => setLoading(false));
  }, [recompute]);

  // Debounced recompute when the user changes the pinned context / KV type.
  useEffect(() => {
    if (!initialized.current) return;
    const id = setTimeout(() => {
      recompute(typeof contextLen === "number" ? contextLen : undefined, kvType || undefined).catch(
        () => {}
      );
    }, 350);
    return () => clearTimeout(id);
  }, [contextLen, kvType, recompute]);

  const rec = result?.recommendation;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4">
      <div className="mb-3 flex items-center gap-2">
        <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs text-muted-foreground" onClick={onBack}>
          <X className="size-3.5" />
          Back
        </Button>
        <span className="truncate font-mono text-sm">{target.filename}</span>
        {rec && <FitBadge fit={rec.fit} />}
      </div>

      {loading && !rec ? (
        <Skeleton className="h-40 w-full" />
      ) : rec ? (
        <div className="space-y-4">
          <div className="space-y-4 rounded-lg border border-border bg-background/40 p-3">
            {(() => {
              const ctxMax = Math.max(rec.model_max_ctx || 0, rec.context_length, 4096);
              const ctxValue = typeof contextLen === "number" ? contextLen : rec.context_length;
              return (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-[11px] font-medium text-muted-foreground">Context length</label>
                    <span className="font-mono text-xs text-foreground">{ctxValue.toLocaleString()} tokens</span>
                  </div>
                  <Slider
                    min={512}
                    max={ctxMax}
                    step={1024}
                    value={Math.min(ctxValue, ctxMax)}
                    onValueChange={setContextLen}
                    aria-label="Context length"
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground/60">
                    <span>512</span>
                    <span>{ctxMax.toLocaleString()} max</span>
                  </div>
                </div>
              );
            })()}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-muted-foreground">KV cache type</label>
                <select
                  value={kvType}
                  onChange={(e) => setKvType(e.target.value)}
                  className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  {KV_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                {loading && <Loader2 className="mb-2 size-4 animate-spin text-muted-foreground" />}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-lg border border-border bg-background/40 p-3 text-xs sm:grid-cols-3">
            <Stat label="GPU layers (-ngl)" value={rec.n_gpu_layers === 999 ? "all" : rec.n_gpu_layers} />
            <Stat label="Context (-c)" value={rec.context_length.toLocaleString()} />
            <Stat label="KV cache" value={rec.kv_cache_type} />
            <Stat label="Flash attention" value={rec.flash_attn ? "on" : "off"} />
            <Stat label="Batch / ubatch" value={`${rec.batch_size} / ${rec.ubatch_size}`} />
            <Stat label="Threads" value={rec.threads} />
            <Stat label="Est. VRAM" value={formatMb(rec.est_vram_mb)} />
            <Stat label="Est. RAM" value={formatMb(rec.est_ram_mb)} />
            <Stat label="Confidence" value={rec.confidence} />
          </div>

          {(rec.fit === "gpu" || rec.fit === "partial") && rec.est_vram_mb > 0 && (
            <div className="grid grid-cols-3 gap-x-4 gap-y-1.5 rounded-lg border border-border bg-background/40 p-3 text-xs">
              <Stat label="VRAM · weights" value={formatMb(rec.weights_mb ?? 0)} />
              <Stat label="VRAM · KV cache" value={formatMb(rec.kv_cache_mb ?? 0)} />
              <Stat label="VRAM · overhead" value={formatMb(rec.overhead_mb ?? 0)} />
            </div>
          )}

          {rec.notes.length > 0 && (
            <ul className="space-y-1 text-[11px] text-muted-foreground">
              {rec.notes.map((n, i) => (
                <li key={i} className="flex gap-1.5">
                  <span className="text-muted-foreground/50">•</span>
                  {n}
                </li>
              ))}
            </ul>
          )}

          <p className="text-[11px] text-muted-foreground">
            <strong className="text-foreground">To apply</strong> the context (<code>-c</code>) and GPU
            layers (<code>-ngl</code>): with the bundled <code>llamacpp</code> service, set{" "}
            <code>LLAMACPP_CTX</code> (and <code>LLAMACPP_NGL</code>) in <code>.env</code> and run{" "}
            <code>docker compose --profile llamacpp up -d llamacpp</code>. Otherwise run the command
            below on your GPU machine. The app connects to whatever the server serves — it doesn't
            launch it.
          </p>
          {downloaded ? (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-background/40 p-3">
              <ConfirmLoadButton
                filename={target.filename}
                requestedContext={typeof contextLen === "number" ? contextLen : undefined}
                kvType={kvType || undefined}
                label="Load with these settings"
                onResult={setLoadResult}
              />
              {loadResult?.ok && (
                <span className="text-[11px] text-emerald-400">Now serving {loadResult.served_model}.</span>
              )}
              {loadResult && !loadResult.ok && (
                <span className={cn("text-[11px]", loadResult.unmanaged ? "text-amber-300/90" : "text-destructive")}>
                  {loadResult.error}
                </span>
              )}
              <span className="text-[11px] text-muted-foreground">
                Serves it on the local llama-server with the context above.
              </span>
            </div>
          ) : null}
          <CommandBlock command={result?.command ?? ""} label="llama-server (local file)" />
          {result?.command_hf && (
            <CommandBlock command={result.command_hf} label="llama-server (-hf: server downloads it)" />
          )}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Couldn't compute a recommendation.</p>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground/70">{label}</span>
      <span className="font-mono text-foreground">{value}</span>
    </div>
  );
}

// Browse modes for the Discover tab (HuggingFace sort keys → labels).
const SORTS: { key: string; label: string }[] = [
  { key: "trendingScore", label: "Trending" },
  { key: "downloads", label: "Popular" },
  { key: "createdAt", label: "New" },
];

// Module-level cache of the Discover browse view (results + which repo is expanded), so switching
// tabs / reopening Settings restores the exact view — including the expanded repo whose in-flight
// download bar (progress from AppContext) should stay visible. Reset on a full page reload, which is
// fine: the list re-fetches and active downloads repopulate from the backend.
const discoverCache: {
  query: string;
  sort: string;
  results: HfModel[];
  openRepo: string | null;
  files: Record<string, HfFile[]>;
} = { query: "", sort: "trendingScore", results: [], openRepo: null, files: {} };

/** Discover: browse new/popular GGUF models (or search), expand a repo to its quantizations
 *  (with fit badges), download. */
function DiscoverTab({ onConfigure }: { onConfigure: (t: ConfigTarget) => void }) {
  // Download progress lives in AppContext (keyed `${repo}/${path}`) so it survives switching tabs,
  // closing the Settings dialog, and a page refresh — the bug being fixed here.
  const { startDownload, downloads, localModels } = useApp();
  const [query, setQuery] = useState(discoverCache.query);
  const [sort, setSort] = useState(discoverCache.sort);
  const [results, setResults] = useState<HfModel[]>(discoverCache.results);
  const [loading, setLoading] = useState(discoverCache.results.length === 0);
  const [openRepo, setOpenRepo] = useState<string | null>(discoverCache.openRepo);
  const [files, setFiles] = useState<Record<string, HfFile[]>>(discoverCache.files);
  const [filesLoading, setFilesLoading] = useState<string | null>(null);
  const didInit = useRef(false);

  const installed = useMemo(() => new Set(localModels.map((m) => m.filename)), [localModels]);

  // Write the browse view back to the cache each render so a remount restores it verbatim.
  useEffect(() => {
    discoverCache.query = query;
    discoverCache.sort = sort;
    discoverCache.results = results;
    discoverCache.openRepo = openRepo;
    discoverCache.files = files;
  });

  const load = useCallback(async (q: string, s: string) => {
    setLoading(true);
    try {
      setResults(await api.searchModels(q, s));
    } finally {
      setLoading(false);
    }
  }, []);

  // Browse on open and whenever the sort changes (an empty query → HF's ranked new/popular list);
  // the search box narrows it on Enter. `query` is intentionally not a dep so typing doesn't refetch.
  // On remount we skip the fetch when the cache already has results (keeps the restored view + bar).
  useEffect(() => {
    if (!didInit.current) {
      didInit.current = true;
      if (results.length > 0) return; // restored from cache — don't clobber the view
    }
    load(query.trim(), sort);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sort, load]);

  const runSearch = () => load(query.trim(), sort);

  const toggleRepo = async (repoId: string) => {
    if (openRepo === repoId) {
      setOpenRepo(null);
      return;
    }
    setOpenRepo(repoId);
    if (!files[repoId]) {
      setFilesLoading(repoId);
      try {
        const list = await api.listRepoFiles(repoId);
        setFiles((prev) => ({ ...prev, [repoId]: list }));
      } finally {
        setFilesLoading(null);
      }
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="space-y-2 p-4 pb-2">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSearch()}
              placeholder="Search HuggingFace GGUF models — or browse new & popular below…"
              className="h-8 w-full rounded-md border border-input bg-background pl-7 pr-2 text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
          </div>
          <Button size="sm" className="h-8 gap-1 px-3 text-xs" onClick={runSearch} disabled={loading}>
            {loading ? <Loader2 className="size-3.5 animate-spin" /> : <Search className="size-3.5" />}
            Search
          </Button>
        </div>
        <div className="flex items-center gap-1">
          <span className="mr-1 text-[10px] uppercase tracking-wide text-muted-foreground/60">Browse</span>
          {SORTS.map((s) => (
            <button
              key={s.key}
              type="button"
              onClick={() => setSort(s.key)}
              className={cn(
                "rounded-md px-2 py-0.5 text-xs transition-colors",
                sort === s.key ? "bg-white/10 text-foreground" : "text-muted-foreground hover:bg-white/5"
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-4 pt-2">
        {loading && results.length === 0 ? (
          Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12 w-full" />)
        ) : results.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
            <Boxes className="size-6 opacity-40" />
            {query.trim() ? "No GGUF models matched your search." : "Nothing to show — try a different sort or search."}
          </div>
        ) : (
          results.map((repo) => {
            const open = openRepo === repo.repo_id;
            const repoFiles = files[repo.repo_id] ?? [];
            return (
              <div key={repo.repo_id} className="rounded-lg border border-border bg-background/40">
                <button
                  type="button"
                  onClick={() => toggleRepo(repo.repo_id)}
                  className="flex w-full items-center gap-2 p-3 text-left"
                >
                  {open ? <ChevronDown className="size-4 shrink-0" /> : <ChevronRight className="size-4 shrink-0" />}
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-mono text-sm">{repo.repo_id}</div>
                    <div className="text-[11px] text-muted-foreground">
                      ↓ {repo.downloads.toLocaleString()} · ♥ {repo.likes.toLocaleString()}
                      {repo.gated && " · gated"}
                    </div>
                  </div>
                </button>
                {open && (
                  <div className="space-y-1 border-t border-border p-2">
                    {filesLoading === repo.repo_id ? (
                      <Skeleton className="h-8 w-full" />
                    ) : repoFiles.length === 0 ? (
                      <p className="p-2 text-xs text-muted-foreground">No .gguf files found in this repo.</p>
                    ) : (
                      repoFiles.map((file) => {
                        const key = `${repo.repo_id}/${file.path}`;
                        const dl = downloads[key];
                        const have = installed.has(file.filename) || dl?.status === "done";
                        const pct = dl && dl.total ? (dl.downloaded / dl.total) * 100 : 0;
                        return (
                          <div key={file.path} className="rounded-md p-2 hover:bg-white/5">
                            <div className="flex items-center gap-2">
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2">
                                  <span className="truncate font-mono text-xs">{file.quant || file.filename}</span>
                                  <FitBadge fit={file.fit} />
                                  {file.shards > 1 && (
                                    <Badge variant="secondary" className="text-[10px]">
                                      {file.shards} shards
                                    </Badge>
                                  )}
                                </div>
                                <span className="text-[11px] text-muted-foreground">{formatBytes(file.size_bytes)}</span>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 gap-1 px-2 text-xs text-muted-foreground"
                                onClick={() =>
                                  onConfigure({
                                    repo_id: repo.repo_id,
                                    filename: file.filename,
                                    size_bytes: file.size_bytes,
                                    quant: file.quant,
                                  })
                                }
                                title="Advanced settings + launch command"
                              >
                                <Settings2 className="size-3.5" />
                                Configure
                              </Button>
                              {have ? (
                                <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" disabled>
                                  <Check className="size-3.5 text-primary" />
                                  Downloaded
                                </Button>
                              ) : (
                                <Button
                                  size="sm"
                                  className="h-7 gap-1 text-xs"
                                  onClick={() => startDownload(repo.repo_id, file)}
                                  disabled={dl?.status === "downloading"}
                                >
                                  {dl?.status === "downloading" ? (
                                    <Loader2 className="size-3.5 animate-spin" />
                                  ) : (
                                    <Download className="size-3.5" />
                                  )}
                                  Download
                                </Button>
                              )}
                            </div>
                            {dl?.status === "downloading" && (
                              <div className="mt-1.5 space-y-1">
                                <Progress value={pct} />
                                <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                                  <span>
                                    {formatBytes(dl.downloaded)} / {formatBytes(dl.total)} ({pct.toFixed(0)}%)
                                  </span>
                                  <span>
                                    {formatSpeed(dl.speed_bps)} · {formatEta(dl.eta_seconds)} left
                                  </span>
                                </div>
                              </div>
                            )}
                            {dl?.status === "error" && (
                              <p className="mt-1 text-[11px] text-destructive">{dl.message}</p>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

/** Library: the downloaded GGUFs — load onto the server, configure (launch command), or remove. */
function LibraryTab({ onConfigure }: { onConfigure: (t: ConfigTarget) => void }) {
  const { localModels, deleteModel, llamacppStatus } = useApp();
  const [notice, setNotice] = useState<{ kind: "error" | "unmanaged"; text: string } | null>(null);

  const isActive = (m: LocalModel) =>
    !!llamacppStatus &&
    (llamacppStatus.served_model === m.label || llamacppStatus.models.includes(m.label));

  const onResult = (res: LoadModelResult) => {
    if (res.ok) {
      setNotice(null);
      return;
    }
    setNotice({
      kind: res.unmanaged ? "unmanaged" : "error",
      text: res.error || "Couldn't load the model.",
    });
  };

  if (localModels.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center text-sm text-muted-foreground">
        <HardDrive className="size-6 opacity-40" />
        No models downloaded yet. Find one in the Discover tab.
      </div>
    );
  }
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {notice && (
        <div
          className={cn(
            "mx-4 mt-4 flex items-start gap-2 rounded-lg border p-3 text-[11px]",
            notice.kind === "unmanaged"
              ? "border-amber-500/30 bg-amber-500/10 text-amber-300/90"
              : "border-rose-500/30 bg-rose-500/10 text-rose-300/90"
          )}
        >
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <p>
            {notice.text}
            {notice.kind === "unmanaged" &&
              " The llama-server isn't a local Docker container this app can restart — use Configure to copy the launch command and run it on the GPU host."}
          </p>
        </div>
      )}
      <div className="grid grid-cols-1 gap-2 overflow-y-auto p-4 sm:grid-cols-2">
        {localModels.map((m: LocalModel) => (
          <div key={m.filename} className="flex flex-col gap-2 rounded-lg border border-border bg-background/40 p-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <div className="truncate font-mono text-sm">{m.filename}</div>
                {isActive(m) && (
                  <Badge className="shrink-0 border-emerald-500/30 bg-emerald-500/15 text-[10px] text-emerald-400">
                    Active
                  </Badge>
                )}
              </div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">
                {m.quant && <span className="mr-2">{m.quant}</span>}
                {formatBytes(m.size_bytes)}
                {m.repo_id && <span className="ml-2 truncate">· {m.repo_id}</span>}
              </div>
            </div>
            <div className="mt-auto flex items-center justify-end gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 gap-1 px-2 text-xs text-muted-foreground"
                onClick={() =>
                  onConfigure({
                    repo_id: m.repo_id ?? "",
                    filename: m.filename,
                    size_bytes: m.size_bytes,
                    quant: m.quant,
                  })
                }
              >
                <Settings2 className="size-3.5" />
                Configure
              </Button>
              {isActive(m) ? (
                <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" disabled>
                  <Check className="size-3.5 text-primary" />
                  Serving
                </Button>
              ) : (
                <ConfirmLoadButton
                  filename={m.filename}
                  disabled={notice?.kind === "unmanaged"}
                  onResult={onResult}
                />
              )}
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1 text-xs"
                onClick={() => deleteModel(m.filename)}
                title="Delete this model file"
              >
                <Trash2 className="size-3.5 text-muted-foreground" />
                Remove
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** VRAM (MB) for a known GPU name from the catalog (substring match, longest key first), mirroring
 *  the backend's `hardware.vram_for_name`. Used to auto-fill VRAM when a card is named. */
function vramForName(name: string, catalog?: Record<string, number>): number | null {
  if (!catalog) return null;
  const lowered = name.toLowerCase();
  for (const key of Object.keys(catalog).sort((a, b) => b.length - a.length)) {
    if (lowered.includes(key)) return catalog[key];
  }
  return null;
}

/** Hardware: the editable profile (source of truth) + best-effort NVIDIA auto-detect. */
function HardwareTab() {
  const { hardware, saveHardware, detectHardware, config } = useApp();
  const [gpus, setGpus] = useState<GpuInfo[]>([]);
  const [ram, setRam] = useState<number>(0);
  const [threads, setThreads] = useState<number>(0);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);

  // Seed the form from the loaded profile.
  const loadFrom = useCallback((p: HardwareProfile | null) => {
    setGpus(p?.gpus ?? []);
    setRam(p?.system_ram_mb ?? 0);
    setThreads(p?.cpu_threads ?? 0);
  }, []);
  useEffect(() => {
    loadFrom(hardware);
  }, [hardware, loadFrom]);

  const detect = async () => {
    setDetecting(true);
    try {
      loadFrom(await detectHardware());
    } finally {
      setDetecting(false);
    }
  };

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await saveHardware({ gpus, system_ram_mb: ram, cpu_threads: threads });
      setSaved(true);
      setTimeout(() => setSaved(false), 1800);
    } catch {
      setSaveError("Couldn't save — check the backend and try again.");
    } finally {
      setSaving(false);
    }
  };

  // Name a GPU → auto-fill its VRAM from the known-GPU catalog (non-destructive: never overwrites a
  // VRAM the user already typed). This + Detect are the fix for the historical `vram_mb: 0` profiles.
  const setGpuName = (i: number, name: string) =>
    setGpus((prev) =>
      prev.map((x, j) => {
        if (j !== i) return x;
        const next = { ...x, name };
        if (!next.vram_mb) {
          const known = vramForName(name, config?.known_gpus);
          if (known) next.vram_mb = known;
        }
        return next;
      })
    );

  const vramTotal = gpus.reduce((s, g) => s + (g.vram_mb || 0), 0);

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
      <div className="flex items-start gap-2 rounded-lg border border-border bg-background/40 p-3 text-xs text-muted-foreground">
        <Cpu className="mt-0.5 size-4 shrink-0 text-primary" />
        <p>
          This profile is the source of truth for recommendations. The backend can't see a GPU on
          another machine, so set it here. <strong>Detect</strong> reads this host's GPU via{" "}
          <code>nvidia-smi</code> when one is visible — a starting point you can edit.
          {hardware && (
            <Badge variant="secondary" className="ml-2 text-[10px]">
              source: {hardware.source}
            </Badge>
          )}
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground">GPUs (NVIDIA)</label>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 px-2 text-xs text-muted-foreground"
            onClick={detect}
            disabled={detecting}
          >
            {detecting ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
            Detect (NVIDIA)
          </Button>
        </div>
        {gpus.map((g, i) => (
          <div key={i} className="flex items-center gap-2">
            <TextInput
              value={g.name}
              placeholder="GPU name (e.g. RTX 4090)"
              onChange={(e) => setGpuName(i, e.target.value)}
              className="flex-1"
            />
            <TextInput
              type="number"
              min={0}
              value={g.vram_mb || ""}
              placeholder="VRAM (MB)"
              onChange={(e) =>
                setGpus((prev) => prev.map((x, j) => (j === i ? { ...x, vram_mb: Number(e.target.value) || 0 } : x)))
              }
              className="w-32"
            />
            <Button
              variant="ghost"
              size="icon"
              className="size-8 shrink-0 text-muted-foreground"
              onClick={() => setGpus((prev) => prev.filter((_, j) => j !== i))}
              aria-label="Remove GPU"
            >
              <X className="size-4" />
            </Button>
          </div>
        ))}
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs"
          onClick={() => setGpus((prev) => [...prev, { name: "", vram_mb: 0 }])}
        >
          + Add GPU
        </Button>
        {gpus.length > 0 && (
          <p className="text-[11px] text-muted-foreground">Total VRAM: {(vramTotal / 1024).toFixed(1)} GB</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">System RAM (MB)</label>
          <TextInput type="number" min={0} value={ram || ""} onChange={(e) => setRam(Number(e.target.value) || 0)} />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">CPU threads</label>
          <TextInput
            type="number"
            min={0}
            value={threads || ""}
            onChange={(e) => setThreads(Number(e.target.value) || 0)}
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-3">
        {saved && (
          <span className="flex items-center gap-1 text-xs text-emerald-400">
            <Check className="size-3.5" />
            Saved
          </span>
        )}
        {saveError && <span className="text-xs text-destructive">{saveError}</span>}
        <Button size="sm" onClick={save} disabled={saving}>
          {saving ? <Loader2 className="mr-1 size-3.5 animate-spin" /> : null}
          Save profile
        </Button>
      </div>
    </div>
  );
}

/** Server: llama-server connectivity + what model it currently serves. */
function ServerTab() {
  const { llamacppStatus, refreshLlamacppStatus } = useApp();
  const s = llamacppStatus;
  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-medium">
          <Server className="size-4 text-primary" />
          llama-server
        </h3>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1 px-2 text-xs text-muted-foreground"
          onClick={() => refreshLlamacppStatus()}
        >
          <RefreshCw className="size-3.5" />
          Refresh
        </Button>
      </div>
      <div className="space-y-2 rounded-lg border border-border bg-background/40 p-3 text-xs">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "inline-block size-2 rounded-full",
              s?.reachable ? "bg-emerald-400" : "bg-rose-400"
            )}
          />
          <span className={s?.reachable ? "text-emerald-400" : "text-rose-400"}>
            {s?.reachable ? "Reachable" : "Unreachable"}
          </span>
        </div>
        <Stat label="Base URL" value={s?.base_url ?? "—"} />
        <Stat label="Serving model" value={s?.served_model ?? "—"} />
        {s && s.models.length > 1 && <Stat label="Models" value={s.models.join(", ")} />}
      </div>
      {s?.reachable ? (
        <p className="text-[11px] text-muted-foreground">
          When llama-server runs as the bundled local Docker container, you can switch models from the{" "}
          <strong>Library</strong> tab — <strong>Load</strong> restarts it onto the chosen GGUF. A
          hand-run or remote server can't be switched from here (use the <strong>Configure</strong>{" "}
          launch command instead).
        </p>
      ) : (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-[11px] text-amber-300/90">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <p>
            The app connects to an external llama-server but does not start it. Download a model in
            Discover, then either <strong>Load</strong> it from the Library (bundled Docker server) or
            copy its <strong>Configure</strong> launch command and run it on the GPU machine. Set{" "}
            <code>LLAMACPP_BASE_URL</code> to point here.
          </p>
        </div>
      )}
    </div>
  );
}

/** The Model Manager panel: discover/download GGUFs from HuggingFace, see what fits the hardware,
 *  load one onto the local llama-server, and manage the library. Rendered inside the Settings page's
 *  "Models" tab (no dialog chrome of its own). */
export function ModelsPanel() {
  const [tab, setTab] = useState("discover");
  const [configTarget, setConfigTarget] = useState<ConfigTarget | null>(null);

  const onConfigure = (t: ConfigTarget) => setConfigTarget(t);

  return (
    <div className="flex h-full flex-col">
      <header className="shrink-0 border-b border-border p-4">
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <Boxes className="size-4 text-primary" />
          Model Manager
        </h2>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Download GGUF models from HuggingFace, see what fits your NVIDIA hardware, and load one onto
          the local llama.cpp server.
        </p>
      </header>

      {configTarget ? (
        <AdvancedConfig target={configTarget} onBack={() => setConfigTarget(null)} />
      ) : (
        <Tabs defaultValue="discover" value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
          <div className="px-4 pt-3">
            <TabsList className="w-full">
              <TabsTrigger value="discover">
                <Search className="size-3.5" /> Discover
              </TabsTrigger>
              <TabsTrigger value="library">
                <HardDrive className="size-3.5" /> Library
              </TabsTrigger>
              <TabsTrigger value="hardware">
                <Cpu className="size-3.5" /> Hardware
              </TabsTrigger>
              <TabsTrigger value="server">
                <Server className="size-3.5" /> Server
              </TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="discover" className="flex min-h-0 flex-1 flex-col">
            <DiscoverTab onConfigure={onConfigure} />
          </TabsContent>
          <TabsContent value="library" className="flex min-h-0 flex-1 flex-col">
            <LibraryTab onConfigure={onConfigure} />
          </TabsContent>
          <TabsContent value="hardware" className="flex min-h-0 flex-1 flex-col">
            <HardwareTab />
          </TabsContent>
          <TabsContent value="server" className="flex min-h-0 flex-1 flex-col">
            <ServerTab />
          </TabsContent>
        </Tabs>
      )}
      <DownloadsBar />
    </div>
  );
}

/** A persistent strip listing active downloads with progress + speed + ETA, shown across all Model
 *  Manager tabs (Discover/Library/Hardware/Server) so navigating away from Discover never hides an
 *  in-flight download. Reads the shared `downloads` map, so it also survives refresh + reopening. */
function DownloadsBar() {
  const { downloads } = useApp();
  const active = Object.entries(downloads).filter(([, d]) => d.status === "downloading");
  if (active.length === 0) return null;
  return (
    <div className="shrink-0 space-y-2.5 border-t border-border bg-background/60 px-4 py-2.5">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground/70">
        <Download className="size-3 animate-pulse" />
        Downloading {active.length > 1 ? `(${active.length})` : ""}
      </div>
      {active.map(([key, d]) => {
        const pct = d.total ? (d.downloaded / d.total) * 100 : 0;
        const name = key.split("/").pop();
        return (
          <div key={key} className="space-y-1">
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="truncate font-mono text-foreground/90">{name}</span>
              <span className="shrink-0 tabular-nums text-muted-foreground">
                {formatSpeed(d.speed_bps)} · {formatEta(d.eta_seconds)} left
              </span>
            </div>
            <Progress value={pct} />
            <div className="text-[10px] tabular-nums text-muted-foreground">
              {formatBytes(d.downloaded)} / {formatBytes(d.total)} ({pct.toFixed(0)}%)
            </div>
          </div>
        );
      })}
    </div>
  );
}
