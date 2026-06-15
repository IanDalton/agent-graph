import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, GripVertical } from "lucide-react";

import { Button } from "@/components/ui/button";
import { AppProvider } from "@/state/AppContext";
import { Sidebar } from "@/panes/Sidebar";
import { Canvas } from "@/panes/Canvas";
import { ContextPane } from "@/panes/ContextPane";
import { cn } from "@/lib/utils";

const DEFAULT_LEFT_WIDTH = 260;
const DEFAULT_RIGHT_WIDTH = 440;
const MIN_LEFT_WIDTH = 220;
const MIN_RIGHT_WIDTH = 320;
const MIN_CENTER_WIDTH = 420;
const COLLAPSED_WIDTH = 56;
const HANDLE_WIDTH = 12;

const LAYOUT_KEYS = {
  leftWidth: "agent-graph:layout:left-width",
  rightWidth: "agent-graph:layout:right-width",
  leftCollapsed: "agent-graph:layout:left-collapsed",
  rightCollapsed: "agent-graph:layout:right-collapsed",
} as const;

type ResizeState =
  | { side: "left"; startX: number; startWidth: number }
  | { side: "right"; startX: number; startWidth: number };

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function readStoredNumber(key: string, fallback: number) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function readStoredBoolean(key: string, fallback: boolean) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return raw === "true";
  } catch {
    return fallback;
  }
}

function ToggleRail({
  side,
  label,
  onToggle,
}: {
  side: "left" | "right";
  label: string;
  onToggle: () => void;
}) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 bg-card px-1 text-muted-foreground">
      <Button
        variant="ghost"
        size="icon"
        className="size-9"
        onClick={onToggle}
        aria-label={`Expand ${label}`}
        title={`Expand ${label}`}
      >
        {side === "left" ? <ChevronRight /> : <ChevronLeft />}
      </Button>
      <span className="select-none text-[10px] font-medium uppercase tracking-[0.35em] [writing-mode:vertical-rl] rotate-180">
        {label}
      </span>
    </div>
  );
}

function ResizeHandle({
  onPointerDown,
}: {
  onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void;
}) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      tabIndex={-1}
      onPointerDown={onPointerDown}
      className={cn(
        "group relative z-10 flex h-full w-3 shrink-0 cursor-col-resize items-center justify-center",
        "touch-none select-none bg-transparent transition-colors hover:bg-border/40"
      )}
    >
      <div className="flex h-8 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm transition-colors group-hover:bg-accent group-hover:text-accent-foreground">
        <GripVertical className="size-3.5" />
      </div>
    </div>
  );
}

function Shell() {
  // Bumped after every completed turn so the right pane re-fetches the summary.
  const [summaryKey, setSummaryKey] = useState(0);
  const onTurnComplete = useCallback(() => setSummaryKey((k) => k + 1), []);

  const [viewportWidth, setViewportWidth] = useState(() => {
    if (typeof window === "undefined") return 1440;
    return window.innerWidth;
  });
  const [leftWidth, setLeftWidth] = useState(() =>
    clamp(readStoredNumber(LAYOUT_KEYS.leftWidth, DEFAULT_LEFT_WIDTH), MIN_LEFT_WIDTH, 640)
  );
  const [rightWidth, setRightWidth] = useState(() =>
    clamp(readStoredNumber(LAYOUT_KEYS.rightWidth, DEFAULT_RIGHT_WIDTH), MIN_RIGHT_WIDTH, 800)
  );
  const [leftCollapsed, setLeftCollapsed] = useState(() =>
    readStoredBoolean(LAYOUT_KEYS.leftCollapsed, false)
  );
  const [rightCollapsed, setRightCollapsed] = useState(() =>
    readStoredBoolean(LAYOUT_KEYS.rightCollapsed, false)
  );
  const [resizeState, setResizeState] = useState<ResizeState | null>(null);
  const leftWidthRef = useRef(leftWidth);
  const rightWidthRef = useRef(rightWidth);

  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    leftWidthRef.current = leftWidth;
  }, [leftWidth]);

  useEffect(() => {
    rightWidthRef.current = rightWidth;
  }, [rightWidth]);

  useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_KEYS.leftWidth, String(leftWidth));
    } catch {
      // Ignore disabled storage.
    }
  }, [leftWidth]);

  useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_KEYS.rightWidth, String(rightWidth));
    } catch {
      // Ignore disabled storage.
    }
  }, [rightWidth]);

  useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_KEYS.leftCollapsed, String(leftCollapsed));
    } catch {
      // Ignore disabled storage.
    }
  }, [leftCollapsed]);

  useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_KEYS.rightCollapsed, String(rightCollapsed));
    } catch {
      // Ignore disabled storage.
    }
  }, [rightCollapsed]);

  useEffect(() => {
    if (!resizeState) return;

    const handleMove = (event: PointerEvent) => {
      if (resizeState.side === "left") {
        const next = clamp(
          resizeState.startWidth + (event.clientX - resizeState.startX),
          MIN_LEFT_WIDTH,
          Math.max(
            MIN_LEFT_WIDTH,
            viewportWidth - (rightCollapsed ? COLLAPSED_WIDTH : rightWidthRef.current) -
              MIN_CENTER_WIDTH -
              HANDLE_WIDTH * 2
          )
        );
        setLeftWidth(next);
        return;
      }

      const next = clamp(
        resizeState.startWidth - (event.clientX - resizeState.startX),
        MIN_RIGHT_WIDTH,
        Math.max(
          MIN_RIGHT_WIDTH,
          viewportWidth - (leftCollapsed ? COLLAPSED_WIDTH : leftWidthRef.current) -
            MIN_CENTER_WIDTH -
            HANDLE_WIDTH * 2
        )
      );
      setRightWidth(next);
    };

    const handleEnd = () => setResizeState(null);

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleEnd);
    window.addEventListener("pointercancel", handleEnd);

    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleEnd);
      window.removeEventListener("pointercancel", handleEnd);
    };
  }, [leftCollapsed, resizeState, rightCollapsed, viewportWidth]);

  const leftRenderedWidth = leftCollapsed
    ? COLLAPSED_WIDTH
    : clamp(
        leftWidth,
        MIN_LEFT_WIDTH,
        Math.max(
          MIN_LEFT_WIDTH,
          viewportWidth - (rightCollapsed ? COLLAPSED_WIDTH : rightWidth) -
            MIN_CENTER_WIDTH -
            HANDLE_WIDTH * 2
        )
      );

  const rightRenderedWidth = rightCollapsed
    ? COLLAPSED_WIDTH
    : clamp(
        rightWidth,
        MIN_RIGHT_WIDTH,
        Math.max(
          MIN_RIGHT_WIDTH,
          viewportWidth - (leftCollapsed ? COLLAPSED_WIDTH : leftWidth) -
            MIN_CENTER_WIDTH -
            HANDLE_WIDTH * 2
        )
      );

  const beginLeftResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (leftCollapsed) return;
    event.preventDefault();
    setResizeState({ side: "left", startX: event.clientX, startWidth: leftWidthRef.current });
  }, [leftCollapsed]);

  const beginRightResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (rightCollapsed) return;
    event.preventDefault();
    setResizeState({ side: "right", startX: event.clientX, startWidth: rightWidthRef.current });
  }, [rightCollapsed]);

  const collapseLeft = useCallback(() => {
    setResizeState(null);
    setLeftCollapsed((value) => !value);
  }, []);

  const collapseRight = useCallback(() => {
    setResizeState(null);
    setRightCollapsed((value) => !value);
  }, []);

  return (
    <div className="flex h-full overflow-hidden bg-background">
      <div className="relative h-full shrink-0 overflow-hidden" style={{ width: leftRenderedWidth }}>
        {leftCollapsed ? (
          <ToggleRail side="left" label="Chats" onToggle={collapseLeft} />
        ) : (
          <div className="relative h-full">
            <Sidebar />
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-2 top-2 z-20 size-8 bg-card/90 text-muted-foreground shadow-sm backdrop-blur hover:bg-accent hover:text-accent-foreground"
              onClick={collapseLeft}
              aria-label="Collapse conversations sidebar"
              title="Collapse conversations sidebar"
            >
              <ChevronLeft className="size-4" />
            </Button>
          </div>
        )}
      </div>
      {!leftCollapsed && <ResizeHandle onPointerDown={beginLeftResize} />}
      <main className="min-w-0 flex-1 overflow-hidden">
        <Canvas onTurnComplete={onTurnComplete} />
      </main>
      {!rightCollapsed && <ResizeHandle onPointerDown={beginRightResize} />}
      <div className="relative h-full shrink-0 overflow-hidden" style={{ width: rightRenderedWidth }}>
        {rightCollapsed ? (
          <ToggleRail side="right" label="Context" onToggle={collapseRight} />
        ) : (
          <div className="relative h-full">
            <ContextPane refreshKey={summaryKey} />
            <Button
              variant="ghost"
              size="icon"
              className="absolute left-2 top-2 z-20 size-8 bg-card/90 text-muted-foreground shadow-sm backdrop-blur hover:bg-accent hover:text-accent-foreground"
              onClick={collapseRight}
              aria-label="Collapse context sidebar"
              title="Collapse context sidebar"
            >
              <ChevronRight className="size-4" />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
