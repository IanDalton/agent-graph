import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * A tiny self-contained popover: a trigger plus a floating panel that closes on outside
 * pointerdown or Escape. Built by hand rather than pulling in `@radix-ui/react-popover` to keep
 * the dependency surface small (the rest of the UI does the same). It anchors *upward* by default
 * (`mb-2 bottom-full`) since its first use is the chat composer pinned to the bottom of the screen.
 */
export function Popover({
  trigger,
  children,
  align = "start",
  className,
}: {
  trigger: (props: { open: boolean; toggle: () => void }) => React.ReactNode;
  children: (props: { close: () => void }) => React.ReactNode;
  align?: "start" | "end";
  className?: string;
}) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      {trigger({ open, toggle: () => setOpen((v) => !v) })}
      {open && (
        <div
          role="menu"
          className={cn(
            "absolute bottom-full z-50 mb-2 min-w-[12rem] max-h-72 overflow-y-auto rounded-xl border border-white/10 bg-slate-900/95 p-1 shadow-xl shadow-black/40 backdrop-blur",
            align === "end" ? "right-0" : "left-0",
            className
          )}
        >
          {children({ close: () => setOpen(false) })}
        </div>
      )}
    </div>
  );
}
