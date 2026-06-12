import * as React from "react";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";

/**
 * A minimal modal dialog: a full-screen backdrop plus a centered panel that closes on backdrop
 * click, the Escape key, or the corner X. Built by hand rather than pulling in
 * `@radix-ui/react-dialog` to keep the dependency surface small (the rest of the UI does the same,
 * see Popover/Select). Open state is controlled by the caller. While open, body scroll is locked.
 */
export function Dialog({
  open,
  onClose,
  children,
  className,
  title,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onPointerDown={(e) => {
        // Close only when the backdrop itself (not a child) is pressed.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={cn(
          "relative flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-2xl",
          className
        )}
      >
        <button
          type="button"
          onClick={onClose}
          title="Close"
          className="absolute right-3 top-3 z-10 rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          <X className="size-4" />
        </button>
        {children}
      </div>
    </div>
  );
}
