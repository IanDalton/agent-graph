import { useEffect, useLayoutEffect, useRef } from "react";

import type { ChatMessage } from "@/types";

/** Distance (px) from the bottom still counted as "parked at the bottom". */
const PIN_THRESHOLD = 80;

/**
 * Keeps a scroll viewport pinned to the bottom *only while the user wants it there*.
 *
 * - On mount and whenever the user sends a message, snap to the bottom.
 * - While the assistant streams, follow along only if the user is still parked at
 *   the bottom. The moment they scroll up they're left alone — no forced jumps.
 * - Scrolling back down re-pins them so the next stream follows again.
 *
 * Returns the ref to attach to the scrollable viewport (`ScrollArea`'s `viewportRef`).
 */
export function useStickyScroll(messages: ChatMessage[]) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const prevLenRef = useRef(messages.length);

  // Track whether the user is at the bottom. Once they scroll up we stop forcing
  // them down; returning to the bottom re-pins.
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onScroll = () => {
      const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
      pinnedRef.current = dist <= PIN_THRESHOLD;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useLayoutEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const grew = messages.length > prevLenRef.current;
    const lastIsUser = messages[messages.length - 1]?.role === "user";
    prevLenRef.current = messages.length;
    // A freshly-sent user message always snaps to the bottom and re-pins;
    // assistant/streaming updates only scroll while the user is still pinned.
    if (grew && lastIsUser) pinnedRef.current = true;
    if (pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  return viewportRef;
}
