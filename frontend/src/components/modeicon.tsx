import { MessageCircle, Microscope, Network, Scale } from "lucide-react";

import type { Mode } from "@/types";

/** Maps a conversation mode to its sidebar icon. Only "regular" is reachable today;
 *  the rest are wired so future modes need no sidebar change. */
const ICONS: Record<Mode, typeof MessageCircle> = {
  regular: MessageCircle,
  research: Microscope,
  swarm: Network,
  council: Scale,
};

export function ModeIcon({ mode, className }: { mode: Mode; className?: string }) {
  const Icon = ICONS[mode] ?? MessageCircle;
  return <Icon className={className} />;
}
