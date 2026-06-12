import { useCallback, useState } from "react";

import { AppProvider } from "@/state/AppContext";
import { Sidebar } from "@/panes/Sidebar";
import { Canvas } from "@/panes/Canvas";
import { ContextPane } from "@/panes/ContextPane";

function Shell() {
  // Bumped after every completed turn so the right pane re-fetches the summary.
  const [summaryKey, setSummaryKey] = useState(0);
  const onTurnComplete = useCallback(() => setSummaryKey((k) => k + 1), []);

  return (
    <div className="grid h-full grid-cols-[260px_1fr_440px] overflow-hidden">
      <Sidebar />
      <main className="min-w-0 overflow-hidden">
        <Canvas onTurnComplete={onTurnComplete} />
      </main>
      <ContextPane refreshKey={summaryKey} />
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
