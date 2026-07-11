import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { getHealth } from "@/lib/api";
import { Sidebar, type View } from "@/components/Sidebar";
import { DocumentChat } from "@/views/DocumentChat";
import { GlobalSearch } from "@/views/GlobalSearch";

export default function App() {
  const [view, setView] = useState<View>("document");
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: getHealth });

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar view={view} onView={setView} />
      <main className="flex min-w-0 flex-1 flex-col">
        {health?.status === "degraded" && (
          <div className="flex items-center gap-2 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span className="truncate">
              Backend not fully ready: {health.detail ?? "pipeline unavailable"}
            </span>
          </div>
        )}
        <div className="min-h-0 flex-1">
          {view === "document" ? <DocumentChat /> : <GlobalSearch />}
        </div>
      </main>
    </div>
  );
}
