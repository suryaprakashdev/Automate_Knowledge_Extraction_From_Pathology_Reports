import { FileText, Globe, Microscope } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getReports } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";

export type View = "document" | "global";

const NAV: { id: View; label: string; icon: typeof FileText; desc: string }[] = [
  { id: "document", label: "Document Chat", icon: FileText, desc: "Chat with one report" },
  { id: "global", label: "Global Search", icon: Globe, desc: "Search across reports" },
];

export function Sidebar({ view, onView }: { view: View; onView: (v: View) => void }) {
  const { data: reports } = useQuery({ queryKey: ["reports"], queryFn: getReports });

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-card/50 backdrop-blur">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Microscope className="h-5 w-5" />
        </div>
        <div>
          <div className="text-sm font-semibold leading-tight">Pathology RAG</div>
          <div className="text-xs text-muted-foreground">Clinical document AI</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1 px-3">
        {NAV.map((item) => {
          const Icon = item.icon;
          const active = view === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onView(item.id)}
              className={cn(
                "group flex items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors",
                active ? "bg-primary/10 text-primary" : "hover:bg-muted text-foreground",
              )}
            >
              <Icon className={cn("mt-0.5 h-4 w-4", active ? "text-primary" : "text-muted-foreground")} />
              <span>
                <span className="block text-sm font-medium">{item.label}</span>
                <span className="block text-xs text-muted-foreground">{item.desc}</span>
              </span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto space-y-3 px-5 py-5">
        <div className="rounded-lg border border-border bg-background/60 px-3 py-2.5">
          <div className="text-xs text-muted-foreground">Documents indexed</div>
          <div className="text-lg font-semibold">{reports?.length ?? "—"}</div>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Appearance</span>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
