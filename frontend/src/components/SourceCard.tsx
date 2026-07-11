import { FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

export function SourceCard({
  source,
  onClick,
  active,
}: {
  source: Source;
  onClick?: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full rounded-lg border bg-card px-3 py-2.5 text-left transition-colors",
        active ? "border-primary ring-1 ring-primary" : "border-border hover:border-primary/60",
      )}
    >
      <div className="mb-1 flex items-center gap-2 text-xs">
        <span className="flex h-5 w-5 items-center justify-center rounded bg-primary/15 font-semibold text-primary">
          {source.index}
        </span>
        <FileText className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="truncate font-medium">{source.filename}</span>
        {source.page != null && (
          <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
            page {source.page}
          </span>
        )}
        <span className="ml-auto text-[11px] text-muted-foreground">{source.score.toFixed(3)}</span>
      </div>
      <p className="line-clamp-2 text-xs text-muted-foreground">{source.text}</p>
    </button>
  );
}
