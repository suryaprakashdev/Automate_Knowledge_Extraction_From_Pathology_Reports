import { Check, ChevronDown, X } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

export function ReportPicker({
  reports,
  selected,
  onChange,
}: {
  reports: string[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  const [open, setOpen] = useState(false);

  const toggle = (r: string) =>
    onChange(selected.includes(r) ? selected.filter((x) => x !== r) : [...selected, r]);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm"
      >
        <span className="text-muted-foreground">Scope:</span>
        <span className="flex-1 truncate text-left">
          {selected.length === 0 ? "All documents" : `${selected.length} selected`}
        </span>
        <ChevronDown className="h-4 w-4 text-muted-foreground" />
      </button>

      {selected.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {selected.map((r) => (
            <span
              key={r}
              className="inline-flex items-center gap-1 rounded-full bg-primary/12 px-2 py-0.5 text-xs text-primary"
            >
              {r}
              <button onClick={() => toggle(r)}>
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute z-20 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-border bg-card p-1 shadow-lg">
            {reports.length === 0 && (
              <div className="px-3 py-2 text-xs text-muted-foreground">No documents indexed.</div>
            )}
            {reports.map((r) => {
              const on = selected.includes(r);
              return (
                <button
                  key={r}
                  onClick={() => toggle(r)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm hover:bg-muted",
                    on && "text-primary",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-4 w-4 items-center justify-center rounded border",
                      on ? "border-primary bg-primary text-primary-foreground" : "border-border",
                    )}
                  >
                    {on && <Check className="h-3 w-3" />}
                  </span>
                  <span className="truncate">{r}</span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
