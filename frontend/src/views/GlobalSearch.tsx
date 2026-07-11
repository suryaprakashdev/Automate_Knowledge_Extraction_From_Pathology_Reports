import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { getReports, type Source } from "@/lib/api";
import { useChat } from "@/lib/useChat";
import { ChatThread } from "@/components/ChatThread";
import { ChatInput } from "@/components/ChatInput";
import { ReportPicker } from "@/components/ReportPicker";
import { SourceCard } from "@/components/SourceCard";
import { PdfViewer } from "@/components/PdfViewer";
import { ExamplePrompts } from "@/components/ExamplePrompts";

const EXAMPLES = [
  "What ER/PR/HER2 receptor patterns appear across reports?",
  "Which reports mention lymph node involvement?",
  "Summarize the most common diagnoses.",
];

interface Preview {
  filename: string;
  page: number;
  highlights: number[][];
}

export function GlobalSearch() {
  const { data: reports } = useQuery({ queryKey: ["reports"], queryFn: getReports });
  const [selected, setSelected] = useState<string[]>([]);
  const [preview, setPreview] = useState<Preview | null>(null);
  const { messages, busy, send, stop } = useChat();

  const submit = (text: string) =>
    send(text, { report_names: selected.length ? selected : null });

  const openSource = (s: Source) =>
    setPreview({ filename: s.filename, page: s.page ?? 1, highlights: s.line_bboxes });

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col p-4">
      <div className="pb-3">
        <ReportPicker reports={reports ?? []} selected={selected} onChange={setSelected} />
      </div>

      <ChatThread
        messages={messages}
        empty={<ExamplePrompts title="Search across all reports" prompts={EXAMPLES} onPick={submit} />}
        renderExtra={(m) =>
          m.role === "assistant" && m.sources && m.sources.length > 0 ? (
            <div className="grid gap-2 sm:grid-cols-2">
              {m.sources.map((s) => (
                <SourceCard key={s.index} source={s} onClick={() => openSource(s)} />
              ))}
            </div>
          ) : null
        }
      />

      <div className="pt-2">
        <ChatInput onSend={submit} onStop={stop} busy={busy} placeholder="Ask across your reports…" />
      </div>

      {preview && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-6" onClick={() => setPreview(null)}>
          <div
            className="flex h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-border bg-background"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
              <span className="truncate text-sm font-medium">{preview.filename}</span>
              <button onClick={() => setPreview(null)} className="rounded-md p-1 hover:bg-muted">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 p-3">
              <PdfViewer
                filename={preview.filename}
                page={preview.page}
                highlights={preview.highlights}
                onPageChange={(p) => setPreview((prev) => (prev ? { ...prev, page: p, highlights: [] } : prev))}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
