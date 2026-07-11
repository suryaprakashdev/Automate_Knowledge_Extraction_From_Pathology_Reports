import { useCallback, useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getReports, type Source } from "@/lib/api";
import { useChat } from "@/lib/useChat";
import { PdfViewer } from "@/components/PdfViewer";
import { UploadPanel } from "@/components/UploadPanel";
import { ChatThread } from "@/components/ChatThread";
import { ChatInput } from "@/components/ChatInput";
import { ExamplePrompts } from "@/components/ExamplePrompts";
import type { Message } from "@/components/MessageBubble";

const EXAMPLES = [
  "What are the key abnormal findings?",
  "What is the tumor grade and stage?",
  "Summarize the ER / PR / HER2 receptor status.",
];

export function DocumentChat() {
  const qc = useQueryClient();
  const { data: reports } = useQuery({ queryKey: ["reports"], queryFn: getReports });
  const [activeDoc, setActiveDoc] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [highlights, setHighlights] = useState<number[][]>([]);

  // Focus the viewer on the top source's page and highlight its lines.
  const focusSources = useCallback((sources: Source[]) => {
    const top = sources.find((s) => s.page != null);
    if (!top || top.page == null) {
      setHighlights([]);
      return;
    }
    const boxes = sources
      .filter((s) => s.page === top.page)
      .flatMap((s) => s.line_bboxes);
    setPage(top.page);
    setHighlights(boxes);
  }, []);

  const { messages, busy, send, stop, reset } = useChat((sources) => focusSources(sources));

  // When the active document changes, start a fresh conversation.
  useEffect(() => {
    reset();
    setPage(1);
    setHighlights([]);
  }, [activeDoc, reset]);

  const onCitation = (message: Message, index: number) => {
    const src = message.sources?.find((s) => s.index === index);
    if (src?.page != null) {
      setPage(src.page);
      setHighlights(src.line_bboxes);
    }
  };

  const submit = (text: string) => {
    if (!activeDoc) return;
    send(text, { report_name: activeDoc });
  };

  return (
    <div className="flex h-full gap-4 p-4">
      {/* LEFT — document */}
      <div className="flex w-[54%] flex-col gap-3">
        <div className="flex items-center gap-2">
          <select
            value={activeDoc ?? ""}
            onChange={(e) => setActiveDoc(e.target.value || null)}
            className="flex-1 rounded-lg border border-border bg-card px-3 py-2 text-sm outline-none"
          >
            <option value="">Select a document…</option>
            {reports?.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <div className="min-h-0 flex-1">
          <PdfViewer filename={activeDoc} page={page} highlights={highlights} onPageChange={setPage} />
        </div>
        <UploadPanel
          onIndexed={(name) => {
            qc.invalidateQueries({ queryKey: ["reports"] });
            setActiveDoc(name);
          }}
        />
      </div>

      {/* RIGHT — chat */}
      <div className="flex min-w-0 flex-1 flex-col">
        <ChatThread
          messages={messages}
          onCitation={onCitation}
          empty={
            activeDoc ? (
              <ExamplePrompts title={`Ask about ${activeDoc}`} prompts={EXAMPLES} onPick={submit} />
            ) : (
              <div className="max-w-sm text-center text-sm text-muted-foreground">
                Upload a pathology PDF or select an indexed document to start chatting with it.
              </div>
            )
          }
        />
        <div className="pt-2">
          <ChatInput
            onSend={submit}
            onStop={stop}
            busy={busy}
            disabled={!activeDoc}
            placeholder={activeDoc ? "Ask about this document…" : "Select a document first"}
          />
        </div>
      </div>
    </div>
  );
}
