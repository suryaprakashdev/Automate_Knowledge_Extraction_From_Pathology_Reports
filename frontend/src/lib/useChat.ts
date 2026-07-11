import { useCallback, useRef, useState } from "react";
import { streamChat, type Source, type ChatTurn } from "./api";
import type { Message } from "@/components/MessageBubble";

let idCounter = 0;
const nextId = () => `m${++idCounter}`;

export interface SendScope {
  report_name?: string | null;
  report_names?: string[] | null;
  top_k?: number;
}

export function useChat(onSources?: (sources: Source[], forMessageId: string) => void) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patch = (id: string, fn: (m: Message) => Message) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));

  const send = useCallback(
    (text: string, scope: SendScope) => {
      if (busy) return;
      const history: ChatTurn[] = messages.map((m) => ({ role: m.role, content: m.content }));

      const userMsg: Message = { id: nextId(), role: "user", content: text };
      const botId = nextId();
      const botMsg: Message = { id: botId, role: "assistant", content: "", streaming: true };
      setMessages((prev) => [...prev, userMsg, botMsg]);
      setBusy(true);

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      streamChat(
        {
          question: text,
          report_name: scope.report_name ?? null,
          report_names: scope.report_names ?? null,
          top_k: scope.top_k ?? 5,
          history,
        },
        {
          onSources: (sources) => {
            patch(botId, (m) => ({ ...m, sources }));
            onSources?.(sources, botId);
          },
          onToken: (t) => patch(botId, (m) => ({ ...m, content: m.content + t })),
          onDone: () => {
            patch(botId, (m) => ({ ...m, streaming: false }));
            setBusy(false);
          },
          onError: (msg) => {
            patch(botId, (m) => ({
              ...m,
              streaming: false,
              content: m.content || `⚠️ ${msg}`,
            }));
            setBusy(false);
          },
        },
        ctrl.signal,
      );
    },
    [busy, messages, onSources],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
    setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setBusy(false);
  }, []);

  return { messages, busy, send, stop, reset };
}
