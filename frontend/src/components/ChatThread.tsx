import { useEffect, useRef } from "react";
import { MessageBubble, type Message } from "./MessageBubble";

export function ChatThread({
  messages,
  onCitation,
  renderExtra,
  empty,
}: {
  messages: Message[];
  onCitation?: (message: Message, index: number) => void;
  renderExtra?: (message: Message) => React.ReactNode;
  empty?: React.ReactNode;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0 && empty) {
    return <div className="flex flex-1 items-center justify-center p-6">{empty}</div>;
  }

  return (
    <div className="flex-1 space-y-5 overflow-y-auto px-1 py-4">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} onCitation={(idx) => onCitation?.(m, idx)}>
          {renderExtra?.(m)}
        </MessageBubble>
      ))}
      <div ref={endRef} />
    </div>
  );
}
