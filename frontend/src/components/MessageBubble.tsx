import ReactMarkdown from "react-markdown";
import { Microscope, User } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
}

/** Turn `[3]` reference markers into markdown links (#cite-3) so we can render
 *  them as interactive citation chips via the `a` component override. */
function withCitationLinks(text: string): string {
  return text.replace(/\[(\d+)\]/g, "[$1](#cite-$1)");
}

export function MessageBubble({
  message,
  onCitation,
  children,
}: {
  message: Message;
  onCitation?: (index: number) => void;
  children?: React.ReactNode;
}) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3 animate-fade-in", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
          isUser ? "bg-muted text-foreground" : "bg-primary text-primary-foreground",
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Microscope className="h-4 w-4" />}
      </div>

      <div className={cn("min-w-0 max-w-[85%] space-y-2", isUser && "flex flex-col items-end")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm",
            isUser
              ? "bg-primary text-primary-foreground"
              : "border border-border bg-card text-card-foreground",
          )}
        >
          {isUser ? (
            <span className="whitespace-pre-wrap">{message.content}</span>
          ) : (
            <div className="markdown">
              <ReactMarkdown
                components={{
                  a: ({ href, children: c }) => {
                    const m = href?.match(/^#cite-(\d+)$/);
                    if (m) {
                      const idx = Number(m[1]);
                      return (
                        <button
                          onClick={() => onCitation?.(idx)}
                          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-primary/15 px-1 align-middle text-xs font-semibold text-primary hover:bg-primary/25"
                          title={`Jump to source ${idx}`}
                        >
                          {idx}
                        </button>
                      );
                    }
                    return (
                      <a href={href} target="_blank" rel="noreferrer" className="text-primary underline">
                        {c}
                      </a>
                    );
                  },
                }}
              >
                {withCitationLinks(message.content) || "​"}
              </ReactMarkdown>
              {message.streaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-sm bg-primary" />
              )}
            </div>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}
