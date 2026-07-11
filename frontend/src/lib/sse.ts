export interface SSEvent {
  event: string;
  data: string;
}

/**
 * Parse a Server-Sent Events stream from a fetch Response body into an async
 * iterator of {event, data}. Used for POST-based SSE where EventSource (GET
 * only) can't be used.
 */
export async function* parseSSE(response: Response): AsyncGenerator<SSEvent> {
  if (!response.body) throw new Error("Response has no body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // Normalize CRLF (sse-starlette uses "\r\n" line endings) so a single
    // "\n\n" boundary check works regardless of the server's separator.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let sep: number;
    // Events are separated by a blank line.
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let event = "message";
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      }
      if (dataLines.length) yield { event, data: dataLines.join("\n") };
    }
  }
}
