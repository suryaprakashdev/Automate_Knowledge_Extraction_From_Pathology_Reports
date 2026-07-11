import { parseSSE } from "./sse";

export interface Source {
  index: number;
  filename: string;
  page: number | null;
  line_bboxes: number[][];
  text: string;
  score: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  question: string;
  report_name?: string | null;
  report_names?: string[] | null;
  top_k?: number;
  history?: ChatTurn[];
}

export interface ChatCallbacks {
  onSources?: (sources: Source[]) => void;
  onToken?: (text: string) => void;
  onDone?: (numSources: number) => void;
  onError?: (message: string) => void;
}

export async function getReports(): Promise<string[]> {
  const r = await fetch("/api/reports");
  if (!r.ok) throw new Error("Failed to load reports");
  return (await r.json()).reports as string[];
}

export interface Health {
  status: "ok" | "degraded";
  num_documents: number;
  detail?: string | null;
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  return r.json();
}

/** Stream a chat answer. Resolves when the stream ends. */
export async function streamChat(
  req: ChatRequest,
  cb: ChatCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

  if (!resp.ok) {
    const detail = await resp.text().catch(() => "");
    cb.onError?.(`Request failed (${resp.status}). ${detail}`);
    return;
  }

  for await (const evt of parseSSE(resp)) {
    const data = safeParse(evt.data);
    if (evt.event === "sources") cb.onSources?.(data.sources ?? []);
    else if (evt.event === "token") cb.onToken?.(data.text ?? "");
    else if (evt.event === "done") cb.onDone?.(data.num_sources ?? 0);
    else if (evt.event === "error") cb.onError?.(data.message ?? "Unknown error");
  }
}

export interface UploadProgress {
  stage: string; // ocr | embedding | indexing | done | complete
  page?: number;
  total?: number;
  num_chunks?: number;
  vectors_added?: number;
}

/** Upload a PDF, then stream OCR/embedding/indexing progress until complete. */
export async function uploadAndTrack(
  file: File,
  onProgress: (p: UploadProgress) => void,
  onComplete: (stats: Record<string, unknown>) => void,
  onError: (message: string) => void,
): Promise<void> {
  const form = new FormData();
  form.append("file", file);

  const start = await fetch("/api/upload", { method: "POST", body: form });
  if (!start.ok) {
    onError(`Upload failed (${start.status}).`);
    return;
  }
  const { job_id } = await start.json();

  const events = await fetch(`/api/upload/${job_id}/events`);
  if (!events.ok) {
    onError("Failed to open progress stream.");
    return;
  }

  for await (const evt of parseSSE(events)) {
    const data = safeParse(evt.data);
    if (evt.event === "progress") onProgress(data as UploadProgress);
    else if (evt.event === "complete") onComplete(data);
    else if (evt.event === "error") onError(data.message ?? "Processing failed");
  }
}

export function pdfUrl(filename: string): string {
  return `/api/pdf/${encodeURIComponent(filename)}`;
}

function safeParse(s: string): any {
  try {
    return JSON.parse(s);
  } catch {
    return {};
  }
}
