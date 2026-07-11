import { useCallback, useRef, useState } from "react";
import { UploadCloud, Loader2, CheckCircle2 } from "lucide-react";
import { uploadAndTrack, type UploadProgress } from "@/lib/api";
import { cn } from "@/lib/utils";

const STAGE_LABEL: Record<string, string> = {
  ocr: "Reading pages (OCR)",
  embedding: "Generating embeddings",
  indexing: "Updating index",
  done: "Finishing up",
};

export function UploadPanel({ onIndexed }: { onIndexed: (filename: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [doneName, setDoneName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handle = useCallback(
    (file: File) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setError("Please choose a PDF file.");
        return;
      }
      setBusy(true);
      setError(null);
      setDoneName(null);
      setProgress({ stage: "ocr" });
      const stem = file.name.replace(/\.pdf$/i, "");
      uploadAndTrack(
        file,
        (p) => setProgress(p),
        () => {
          setBusy(false);
          setProgress(null);
          setDoneName(stem);
          onIndexed(stem);
        },
        (msg) => {
          setBusy(false);
          setProgress(null);
          setError(msg);
        },
      );
    },
    [onIndexed],
  );

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]);
        }}
        onClick={() => !busy && inputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors",
          drag ? "border-primary bg-primary/5" : "border-border hover:border-primary/50",
          busy && "cursor-default opacity-80",
        )}
      >
        {busy ? (
          <>
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <div className="text-sm font-medium">
              {progress ? STAGE_LABEL[progress.stage] ?? "Processing" : "Processing"}
              {progress?.page && progress?.total ? ` · page ${progress.page}/${progress.total}` : ""}
            </div>
            <ProgressBar progress={progress} />
          </>
        ) : doneName ? (
          <>
            <CheckCircle2 className="h-6 w-6 text-emerald-500" />
            <div className="text-sm font-medium">Indexed “{doneName}”</div>
            <div className="text-xs text-muted-foreground">Click to upload another</div>
          </>
        ) : (
          <>
            <UploadCloud className="h-6 w-6 text-muted-foreground" />
            <div className="text-sm font-medium">Drop a pathology PDF or click to browse</div>
            <div className="text-xs text-muted-foreground">OCR + embedding runs automatically</div>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
        />
      </div>
      {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
    </div>
  );
}

function ProgressBar({ progress }: { progress: UploadProgress | null }) {
  // Coarse stage-based fill; OCR uses page fraction when available.
  let pct = 10;
  if (progress?.stage === "ocr") {
    pct = progress.page && progress.total ? 10 + (progress.page / progress.total) * 50 : 30;
  } else if (progress?.stage === "embedding") pct = 70;
  else if (progress?.stage === "indexing") pct = 88;
  else if (progress?.stage === "done") pct = 96;
  return (
    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div className="h-full rounded-full bg-primary transition-all duration-500" style={{ width: `${pct}%` }} />
    </div>
  );
}
