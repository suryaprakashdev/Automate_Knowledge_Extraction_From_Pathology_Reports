import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Loader2 } from "lucide-react";
import { pdfUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";

// pdf.js worker (bundled by Vite)
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

export function PdfViewer({
  filename,
  page,
  highlights,
  onPageChange,
}: {
  filename: string | null;
  page: number;
  highlights: number[][];
  onPageChange: (p: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(600);
  const [numPages, setNumPages] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [renderScale, setRenderScale] = useState(1); // px per PDF point

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver(() => setContainerWidth(el.clientWidth - 32));
    ro.observe(el);
    setContainerWidth(el.clientWidth - 32);
    return () => ro.disconnect();
  }, []);

  const file = useMemo(() => (filename ? pdfUrl(filename) : null), [filename]);
  const pageWidth = Math.max(200, containerWidth * zoom);

  if (!file) {
    return (
      <div className="flex h-full items-center justify-center rounded-xl border-2 border-dashed border-border text-sm text-muted-foreground">
        Upload or select a document to preview it here
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col rounded-xl border border-border bg-muted/30">
      {/* toolbar */}
      <div className="flex items-center gap-1 border-b border-border px-3 py-2">
        <Button
          variant="ghost"
          size="icon"
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="min-w-[90px] text-center text-xs text-muted-foreground">
          Page {page} / {numPages || "…"}
        </span>
        <Button
          variant="ghost"
          size="icon"
          disabled={numPages > 0 && page >= numPages}
          onClick={() => onPageChange(page + 1)}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <div className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="icon" onClick={() => setZoom((z) => Math.max(0.6, z - 0.15))}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="w-10 text-center text-xs text-muted-foreground">{Math.round(zoom * 100)}%</span>
          <Button variant="ghost" size="icon" onClick={() => setZoom((z) => Math.min(2.5, z + 0.15))}>
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* page */}
      <div ref={containerRef} className="flex-1 overflow-auto p-4">
        <Document
          file={file}
          onLoadSuccess={(pdf) => setNumPages(pdf.numPages)}
          loading={
            <div className="flex h-40 items-center justify-center text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading document…
            </div>
          }
          error={
            <div className="flex h-40 items-center justify-center text-sm text-red-500">
              Failed to load PDF.
            </div>
          }
          className="flex justify-center"
        >
          <div className="relative shadow-lg" style={{ width: pageWidth }}>
            <Page
              pageNumber={page}
              width={pageWidth}
              renderTextLayer={false}
              renderAnnotationLayer={false}
              onLoadSuccess={(p) => setRenderScale(pageWidth / p.originalWidth)}
            />
            {/* highlight overlay (PDF points → pixels via renderScale) */}
            <div className="pointer-events-none absolute inset-0">
              {highlights.map((b, i) => (
                <div
                  key={i}
                  className="absolute rounded-sm animate-pulse-ring"
                  style={{
                    left: b[0] * renderScale,
                    top: b[1] * renderScale,
                    width: (b[2] - b[0]) * renderScale,
                    height: (b[3] - b[1]) * renderScale,
                    background: "hsl(var(--highlight) / 0.38)",
                    outline: "1px solid hsl(var(--highlight) / 0.8)",
                  }}
                />
              ))}
            </div>
          </div>
        </Document>
      </div>
    </div>
  );
}
