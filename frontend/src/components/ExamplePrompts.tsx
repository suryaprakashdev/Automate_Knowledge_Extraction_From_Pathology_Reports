import { Sparkles } from "lucide-react";

export function ExamplePrompts({
  title,
  prompts,
  onPick,
}: {
  title: string;
  prompts: string[];
  onPick: (p: string) => void;
}) {
  return (
    <div className="max-w-md text-center">
      <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
        <Sparkles className="h-5 w-5" />
      </div>
      <h2 className="mb-1 text-lg font-semibold">{title}</h2>
      <p className="mb-4 text-sm text-muted-foreground">Try one of these to get started</p>
      <div className="flex flex-col gap-2">
        {prompts.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className="rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:border-primary/60 hover:bg-muted"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}
