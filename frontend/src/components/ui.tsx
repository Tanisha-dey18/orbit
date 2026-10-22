import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from "lucide-react";
import type { ReactNode } from "react";

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return <Loader2 className={`${className} animate-spin`} aria-hidden />;
}

export function Panel({ title, icon, children, className = "", actions }: {
  title?: string; icon?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode;
}) {
  return (
    <section className={`panel p-4 ${className}`}>
      {title && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h2 className="flex items-center gap-2 label !text-muted">{icon}{title}</h2>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

type Tone = "info" | "warn" | "error" | "ok";
const toneStyle: Record<Tone, string> = {
  info: "border-cyan/30 bg-cyan/5 text-cyan",
  warn: "border-warn/30 bg-warn/5 text-warn",
  error: "border-change/40 bg-change/10 text-change",
  ok: "border-ok/30 bg-ok/5 text-ok",
};
const toneIcon: Record<Tone, ReactNode> = {
  info: <Info className="h-4 w-4 shrink-0" />,
  warn: <AlertTriangle className="h-4 w-4 shrink-0" />,
  error: <XCircle className="h-4 w-4 shrink-0" />,
  ok: <CheckCircle2 className="h-4 w-4 shrink-0" />,
};

export function Notice({ tone = "info", title, children }: { tone?: Tone; title?: string; children?: ReactNode }) {
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`flex gap-2.5 rounded-lg border px-3 py-2.5 text-sm ${toneStyle[tone]}`}>
      <span className="mt-0.5">{toneIcon[tone]}</span>
      <div className="min-w-0">
        {title && <p className="font-medium">{title}</p>}
        {children && <div className="text-ink/80 break-words">{children}</div>}
      </div>
    </div>
  );
}

export function Badge({ children, tone = "cyan" }: { children: ReactNode; tone?: "cyan" | "violet" | "warn" | "change" | "ok" | "muted" }) {
  const map = {
    cyan: "border-cyan/40 text-cyan bg-cyan/10",
    violet: "border-violet/40 text-violet bg-violet/10",
    warn: "border-warn/40 text-warn bg-warn/10",
    change: "border-change/40 text-change bg-change/10",
    ok: "border-ok/40 text-ok bg-ok/10",
    muted: "border-line text-muted bg-space-800",
  } as const;
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${map[tone]}`}>
      {children}
    </span>
  );
}

export function Stat({ label, value, hint, accent = false }: { label: string; value: ReactNode; hint?: ReactNode; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-space-900/60 px-3 py-2.5">
      <p className="label">{label}</p>
      <p className={`mt-1 font-display text-xl font-semibold tabular-nums ${accent ? "text-cyan" : "text-ink"}`}>{value}</p>
      {hint && <p className="mt-0.5 text-[11px] leading-snug text-dim">{hint}</p>}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="rounded-full border border-line bg-space-800 p-3 text-cyan">{icon}</div>
      <p className="font-display text-lg text-ink">{title}</p>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Button({ children, variant = "secondary", className = "", ...rest }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "danger" | "ghost" }) {
  const v = {
    primary: "bg-cyan text-space-950 hover:bg-cyan/90 font-semibold shadow-[0_0_24px_-6px_rgb(34_211_238/0.6)]",
    secondary: "border border-line bg-space-800 text-ink hover:border-cyan/50 hover:text-cyan",
    danger: "border border-change/40 bg-change/10 text-change hover:bg-change/20",
    ghost: "text-muted hover:text-cyan",
  }[variant];
  return (
    <button
      {...rest}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${v} ${className}`}
    >
      {children}
    </button>
  );
}
