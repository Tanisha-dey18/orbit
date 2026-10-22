import { Satellite } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import type { BackendState } from "../hooks/useBackend";

const links = [
  { to: "/", label: "Overview", end: true },
  { to: "/analyze", label: "Analyze" },
  { to: "/history", label: "History" },
  { to: "/methodology", label: "Methodology" },
];

function StatusDot({ state }: { state: BackendState }) {
  if (state.kind === "loading") return <span className="label">connecting…</span>;
  if (state.kind === "offline")
    return (
      <span className="flex items-center gap-1.5 font-mono text-[11px] text-change" title={state.error}>
        <span className="h-1.5 w-1.5 rounded-full bg-change" /><span className="hidden sm:inline">API offline</span>
      </span>
    );
  const ready = state.health.model.ready;
  return (
    <span className="flex items-center gap-3 font-mono text-[11px]">
      <span className="flex items-center gap-1.5 text-ok"><span className="h-1.5 w-1.5 rounded-full bg-ok" /><span className="hidden sm:inline">API online</span></span>
      <span className={`hidden items-center gap-1.5 sm:flex ${ready ? "text-violet" : "text-dim"}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-violet" : "bg-dim"}`} />
        {ready ? "model loaded" : "no DL model"}
      </span>
    </span>
  );
}

export default function Layout({ backend }: { backend: BackendState }) {
  return (
    <div className="grid-bg min-h-screen">
      <header className="sticky top-0 z-30 border-b border-line bg-space-950/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-3 px-4 sm:gap-6">
          <NavLink to="/" className="flex shrink-0 items-center gap-2" aria-label="PROJECT ORBIT home">
            <Satellite className="h-5 w-5 text-cyan" />
            <span className="whitespace-nowrap font-display text-[15px] font-bold tracking-[0.18em]">
              <span className="hidden sm:inline">PROJECT </span><span className="text-cyan">ORBIT</span>
            </span>
          </NavLink>
          <nav className="flex min-w-0 items-center gap-1 overflow-x-auto">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                className={({ isActive }) =>
                  `whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors ${isActive ? "bg-cyan/10 text-cyan" : "text-muted hover:text-ink"}`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto shrink-0"><StatusDot state={backend} /></div>
        </div>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="border-t border-line py-6 text-center font-mono text-[11px] text-dim">
        PROJECT ORBIT · satellite change detection for research &amp; education · results require expert verification
      </footer>
    </div>
  );
}
