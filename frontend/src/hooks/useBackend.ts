import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api";
import type { Health, ModelsResponse } from "../types/api";

export type BackendState =
  | { kind: "loading" }
  | { kind: "offline"; error: string }
  | { kind: "online"; health: Health; models: ModelsResponse };

/** Polls backend health + model availability. */
export function useBackend(pollMs = 30000) {
  const [state, setState] = useState<BackendState>({ kind: "loading" });

  const refresh = useCallback(async () => {
    try {
      const [health, models] = await Promise.all([api.health(), api.models()]);
      setState({ kind: "online", health, models });
    } catch (e) {
      setState({ kind: "offline", error: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, pollMs);
    return () => window.clearInterval(t);
  }, [refresh, pollMs]);

  return { state, refresh };
}
