import { Brain, Cpu, ShieldAlert } from "lucide-react";
import type { AnalysisResult } from "../types/api";
import { Badge } from "./ui";

/** Always makes it obvious whether a result came from the classical baseline or a trained model. */
export default function MethodBadge({ result }: { result: AnalysisResult }) {
  const m = result.method;
  const dataset = (m.model?.dataset as string | undefined) ?? "";
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {m.is_trained_model ? (
        <Badge tone="violet"><Brain className="h-3 w-3" /> Deep model output</Badge>
      ) : (
        <Badge tone="cyan"><Cpu className="h-3 w-3" /> Classical baseline · not AI</Badge>
      )}
      {dataset.startsWith("synthetic") && <Badge tone="warn">synthetic-trained checkpoint</Badge>}
      {result.status === "unreliable" ? (
        <Badge tone="change"><ShieldAlert className="h-3 w-3" /> unreliable comparison</Badge>
      ) : result.status === "completed_with_warnings" ? (
        <Badge tone="warn">completed with warnings</Badge>
      ) : (
        <Badge tone="ok">completed</Badge>
      )}
    </div>
  );
}
