"""PDF report generation (ReportLab).

The report is built exclusively from the stored analysis record and its artifacts,
so it always reflects the actual computed results.
"""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from app.database.models import Analysis
from app.services.storage import ArtifactStore

NAVY = colors.HexColor("#0a1628")
CYAN = colors.HexColor("#00a8c6")
GREY = colors.HexColor("#5b6b80")
REPORT_IMAGE_PX = 900


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=20, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], textColor=GREY, fontSize=9),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12.5),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.8, leading=10, textColor=GREY),
        "caption": ParagraphStyle("c", parent=ss["Normal"], fontSize=8, alignment=1, textColor=GREY),
        "warn": ParagraphStyle("w", parent=ss["Normal"], fontSize=8.5, leading=11.5, textColor=colors.HexColor("#8a4b00")),
    }


def _table(rows, col_widths, header=True):
    t = Table(rows, colWidths=col_widths)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d3df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")]),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def _img(store: ArtifactStore, rec: Analysis, kind: str, width: float):
    art = next((a for a in rec.artifacts if a.artifact_type == kind), None)
    if art is None:
        return Paragraph("(not available)", _styles()["small"])
    path = store.path_for(rec.id, art.stored_filename)
    # Re-encode as a downscaled JPEG: keeps the PDF small even for 2048 px analyses.
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((REPORT_IMAGE_PX, REPORT_IMAGE_PX), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        w, h = im.size
    buf.seek(0)
    return RLImage(buf, width=width, height=width * h / max(w, 1))


def _fmt(v, digits=2):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def build_report(rec: Analysis, store: ArtifactStore) -> bytes:
    st = _styles()
    r = rec.result_json
    buf = io.BytesIO()

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 1.1 * cm, A4[0], 1.1 * cm, fill=1, stroke=0)
        canvas.setFillColor(CYAN)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(1.5 * cm, A4[1] - 0.72 * cm, "PROJECT ORBIT")
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - 1.5 * cm, A4[1] - 0.72 * cm, "Satellite Change Detection Report")
        canvas.setFillColor(GREY)
        canvas.drawString(1.5 * cm, 0.8 * cm, f"Analysis {rec.id}")
        canvas.drawRightString(A4[0] - 1.5 * cm, 0.8 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.5 * cm, title=f"ORBIT report {rec.id}",
                            author="PROJECT ORBIT")
    full_w = A4[0] - 3 * cm
    story = []
    method = r["method"]
    stats = r["statistics"]

    story.append(Paragraph("Change Detection Analysis", st["title"]))
    story.append(Paragraph(
        f"ID <b>{rec.id}</b> &nbsp;|&nbsp; Created {rec.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"&nbsp;|&nbsp; Status <b>{escape(r['status'])}</b>", st["sub"]))
    story.append(Spacer(1, 6))
    tag = ("TRAINED MODEL OUTPUT" if method["is_trained_model"] else "CLASSICAL BASELINE (not a trained AI prediction)")
    story.append(Paragraph(f"<b>Method:</b> {escape(method['label'])} - <b>{tag}</b>", st["body"]))
    if not r.get("reliable", True):
        story.append(Paragraph("<b>This comparison was flagged as UNRELIABLE.</b> See warnings.", st["warn"]))

    # Images ---------------------------------------------------------------------------------
    cw = (full_w - 0.6 * cm) / 2
    grid = Table([
        [_img(store, rec, "before", cw), _img(store, rec, "after", cw)],
        [Paragraph(f"BEFORE - {escape(rec.before_filename)}", st["caption"]),
         Paragraph(f"AFTER (aligned) - {escape(rec.after_filename)}", st["caption"])],
        [_img(store, rec, "heatmap", cw), _img(store, rec, "overlay", cw)],
        [Paragraph("Change intensity heatmap (blue low - red high)", st["caption"]),
         Paragraph(f"Change overlay (threshold {r['threshold']['value']:.3f})", st["caption"])],
    ], colWidths=[cw + 0.3 * cm] * 2)
    grid.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [Spacer(1, 6), grid]

    # Summary ---------------------------------------------------------------------------------
    story.append(Paragraph("Automated summary", st["h2"]))
    for line in r["interpretation"]["summary"]:
        story.append(Paragraph("&bull; " + escape(line), st["body"]))

    # Statistics ----------------------------------------------------------------------------
    area = r["area"]
    rows = [["Metric", "Value"],
            ["Valid compared pixels", _fmt(stats["valid_pixel_count"])],
            ["Changed pixels", _fmt(stats["changed_pixel_count"])],
            ["Changed-pixel percentage (of valid pixels)", f"{stats['changed_pixel_percentage']:.3f} %"],
            ["Connected changed regions", _fmt(stats["region_count"])],
            ["Largest region (pixels)", _fmt(stats["largest_region_px"])],
            ["Mean intensity in changed pixels", _fmt(stats["mean_intensity_changed"], 3)],
            ["Threshold", f"{r['threshold']['value']:.3f} ({r['threshold']['mode']})"],
            ["Processed size", f"{r['processed_size'][0]} x {r['processed_size'][1]} px"],
            ["Approx. physical changed area",
             (f"{area['changed_area_m2']:,.1f} m^2 ({area['changed_area_km2']:.4f} km^2)"
              if area["changed_area_m2"] is not None else "not estimated")],
            ["Inference / total time", f"{r['timing']['inference_ms']:.0f} ms / {r['timing'].get('total_ms', 0):.0f} ms"]]
    story += [Paragraph("Change statistics", st["h2"]), _table(rows, [full_w * 0.55, full_w * 0.45])]
    story.append(Paragraph(escape(area["explanation"]), st["small"]))

    # Method & model ------------------------------------------------------------------------
    story.append(Paragraph("Method and model", st["h2"]))
    mrows = [["Field", "Value"], ["Method type", method["type"]], ["Variant", method["variant"]],
             ["Description", Paragraph(escape(method["description"]), st["body"])]]
    if method.get("model"):
        m = method["model"]
        mrows += [["Model name / version", f"{m.get('name')} / {m.get('version')}"],
                  ["Training dataset (per checkpoint)", str(m.get("dataset"))],
                  ["Epochs trained", _fmt(m.get("epochs_trained"))],
                  ["Checkpoint SHA-256", Paragraph(str(m.get("sha256"))[:64], st["small"])],
                  ["Validation metrics (recorded in checkpoint)",
                   Paragraph(escape(str(m.get("best_val_recorded_in_checkpoint"))), st["small"])]]
    al = r["alignment"]
    mrows += [["Alignment", f"{al.get('method_used')} - {al.get('status')} (valid {al.get('valid_fraction', 0) * 100:.1f}%)"],
              ["Radiometric normalisation", str(al.get("normalization"))]]
    story.append(_table(mrows, [full_w * 0.35, full_w * 0.65]))

    # Interpretation ------------------------------------------------------------------------
    cats = r["interpretation"]["categories"]
    crow = [["Heuristic category", "Regions", "Share of changed pixels"]]
    for v in cats.values():
        crow.append([v["label"], _fmt(v["regions"]), f"{v['share_of_change'] * 100:.1f} %"])
    story.append(KeepTogether([Paragraph("Disaster-intelligence interpretation (heuristic)", st["h2"]),
                               _table(crow, [full_w * 0.6, full_w * 0.15, full_w * 0.25]),
                               Paragraph(escape(r["interpretation"]["notice"]), st["small"])]))

    if r.get("evaluation"):
        e = r["evaluation"]["metrics"]
        erows = [["Metric", "Value"]] + [[k, _fmt(e.get(k), 4)] for k in ("precision", "recall", "f1", "iou", "tp", "fp", "fn", "tn")]
        story += [Paragraph("Evaluation against uploaded ground truth", st["h2"]),
                  _table(erows, [full_w * 0.5, full_w * 0.5])]

    # Warnings & limitations ----------------------------------------------------------------
    story.append(Paragraph("Warnings", st["h2"]))
    if r["warnings"]:
        for w in r["warnings"]:
            story.append(Paragraph("&bull; " + escape(w), st["warn"]))
    else:
        story.append(Paragraph("No warnings were raised.", st["body"]))
    story.append(Paragraph("Limitations", st["h2"]))
    for lim in r["limitations"]:
        story.append(Paragraph("&bull; " + escape(lim), st["small"]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()
