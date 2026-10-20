"""One-page customer PDF for the account manager (English / Latin text only)."""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DEC_COLORS = {"monitor": "#1e90ff", "act": "#1f9d55", "review": "#d98e04",
              "distrust": "#d64545", "no_roi": "#6b7a90"}


def _money(v):
    return ("-" if v < 0 else "") + "$" + format(abs(round(v)), ",")


def _waterfall_png(info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [info["story_base_label"]] + [n for n, _ in info["story"]] + [info["story_final_label"]]
    base = info["base_risk"]
    vals = [v for _, v in info["story"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=170)
    cum = base
    ax.barh(0, base, color="#1e90ff")
    ax.text(base + 0.6, 0, "%.0f%%" % base, va="center", fontsize=7)
    for i, v in enumerate(vals, start=1):
        left = cum if v >= 0 else cum + v
        ax.barh(i, abs(v), left=left, color="#d64545" if v >= 0 else "#1f9d55")
        ax.text(max(cum, cum + v) + 0.6, i, "%+.0f" % v, va="center", fontsize=7)
        cum += v
    n = len(vals) + 1
    ax.barh(n, info["p"] * 100, color="#1e90ff")
    ax.text(info["p"] * 100 + 0.6, n, "%.0f%%" % (info["p"] * 100), va="center", fontsize=7)
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(100, info["p"] * 100 + 8, cum + 8))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("churn risk (%)", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(info):
    """info: dict prepared in app.py (all strings English). Returns PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=11 * mm, bottomMargin=10 * mm,
                            title="ChurnLens customer report", author="Team Visionaries")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=20, alignment=0, spaceAfter=0)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=8, textColor=colors.HexColor("#6b7a90"))
    h2 = ParagraphStyle("h2", parent=ss["Heading3"], fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2,
                        textColor=colors.HexColor("#0b3d75"))
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=8.6, leading=11)
    small = ParagraphStyle("small", parent=body, fontSize=7.2, leading=9, textColor=colors.HexColor("#6b7a90"))
    white = ParagraphStyle("white", parent=body, textColor=colors.white, fontSize=9.5, leading=12)

    el = [Paragraph("ChurnLens: customer report", h1),
          Paragraph("Team Visionaries &middot; Explainable, actionable, trust-scored churn", sub), Spacer(1, 4)]

    # headline numbers
    head = [["Churn risk", "Risk level", "Trust score", "Revenue at risk (%d mo)" % info["horizon"]],
            ["%d%%" % round(info["p"] * 100), info["level"], "%d / 100 (%s)" % (info["trust"], info["trust_level"]),
             _money(info["rar"])]]
    t = Table(head, colWidths=[40 * mm, 40 * mm, 50 * mm, 52 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4fb")),
        ("FONTSIZE", (0, 0), (-1, 0), 7.5), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#4a5b73")),
        ("FONTSIZE", (0, 1), (-1, 1), 12), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d6e6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde6f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    el += [t, Spacer(1, 5)]

    # decision box
    box = Table([[Paragraph("<b>DECISION: %s</b><br/>%s" % (info["dec_name"], info["dec_body"]), white)]],
                colWidths=[182 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(DEC_COLORS[info["dec"]])),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    el += [box]

    # profile
    el += [Paragraph("Customer snapshot", h2), Paragraph(info["profile"], body)]

    # story chart
    el += [Paragraph("Why this risk? (average customer to this customer)", h2),
           Image(_waterfall_png(info), width=170 * mm, height=59 * mm)]

    # offers
    el += [Paragraph("Retention offers, ranked by net gain", h2)]
    rows = [["Offer", "New risk", "Risk drop", "Offer cost", "Revenue saved", "Net gain"]]
    for r in info["rows"][:4]:
        rows.append([r["label"], "%d%%" % round(r["new_p"] * 100), "-%d pts" % round(r["delta"] * 100),
                     _money(r["cost"]), _money(r["saved"]), _money(r["net"])])
    if len(rows) == 1:
        rows.append(["No applicable offer", "", "", "", "", ""])
    ot = Table(rows, colWidths=[66 * mm, 20 * mm, 22 * mm, 24 * mm, 26 * mm, 24 * mm])
    ot.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d75")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d6e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    el += [ot]

    # trust + fairness + message
    el += [Paragraph("Trust and fairness", h2),
           Paragraph(info["trust_line"], body), Paragraph(info["fair_line"], body),
           Paragraph("Suggested message", h2), Paragraph(info["sms"], body), Spacer(1, 4),
           Paragraph(info["footnote"], small)]
    doc.build(el)
    return buf.getvalue()
