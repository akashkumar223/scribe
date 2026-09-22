"""
Generates professional PDF reports for document summaries, dataset analytics,
and meeting/transcript reports. Uses reportlab (free, no external service).
"""
import io
import matplotlib
matplotlib.use("Agg")  # no display needed, just render to image files
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20, spaceAfter=4)
SUBTITLE_STYLE = ParagraphStyle("SubtitleStyle", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=20)
H2_STYLE = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=13, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#2C4A7C"))
BODY_STYLE = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=10.5, leading=16)
BULLET_STYLE = ParagraphStyle("BulletStyle", parent=styles["Normal"], fontSize=10.5, leading=16, leftIndent=14, bulletIndent=4)


def _text_to_paragraphs(text: str, style=BODY_STYLE):
    """Split text on newlines into separate Paragraph flowables (keeps line breaks sane)."""
    flowables = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 6))
            continue
        # Turn "- " or "* " prefixed lines into bullets
        if line.startswith(("- ", "* ")):
            flowables.append(Paragraph(f"&bull; {line[2:]}", BULLET_STYLE))
        else:
            flowables.append(Paragraph(line, style))
    return flowables


import re


def _markdown_line_to_flowable(line: str):
    """
    Converts one line of lightweight markdown (as commonly returned by the LLM)
    into a reportlab flowable: headings, bold/italic, bullets, table-ish rows.
    """
    line = line.rstrip()

    if not line.strip():
        return Spacer(1, 6)

    if line.strip() == "---":
        return Spacer(1, 4)

    # Headings: #, ##, ###
    heading_match = re.match(r"^(#{1,4})\s+(.*)", line.strip())
    if heading_match:
        text = heading_match.group(2)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        return Paragraph(text, H2_STYLE)

    # Bullets: "- " or "* "
    stripped = line.strip()
    if stripped.startswith(("- ", "* ")):
        content = stripped[2:]
        content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
        content = re.sub(r"\*(.+?)\*", r"<i>\1</i>", content)
        return Paragraph(f"&bull; {content}", BULLET_STYLE)

    # Table-ish rows (markdown pipes) — render as plain readable text, not a real table
    if "|" in stripped and stripped.startswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} for c in cells if c):
            return Spacer(1, 2)  # skip separator rows like |---|---|
        content = "  •  ".join(cells)
        return Paragraph(content, BODY_STYLE)

    # Regular paragraph with inline bold/italic
    content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
    content = re.sub(r"\*(.+?)\*", r"<i>\1</i>", content)
    return Paragraph(content, BODY_STYLE)


def build_generic_text_pdf(title: str, raw_text: str) -> bytes:
    """
    Exports any block of text (e.g. a chat answer) as a formatted PDF,
    rendering lightweight markdown (headings, bold, bullets) properly.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = [Paragraph(title, TITLE_STYLE), Spacer(1, 10)]
    for line in raw_text.split("\n"):
        story.append(_markdown_line_to_flowable(line))
    doc.build(story)
    return buf.getvalue()


def build_document_summary_pdf(filename: str, summary_text: str) -> bytes:
    """Professional one-document summary report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = [
        Paragraph("Document Summary Report", TITLE_STYLE),
        Paragraph(f"Source: {filename}", SUBTITLE_STYLE),
        Paragraph("Summary", H2_STYLE),
    ]
    story.extend(_text_to_paragraphs(summary_text))
    doc.build(story)
    return buf.getvalue()


def _make_chart_image(chart_data: dict) -> io.BytesIO:
    """Render a chart_data dict (from analytics.py) as a PNG image for embedding in the PDF."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(chart_data["labels"], chart_data["values"], color="#3E6B52")
    ax.set_title(f"Distribution — {chart_data['column']}", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=150)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf


def build_analytics_report_pdf(filename: str, analytics: dict) -> bytes:
    """Full end-to-end analytics report: overview, quality score, insights, correlations, chart, column table."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = [
        Paragraph("Dataset Analytics Report", TITLE_STYLE),
        Paragraph(f"Source: {filename}", SUBTITLE_STYLE),
    ]

    story.append(Paragraph("Overview", H2_STYLE))
    overview_table = Table([
        ["Rows", str(analytics["shape"]["rows"])],
        ["Columns", str(analytics["shape"]["columns"])],
    ], colWidths=[6*cm, 6*cm])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F2F5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(overview_table)

    sb = analytics.get("summary_board")
    if sb:
        story.append(Paragraph(f"Data Quality Score: {sb['quality_score']} / 100", H2_STYLE))
        story.append(Paragraph("Key Insights", H2_STYLE))
        for insight in sb["insights"]:
            story.append(Paragraph(f"&bull; {insight}", BULLET_STYLE))

        if sb.get("top_correlations"):
            story.append(Paragraph("Strongest Relationships", H2_STYLE))
            rows = [["Column A", "Column B", "Correlation (r)"]]
            for p in sb["top_correlations"]:
                rows.append([p["col_a"], p["col_b"], str(p["r"])])
            t = Table(rows, colWidths=[5*cm, 5*cm, 4*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C4A7C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t)

        if sb.get("outliers"):
            story.append(Paragraph("Potential Outliers", H2_STYLE))
            rows = [["Column", "Outlier Count"]]
            for o in sb["outliers"]:
                rows.append([o["column"], str(o["outlier_count"])])
            t = Table(rows, colWidths=[7*cm, 5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B3A2E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]))
            story.append(t)

    if analytics.get("chart_data"):
        story.append(Paragraph("Distribution Chart", H2_STYLE))
        img_buf = _make_chart_image(analytics["chart_data"])
        story.append(Image(img_buf, width=15*cm, height=7.5*cm))

    story.append(Paragraph("Column Overview", H2_STYLE))
    rows = [["Column", "Type", "Unique", "Missing %"]]
    for c in analytics["columns"]:
        rows.append([c["name"], c["dtype"], str(c["unique"]), f"{c['missing_pct']}%"])
    t = Table(rows, colWidths=[5*cm, 3*cm, 3*cm, 3*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C4A7C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    doc.build(story)
    return buf.getvalue()


def build_transcript_report_pdf(filename: str, transcript: dict) -> bytes:
    """Full meeting/interview report: brief, summary, MOM, sentiment, and duration/language metadata."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = [
        Paragraph("Meeting / Interview Report", TITLE_STYLE),
        Paragraph(f"Source: {filename}", SUBTITLE_STYLE),
    ]

    meta_rows = [
        ["Duration", f"{round(transcript['duration'])}s" if transcript.get("duration") else "Live session"],
        ["Language", transcript.get("language", "—")],
    ]
    if transcript.get("sentiment"):
        meta_rows.append(["Overall Sentiment", transcript["sentiment"].get("overall", "—")])
    meta_table = Table(meta_rows, colWidths=[5*cm, 8*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F2F5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)

    story.append(Paragraph("Brief", H2_STYLE))
    story.extend(_text_to_paragraphs(transcript.get("brief", "")))

    story.append(Paragraph("General Summary", H2_STYLE))
    story.extend(_text_to_paragraphs(transcript.get("summary", "")))

    if transcript.get("sentiment"):
        story.append(Paragraph("Sentiment Analysis", H2_STYLE))
        story.extend(_text_to_paragraphs(transcript["sentiment"].get("explanation", "")))

    story.append(Paragraph("Minutes of Meeting", H2_STYLE))
    story.extend(_text_to_paragraphs(transcript.get("mom", "")))

    story.append(PageBreak())
    story.append(Paragraph("Full Transcript", H2_STYLE))
    story.extend(_text_to_paragraphs(transcript.get("text", "")))

    doc.build(story)
    return buf.getvalue()