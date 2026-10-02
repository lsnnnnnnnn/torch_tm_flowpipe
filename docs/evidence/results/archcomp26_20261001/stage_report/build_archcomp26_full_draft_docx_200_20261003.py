"""Build the editable ARCH-COMP26 16-section report at the 200-attempt cutoff.

Reads the existing Markdown and coverage overlay. It never starts an experiment
or computes a content digest. The PDF is exported from the DOCX in Word.
"""

import os
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL import Image


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md"
OUT = ROOT / "docs/evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_FULL_STAGE_DRAFT_200_20261003.docx"
FONT = "Hiragino Sans GB"
PAGE_WIDTH_CM = 21
CONTENT_WIDTH_CM = 17.5
INLINE = re.compile(r"(!?\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`)")
IMAGE_LINK = re.compile(r"!?\[([^\]]+)\]\(([^)]+\.png)\)")
NUMBERED_SECTION = re.compile(r"^## (\d{1,2})\. ")
EMBEDDED = set()


def set_font(font, element):
    font.name = FONT
    rpr = element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for kind in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{kind}"), FONT)


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    tag = OxmlElement("w:tblHeader")
    tag.set(qn("w:val"), "true")
    trpr.append(tag)


def avoid_row_split(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def resolved_target(target):
    if re.match(r"^[a-zA-Z]+://", target):
        return target
    base, marker, anchor = target.partition("#")
    source_path = (SOURCE.parent / base).resolve() if base else SOURCE
    relative = os.path.relpath(source_path, OUT.parent)
    return relative + ("#" + anchor if marker else "")


def add_hyperlink(paragraph, label, target):
    rid = paragraph.part.relate_to(resolved_target(target), RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1D4D71")
    rpr.append(color)
    under = OxmlElement("w:u")
    under.set(qn("w:val"), "single")
    rpr.append(under)
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:eastAsia"), FONT)
    fonts.set(qn("w:ascii"), FONT)
    rpr.append(fonts)
    run.append(rpr)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    h.append(run)
    paragraph._p.append(h)


def inline(paragraph, value, *, bold=False):
    """Keep source wording, code spans, emphasis, and evidence links."""
    pos = 0
    for match in INLINE.finditer(value):
        if match.start() > pos:
            r = paragraph.add_run(value[pos:match.start()])
            r.bold = bold
        token = match.group()
        if token.startswith("**"):
            inline(paragraph, token[2:-2], bold=True)
        elif token.startswith("`"):
            r = paragraph.add_run(token[1:-1])
            r.bold = bold
            r.font.name = "Menlo"
            r.font.size = Pt(8)
        else:
            link = re.match(r"!?\[([^\]]+)\]\(([^)]+)\)", token)
            assert link is not None
            add_hyperlink(paragraph, link.group(1), link.group(2))
        pos = match.end()
    if pos < len(value):
        r = paragraph.add_run(value[pos:])
        r.bold = bold


def add_text(doc, value, style=None, *, lead=None):
    p = doc.add_paragraph(style=style)
    if lead:
        run = p.add_run(lead)
        run.bold = True
    inline(p, value)
    return p


def add_figure(doc, label, href):
    path = (SOURCE.parent / href).resolve()
    if not path.is_file() or path in EMBEDDED:
        return
    EMBEDDED.add(path)
    with Image.open(path) as image:
        width, height = image.size
    ratio = width / height
    display_width = min(15.8, 8.6 * ratio)
    display_height = display_width / ratio
    p = doc.add_paragraph(style="Figure")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(display_width), height=Cm(display_height))
    caption = add_text(doc, label + "（保存范围；性质与计时资格见正文）。", "Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER


def maybe_add_figures(doc, line):
    for label, href in IMAGE_LINK.findall(line):
        add_figure(doc, label, href)


def table_cells(line):
    return [part.strip() for part in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def is_divider(row):
    return all(re.fullmatch(r":?-{3,}:?", x) for x in row)


def compact_table(doc, headers, rows):
    cols = len(headers)
    assert 2 <= cols <= 5
    widths = {
        3: [4.8, 6.3, 6.4],
        4: [3.4, 3.7, 5.2, 5.2],
        5: [4.4, 3.275, 3.275, 3.275, 3.275],
    }.get(cols, [8.75, 8.75])
    if cols == 5 and len(rows) == 16:
        widths = [6.3, 2.8, 2.8, 2.8, 2.8]
    table = doc.add_table(rows=1, cols=cols)
    table.autofit = False
    table.style = "Table Grid"
    for i, values in enumerate([headers, *rows]):
        cells = table.rows[0].cells if i == 0 else table.add_row().cells
        for j, value in enumerate(values):
            cell = cells[j]
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shade(cell, "E6EDF2" if i == 0 else ("F8FAFB" if i % 2 == 0 else "FFFFFF"))
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.05
            if j:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            inline(p, value)
            for run in p.runs:
                run.font.size = Pt(7.7 if cols >= 4 else 8.1)
                run.bold = i == 0
        avoid_row_split(table.rows[i])
    repeat_header(table.rows[0])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def long_evidence_records(doc, rows):
    for scope, known, qualification in rows:
        heading = add_text(doc, scope, "Evidence Name")
        heading.paragraph_format.keep_with_next = True
        add_text(doc, known, "Evidence Body", lead="已知状态  ")
        add_text(doc, qualification, "Evidence Body", lead="结果资格  ")


def add_table_block(doc, block):
    values = [table_cells(line) for line in block]
    assert len(values) >= 3 and is_divider(values[1])
    assert all(len(row) == len(values[0]) for row in values)
    headers, rows = values[0], values[2:]
    if len(headers) == 3 and len(rows) > 20:
        long_evidence_records(doc, rows)
    else:
        compact_table(doc, headers, rows)


def initialize(doc):
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(PAGE_WIDTH_CM)
    section.left_margin = section.right_margin = Cm((PAGE_WIDTH_CM - CONTENT_WIDTH_CM) / 2)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.75)
    section.header_distance = Cm(0.7)
    section.footer_distance = Cm(0.7)
    normal = doc.styles["Normal"]
    set_font(normal.font, normal._element)
    normal.font.size = Pt(9)
    normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.space_after = Pt(3)
    for name, size, before, after in (
        ("Title", 17.5, 0, 8),
        ("Heading 1", 12.5, 12, 5),
        ("Heading 2", 10.5, 9, 4),
    ):
        style = doc.styles[name]
        set_font(style.font, style._element)
        style.font.size = Pt(size)
        style.font.bold = name != "Title"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    title_ppr = doc.styles["Title"]._element.get_or_add_pPr()
    for border in title_ppr.findall(qn("w:pBdr")):
        title_ppr.remove(border)
    for name, size, before, after in (
        ("Stage Note", 8.7, 0, 5),
        ("Evidence Name", 9.2, 5, 2),
        ("Evidence Body", 8.7, 0, 3),
        ("Figure", 9, 5, 1),
    ):
        style = doc.styles.add_style(name, 1)
        style.base_style = normal
        set_font(style.font, style._element)
        style.font.size = Pt(size)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
    doc.styles["Evidence Name"].font.bold = True
    doc.styles["Evidence Name"].font.color.rgb = RGBColor(0, 0, 0)
    for name in ("Caption", "List Bullet"):
        style = doc.styles[name]
        set_font(style.font, style._element)
        style.font.size = Pt(8.2 if name == "Caption" else 9)
    doc.styles["Caption"].font.color.rgb = RGBColor(0, 0, 0)
    header = section.header.paragraphs[0]
    header.text = "ARCH COMP26 非 VCAS 四方实验  ·  可审阅阶段稿"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        set_font(run.font, run._element)
        run.font.size = Pt(7.8)
        run.font.color.rgb = RGBColor(0, 0, 0)
    footer = section.footer.paragraphs[0]
    footer.text = "2026 年 10 月 3 日  ·  仍非最终成绩  ·  "
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_page_field(footer)
    for run in footer.runs:
        set_font(run.font, run._element)
        run.font.size = Pt(7.8)
    doc.core_properties.title = "ARCH COMP26 非 VCAS 四方实验阶段报告"
    doc.core_properties.subject = "可审阅 16 节阶段稿"


def build():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    sections = [NUMBERED_SECTION.match(line).group(1) for line in lines if NUMBERED_SECTION.match(line)]
    assert sections == [str(i) for i in range(1, 17)], sections
    assert "200 条本轮尝试" in SOURCE.read_text(encoding="utf-8")
    assert "38 格本轮新全程、8 格同合同旧全程、14 格无全程、4 格 Airplane discrete 合同阻塞" in SOURCE.read_text(encoding="utf-8")
    doc = Document()
    initialize(doc)
    doc.add_paragraph("ARCH COMP26 非 VCAS 四方实验阶段报告", "Title")
    note = add_text(doc, "可审阅阶段稿  ·  200 条本轮尝试截点  ·  2026 年 10 月 3 日  ·  16 个实例 × 4 种方法", "Stage Note")
    note.paragraph_format.space_after = Pt(8)
    add_text(doc, "本稿以 200 条本轮尝试及八格同合同历史全程为截点；64 格中，38 格本轮完成数值时域、8 格经审计可复用旧全程、14 格无完整数值时域、4 格 Airplane 离散合同阻塞。数值全程不自动等于性质证明、独立端到端浮点 NNCS 证书或稳定速度排名。", "Stage Note")
    last_content_line = max(j for j, value in enumerate(lines) if value.strip())
    i = 1  # Replace only the source title; retain all other source content.
    current_numbered_section = None
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table_block(doc, block)
            continue
        if line.startswith("## "):
            numbered = NUMBERED_SECTION.match(line)
            current_numbered_section = int(numbered.group(1)) if numbered else None
            doc.add_heading(line[3:], level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("> "):
            p = add_text(doc, line[2:], "Stage Note")
            p.paragraph_format.left_indent = Cm(0.35)
        elif line.startswith("- "):
            add_text(doc, line[2:], "List Bullet")
            maybe_add_figures(doc, line)
            if current_numbered_section == 14 and line.startswith("- **时间、宽度、图和复现：**"):
                add_figure(
                    doc,
                    "TORA reach-sigmoid 官方 u=11f：四方 500 步 x1/x2 tube 与 T=5 终点差值；Huan/Xiangru 曲线重合",
                    "evidence/results/archcomp26_20261001/tora_reach_sigmoid_u11_fourway_saved_figure_20261002/tora_reach_sigmoid_u11_fourway_saved.png",
                )
            if current_numbered_section == 15 and line.startswith("- **时间、宽度、图和复现：**"):
                add_figure(
                    doc,
                    "TORA reach-tanh 历史四方 x1/x2 保存 tube；旧 P3 引擎不是当前 working P3",
                    "evidence/results/archcomp26_20261001/tora_reach_tanh_historical_fourway_saved_20261002/tora_reach_tanh_u11_historical_fourway_saved.png",
                )
            if current_numbered_section == 16 and line.startswith("- **时间、宽度、图和复现：**"):
                add_figure(
                    doc,
                    "Unicycle 论文常值 w：四方 x1/x3 整步 tube 与 T=10 终点盒；目标只在终点判定",
                    "evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/unicycle_paper_speed_fourway_saved.png",
                )
        elif line.startswith("!["):
            match = IMAGE_LINK.search(line)
            if match:
                add_figure(doc, match.group(1), match.group(2))
            else:
                add_text(doc, line)
        else:
            paragraph = add_text(doc, line)
            if i == last_content_line:
                for run in paragraph.runs:
                    run.font.size = Pt(8.8)
            maybe_add_figures(doc, line)
            if current_numbered_section == 13 and line.startswith("**另列数值步长变体：**"):
                add_figure(
                    doc,
                    "TORA remain h=0.05：四方 12 盒、400 步 x4 整步 tube；与固定 h=0.1 主表分列",
                    "evidence/results/archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/tora_remain_h005_fourway_t_x4.png",
                )
        i += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"DOCX {OUT}")
    print(f"Numbered sections {len(sections)}; embedded figures {len(EMBEDDED)}")


if __name__ == "__main__":
    build()
