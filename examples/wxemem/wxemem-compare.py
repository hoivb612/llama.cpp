"""Generate the approved wxemem A/B slide from local JSON reports."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

MIB = 2**20
CATEGORIES = (
    ("Kernel pool", "Paged + nonpaged", "4263EB"),
    ("Kernel + driver code", "Resident", "12A594"),
    ("System cache", "Resident", "EEA62B"),
    ("Modified list", "Dirty pages", "E76F51"),
    ("User mode", "Total WS Process + services", "A4B6CE"),
)


@dataclass(frozen=True)
class Report:
    path: Path
    label: str
    digest: str
    total: int
    used: int
    parts: tuple[int, ...]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def byte_value(data, section, field):
    group = data.get(section)
    if not isinstance(group, dict):
        raise ValueError(f"Missing or invalid object: {section}")
    value = group.get(field)
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ValueError(f"{section}.{field} must be a nonnegative 64-bit integer")
    return value


def short_text(value, maximum, name):
    if not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must contain 1-{maximum} printable characters")
    return value


def infer_label(path):
    match = re.match(r"wxemem_(.+?)_json(?:_|$)", path.stem, re.IGNORECASE)
    if match:
        context = " / ".join((path.parent.parent.parent.name,
                              path.parent.parent.name, path.parent.name))
        flight = " (dev_flight)" if "dev_flight" in path.parts else ""
        return f"{match[1]}{flight} | {context}"
    return path.stem


def load_report(path, label=None):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(data, dict):
        raise ValueError("Report must be a JSON object")
    if data.get("wxemem_version") != "1.1.0":
        raise ValueError("Unsupported wxemem_version; expected 1.1.0. Review its formula before enabling another version.")
    total = byte_value(data, "physical", "total_bytes")
    available = byte_value(data, "physical", "available_bytes")
    used = byte_value(data, "physical", "used_bytes")
    if total == 0 or available > total or used != total - available:
        raise ValueError("Physical RAM is inconsistent: require total > 0 and Used = Total - Available")
    pool = byte_value(data, "kernel", "paged_pool_bytes") + byte_value(data, "kernel", "non_paged_pool_bytes")
    code = byte_value(data, "kernel", "driver_code_bytes") + byte_value(data, "kernel", "system_code_bytes")
    cache = byte_value(data, "kernel", "system_cache_bytes")
    modified = byte_value(data, "memory_list", "modified_bytes")
    residual = used - pool - code - cache - modified
    if residual < 0:
        raise ValueError("Approximate buckets exceed Used RAM; cannot draw a reconciled stack")
    return Report(path, short_text(label if label is not None else infer_label(path), 90, "Report label"),
                  hashlib.sha256(raw).hexdigest(), total, used,
                  (pool, code, cache, modified, residual))


def make_deck(a, b, title):
    short_text(title, 65, "Title")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string("F7F9FC")

    def rect(x, y, w, h, color):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(color)
        shape.line.fill.background()
        return shape

    def text(x, y, w, h, value, size=16, color="182A43", bold=False, align=PP_ALIGN.LEFT):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = shape.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, row in enumerate(value.split("\n")):
            p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            p.text, p.alignment = row, align
            p.font.name, p.font.size, p.font.bold = "Aptos", Pt(size), bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.space_after = Pt(0)

    def line(x1, y1, x2, y2):
        shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        shape.line.color.rgb = RGBColor.from_string("DCE3ED")
        shape.line.width = Pt(1)

    rect(0, 0, 0.14, 7.5, "4263EB")
    text(0.55, 0.35, 12.1, 0.55, title, 30 if len(title) < 50 else 24, bold=True)
    text(0.57, 0.95, 12.1, 0.45, f"A: {a.label}\nB: {b.label}", 11, "586B83")
    line(0.55, 1.48, 12.75, 1.48)
    text(0.60, 1.65, 1.0, 0.3, "RAM (MiB)", 11, "586B83")
    text(1.35, 1.70, 1.90, 0.3, "Breakdown A", 17, bold=True, align=PP_ALIGN.CENTER)
    text(3.45, 1.70, 1.90, 0.3, "Breakdown B", 17, bold=True, align=PP_ALIGN.CENTER)
    bottom, plot_height = 6.03, 3.85
    maximum = max(a.used, b.used) / MIB
    raw_step = max(maximum / 5, 0.2)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(v * magnitude for v in (1, 2, 5, 10) if v * magnitude >= raw_step)
    scale = max(step, maximum * 1.1)
    tick = 0
    while tick <= scale:
        y = bottom - tick / scale * plot_height
        line(1.12, y, 5.65, y)
        text(0.25, y - 0.12, 0.76, 0.25, f"{tick:,.0f}" if step >= 1 else f"{tick:,.1f}",
             10, "677A92", align=PP_ALIGN.RIGHT)
        tick += step
    for x, report, letter in ((1.80, a, "A"), (3.92, b, "B")):
        y = bottom
        for index, amount in enumerate(report.parts):
            h = amount / MIB / scale * plot_height
            y -= h
            if amount:
                shape = rect(x, y, 1.04, h, CATEGORIES[index][2])
                shape.name = f"Stack-{letter}-{index}"
        text(x - 0.43, y - 0.43, 1.90, 0.32, f"{report.used / MIB:,.2f}",
             21, bold=True, align=PP_ALIGN.CENTER)
        text(x - 0.43, 6.13, 1.90, 0.3, f"Report {letter}", 12, "586B83", align=PP_ALIGN.CENTER)

    text(6.15, 1.72, 6.25, 0.4, "Used RAM Contributions", 21, bold=True)
    text(10.20, 2.13, 1.10, 0.25, "A (MiB)", 12, "586B83", align=PP_ALIGN.RIGHT)
    text(11.48, 2.13, 1.10, 0.25, "B (MiB)", 12, "586B83", align=PP_ALIGN.RIGHT)
    for row, index in enumerate(reversed(range(len(CATEGORIES)))):
        name, desc, color = CATEGORIES[index]
        y = 2.40 + row * 0.66
        rect(6.17, y + 0.09, 0.17, 0.36, color)
        text(6.51, y, 3.60, 0.30, name, 17, bold=True)
        text(6.51, y + 0.32, 3.60, 0.25, desc, 11, "586B83")
        for x, report in ((10.20, a), (11.48, b)):
            value = f"{report.parts[index] / MIB:,.2f}"
            text(x, y + 0.03, 1.10, 0.32, value,
                 min(17, 120 / len(value)), bold=True, align=PP_ALIGN.RIGHT)
    delta = a.used - b.used
    percent = f"{delta / b.used:+.1%} vs. B" if b.used else "N/A: B is zero"
    text(6.17, 5.93, 6.45, 0.48,
         f"Used RAM: A - B = {delta / MIB:+,.2f} MiB  ({percent})", 14, "4263EB", True)
    line(0.55, 6.64, 12.75, 6.64)
    footer = (
        "Approximate accounting, not an exact physical-page partition. Paged pool includes nonresident allocations.\n"
        "Process WS sums are excluded (shared-page double counting). Residual is not apps alone. Rounded to 0.01 MiB."
    )
    if a.total != b.total:
        footer += "\nCAUTION: Reports have different physical RAM capacities; compare conditions before attributing changes."
    text(0.57, 6.77, 12.15, 0.62, footer, 10, "586B83")
    notes = [
        "Generated locally: " + datetime.now(timezone.utc).isoformat(),
        f"A: {a.path}\nSHA-256: {a.digest}\nTotal physical bytes: {a.total}",
        f"B: {b.path}\nSHA-256: {b.digest}\nTotal physical bytes: {b.total}",
        "Accounting: wxemem 1.1.0 Memory Breakdown (Approximate).",
        "Used = Total - Available; pool = paged + nonpaged; code = driver + system code.",
        "Residual = Used - pool - code - resident cache - modified list.",
        "Paged pool is total paged pool, NOT resident paged pool. The stack reconciles by construction, not by disjoint page measurement.",
        "Residual includes process pages and other unattributed usage; VBS/Hyper-V/GPU amounts are not separately measured.",
        "Process working-set sums double-count shared pages and must not be added to the stack.",
        "Units: MiB = 2^20 bytes. Delta = A minus B; percent denominator = Used B.",
        "Match hardware, capture phase, runtime conditions and collection settings before drawing causal conclusions.",
    ]
    for letter, report in (("A", a), ("B", b)):
        notes.append(f"{letter} Used bytes: {report.used}")
        notes.extend(f"{letter} {category[0]} bytes: {amount}" for category, amount in zip(CATEGORIES, report.parts))
    slide.notes_slide.notes_text_frame.text = "\n\n".join(notes)
    prs.core_properties.title = title
    prs.core_properties.subject = f"A: {a.label}; B: {b.label}"
    prs.core_properties.author = "wxemem-compare"
    output = BytesIO()
    prs.save(output)
    content = output.getvalue()
    verify_deck(content, a, b, scale, plot_height)
    return content


def verify_deck(content, a, b, scale, plot_height):
    deck = Presentation(BytesIO(content))
    if len(deck.slides) != 1:
        raise ValueError("Generated deck must contain exactly one slide")
    slide = deck.slides[0]
    labels = [s.text for s in slide.shapes if s.has_text_frame]
    names = {c[0] for c in CATEGORIES}
    if [s for s in labels if s in names] != [c[0] for c in reversed(CATEGORIES)]:
        raise ValueError("Category order mismatch")
    for shape in slide.shapes:
        if not (shape.left >= 0 and shape.top >= 0
                and shape.left + shape.width <= deck.slide_width
                and shape.top + shape.height <= deck.slide_height):
            raise ValueError("Generated shape exceeds slide bounds")
    for letter, report in (("A", a), ("B", b)):
        if sum(report.parts) != report.used:
            raise ValueError("Stack does not reconcile")
        for amount in (*report.parts, report.used):
            if f"{amount / MIB:,.2f}" not in labels:
                raise ValueError("Displayed value missing")
        height = sum(s.height for s in slide.shapes if s.name.startswith(f"Stack-{letter}-"))
        expected = Inches(report.used / MIB / scale * plot_height)
        if abs(height - expected) > len(CATEGORIES):
            raise ValueError("Stack scale mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="Report A (JSON or JSON-in-TXT)")
    parser.add_argument("--b", required=True, type=Path, help="Report B; baseline for delta percentage")
    parser.add_argument("--output", required=True, type=Path, help="New .pptx path; never overwrites")
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--title", default="WXE memory footprint comparisons")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output.suffix.lower() != ".pptx":
            raise ValueError("Output must have a .pptx extension")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        a, b = load_report(args.a, args.label_a), load_report(args.b, args.label_b)
        if a.path.samefile(b.path):
            raise ValueError("A and B must be distinct report files")
        content = make_deck(a, b, args.title)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            try:
                stream.write(content)
            except OSError:
                stream.close()
                output.unlink()
                raise
        print(f"Created: {output}")
        for letter, report in (("A", a), ("B", b)):
            print(f"{letter}: {report.label} | Used {report.used / MIB:,.2f} MiB")
        print(f"A - B: {(a.used - b.used) / MIB:+,.2f} MiB")
        if a.total != b.total:
            print("WARNING: A and B have different total physical RAM capacities.", file=sys.stderr)
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
