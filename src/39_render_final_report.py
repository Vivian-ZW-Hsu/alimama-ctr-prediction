"""Publication only: render the checked-in Chinese report, never load model/data code."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape
import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font', type=Path, required=True, help='CJK TrueType font with embedding permission (e.g. Arial Unicode.ttf)')
    p.add_argument('--output', type=Path, required=True, help='New PDF path; existing outputs are never overwritten')
    p.add_argument('--render-dir', type=Path, required=True, help='New directory for page PNGs and PDF QA JSON')
    # The pagination check used to hard-code 12. The Chinese report gained sections,
    # so the expected count is now explicit and recorded, instead of a magic number
    # that silently blocks a legitimate content change.
    p.add_argument('--expected-pages', type=int, required=True,
                   help='Exact page count the rendered PDF must have')
    p.add_argument('--min-cjk-per-page', type=int, default=35,
                   help='Minimum CJK characters extracted per page')
    a = p.parse_args()
    if a.output.exists() or a.render_dir.exists():
        raise FileExistsError('Choose fresh publication paths; preserve existing outputs')
    source = ROOT / 'docs/FINAL_REPORT_ZH.md'
    text = source.read_text()
    font = TTFont('CJK', str(a.font))
    pdfmetrics.registerFont(font)
    missing = sorted({c for c in text if ord(c) > 127 and ord(c) not in font.face.charToGlyph})
    assert not missing, f'Font missing characters: {missing}'
    w, h = landscape(A4)
    width = w - 80
    styles = {
        'body': ParagraphStyle('body', fontName='CJK', fontSize=10.8, leading=16, spaceAfter=7, wordWrap='CJK'),
        'title': ParagraphStyle('title', fontName='CJK', fontSize=21, leading=29, spaceAfter=12, textColor=colors.HexColor('#15354A'), wordWrap='CJK'),
        'sub': ParagraphStyle('sub', fontName='CJK', fontSize=13, leading=19, spaceAfter=10, textColor=colors.HexColor('#0072B2'), wordWrap='CJK'),
        'cell': ParagraphStyle('cell', fontName='CJK', fontSize=9.5, leading=13, wordWrap='CJK'),
        'caption': ParagraphStyle('caption', fontName='CJK', fontSize=9.1, leading=13, spaceAfter=7, textColor=colors.HexColor('#435566'), wordWrap='CJK'),
    }
    # The source is Markdown, so inline emphasis must become ReportLab markup
    # rather than reaching the page as literal asterisks. Escaping happens first,
    # so the tags inserted afterwards are the only markup in the string.
    def paragraph(t, style='body'):
        body = escape(t)
        body = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', body)
        # Code spans lose their backticks rather than switching font, which keeps
        # every glyph in the embedded CJK face and avoids mixed-font page checks.
        body = re.sub(r'`([^`]+)`', r'\1', body)
        return Paragraph(body, styles[style])
    story = []
    page_image_count = 0
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
        elif line.startswith('# '):
            story.append(paragraph(line[2:], 'title'))
        elif line.startswith('## '):
            story.append(paragraph(line[3:], 'sub'))
        elif line.startswith('|'):
            rows, raw = [], []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append([paragraph(c, 'cell') for c in cells])
                    raw.append(cells)
                i += 1
            n = len(rows[0])
            ratios = {2: [.10, .90], 3: [.20, .40, .40], 4: [.33, .225, .225, .22],
                      5: [.26, .11, .18, .32, .13], 6: [.22, .13, .15, .22, .15, .13]}[n]
            if n == 2:
                # The default two-column split assumes a narrow label and a wide
                # prose column. A label/number table is the opposite shape, so the
                # split follows the widest cell actually present in each column.
                widest = [max(len(r[c]) for r in raw) for c in range(2)]
                if widest[0] > widest[1]:
                    first = min(0.75, max(0.35, widest[0] / (widest[0] + widest[1])))
                    ratios = [first, 1 - first]
            # Date and comparison tables need a wide second column. An override only
            # applies when its width count matches the table, so a four-width rule
            # can never be attached to a five-column table.
            header = lines[i-len(rows)-1] if i-len(rows)-1 >= 0 else ''
            for marker, override in (('冻结比较', [.15, .43, .21, .21]),
                                     ('北京时间日期', [.15, .30, .20, .35])):
                if marker in header and len(override) == n:
                    ratios = override
            table = Table(rows, colWidths=[width*r for r in ratios], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DFEAF1')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F4F7F9')]),
                ('LINEBELOW', (0,0), (-1,0), .7, colors.HexColor('#ABC0CF')),
                ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            # Keep a table on one page. Splitting one leaves a lone data row at a
            # page foot and a repeated header at the next page head.
            story.extend([KeepTogether(table), Spacer(1,8)])
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*)\]\((.*)\)', line)
            caption, path = match.groups()
            img = Image(str((source.parent/path).resolve()))
            scale = width/img.imageWidth
            # Table+figure pages retain original aspect ratio, with compact charts.
            max_height = 198 if page_image_count in [0, 1] else 277
            scale = min(scale, max_height/img.imageHeight)
            img.drawWidth = img.imageWidth*scale
            img.drawHeight = img.imageHeight*scale
            story.extend([img, paragraph(caption, 'caption')])
            page_image_count += 1
        else:
            story.append(paragraph(line))
        i += 1
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#C8D3DA'))
        canvas.line(40,32,w-40,32)
        canvas.setFont('CJK',8)
        canvas.setFillColor(colors.HexColor('#546677'))
        canvas.drawString(40,20,'CTR 冻结研究报告 · 仅使用已有证据 · Release audit')
        canvas.drawRightString(w-40,20,str(doc.page))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(a.output), pagesize=(w,h), leftMargin=40,rightMargin=40,topMargin=30,bottomMargin=43,
                            title='严格时间切分下的 CTR 预测：最终中文报告',author='CTR project',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pdf = fitz.open(a.output)
    assert len(pdf) == a.expected_pages, (
        f'Unexpected pagination: {len(pdf)} pages instead of {a.expected_pages}')
    a.render_dir.mkdir(parents=True)
    pages = []
    for i, page in enumerate(pdf):
        s = page.get_text()
        assert '\ufffd' not in s
        assert sum('\u4e00' <= c <= '\u9fff' for c in s) > a.min_cjk_per_page
        bad = []
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0: continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0 < 35 or y0 < 10 or x1 > w-35 or y1 > h-10: bad.append(span['text'])
        assert not bad, (i+1, bad)
        embedded = [bool(pdf.extract_font(f[0])[3]) for f in page.get_fonts() if f[2] == 'TrueType']
        assert embedded and all(embedded)
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(a.render_dir/f'page-{i+1:02}.png')
        pages.append({'page':i+1,'extracted_characters':len(s),'embedded_truetype':True,'text_within_page':True})
    record = {'pages':pages,'page_count':len(pdf),'expected_pages':a.expected_pages,
              'min_cjk_per_page':a.min_cjk_per_page,'font_name':a.font.name,'font_sha256':hashlib.sha256(a.font.read_bytes()).hexdigest(),
              'pdf_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'visual_review':'Required separately; automatic geometry/font checks do not replace inspection'}
    (a.render_dir/'pdf_checks.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
