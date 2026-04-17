import sys
import fitz
doc = fitz.open('../backup/Cópia Banner.pptx.pdf')
with open('parsed_pdf.txt', 'w', encoding='utf-8') as f:
    for page in doc:
        f.write(page.get_text())
