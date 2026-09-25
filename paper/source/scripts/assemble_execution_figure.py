"""Rearrange the supplied vector panels without recalculating plotted values."""
from pathlib import Path
import fitz
import cairosvg
p=Path(__file__).resolve().parents[1]/"figures/narrative"
source=fitz.open(stream=cairosvg.svg2pdf(bytestring=(p/"figure6_source_edited.svg").read_bytes()),filetype="pdf")
out=fitz.open(); page=out.new_page(width=514,height=380)
page.insert_text((70,10), "Perceived PT delay: +15 min; physical timetable unchanged", fontsize=9, fontname="hebo")
page.show_pdf_page(fitz.Rect(0,20,514,203.5),source,0,clip=fitz.Rect(0,0,514,183.5))
page.show_pdf_page(fitz.Rect(124,201,390,380),source,0,clip=fitz.Rect(0,189,264.5,370.001027))
out.save(p/"figure6_execution_response.pdf",garbage=4,deflate=True)
speed=fitz.open();page=speed.new_page(width=260,height=190)
page.show_pdf_page(page.rect,source,0,clip=fitz.Rect(261,189,513.673906,370.001027),keep_proportion=True)
speed.save(p/"figureS_speed_sensitivity.pdf",garbage=4,deflate=True)
