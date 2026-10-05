#!/usr/bin/env python3
"""Python ReportLab chart from saved driver timings; no numerical experiments."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import hashlib, subprocess, tempfile
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from reportlab.pdfbase.pdfdoc import PDFDocument
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/results/archcomp26_report_20261005/figures'
OUT.mkdir(exist_ok=True)
rows=[('ACC',4.720203388,3.981541836),('NAV robust',15.531658911,12.666988223),('QUAD',1350.525528709,997.675007039),('TORA sigmoid',9.793604707,5.255259025),('Unicycle',13.617203456,7.652434832)]
def prohibited(*a,**kw): raise RuntimeError('Content digest operations prohibited')
with ExitStack() as stack:
 for name in (*hashlib.algorithms_guaranteed,'new','file_digest'):stack.enter_context(patch.object(hashlib,name,prohibited))
 # A fixed document ID is metadata only; suppress ReportLab's default digest-based ID.
 stack.enter_context(patch.object(PDFDocument,'ID',lambda self: b'\n[<30303030303030303030303030303030><30303030303030303030303030303030>]\n'))
 d=Drawing(850,280)
 d.add(String(425,264,'Five complete P3 candidates',fontName='Helvetica-Bold',fontSize=15,textAnchor='middle'))
 d.add(String(425,244,'Driver elapsed in seconds | one new process each | saved widths unchanged',fontSize=10,textAnchor='middle'))
 for i,(name,old,new) in enumerate(rows):
  x=24+i*169
  chart=VerticalBarChart();chart.x=x+24;chart.y=50;chart.width=110;chart.height=143;chart.data=[(old,new)]
  chart.categoryAxis.categoryNames=['Saved','New'];chart.categoryAxis.labels.fontSize=9
  chart.valueAxis.valueMin=0;chart.valueAxis.valueMax=old*1.25;chart.valueAxis.valueStep=old/4
  chart.valueAxis.labels.fontSize=7;chart.valueAxis.labelTextFormat=lambda v:f'{v:.1f}'
  chart.bars[(0,0)].fillColor=colors.HexColor('#8797a4');chart.bars[(0,1)].fillColor=colors.HexColor('#246c90');chart.bars.strokeWidth=0
  chart.barLabels.fontSize=9;chart.barLabels.nudge=6;chart.barLabelFormat='%.3f'
  d.add(chart);d.add(String(x+78,220,name,fontName='Helvetica-Bold',fontSize=11,textAnchor='middle'))
  d.add(String(x+78,203,f'{100*(old-new)/old:.1f}% lower',fontSize=10,textAnchor='middle'))
 d.add(String(425,17,'Different subplot scales. Reference and candidate use the same internal timing boundary.',fontSize=9,textAnchor='middle'))
 with tempfile.TemporaryDirectory(prefix='archcomp26-chart-') as td:
  pdf=Path(td)/'chart.pdf';renderPDF.drawToFile(d,str(pdf))
  subprocess.run(['/Users/shengenli/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm','-png','-singlefile','-r','170',str(pdf),str(OUT/'driver_before_after')],check=True,capture_output=True)
print('Created Python chart from saved driver times')
