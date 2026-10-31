"""Writes digits_dataset.pdf and digits_dataset.csv: the scikit-learn digits data used in the T24 project.
Run:  python make_dataset_pdf.py [output folder]"""
import csv, os, sys
import numpy as np
from sklearn.datasets import load_digits
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
AUTHORS = "Saurabh Suresh (24BCE2750) and Advaith Sivakumar (24BCE2715)"
TEAM = "Team 24, Communication Networks, School of Computer Science Engineering (SCOPE), Vellore Institute of Technology"
ROWS_PER_PAGE = 100

d = load_digits()
X, y = d.data.astype(int), d.target.astype(int)
assert X.shape == (1797, 64) and X.min() == 0 and X.max() == 16

with open(os.path.join(OUT, "digits_dataset.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow([f"p{i}" for i in range(64)] + ["label"])
    for xi, yi in zip(X, y):
        w.writerow(list(xi) + [yi])

W, H = landscape(A4)
M = 36
n_pages = 1 + int(np.ceil(len(y) / ROWS_PER_PAGE))
c = canvas.Canvas(os.path.join(OUT, "digits_dataset.pdf"), pagesize=(W, H))
c.setTitle("Digits data set used in the T24 project")
c.setAuthor("Saurabh Suresh and Advaith Sivakumar")
c.setSubject("scikit-learn load_digits(), 1,797 samples")


def footer(page):
    c.setFont("Helvetica", 7.5)
    c.setFillGray(0.35)
    c.drawString(M, 20, f"T24 digits data set  ·  {AUTHORS}")
    c.drawRightString(W - M, 20, f"page {page} of {n_pages}")
    c.setFillGray(0)


# ---------------------------------------------------------------- page 1
yy = H - M - 14
c.setFont("Helvetica-Bold", 17)
c.drawString(M, yy, "Digits data set used in the T24 project")
yy -= 17
c.setFont("Helvetica", 10.5)
c.drawString(M, yy, AUTHORS)
yy -= 13
c.setFont("Helvetica", 8.5)
c.setFillGray(0.3)
c.drawString(M, yy, TEAM)
c.setFillGray(0)
yy -= 20
c.setFont("Helvetica", 8.5)
for line in [
    "Source: scikit-learn load_digits(), a copy of the test set of the UCI 'Optical Recognition of Handwritten Digits' data set "
    "(E. Alpaydin, C. Kaynak, 1998; CC BY 4.0).",
    "1,797 samples. Each sample is an 8 x 8 grey-scale image flattened to 64 pixel values (integers 0-16) with a class label 0-9.",
    "Use in the project: for every seed, 450 random samples are held out for testing and 1,347 are split across 50 devices "
    "(Dirichlet 0.5 label skew). Pixels are divided by 16.",
    "Machine-readable copy: digits_dataset.csv (columns p0 ... p63, label), identical to the values listed from page 2 on.",
]:
    c.drawString(M, yy, line)
    yy -= 12
yy -= 8
c.setFont("Helvetica-Bold", 10)
c.drawString(M, yy, "Samples per class")
yy -= 14
cnt = np.bincount(y, minlength=10)
c.setFont("Courier", 8.5)
c.drawString(M, yy, "digit : " + "".join(f"{k:>6}" for k in range(10)) + "   total")
yy -= 11
c.drawString(M, yy, "count : " + "".join(f"{v:>6}" for v in cnt) + f"{cnt.sum():>8}")
yy -= 22
c.setFont("Helvetica-Bold", 10)
c.drawString(M, yy, "Example images (first three samples of each digit)")
yy -= 16
cell, gap = 5.0, 14
for k in range(10):
    x0 = M + 10 + k * (8 * cell + gap + 16)
    c.setFont("Helvetica", 9)
    c.drawCentredString(x0 + 4 * cell, yy, str(k))
    for r, idx in enumerate(np.where(y == k)[0][:3]):
        img = X[idx].reshape(8, 8)
        top = yy - 8 - r * (8 * cell + 8)
        for i in range(8):
            for j in range(8):
                c.setFillGray(1 - img[i, j] / 16)
                c.rect(x0 + j * cell, top - (i + 1) * cell, cell, cell, stroke=0, fill=1)
        c.setFillGray(0)
yy -= 8 + 3 * (8 * cell + 8) + 10
c.setFont("Helvetica", 8.5)
c.drawString(M, yy, "The following pages list every sample: row number, label, then the 64 pixel values "
                    "(p0 to p63, left to right and top to bottom).")
footer(1)
c.showPage()

# ---------------------------------------------------------------- listing
COLX = [M, M + 22] + [M + 46 + 11.0 * i for i in range(64)]
head = ["row", "lbl"] + [str(i) for i in range(64)]
for p in range(n_pages - 1):
    lo, hi = p * ROWS_PER_PAGE, min(len(y), (p + 1) * ROWS_PER_PAGE)
    top = H - M
    c.setFont("Courier-Bold", 5.6)
    for x, t in zip(COLX, head):
        c.drawRightString(x + 10, top, t)
    c.setLineWidth(0.3); c.line(M, top - 2.5, W - M, top - 2.5)
    c.setFont("Courier", 5.6)
    for r in range(lo, hi):
        yr = top - 9 - (r - lo) * 5.15
        vals = [str(r), str(y[r])] + [str(v) for v in X[r]]
        for x, t in zip(COLX, vals):
            c.drawRightString(x + 10, yr, t)
    footer(p + 2)
    c.showPage()
c.save()
print("written", n_pages, "pages")
