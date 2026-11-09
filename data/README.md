# data

Local input and output files. Everything here is ignored by git except this
file. To track a small essential file, add an explicit exception to
`.gitignore` and record its origin below.

## Tracked files

Bin contents from the Super-Kamiokande DSNB paper (arXiv:2109.11174v2), recovered
with PyMuPDF from the vector graphics of its figures and calibrated on the axis
ticks. They are approximate copies of published plots, not collaboration data.

| File | Source | Content |
|---|---|---|
| `sk4_fig24_spectrum.csv` | Fig. 24 | Backgrounds, data and DSNB, events per 2 MeV |
| `sk4_fig18_efficiency.csv` | Fig. 18, top | Cumulative signal efficiency in percent, one tagged neutron |
| `sk4_fig19_candidates.csv` | Fig. 19 | Data events per 2 MeV after successive reduction steps |
