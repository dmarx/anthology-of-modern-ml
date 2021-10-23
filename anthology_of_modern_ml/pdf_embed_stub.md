---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Read this PDF

This thing is amazing and you should read it.

```{code-cell} ipython3
:tags: [hide-input]

import panel as pn
pn.extension()
pdf_pane = pn.pane.PDF('https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf', width=700, height=1000)
pdf_pane
```
