---
layout: article
title: Jupyter notebook to slides with hidden code
categories: blog
modified: 2019-01-19
tags: [jupyter]
---

How to export a JupyterLab notebook to a slide presentation, without showing the code.

From time to time I need to make a presentation out of an analysis I have been working on in a Jupyter notebook. People who sit at these meetings are not really interested in the code, so I needed a way of hiding it while turning markdown comments and figures into a presentation. Figuring this out required some heavy googling, so I thought about summarizing it here.

A general guide (for the original jupyter notebook environment) can be found in this article: [Presenting Code Using Jupyter Notebook Slides](https://medium.com/@mjspeck/presenting-code-using-jupyter-notebook-slides-a8a3c3b59d67). In JupyterLab, each cell rendering for the presentation is controlled from the `Cell Inspector` tab in the left sidebar. If you want to hide the code of a given cell, add `"tags": ["to_remove"],` to the cell's metadata, that should look like this (for a fragment cell for instance):

```
{
    "tags": [
        "to_remove"
    ],
    "slideshow": {
        "slide_type": "fragment"
    }
}
```

Unfortunately, this has to be done for each cell. I haven't found a way to simultaneously edit the metadata of multiple cells.

Once done with formatting the metadata of all cells, build the html presentation with:

```
jupyter nbconvert presentation.ipynb --to slides --no-prompt --TagRemovePreprocessor.remove_input_tags={\"to_remove\"} --post serve --SlidesExporter.reveal_theme=simple
```

Seen in [ Edit meta-data to suppress code input and cell numbers for reveal.js](https://github.com/jupyterlab/jupyterlab/issues/4100#issuecomment-370938358).
