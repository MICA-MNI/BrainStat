.. _python_reporting:

Display-free SLM reports
========================

For Colab, SSH notebooks and machines without a display, export a fitted SLM
as a self-contained HTML report. The report uses Nilearn's browser surface
viewer with embedded JavaScript. No VTK render window, Xvfb, EGL or OSMesa is
required. Geometry supplied as BrainSpace objects still uses VTK data objects;
rendering happens in the browser. No external viewer code is vendored.

.. code-block:: python

    from brainstat.reporting import make_slm_report

    report = make_slm_report(
        slm,
        filename="slm_report.html",
        surface="fsaverage5",
        maps=["t", "q"],
    )

The same helper is available as ``slm.to_html_report(...)``. Both return a
``pathlib.Path``. Open the file in a browser, or use a notebook link:

.. code-block:: python

    from IPython.display import FileLink, display
    display(FileLink(str(report)))

Surface and map inputs
----------------------

Use a BrainStat template name, ``(surf_lh, surf_rh)``, or a dictionary with
``lh`` and ``rh`` surface objects. Without an explicit surface, the helper uses
the SLM template or combined surface. A combined surface must have equal-size
hemispheres ordered left then right with no triangles crossing the split;
pass an explicit pair for unequal hemispheres.

Maps must be univariate and contain one value per vertex. Expand parcellated
maps to vertices before exporting. Available map names are:

* ``t``: fitted t-statistics.
* ``p``: RFT-corrected vertex p-values, when present in ``slm.P``.
* ``cluster_p``: RFT-corrected cluster p-values, when available.
* ``q``: FDR q-values, when present in ``slm.Q``.

By default every available map is included. An explicitly requested missing
map raises an error. P/q values are unthresholded and use a color range of
0 to 1; ``color_range="sym"`` or a numeric pair overrides the range. Excluded
vertices and missing values are represented as NaN. The report records the
contrast, design size, correction settings, mask coverage and package versions.

Static plotting and QC
----------------------

The existing ``plot_hemispheres`` and ``SLM.qc`` workflows remain available for
local static plots. HTML export contains fitted surface maps and provenance;
it does not add residual-QC figures or volumetric rendering.
