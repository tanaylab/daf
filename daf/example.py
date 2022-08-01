"""
Example data for demonstrating the ``daf`` package.

This example data is in the `.files` storage format and is included as part of the ``daf`` distribution. The path to the
example directory is provided in `.DAF_EXAMPLE_PATH`.

This is a very small data to serve as a basis for executable code examples in the documentation. Real data sets would
typically be much larger and may contain completely different types of data (axes and properties); we picked single-cell
RNA sequencing (scRNA-seq) data as an example as this is the problem domain we originally developed ``daf`` for. The
example data set contains the following:

``created`` **0D data**
    The date this example data was created.

``batch#`` **axis**
    One entry per batch of cells. The names of the example batches are simply ``Batch_x`` where ``x`` is between 0 and
    12 (a total of 13 batches). Real scRNA-seq data sets often combine cells from different such "batches" (e.g., taken
    at different times and/or from different donors). The batch names are typically more descriptive, and are often used
    as keys to external databases with additional per-batch data.

``batch#age`` **1D data**
    For each batch, the age in years of the donor the batch of cells was obtained from.

``batch#sex`` **1D data**
    For each batch, either ``male`` or ``female`` depending on the sex of the donor the batch of cells was obtained
    from.

 ``sex#`` **axis**
    An axis with two entries, ``male`` and ``female``.

``cell#`` **axis**
    One entry per each sequenced cell. The names of the example entries are simply ``Cell_x`` where ``x`` is between 0
    and 523 (a total of 524 cells). Typically in real scRNA-seq data sets there would be anywhere between tens of
    thousands and millions of cells, each uniquely identified by some ``ACTG`` based barcode combined with additional
    data to make the name unique in the data set (e.g. the batch name).

``cell#batch`` **1D data**
    For each cell, the 0-based index of the batch it was obtained from.

``gene#`` **axis**
    One entry per each sequenced gene. The names of the entries are gene names, e.g. ``FOXA1``. The example data
    contains just 10 genes. Typically in a real scRNA-seq data set there would be tens of thousands of genes.

``cell,gene#UMIs`` **2D data**
    For each gene of each cell, the number of observed unique molecular identifiers (UMIs). This serves as an estimate
    of the number of mRNA molecules actively transcribing the specific gene in the specific cell. As the technology only
    counts a small fraction of the molecules, this is very sparse data; a typically cell has ~500K such molecules (this
    varies greatly with the cell type) and the total number of observed UMIs is only a few thousands out of these. Even
    so, being able to do this is a highly impressive technical feat, so we can't really complain.

``metacell#`` **axis**
    Since single-cell data is very sparse, cells are typically grouped in some ways to compute more robust statistics.
    There are many ways to do this; in this example we grouped the cells into `metacells
    <https://pypi.org/project/metacells/>`_. The names of the example entries are simply ``Metacell_x`` where ``x`` is
    between 0 and 9 (a total of 10 metacells).

``cell#metacell`` **1D data**
    For each cell, the 0-based index of the metacell it is grouped into.

``metacell,gene#UMIs`` **2D Data**
    For each gene of each metacell, the sum of the UMIs of the gene in the cells grouped into the metacell. Since this
    data is the sum of many dozens of cells, it is much less sparse and much more robust than the single-cell data, and
    therefore easier to analyze. The trick, of course, is to sum together cells which have the "same" transcriptional
    state, which is the point of the metacells algorithm.

``metacell#umap_x`` and ``metacell#umap_y`` **1D Data**
    For each metacell, its coordinates in a 2D UMAP projection. These are convenient for visualizing the data but must
    not take the place of proper analysis of the relationships between the (meta)cell states, which are too complex to
    be captured in any 2D projection.

``cell_type#`` **axis**
    One entry per cell type. Cell type annotations are a simple and common (if imprecise) way of categorizing the
    transcriptional state of (meta)cells. More precise (and much more difficult) methods describe (meta)cell states as a
    combination of active gene programs/modules, possibly along some de/activation gradient.

``metacell#cell_type`` **1D Data**
    For each metacell, the name of the cell type that it belongs to.

``cell_type#color`` **1D data**
    A color for each cell type for visualizations. For example, the 2D UMAP projection is often shown where each point
    (metacell) is colored according to its cell type.
"""

__all__ = [
    "DAF_EXAMPLE_PATH",
]


#: The path to the ``daf`` example files directory.
DAF_EXAMPLE_PATH = __file__[:-3]
