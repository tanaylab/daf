DAF 0.1.0-dev.1 - Data in Axes in Files
=======================================

.. image:: https://readthedocs.org/projects/daf?version=latest
    :target: https://daf.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

The daf package provides a uniform generic interface for accessing 1D and 2D data arranged along some set of axes. This
is a much-needed generalization of the `annotated data <https://pypi.org/project/anndata>`_ functionality. The key
features are:

* Support both in-memory and persistent (file based) data storage.

* Allows storing the data in a collection of simple memory-mapped files, as well as in arbitrary complex formats using
  format adapters (in particular, allow using daf to access ``AnnData``, e.g. to access ``h5ad`` files).

* The data model is based on (1) some axes, (2) 1-D vectors of data indexed by a single axis, (3) 2-D matrices indexed
  by a pair of axis, and (4) 0-D data (arbitrary blobs).

* There is explicit control over matrix layout (row or column major), and support for both dense and sparse matrices,
  both of which are crucial for performance.

* Allows accessing the data as either plain numpy arrays, scipy csr/csc sparse matrices, or as pandas series/frames.

Usage
-----

.. code-block:: python

    import daf
    import numpy as np
    import scipy.sparse as sp

    # Open an existing DAF storage
    data = daf.open("daf.yaml")

    # Access an arbitrary blob.
    name = daf.get_datum("name")

    # Get a numpy 1D vector by axis and name.
    metacell_types = data.get_vector("metacell:type")

    # Get a Pandas series by axis and name.
    type_colors = data.get_series("type:color")

    # Combine these to get a Pandas series of the color of each metacell.
    metacell_colors = type_colors[metacell_types]

    # Get a 2D matrix by two axes and a name.
    umis_matrix = data.get_matrix("cell,gene:UMIs", layout="row_major")

    if isinstance(umis_matrix, np.ndarray):
        # Umis matrix is dense
        ...
    else:
        assert isinstance(umis_matrix, sp.csr_matrix)
        # Umis matrix is sparse
        ...

    # Get a Pandas data frame by two axes and a name.
    type_marker_genes = data.get_frame("gene,type:marker", layout="column_major")

    # Access the mask of marker genes for a specific type.
    type_marker_genes["T"]

    # Get a Pandas data frame containing multiple named vectors (columns) of the same axis (always column-major).
    genes_masks = data.get_table("gene", ["forbidden", "significant"])

    # Access the mask of significant genes.
    genes_masks["significant"]

See the `documentation <https://daf.readthedocs.io/en/latest/?badge=latest>`_ for the full API details.

Installation
------------

In short: ``pip install daf``. Note that ``metacells`` requires many "heavy" dependencies, most notably ``numpy``,
``pandas``, ``scipy``, ``scanpy``, which ``pip`` should automatically install for you. If you are running inside a
``conda`` environment, you might prefer to use it to first install these dependencies, instead of having ``pip`` install
them from ``PyPI``.

License (MIT)
-------------

Copyright © 2022 Weizmann Institute of Science

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
