daf.storage formats
===================

For in-memory storage use the `.memory` storage.

For disk storage, use either `.files` or `.h5fs` storage. The trade-off is that files storage allows for (much) better
performance for large data sets (even allowing for accessing data larger than the available RAM); and it stores each
data (array) in a different file in a single directory, allowing the use a multitude of file-based tools, most notably
build tools like ``make`` for automating reproducible computation pipelines. In contrast, ``h5fs`` storage places
everything in a single file, and provides advanced storage features such as compression (though these aren't easy to
apply through the ``daf`` API).

Finally, if you need to read or write ``AnnData``, use `.anndata`, and to disable caching of data, use `.none`.

.. toctree::

    daf.storage.memory
    daf.storage.files
    daf.storage.h5fs
    daf.storage.anndata
    daf.storage.none
