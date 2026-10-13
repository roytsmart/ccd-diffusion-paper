Introduction
============

This is the documentation for the code behind an article measuring the
depth-dependent charge-diffusion kernel of the IRIS CCDs in orbit from
glancing particle tracks, for the Journal of Astronomical Telescopes,
Instruments, and Systems.


The article itself is built by the tests and published as
`ccd-diffusion.pdf <../ccd-diffusion.pdf>`_.


Data
====

Every track in the package and a frame from each camera of every
campaign, to inspect one by one.

.. toctree::
    :maxdepth: 1

    browser
    frames


Reports
=======

Notebooks which explore the measurement in more detail than the article
itself has room for.

.. toctree::
    :maxdepth: 2

    reports/tracks


API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    ccd_diffusion


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
