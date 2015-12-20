.. NJU_DMRG documentation master file, created by
   sphinx-quickstart on Fri Nov  6 16:32:42 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NJU_DMRG's documentation - Lattice and Hamiltonian
====================================

.. toctree::
   :maxdepth: 2

****************************************************
How to Build Lattices and Momentum Spaces
****************************************************

Variaty Kinds of Lattices
----------------

.. automodule:: lattice.latticelib

To construct a specific type of lattice, you may use the function

.. autofunction:: lattice.latticelib.construct_lattice

It will return a instance of one of the following derivative classes of <Lattice>,

.. autoclass:: lattice.latticelib.Chain
   :members:
.. autoclass:: lattice.latticelib.Square_Lattice
   :members:
.. autoclass:: lattice.latticelib.Triangular_Lattice
   :members:
.. autoclass:: lattice.latticelib.Honeycomb_Lattice
   :members:

The base class <Lattice> is a special kind(derivative) of <Structure>, which is featured with repeated units.
To know more about the abilities of our <Lattice> and <Structure> instances,

.. autoclass:: lattice.structure.Structure
   :members:
.. autoclass:: lattice.lattice.Lattice
   :members:


****************************************************
How to Write Down Hamiltonians
****************************************************

