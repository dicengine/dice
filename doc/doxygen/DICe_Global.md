Copyright 2015 Sandia Corporation.  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
the U.S. Government retains certain rights in this software.

Global DIC
==========

This page provides informatino about building and running the global DIC method in DICe.

Building DICe Global
--------------------

The libraries required for building DICe with global enabled include the following Trilinos libraries:

-  Tpetra (not Epetra)

-  Exodus (made availble by enabling `-D Trilinos_ENABLE_SEACASExodus:BOOL=ON` and `-D Trilinos_ENABLE_SEACASIoss:BOOL=OFF` in the Trilinos configuation script)

To enable global in DICe, the following cmake configuration variable must be set:

    -D DICE_ENABLE_GLOBAL:BOOL=ON


