Copyright 2015 Sandia Corporation.  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
the U.S. Government retains certain rights in this software.

Introduction
============

This section of the documentation describes how to write a custom DIC application with DICe. For the example below, all of the steps assume that a Linux operating system is being used. For Windows, the cmake script needs to be modified to produce nmake makefiles, and the appropriate scripts should be exectuted as batch files.

Code Design and Workflow
------------------------

A basic outline for how DICe performs a correlation is as follows. If
more than one image is correlated, the basic workflow is the same, just
applied numerous times.

Initialization phase:

-   The reference and deformed image are read into memory

-   The correlation parameters are defined

-   Subsets are defined (for the local algorithm)

Execution phase:

-   The correlation is executed

-   Results are written to disk

-   The ref or def images can be changed and the execution phase repeated

All of the initialization data and correlation parameters are stored in a `DICe::Schema`, this
class controls the algorithms, images, and subset locations. The
`DICe::Schema` also stores the correlation results and all other fields.
An image is stored as an array in a `DICe::Image`. A copy
of a portion of the image intensity array is made available through a `DICe::Subset`.
The bulk of the correlation algorithm resides in a `DICe::Objective`.
More details on each of these are provided in the class documentation.

Rules of Thumb
--------------

When developing a custom application using DICe, keep the following points in mind:

-   The number of subsets cannot change once a DICe::Schema is initialized. If the number of subsets needs to change (for instance, using subset adaptivity), re-instantiate a new DICe::Schema.

-   The locations of subset centroids can change after initialization, but not the number.

-   New correlation parameters can be used for an old DICe::Schema by calling the set_params() method and passing in the new parameters as a Teuchos::ParameterList

-   The reference or deformed images can be changed after the DICe::Schema is initialized with a call to `set_ref_image()` or `set_def_image()`. These ref and def images should be the same dimensions, and the new ref or def image cannot change the image dimensions

Custom DIC Application Example
------------------------------

An example project which can be used to build a custom DIC application using DICe methods is
provided in the examples directory in the folder

    dice/tests/examples/custom_app

To use this folder as a starting point for a new project, copy the folder and its contents to a new location.

    cp dice/tests/examples/custom_app ./<new_project_location>

Inside the project folder is a `src` directory which can be removed, this was an artifact of including the `custom_app` example in the regression tests for DICe. It is not needed for building a custom application outside of DICe.

Create a `build` directory in the top level of the project and change directories into it.

    mkdir build
    cd build

Inside the build directory, create a CMake script similar to the example given below called `do-cmake` on Linux or `do-cmake.bat` on Windows:

    #!/bin/bash

    rm -f CMakeCache.txt
    rm -rf CMakeFiles

    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D DICE_TRILINOS_DIR:FILEPATH=<location_of_trilinos_install> \
        -D DICE_LIB_DIR:FILEPATH=<location_of_dice_installed_libraries> \
        -D DICE_HEADER_DIR:FILEPATH=<locaiont_of_dice_headers> \
        -D CMAKE_CXX_FLAGS:STRING="-Wall" \
        ../
	
    # define the tiff directory if libtiff is installed in another location than the system path
    # add "-D TIFF_DIR:FILEPATH="<libtiff_location>" \" to the variables above

Then, run the configuration

    $ ./do-cmake

Here we again use the `$` symbol to denote the command line prompt. After the project is configured, build it with 

    $ make

To test that the application was built correctly, change directories back to the project top level dir (where the input xml files are located) and run the executable

    $ cd ../
    $ ./build/main

The source file `main.cpp` in the top level directory has a number of notes in the comments to help with setting up a DICe::Schema and executing a correlation. This file is a good starting place for developing a custom DIC application.

