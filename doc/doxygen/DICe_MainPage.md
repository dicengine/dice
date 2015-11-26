Copyright 2015 Sandia Corporation.  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
the U.S. Government retains certain rights in this software.

[DICe Home Page] (http://dice.sandia.gov)

[Getting Started] (#GettingStarted)

[User Manual] (#UserManual)

[Tutorial] (DICe_Tutorial.md)

[Obtaining DICe source code] (#DICESource)

[Building DICe] (#BuildingDICE)

[Code Design] (#CodeDesign)

[Legal] (#Legal)


Introduction
============

DICe is an open source digital image correlation (DIC) tool intended for use as a module in an 
external application or as a standalone analysis code. Its primary capability is
computing full-field displacements and strains from sequences of digital images. These
images are typically of a material sample undergoing a characterization
experiment, but DICe is also useful for other applications (for example, trajectory tracking and 
object classification). DICe is machine portable (Windows, Linux and Mac) and can be
effectively deployed on a high performance computing platform (DICe uses MPI parallelism as well as
threaded on-core parallelism). Capabilities from DICe can
be invoked through a customized library interface, via source code integration of DICe classes or
through a standalone executable. 

DIC, in general, has become a popular means of determining full-field displacements from digital images,
it has also become a vital component in material characterization applications that use full-field
information as part of a parameter inversion process. DIC is also used extensively for constitutive 
model development and validation as well as physics code validation. DICe intends to enable more seamless integration of DIC in these types of applications by providing a DIC tool that can be directly incorporated in
an external application.

DICe is different than other available DIC codes in the following ways: First, subsets can be of arbitrary shape. This enables
tracking of oblong objects that otherwise would not be trackable with a square subset. DICe also incudes
a robust simplex optimization
method that does not use image gradients (this method is useful for data sets that are impossible to analyze
with the traditional Lucas-Kanade-type algorithms, for example,  objects without speckles, images with low contrast,
and small subset sizes < 10 pixels). Lastly, DICe also includes a well-posed global DIC formulation that addresses instabilities
associated with the saddle-point problem in DIC (This capability will be released later this year).

The DIC engine concept is meant to represent the code's flexibility in terms of using it as 
a plug-in component in a larger application.
It is also meant to represent the ease with which
various algorithms can be hot-swapped to create a customized DIC kernel for a particular
application.

Features
--------

DICe has a number of attractive features, including the following:

- Both local and global DIC algorithms

- Conformal subsets of arbitrary shape (for local DIC)

- MPI enabled parallelism

- Convolution-based interpolation functions that perform nearly as well as quintic splines at a fraction
of the compute time

- Zero-normalized sum squared differences (ZNSSD) correlation criteria

- Gradient-based optimization as well as simplex-based (that requires no image gradients)

- User specified activation of various shape functions (translation, rotation, normal strain and shearing)

- User specified arrangement of correlation points that can be adaptively refined (for local DIC)

- Robust strain calculation capabilities for treating discontinuities and high strain gradients

- Extensive regression testing and unit tests

Contact Information
-------------------

For questions, contact Dan Turner, email `dzturne@sandia.gov`,
phone (505) 845-7446.

<a name="GettingStarted"></a>Getting Started 
===============

There are three modes in which DICe can be used.

-   As a standalone executable

-   As integrated code in an external application by static linking to DICe

-   As a dynalically linked library

DICe standalone use
-------------------

To use DICe in standalone mode (assuming the `dice` executable is built and in the system path), 
the user simply has to write an input file and invoke DICe with

    $ dice -i <input_file>

The input file is an `.xml` formated set of parameters. To generate
a template set of input files (which includes the input file and the 
correaltion paramters file, described below) add the `-g` option to the `dice` call.

    $ dice -g [file_prefix]

Commented notes on all of the paramters in the input files are given
with the `-g` option in the template files. The input file specifies,
the location of the images and results and also the images to use. The
user can select a correlation of a list of specific images or a sequence of images 
based on which paramters are specified.

If MPI is enabled ([see below] (#MPINotes)) DICe is run in parallel with

    $ mpirun -n <num_procs> dice -i <input_file>

Where `num_procs` specifies the number of processors.

### <a name="MPINotes"></a>Running DICe in parallel with MPI enabled

To run DICe in parallel with MPI enabled, Trilinos must be installed with MPI enabled
by setting the approprate flag in the trilinos CMake script

    -D Trilinos_ENABLE_MPI:BOOL=ON

When DICe is configured before building, CMake will default to using the same 
compilers that were used to build Trilinos. Nothing extra has to be done in the 
DICe CMake scripts to enable MPI parallelism.

There are three ways that the subsets can be decomposed over the set of processors.
If the initialization method is `USE_FIELD_VALUES` (no neighbor solution is needed to
initialize each frame's solution) the subsets will be split up evenly across the number
of processors. If the initialization method is `USE_NEIGHBOR_VALUES` the subsets will
be split into groups that share a common seed. In this case, the maximum number
of processors in use will be equal to the number of seeds that are 
specified in the input. If more processors are requested than seeds, the extra 
processors will remain idle with not subsets to evaluate. A third case involves the
initialization method being `USE_NEIGHBOR_VALUE_FIRST_STEP_ONLY`. In this case,
the decomposition of subsets will be the same as `USE_NEIGHBOR_VALUES`, but only for
the first frame. If the analysis involves more than one frame, for subsequent frames 
the subsets will be split evenly over the number of processors available, regardless
of how many seeds are specified.

The solution output from a parallel run will be concatenated into one file, written by
process 0. Timing files, on the other hand, are written for each processor individually.
The file naming convention for parallel runs is

    <output_file_prefix>_<image_frame>.<num_procs>.txt

The file naming convention for timing files is

    timing.<num_procs>.<proc_id>.txt

<a name="UserManual"></a>User Manual 
===============

### User specified correlation parameters

Correlation paramters (for example, the interpolation method or how
the image gradients are computed) can be set in a separate `.xml` file.
We refer to this file as the correlation parameters file.
To include user defined correlation parameters, specify the following
option in the input file

    <Parameter name="correlation_parameters_file" type="string" value="<file_name>" />

If a correlation parameter is not specified in this file, the default value is used.

### User defined correlation point locations

In many cases, the locations of the correlation points are equally spaced 
on a grid in the image. The user need not add any options to the input parameters
for this default case. In other cases, the user may wish to specify the coordinates
of the correlation points as well as use conformal subsets that trace objects
in the image rather than use square subsets. Any combination of square and conformal 
subsets can be used in DICe for the same analysis. The user can specify custom 
coordinates and conformal subsets using a subset file with a certain syntax (Note: 
The `-g` option will not generate a template subset file). 
If a subset file is used, the coordinate list is mandatory, but the conformal 
subset definitions are optional. If a conformal subset definition is not provided in the
subset file for a subset, it is assumed that the subset is square and will be
sized according to the `subset_size` parameter in the input file. (If all of the subsets
in the subset file are conformal, the user does not have to specify the `subset_size` in
the input file.) The ids of the subsets are assigned according to the given coordinates, the
first set of coordinates being subset id 0. 

The name of the subset file is specified with the following option in the input file:

    <Parameter name="subset_file" type="string" value="<file_name>" />  

If a subset file is specified, the `step_size` parameter should not
be used and will cause an error. The subset locations file should be a text file with the following
syntax. Comments are denoted by `#` characters. Lines beginning with `#` or blank lines will
not be parsed. Upper and lower case can be used, the parser will automatically convert all
text to upper case during parsing. 

The first section of the subset file is a mandatory listing of centroid coordinates. The coordinate
system has its origin at the upper left corner of the image with x positive to the right
and y positive downward. The coordinate listing should begin with the command `BEGIN SUBSET_COORDINATES` and
end with `END SUBSET_COORDINATES` and have the x and y coordinates of each subset listed in between.
For example, if the user would like five subsets with centroids at (126,157), (125,250), (397,139),
(177,314) and (395,405) this section of the subset file would look like:

    BEGIN SUBSET_COORDINATES
      126 157
      # comment row will get skipped
      125 250
      397 139 # comment can go after a number

      # blank row above will get skipped
      177 314
      395 405
    END SUBSET_COORDINATES

If this is all the content in the subset file, five square subsets will be generated with the centroids as given
above and a subset size as specified in the input params file.

### Regions of interest and seeding

In some instances, the user may wish to specify certain regions of an image to correlate, but without having
to define the coordinates of each point. In this case the user can include a REGION_OF_INTEREST block in the
subset file. This block uses common shapes to build up an active sub-area of the image. There are two parts
to a REGION_OF_INTEREST definition, the boundary definition and an optional excluded area. Correlation points
will be evenly spaced in the REGION_OF_INTEREST according to the step_size parameter. Multiple REGION_OF_INTEREST
blocks can be included in the subset file. An example REGION_OF_INTEREST block is given below. Valid shapes
include the same as those defined below for conformal subsets.

    BEGIN REGION_OF_INTEREST
      BEGIN BOUNDARY
        BEGIN RECTANGLE
          CENTER <X> <Y>
          WIDTH <W>
          HEIGHT <H>
        END
        # Other shapes can be defined (the union of these shapes will define the ROI)
      END
      # Use optional BEGIN EXCLUDED to define interior shapes to omit from the ROI
    END

To seed the solution process, the following command block can be used

    BEGIN REGION_OF_INTEREST
      BEGIN BOUNDARY
        ...
      END BOUNDARY
      BEGIN SEED
        LOCATION <X> <Y> # nearest correlation point to these coordinates will be used
        DISPLACEMENT <UX> <UY>
        # OPTIONAL: NORMAL_STRAIN <EX> <EY>
        # OPTIONAL: SHEAR_STRAIN <GAMMA_XY>
        # OPTIONAL: ROTATION <THETA> 
      END
    END

If the optional BEGIN_SEED command is used in a REGION_OF_INTEREST block, the correlation points will be 
computed in an order that begins with the seed location and branches out to the rest of the domain. The 
initializiation method for a seeded analysis should be USE_NEIGHBOR_VALUES or USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY.
Only one seed can be specified for each REGION_OF_INTEREST.

### Conformal subsets

**Note: conformal subsets require the the coordinates of the subsets are specified with a SUBSET_COORDINATES
block in the subset file as described above. Conformal subsets cannot be used in combination with regions of interest.**

The user may wish to use conformal subsets for some or all of the subsets in an analysis. Conformal subsets can 
be helpful if the tracked object is of an odd shape. This allows more speckles to be included in the correlation.
There are also a number of features for conformal subsets that are useful for trajectory tracking. For example,
there are ways to enable tracking objects that cross each other's path or become partially obscured by
another object. Conformal subsets can also evolved through an image sequence to build up the intensity profile
if the object is not fully visible at the start of a sequence.

Conformal subsets
are created by specifying the geometry using shapes. There are three attributes of a conformal subset
that can be specified using sets of shapes. The first the `BOUNDARY` of the subset. This represents the outer circumference
of the subset. The second is the `EXCLUDED` area. This represents any area internal to the subset
that the user wishes to ignore (see below regarding evolving subsets). Lastly, an `OBSTRUCTED` area can be
defined. Obstructions are fixed regions in the image in which pixels should be
deactivated if they fall in this region. For example if the user is tracking a vehicle through the frame,
and it passed behind a light post, the light post should be defined using an obstructed area.

**Note: For trajectory tracking and evolving subsets, the `SL_ROUTINE` `correlation_routine` should be used.**

For each conformal subset, the following syntax should be used to define these three sets of shapes. Note, the
`BOUNDARY` and `SUBSET_ID` are required, but the `EXCLUDED` and `OBSTRUCTED` sections are optional. Continuing with the
example subset file above, after the coordinates are listed, we wish to denote that subset 2 (with centroid coordinates (397 139))
is a conformal subset made up of an odd shape made of two polygons and a circle. The subset file text for this subset 
would be

    BEGIN CONFORMAL_SUBSET
      SUBSET_ID 2 # required id of the subset
      BEGIN BOUNDARY # defines the outer boundary of the subset

        BEGIN POLYGON
          BEGIN VERTICES
            # needs at least 3
            # polygon points only need to be listed once, the last segment will close the polygon
            # by connecting the last point with the first
            307 136
            352 86
            426 109
            421 243
            372 143 # comment after vertex value for testing
          END VERTICES
        END POLYGON

        BEGIN CIRCLE
          CENTER 362 201
          RADIUS 45
        END CIRCLE

        BEGIN POLYGON
          BEGIN VERTICES
            359 228
            455 175
            437 267
          END VERTICES
        END POLYGON

      END BOUNDARY    
    END CONFORMAL_SUBSET

Available shapes include the following and their syntaxes

    BEGIN POLYGON
      BEGIN VERTICES
        <X> <Y>
        ...
      END VERTICES
    END POLYGON

    BEGIN CIRCLE
      CENTER <X> <Y>
      RADIUS <R>
    END CIRCLE

    BEGIN RECTANGLE
      CENTER <X> <Y>
      WIDTH <W>
      HEIGHT <H>
    END RECTANGLE

    BEGIN RECTANGLE
      UPPER_LEFT <X> <Y>
      LOWER_RIGHT <X> <Y>
    END RECTANGLE

Sets of shapes used to define an attribute of a conformal subset can overlap. The pixels inside the overlap will only be included once. An example conformal subset definition that include all three attributes defined is as follows. This subset has a circular
boundary, a triangular region to be excluded and an obstruction along the bottom edge.

    BEGIN CONFORMAL_SUBSET
      SUBSET_ID 3
      BEGIN BOUNDARY
        BEGIN CIRCLE
          CENTER 178 352
          RADIUS 64
        END CIRCLE
      END BOUNDARY
      BEGIN EXCLUDED # (optional) defines internal regions that should initially be excluded because they are blocked
        BEGIN POLYGON
          BEGIN VERTICES
            148 341
            160 374
            205 340
          END VERTICES
        END POLYGON
      END EXCLUDED
      BEGIN OBSTRUCTED # (optional) defines objects that do not move that could obstruct the subset
        BEGIN POLYGON
          BEGIN VERTICES
            130 423
            252 366
            266 402
            148 460
          END VERTICES
        END POLYGON
      END OBSTRUCTED
      BEGIN BLOCKING_SUBSETS # (optional) list of other subset global ids that could block this one
        0  # one subset per line
        2  # the ids are assigned as the order of the subset centroid coordinates vector
      END BLOCKING_SUBSETS
    END CONFORMAL_SUBSET

The optional `BLOCKING_SUBSETS` section of the conformal subset definition above lists other subsets in the analysis
that may cross paths with this subset. After each frame, pixels are deactivated from this subset if their
location coincides with one of the listed blocking subsets. This is useful mostly for trajectory tracking.

An excluded area can be used to generate a hole in the subset and can also 
be used to denote an area that may be initially obstructed by an object in the image and therefore not
visible. The reason a user may wish to treat this area as excluded rather than draw the subset around it
is because if pixels in the excluded area become visible later in the image sequence the user can 
request that these pixels become activated. If the correlation parameter

    <Parameter name="use_subset_evolution" type="bool" value="true" />

is used, after each frame, the pixels in the excluded area are tested to see if they are now visible. If so,
the pixel intensity value from the deformed image is used to evolve the reference pixel intensity that
was initially not known. In this way, the subset intensity profile evolves as more regions
become visible.

When obstructions or blocking subsets are used, the user can specify the size of the buffer that is constructed 
surrouding the obstructions, effectively enlarging them. To specify the buffer size use the following option

    <Parameter name="obstruction_buffer_size" type="int" value="<size>"

The default value for the buffer size is 3 pixels.

The solution values can be seeded for a conformal subset by adding a SEED command block to the CONFORMAL_SUBSET
command block. For example,

    BEGIN CONFORMAL_SUBSET
      ....
      BEGIN SEED
        DISPLACEMENT <UX> <UY>
        NORMAL_STRAIN <EX> <EY>
        SHEAR_STRAIN <GAMMA_XY>
        ROTATION <THETA>
      END SEED
    END CONFORMAL_SUBSET

Note that the location of the seed is automatically the subset centroid and cannot be specified (as it is in
a REGION_OF_INTEREST). The displacement guess for a seed is required. All other initial values (shear strain, etc.) are optional.  

The conformal and square subsets from the example above are shown in the image below. Note the subset
shapes, exclusions etc. are simply a random example to illustrate the syntax, not a meaningful
way to set up an analysis.

![](images/SubsetDefs.png)
@image latex images/SubsetDefs.png

### Shape functions

The user can select which shape functions are used to evaluate the correlation between subsets. By "shape
functions" we are referring to the parameters used in the mapping of a subset from the reference to
the deformed frame of reference. There are four sets of shape functions available in DICe: translation, rotation,
normal strain and shear strain. To manually specify which shape functions should be used, the user can
add the following options to the correlation parameters file

    <Parameter name="enable_translation" type="bool" value="<true/false>" />
    <Parameter name="enable_rotation" type="bool" value="<true/false>" />
    <Parameter name="enable_normal_strain" type="bool" value="<true/false>" />
    <Parameter name="enable_shear_strain" type="bool" value="<true/false>" />

See DICe::Subset for information on how these parameters are used in constructing a deformed subset.

### Output files

The output produced will be ASCII space delimited text files. The default output is
one file per deformed image listing the solution variables for each 
subset. In the default case, the index at the end of the file name refers to the image id or
frame number. 

The user can alternatively request output as a separate file
for each subset listing the solution variables for each deformed image
or frame. To do so set the following option in the input file

    <Parameter name="separate_output_file_for_each_subset" type="bool" value="true" />

If the separate file for each subset option is employed, the index at the end of the filename
refers to the subset id.

The file name prefix to use for the output files can be set with the following option

    <Parameter name="output_prefix" type="string" value="<file_name_prefix>" />

The user can also define an output specification to list the fields in a specific order and 
choose which fields to output. To define the output fields and the order, an `output_spec`
can be added to the correlation parameters. The following is a syntax example that outputs only 
three fields in the order `COORDINATE_X`, `COORDINATE_Y`, `DISPLACEMENT_X`:

    <ParameterList name="output_spec">
       <Parameter name="coordinate_x" type="int" value="0" />
       <Parameter name="coordinate_y" type="int" value="1" />
       <Parameter name="displacement_x" type="int" value="2" />
    </ParameterList>

The integer value for each field name is the rank in the output order (the order is not
assumed from the ordering in the parameter list).

The delimiter used in the output file can be set in the parameters file with the following option:

    <Parameter name="output_delimiter" type="string" value="<value>" />

The user can also request that the row id in the output files be omitted with

    <Parameter name="omit_output_row_id" type="bool" value="true" />

### Plotting results with python

Note: Some extra python modules must be installed to use these python scripts.

The following python script can be used to plot the results from a 
DICe output file. (Note: these are not intended for conformal
subsets if they are defined). If the output files are in the default format of 
one output file per image with a listing of subset variables in columns with 
one subset per row, the following python script can be used to create a two-dimensional
contour plot. This example also assumes that the output specification has the fields
in the default order.

    #Import everything from matplotlib (numpy is accessible via 'np' alias)
    from pylab import *
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    # skiprows used to skip the header comments
    DATA = loadtxt("<results_folder>/DICE_solution_<#>.txt",skiprows=21)
    X = DATA[:,1]
    Y = DATA[:,2]
    DISP_X = DATA[:,3]
    DISP_Y = DATA[:,4]
    SIGMA = DATA[:,9]

    NUMPTS = len(X)
    NUMPTSX = sqrt(NUMPTS)
    LINEDISPY = []
    LINEY = []
    i = NUMPTSX/2
    while i < NUMPTS:
        LINEDISPY.append(DISP_Y[i])
        LINEY.append(Y[i])
        i = i + NUMPTSX

    OUTFILE_DISPLACEMENT_X = "DICEDispX.pdf"
    OUTFILE_DISPLACEMENT_Y = "DICEDispY.pdf"
    OUTFILE_SIGMA = "DICECorrSigma.pdf"
    OUTFILE_LINEY = "DICELineDispY.pdf"

    triang = tri.Triangulation(X, Y)

    font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 8}
    matplotlib.rc('font', **font)

    fig = figure(dpi=150)
    ax1 = tripcolor(triang, DISP_X, shading='flat', cmap=plt.cm.rainbow)
    axis('equal')
    xlabel('X (px)')
    ylabel('Y (px)')
    title('Displacement-X')
    plt.gca().invert_yaxis()
    colorbar()
    savefig(OUTFILE_DISPLACEMENT_X,dpi=150, format='pdf')

    fig = figure(dpi=150)
    ax2 = plot(LINEY,LINEDISPY,'-')
    title('Displacement Y (px)')
    xlabel('Y (px)')
    ylabel('DISP_Y (px)')
    savefig(OUTFILE_LINEY, dpi=150, format='pdf')

    #Other figures similar to above

If separate output files were generated for each subset, a time history
of the solution variables can be plotted for all subsets with the following script.
A separate plot will be created for each subset.

    # Import everything from matplotlib (numpy is accessible via 'np' alias)
    from pylab import *
    import matplotlib.pyplot as plt

    NUM_SUBSETS = <number_of_subsets>
    FILE_PREFIX = "<results_folder>/DICE_solution_"

    font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 8}
    matplotlib.rc('font', **font)

    for i in range(0,NUM_SUBSETS):
       FILE = FILE_PREFIX+str(i)+".txt"
       PDFU = "DispX_"+str(i)+".pdf"
       PDFV = "DispY_"+str(i)+".pdf"
       PDFTHETA = "Theta_"+str(i)+".pdf"
       print(FILE)

       # skiprows used to skip the header comments
       DATA = loadtxt(FILE,skiprows=21)
       IMAGE   = DATA[:,0]
       DISP_X  = DATA[:,3]
       DISP_Y  = DATA[:,4]
       THETA   = DATA[:,5]
       SIGMA   = DATA[:,9]
       FLAG    = DATA[:,10]

       fig = figure(figsize=(12,5), dpi=150)
       plot(IMAGE,THETA,'-b')
       fig.set_tight_layout(True)
       xlabel('Image Number')
       xlim(0.0,18074.0)
       savefig(PDFTHETA,dpi=150, format='pdf')

       fig = figure(figsize=(12,5), dpi=150)
       plot(IMAGE,DISP_X,'-b')
       fig.set_tight_layout(True)
       xlabel('Image Number')
       ylabel('Displacement X')
       xlim(0.0,18074.0)
       savefig(PDFU,dpi=150, format='pdf')
 
       fig = figure(figsize=(12,5), dpi=150)
       plot(IMAGE,DISP_Y,'-b')
       fig.set_tight_layout(True)
       xlabel('Image Number')
       ylabel('Displacement Y')
       xlim(0.0,18074.0)
       savefig(PDFV,dpi=150, format='pdf')

    #show()   

The python scripts above are only meant to provide a simple example.
Obviously, there are many ways python can be used to generate more
sophisticated plots.

Incorporating DICe in an external application with static linking
-----------------------------------------------------------------

To use DICe correlation capabilities in an external application the
developer need only include the DICe header files and link to `libdicecore`,
which resides in the `\dice\build\src\` folder.

The following code is a simple `.cpp` file that can be used as a
template for using DICe in a C++ application. Each line is described in
the comments above the code line. (Note: this example assumes boost is enabled,
so that tiff files can be read)

The process of performing a correlation involves three steps, first a DICe::Schema
must be instantiated by calling its constructor. Then the schema must be initialized
(which resizes the field arrays, etc.). Lastly, the correlation is exectued.

    #include <DICE_Types.h>
    #include <DICE_Schema.h>

    int main(int argc, char *argv[]) {

      // set up the correlation parameters
      Teuchos::RCP<Teuchos::ParameterList> params = 
          rcp(new Teuchos::ParameterList());
      params->set("enable_rotation",false);
      params->set("enable_normal_strain",true);
      params->set("correlation_method", DICE::SSD);
      // many other options possible ...

      // create a schema to orchestrate the correlation
      DICE::Schema<RealT,SizeT> schema("./smoothRef.tif","./smoothDef.tif",params);

      // initialize the schema by setting the spacing of control 
      // points using a constant width and height
      const RealT step_size_x = 20.0; // pixels
      const RealT step_size_y = 20.0; // pixels
      const RealT subset_size = 15.0; // pixels
      schema.initialize(step_size_x,step_size_y,subset_size);

      // There are other ways to initialize a schema 
      // (for example by specifying the number of points and 
      // then setting the coordinates manually)
      
      // perform the correlation
      schema.execute_correlation();
      
      // At any point after initialization, the user can access field values
      // by calling field_value(subset_id,field_name)
      // This call enables getting or setting the field value
      // Setting can be used to initialize values prior to calling execute_correlation()
      const RealT point_0_disp_x = schema.field_value(0,DICE::DISPLACEMENT_X);

      // write the output files to the given folder
      schema.write_output("./");
      
      // print out an image showing the control points and windows 
      // (only enabled if boost is, otherwise this is a no-op)
      schema.write_control_points_image("InitialCPs");

      return 0;
    }

The solution for each subset is contained in the data structures of the DICe::Schema, to access
these values, DICe::Schema provides the `field_value(const Size subset_id, const std::string & field_name)`
method. This method can be used to get or set a field value for a specific subset. Valid field
names are given in DICe_Types.h.

Using DICe as a dynamically linked library
------------------------------------------

DICe can also be used in library mode, but a custom interface to the particular
application for DICe needs to be written first. (An example of a LabView interface
is in `\dice\lib\DICE_api.cpp`.) Although DICe does not have a standard
interface, it is simple to write one that meets the needs of a
particular application.

This library is called `libdice` and resides in the `\dice\build\lib\` folder.

Using DICe as a library involves writing an interface that sets the
parameters and orders the data from the correlation in the right order
for the particular application (for the purposes of data exchange). Once
this interface is compiled as a library, it can be linked and used in an
external application simply by calling the correlation function exposed
via the interface.


<a name="BuildingDICE"></a>Building DICe
=============

Requirements
------------

DICe can be built and run on Mac OS X, Windows, and Linux. The primary
intended platforms are Mac OS X and Linux, but Windows builds are also
possible. The prerequisite tools required for installing DICe include

-   [CMake] (http://www.cmake.org) 

-   [Trilinos] (http://trilinos.org)

-   LAPACK or CLAPACK (for Windows, only CLAPACK is supported)

-   [Boost] (http://www.boost.org)

CMake
-----

DICe makes use of CMake
for build configuration. Version 2.8 or greater is required. Sample CMake scripts for building Trilinos and
DICe are in the folder `dice\scripts`

Installing Trilinos
-------------------

Trilinos contains a set of software packages within an object-oriented software framework used
for the solution of large-scale, complex multi-physics engineering and scientific problems. DICe uses some
of the packages avaiable in Trilinos and requires that Trilinos be installed.

Trilinos can be downloaded from http://trilinos.org and build
instructions can be found on the getting started page. DICe requires
that Trilinos be built with the following packages enabled:

-   Epetra or Tpetra

-   Kokkos

-   BLAS

-   LAPACK

-   TeuchosCore

-   TeuchosParameterList

-   TeuchosNumerics

<a name="DICESource"></a>Obtaining DICe source code
--------------------------

DICe can be cloned from the following git repository

    github.com/dicengine

Please request repository access via the contact information above.

### Setting up your git repository

If this is your first time using git, you will have to set up your
git configurations. To set your user name and email use

    git config --global user.name "Your Name"
    git config --global user.email youremail@email.com

### Windows users

To enable the tests that diff text files to pass, it will be important
for Windows git users to add another option to their git configuration

    git config --global core.autocrlf=true

This checks out text files in Windows CRLF format and checks files into the git
repo with unix LF format.

**Note: this must be done before executing the pull command above, doing so after
pulling the DICe repository will not work.**


Building DICe
-------------

### Mac OSX or Linux

To build DICEe on Mac OSX or Linux, create a folder in the main directory
called build

    $ mkdir build 

Change directory  into the build directory and copy the CMake script from the
scripts folder.

    $ cd build
    $ cp ../scripts/example-do-cmake ./do-cmake
    $ chmod +x ./do-cmake

Edit the script to have the correct path locations. Then build DICe

    $ ./do-cmake
    $ make

### Windows

First, download and install CLAPACK from
`http://www.netlib.org/clapack/`. In the top level directory, create a
file called `do-cmake.bat` with the contents:

    cmake -D CMAKE_INSTALL_PREFIX:PATH=<install prefix> 
    -D CMAKE_BUILD_TYPE:STRING=RELEASE -G "NMake Makefiles" .

Note that the entire do-cmake file should be a single line with no
carriage returns. Then execute

    $ do-cmake.bat
    $ nmake

Once CLAPACK has built, Trilinos must be built. A sample
`do-trilinos-cmake-win.bat` file is provided in the `\dice\scripts`
folder. The same process for building CLAPACK can be used to build
Trilinos and eventually DICe. A sample `do-dice-cmake-win.bat` file is
also included in the repository in the `\dice\scripts` folder.

Boost
-----

Boost is a required dependency.  DICe makes use of the graphics image library in
boost. This enables reading standard image formats like `.tiff` files into DICe.

Testing
-------

To test the installation, from the directory `\dice\build\test` execute
the command (this assumes that DICe has been built in the `build` directory, otherwise
specify the correct path)

    $ ctest

Compiling DICe with debug messages
----------------------------------

To enable debug messages in DICe, simply set the CMake flag
`-D DICE_DEBUG_MSG:BOOL=ON` in the `do-cmake` script.

<a name="CodeDesign"></a>Code Design and Workflow
========================

A basic outline for how DICe performs a correlation is as follows. If
more than one image is correlated, the basic workflow is the same, just
applied numerous times.

-   The reference and deformed image are read into memory

-   The correlation parameters are defined

-   Subsets are defined (for the local algorithm)

-   The correlation is performed

-   Results are written to disk

All of the correlation parameters are stored in a `DICE::Schema`, this
class controls the algorithms, images, and subset locations. The
`DICE::Schema` also stores the correlation results and all other fields.
An image is stored as an array in a `DICE::Image`. A copy
of a portion of the image intensity array is made available through a `DICE::Subset`.
The bulk of the correlation algorithm resides in a `DICE::Objective`.
More details on each of these are provided in the class documentation.

<a name="Legal"></a>Legal
=====

Sandia National Laboratories is a multi-program laboratory managed and
operated by Sandia Corporation, a wholly owned subsidiary of Lockheed
Martin Corporation, for the U.S. Department of Energy’s National Nuclear
Security Administration under contract DE-AC04-94AL85000.

License
-------

Copyright 2015 Sandia Corporation.

Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

3.  Neither the name of the Corporation nor the names of the
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION “AS IS” AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
