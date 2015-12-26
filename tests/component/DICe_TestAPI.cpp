// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 Sandia Corporation.
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact: Dan Turner (dzturne@sandia.gov)
//
// ************************************************************************
// @HEADER

/// \file  DICe_TestAPI.cpp
/// \brief Test that calling a correlation from the library, libdice, works
/// This is a test of the old library signature, and may be depricated
/// There is another test in the regression suite that tests the new
/// library that enables conformal subsets
#include <DICe.h>
#include <DICe_api.h>
// need DICe_Image.h to read in the images into an intensity array
// in a real use case, this would be done by the libdice user with or without DICe
#include <DICe_Image.h>

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_RCP.hpp>

#include <cassert>

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  int_t iprint     = argc - 1;
  // for serial, the global MPI session is a no-op, but in parallel
  // ensures that MPI_Init is called (needed by the schema)
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);
  int_t errorFlag  = 0;
  scalar_t errtol  = 1.0E-4;


  // set up a parameter list to pass to the library call (if null defaults are used)
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::enable_translation,true);
  params->set(DICe::enable_rotation,true);
  params->set(DICe::enable_normal_strain,true);
  params->set(DICe::enable_shear_strain,true);
  params->set(DICe::correlation_routine,DICe::TRACKING_ROUTINE);
  params->set(DICe::interpolation_method,DICe::KEYS_FOURTH);
  params->set(DICe::initialization_method,DICe::USE_FIELD_VALUES);
  params->set(DICe::optimization_method,DICe::SIMPLEX);

  // set up the subsets:
  const int_t subset_size = 21;
  const int_t num_subsets = 4;
  std::vector<int_t> subset_centroids_x(num_subsets);
  std::vector<int_t> subset_centroids_y(num_subsets);
  // small cluster of subsets near the center (49,49)
  // imagage are 100x100
  subset_centroids_x[0] = 47; subset_centroids_y[0] = 47;
  subset_centroids_x[1] = 51; subset_centroids_y[1] = 47;
  subset_centroids_x[2] = 51; subset_centroids_y[2] = 51;
  subset_centroids_x[3] = 47; subset_centroids_y[3] = 51;

  // initialize the input data:
  // initialize x y u v theta sigma
  // this is a similar ordering to the labView calls to libdice
  scalar_t * points = new scalar_t[(int_t)num_subsets*DICE_API_STRIDE];
  for(int_t subsetIt=0;subsetIt<num_subsets;++subsetIt){
    for(int_t i=0;i<DICE_API_STRIDE;++i){
      points[subsetIt*DICE_API_STRIDE+i] = 0.0;
    }
  }
  for(int_t subsetIt=0;subsetIt<num_subsets;++subsetIt){
    points[subsetIt*DICE_API_STRIDE + 0] = subset_centroids_x[subsetIt]; //x0
    points[subsetIt*DICE_API_STRIDE + 1] = subset_centroids_y[subsetIt]; //y0
  }
  // solution with default params changed
  scalar_t * pointsParams = new scalar_t[(int_t)num_subsets*DICE_API_STRIDE];
  for(int_t subsetIt=0;subsetIt<num_subsets;++subsetIt){
    for(int_t i=0;i<DICE_API_STRIDE;++i){
      pointsParams[subsetIt*DICE_API_STRIDE+i] = 0.0;
    }
  }
  for(int_t subsetIt=0;subsetIt<num_subsets;++subsetIt){
    pointsParams[subsetIt*DICE_API_STRIDE + 0] = subset_centroids_x[subsetIt]; //x0
    pointsParams[subsetIt*DICE_API_STRIDE + 1] = subset_centroids_y[subsetIt]; //y0
  }

  // correlate on the set of five images that shift the image by one pixel each
  std::string ref_name = "./images/defSyntheticSpeckled0.tif";
  Teuchos::RCP<DICe::Image> refImg = Teuchos::rcp( new DICe::Image(ref_name.c_str()));
  Teuchos::ArrayRCP<intensity_t> ref_img = refImg->intensities();
  const int_t ref_w = refImg->width();
  const int_t ref_h = refImg->height();

  std::vector<std::string> def_names;
  def_names.push_back("./images/defSyntheticSpeckled0.tif");
  def_names.push_back("./images/defSyntheticSpeckled1.tif");
  def_names.push_back("./images/defSyntheticSpeckled2.tif");
  def_names.push_back("./images/defSyntheticSpeckled3.tif");
  def_names.push_back("./images/defSyntheticSpeckled4.tif");
  def_names.push_back("./images/defSyntheticSpeckled5.tif");

  for(size_t img=0;img<def_names.size();++img){
    *outStream << "correlating image " << ref_name << " WITH " << def_names[img] << std::endl;

    Teuchos::RCP<DICe::Image> defImg = Teuchos::rcp( new DICe::Image(def_names[img].c_str()));
    Teuchos::ArrayRCP<intensity_t> def_img = defImg->intensities();

    // call library using default params
    errorFlag = dice_correlate(points, num_subsets, subset_size,
      ref_img.get(), ref_w, ref_h,
      def_img.get(), ref_w, ref_h);

    *outStream << "results: " << std::endl;
    for(int_t i=0;i<num_subsets;++i){
      *outStream << "     subset " << i << " x " << points[i*DICE_API_STRIDE + 0] <<
          " y " << points[i*DICE_API_STRIDE + 1] <<
          " u " << points[i*DICE_API_STRIDE + 2] <<
          " v " << points[i*DICE_API_STRIDE + 3] <<
          " theta " << points[i*DICE_API_STRIDE + 4] <<
          " sigma " << points[i*DICE_API_STRIDE + 5] <<
          " gamma " << points[i*DICE_API_STRIDE + 6] <<
          " flag " << points[i*DICE_API_STRIDE + 7] << std::endl;
      // test the solution
      if ( std::abs(points[i*DICE_API_STRIDE + 0] - subset_centroids_x[i]) > errtol) {
        *outStream << "Error, Coordinate X is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(points[i*DICE_API_STRIDE + 1] - subset_centroids_y[i]) > errtol) {
        *outStream << "Error, Coordinate Y is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(points[i*DICE_API_STRIDE + 2] - img) > errtol) {
        *outStream << "Error, Displacement X is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(points[i*DICE_API_STRIDE + 3] - img) > errtol) {
        *outStream << "Error, Displacement Y is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(points[i*DICE_API_STRIDE + 4]) > errtol) {
        *outStream << "Error, Theta is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(points[i*DICE_API_STRIDE + 6]) > errtol) {
        *outStream << "Error, Gamma is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(points[i*DICE_API_STRIDE + 8] - 1) > errtol) {
        *outStream << "Error, Flag is not correct for image: " << img << std::endl;
        errorFlag++;
      };
    } // results output

    // try calling the library with params passed in and check the values

    errorFlag = dice_correlate(pointsParams, num_subsets, subset_size,
      ref_img.get(), ref_w, ref_h,
      def_img.get(), ref_w, ref_h, params.getRawPtr());

    *outStream << "results (Params specified): " << std::endl;
    for(int_t i=0;i<num_subsets;++i){
      *outStream << "     subset " << i << " x " << pointsParams[i*DICE_API_STRIDE + 0] <<
          " y " << pointsParams[i*DICE_API_STRIDE + 1] <<
          " u " << pointsParams[i*DICE_API_STRIDE + 2] <<
          " v " << pointsParams[i*DICE_API_STRIDE + 3] <<
          " theta " << pointsParams[i*DICE_API_STRIDE + 4] <<
          " sigma " << pointsParams[i*DICE_API_STRIDE + 5] <<
          " gamma " << pointsParams[i*DICE_API_STRIDE + 6] <<
          " flag " << pointsParams[i*DICE_API_STRIDE + 8] << std::endl;
      // test the solution
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 0] - subset_centroids_x[i]) > errtol) {
        *outStream << "Error, Params, coordinate X is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 1] - subset_centroids_y[i]) > errtol) {
        *outStream << "Error, Params, coordinate Y is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 2] - img) > errtol) {
        *outStream << "Error, Params, displacement X is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 3] - img) > errtol) {
        *outStream << "Error, Params, displacement Y is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 4]) > errtol) {
        *outStream << "Error, Params, theta is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 6]) > errtol) {
        *outStream << "Error, Params, gamma is not correct for image: " << img << std::endl;
        errorFlag++;
      };
      if ( std::abs(pointsParams[i*DICE_API_STRIDE + 8] - 1) > errtol) {
        *outStream << "Error, Params, flag is not correct for image: " << img << std::endl;
        errorFlag++;
      };
    } // results output

  } // image loop


  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  DICe::finalize();

  return 0;

}

