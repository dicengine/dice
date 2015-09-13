// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Subset.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  size_t iprint     = argc - 1;
  size_t errorFlag  = 0;
  scalar_t errorTol = 1.0E-4;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create a subset by centroid, width and height
  *outStream << "creating a subset from cx, cy, width and height " << std::endl;
  size_t cx = 125;
  size_t cy = 250;
  size_t w = 13;
  size_t h = 19;
  Subset square(cx,cy,w,h);
  if(square.num_pixels()!=w*h){
    *outStream << "Error, the square subset is not the right size. "
        "Expected size " << w*h << " actual size " << square.num_pixels() << std::endl;
    errorFlag++;
  }
  if(square.centroid_x()!=cx){
    *outStream << "Error, the x centroid of the square subset is not correct. "
        "Expected cx " << cx << " actual cx " << square.centroid_x() << std::endl;
    errorFlag++;
  }
  if(square.centroid_y()!=cy){
    *outStream << "Error, the y centroid of the square subset is not correct. "
        "Expected cy " << cy << " actual cy " << square.centroid_y() << std::endl;
    errorFlag++;
  }

  // create a subset by array
  *outStream << "creating a subset by array" << std::endl;
  const size_t num_pts = 48;
  Teuchos::ArrayRCP<size_t> x_coords(num_pts,0);
  Teuchos::ArrayRCP<size_t> y_coords(num_pts,0);
  for(size_t i=0;i<num_pts;++i){
    x_coords[i] = i*2 +4; // random point locations
    y_coords[i] = 42+i;
  }
  Subset array(cx,cy,x_coords,y_coords);
  if(array.num_pixels()!=num_pts){
    *outStream << "Error, the number of pixels in the array constructed subset is not correct" << std::endl;
    errorFlag++;
  }
  if(array.centroid_x()!=cx){
    *outStream << "Error, the x centroid of the array subset is not correct. "
        "Expected cx " << cx << " actual cx " << array.centroid_x() << std::endl;
    errorFlag++;
  }
  if(array.centroid_y()!=cy){
    *outStream << "Error, the y centroid of the array subset is not correct. "
        "Expected cy " << cy << " actual cy " << array.centroid_y() << std::endl;
    errorFlag++;
  }
  bool x_coord_error = false;
  bool y_coord_error = false;
  for(size_t i=0;i<num_pts;++i){
    if(array.x(i)!=x_coords[i])
      x_coord_error = true;
    if(array.y(i)!=y_coords[i])
      y_coord_error = true;
  }
  if(x_coord_error || y_coord_error){
    *outStream << "Error, the coordinates are not correct for the array subset" << std::endl;
  }

  // test initializing the subset from an image:
  // create an image:
  Teuchos::RCP<Image> image = Teuchos::rcp(new Image("./images/ImageA.tif"));
  // initialize the square subset
  square.initialize(image);
  // test the subset ref values:
  bool ref_values_error = false;
  for(size_t i=0;i<square.num_pixels();++i){
    //std::cout << "subset: " << square.ref_intensities(i) << " img: " << (*image)(square.x(i),square.y(i)) << std::endl;
    if(square.ref_intensities(i)!=(*image)(square.x(i),square.y(i))){
      ref_values_error = true;
    }
  }
  if(ref_values_error){
    *outStream << "Error, the ref intensity values for the initialized square subset are wrong" << std::endl;
    errorFlag++;
  }
  // initialize the deformed values
  *outStream << "constructing a simple deformed subset" << std::endl;
  Teuchos::RCP<Def_Map> map = Teuchos::rcp (new Def_Map());
  map->u_ = 5;
  map->v_ = 10;
  square.initialize(image,map,BILINEAR,DEF_INTENSITIES);
  square.write_tif("squareSubsetRef.tif",false);
  square.write_tif("squareSubsetDef.tif",true);
  square.write_subset_on_image("squareSubsetMapped.tif",image,map);
  // check simple motion intensity values
  *outStream << "checking the bilinear interpolation" << std::endl;
  bool def_values_error = false;
  for(size_t i=0;i<square.num_pixels();++i){
    if(square.def_intensities(i)!=(*image)(square.x(i)+map->u_,square.y(i)+map->v_))
      def_values_error = true;
  }
  if(def_values_error){
    *outStream << "Error, the def intensity values for the initialized square subset are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "checking the keys fourth order interpolant" << std::endl;
  map->u_ = 15;
  map->v_ = 12;
  square.initialize(image,map,KEYS_FOURTH_ORDER,DEF_INTENSITIES);
  square.write_tif("squareSubsetDefKeys.tif",true);
  bool keys_values_error = false;
  for(size_t i=0;i<square.num_pixels();++i){
    if(square.def_intensities(i)!=(*image)(square.x(i)+map->u_,square.y(i)+map->v_))
      keys_values_error = true;
  }
  if(keys_values_error){
    *outStream << "Error, the def intensity values for the keys initialized square subset are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "checking the mean value of the reference intensities" << std::endl;
  scalar_t ref_mean = 0.0;
  for(size_t i=0;i<square.num_pixels();++i){
    ref_mean += square.ref_intensities(i);
  }
  ref_mean/=square.num_pixels();
  scalar_t func_ref_mean = square.mean(REF_INTENSITIES);
  if(std::abs(func_ref_mean - ref_mean)>errorTol){
    *outStream << "Error, the reference mean is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "checking the mean value of the deformed intensities" << std::endl;
  scalar_t def_mean = 0.0;
  for(size_t i=0;i<square.num_pixels();++i){
    def_mean += square.def_intensities(i);
  }
  def_mean/=square.num_pixels();
  scalar_t func_def_mean = square.mean(DEF_INTENSITIES);
  if(std::abs(func_def_mean - def_mean)>errorTol){
    *outStream << "Error, the deformed mean is not correct" << std::endl;
    errorFlag++;
  }

  // TODO come up with a complex map and check the values

  *outStream << "--- End test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

