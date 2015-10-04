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
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  scalar_t errorTol = 1.0E-3;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create a subset by centroid, width and height
  *outStream << "creating a subset from cx, cy, width and height " << std::endl;
  int_t cx = 125;
  int_t cy = 250;
  int_t w = 13;
  int_t h = 19;
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
  const int_t num_pts = 48;
  Teuchos::ArrayRCP<int_t> x_coords(num_pts,0);
  Teuchos::ArrayRCP<int_t> y_coords(num_pts,0);
  for(int_t i=0;i<num_pts;++i){
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
  for(int_t i=0;i<num_pts;++i){
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
  for(int_t i=0;i<square.num_pixels();++i){
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
  square.initialize(image,DEF_INTENSITIES,map,BILINEAR);
  square.write_tiff("squareSubsetRef.tif",false);
  square.write_tiff("squareSubsetDef.tif",true);
  square.write_subset_on_image("squareSubsetMapped.tif",image,map);
  // check simple motion intensity values
  *outStream << "checking the bilinear interpolation" << std::endl;
  bool def_values_error = false;
  for(int_t i=0;i<square.num_pixels();++i){
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
  square.initialize(image,DEF_INTENSITIES,map,KEYS_FOURTH);
  square.write_tiff("squareSubsetDefKeys.tif",true);
  bool keys_values_error = false;
  for(int_t i=0;i<square.num_pixels();++i){
    if(square.def_intensities(i)!=(*image)(square.x(i)+map->u_,square.y(i)+map->v_))
      keys_values_error = true;
  }
  if(keys_values_error){
    *outStream << "Error, the def intensity values for the keys initialized square subset are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "checking the mean value of the reference intensities" << std::endl;
  scalar_t ref_mean = 0.0;
  scalar_t ref_sum = 0.0;
  for(int_t i=0;i<square.num_pixels();++i){
    ref_mean += square.ref_intensities(i);
  }
  ref_mean/=square.num_pixels();
  for(int_t i=0;i<square.num_pixels();++i){
    ref_sum += (square.ref_intensities(i)-ref_mean)*(square.ref_intensities(i)-ref_mean);
  }
  ref_sum = std::sqrt(ref_sum);
  scalar_t func_ref_sum = 0.0;
  scalar_t func_ref_mean = square.mean(REF_INTENSITIES,func_ref_sum);
  if(std::abs(func_ref_mean - ref_mean)>errorTol){
    *outStream << "Error, the reference mean is not correct" << std::endl;
    errorFlag++;
  }
  if(std::abs(func_ref_sum - ref_sum)>errorTol){
    *outStream << "Error, the reference mean sum is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "checking the mean value of the deformed intensities" << std::endl;
  scalar_t def_mean = 0.0;
  scalar_t def_sum = 0.0;
  for(int_t i=0;i<square.num_pixels();++i){
    def_mean += square.def_intensities(i);
  }
  def_mean/=square.num_pixels();
  for(int_t i=0;i<square.num_pixels();++i){
    def_sum += (square.def_intensities(i)-def_mean)*(square.def_intensities(i)-def_mean);
  }
  def_sum = std::sqrt(def_sum);
  scalar_t func_def_sum = 0.0;
  scalar_t func_def_mean = square.mean(DEF_INTENSITIES,func_def_sum);
  if(std::abs(func_def_mean - def_mean)>errorTol){
    *outStream << "Error, the deformed mean is not correct" << std::endl;
    errorFlag++;
  }
  if(std::abs(func_def_sum - def_sum)>errorTol){
    *outStream << "Error, the deformed mean sum is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "the mean values and mean sum values have been checked" << std::endl;
  // TODO come up with a complex mapping and check the values

  *outStream << "creating a conformal subset" << std::endl;
  std::vector<int_t> shape_1_x(5);
  std::vector<int_t> shape_1_y(5);
  shape_1_x[0] = 940; shape_1_y[0] = 422;
  shape_1_x[1] = 951; shape_1_y[1] = 399;
  shape_1_x[2] = 964; shape_1_y[2] = 413;
  shape_1_x[3] = 980; shape_1_y[3] = 413;
  shape_1_x[4] = 959; shape_1_y[4] = 432;
  Teuchos::RCP<DICe::Polygon> poly1 = Teuchos::rcp(new DICe::Polygon(shape_1_x,shape_1_y));
  std::vector<int_t> shape_2_x(4);
  std::vector<int_t> shape_2_y(4);
  shape_2_x[0] = 982;  shape_2_y[0] = 428;
  shape_2_x[1] = 1000; shape_2_y[1] = 428;
  shape_2_x[2] = 991;  shape_2_y[2] = 456;
  shape_2_x[3] = 978;  shape_2_y[3] = 445;
  Teuchos::RCP<DICe::Polygon> poly2 = Teuchos::rcp(new DICe::Polygon(shape_2_x,shape_2_y));
  DICe::multi_shape boundary;
  boundary.push_back(poly1);
  boundary.push_back(poly2);

  // create an area that is deactivated in the subset:
  Teuchos::RCP<DICe::Rectangle> rect = Teuchos::rcp(new DICe::Rectangle(955,423,11,7));
  DICe::multi_shape excluded;
  excluded.push_back(rect);

  DICe::Conformal_Area_Def subset_def(boundary,excluded);
  int_t ccx = 974;
  int_t ccy = 428;
  Subset conformal_subset(ccx,ccy,subset_def);
  conformal_subset.initialize(image);
  conformal_subset.write_tiff("conformal.tif");
  // read in the image that was just created and compare to a gold copy:
  Image conf_img("./conformal.tif");
  conf_img.write_rawi("conformal.rawi");
  Image conf_img_exact("./images/conformal.rawi");
  //conformal_subset.write_subset_on_image("ConformalOnImage.tiff",image);
  // compare the sizes and intensity values
  if(conf_img.width()!=conf_img_exact.width() || conf_img.height()!=conf_img_exact.height()){
    *outStream << "Error, the number of pixels in the conformal subset image is not correct." << std::endl;
    errorFlag++;
  }
  bool intensity_error = false;
  for(int_t y=0;y<conf_img.height();++y){
    for(int_t x=0;x<conf_img.width();++x){
      if(conf_img(x,y) != conf_img_exact(x,y)){
        intensity_error = true;
      }
    }
  }
  if(intensity_error){
    *outStream << "Error, the conformal subset's intensity values are not correct." << std::endl;
    errorFlag++;
  }
  *outStream << "Conformal subset intensity values have been checked." << std::endl;



  *outStream << "--- End test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

