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
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create an image from file:
  *outStream << "creating an image from a tiff file " << std::endl;
  Image img("./images/ImageA.tif");
  img.write_tif("outImageA.tif");
  if(img.width()!=2048){
    *outStream << "Error, the image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(img.height()!=589){
    *outStream << "Error, the image height is not correct" << std::endl;
    errorFlag +=1;
  }

  // capture a portion of an image from file
  *outStream << "creating an image from a portion of a tiff file " << std::endl;
  Image sub_img("./images/ImageA.tif",100,100,300,200);
  sub_img.write_tif("outSubImageA.tif");
  if(sub_img.width()!=300){
    *outStream << "Error, the sub image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(sub_img.height()!=200){
    *outStream << "Error, the sub image height is not correct" << std::endl;
    errorFlag +=1;
  }
  // the pixel values from the sub image should line up with the image above, if given the right coordinates
  bool intensity_match_error = false;
  for(size_t y=0;y<sub_img.height();++y){
    for(size_t x=0;x<sub_img.width();++x){
      if(sub_img(x,y)!=img(x+sub_img.offset_x(),y+sub_img.offset_y()))
        intensity_match_error = true;
    }
  }
  if(intensity_match_error){
    *outStream << "Error, the intensities for the sub image do not match the global image" << std::endl;
    errorFlag+=1;
  }

  // create an image from an array
  *outStream << "creating an image from an array" << std::endl;
  const size_t array_w = 10;
  const size_t array_h = 10;
  intensity_t * intensities = new intensity_t[array_w*array_h];
  scalar_t * gx = new scalar_t[array_w*array_h];
  scalar_t * gy = new scalar_t[array_w*array_h];
  // populate the intensities with a sin/cos function
  for(size_t y=0;y<array_h;++y){
    for(size_t x=0;x<array_w;++x){
      intensities[y*array_w+x] = 255*std::cos(x/(4*DICE_PI))*std::sin(y/(4*DICE_PI));
      gx[y*array_w+x] = -255*(1/(4*DICE_PI))*std::sin(x/(4*DICE_PI))*std::sin(y/(4*DICE_PI));
      gy[y*array_w+x] = 255*(1/(4*DICE_PI))*std::cos(x/(4*DICE_PI))*std::cos(y/(4*DICE_PI));
    }
  }
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::compute_image_gradients,true);
  Image array_img(intensities,array_w,array_h,params);

  *outStream << "creating an image from an array RCP" << std::endl;
  Teuchos::ArrayRCP<intensity_t> intensityRCP(intensities,0,array_w*array_h,false);
  Image rcp_img(array_w,array_h,intensityRCP);

  bool intensity_value_error = false;
  bool grad_x_error = false;
  bool grad_y_error = false;
  scalar_t grad_tol = 1.0E-3;
  for(size_t y=2;y<array_h-2;++y){
    for(size_t x=2;x<array_w-2;++x){
      if(intensities[y*array_w+x] != array_img(x,y))
        intensity_value_error = true;
      //std::cout << " grad x " << gx[y*array_w+x] << " computed " << array_img.grad_x(x,y) << std::endl;
      if(std::abs(gx[y*array_w+x] - array_img.grad_x(x,y)) > grad_tol)
        grad_x_error = true;
      if(std::abs(gy[y*array_w+x] - array_img.grad_y(x,y)) > grad_tol)
        grad_y_error = true;
    }
  }
  if(intensity_value_error){
    *outStream << "Error, the intensity values are wrong" << std::endl;
    errorFlag++;
  }
  if(grad_x_error){
    *outStream << "Error, the flat x-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  if(grad_y_error){
    *outStream << "Error, the flat y-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "flat image gradients have been checked" << std::endl;

  grad_x_error = false;
  grad_y_error = false;
  // check the hierarchical gradients:
  const size_t team_size = 256;
  array_img.compute_gradients(true,team_size);
  for(size_t y=2;y<array_h-2;++y){
    for(size_t x=2;x<array_w-2;++x){
      if(std::abs(gx[y*array_w+x] - array_img.grad_x(x,y)) > grad_tol)
        grad_x_error = true;
      if(std::abs(gy[y*array_w+x] - array_img.grad_y(x,y)) > grad_tol)
        grad_y_error = true;
    }
  }
  if(grad_x_error){
    *outStream << "Error, hierarchical x-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  if(grad_y_error){
    *outStream << "Error, the hierarchical y-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "hierarchical image gradients have been checked" << std::endl;

  // create an image with a teuchos array and compare the values
  bool rcp_error = false;
  for(size_t y=0;y<array_h;++y){
    for(size_t x=0;x<array_w;++x){
      if(rcp_img(x,y) != array_img(x,y))
        rcp_error = true;
    }
  }
  if(rcp_error){
    *outStream << "Error, the rcp image was not created correctly" << std::endl;
    errorFlag++;
  }

  delete[] intensities;
  delete[] gx;
  delete[] gy;


  // test filtering an image:

  *outStream << "creating gauss filtered image filter size 5: outFilter5.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_5_params = rcp(new Teuchos::ParameterList());
  filter_5_params->set(DICe::gauss_filter_image,true);
  filter_5_params->set(DICe::gauss_filter_mask_size,5);
  Image filter_5_img("./images/ImageA.tif",filter_5_params);
  filter_5_img.write_tif("outFilter5.tif");

  *outStream << "creating gauss filtered image filter size 7: outFilter7.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_7_params = rcp(new Teuchos::ParameterList());
  filter_7_params->set(DICe::gauss_filter_image,true);
  filter_7_params->set(DICe::gauss_filter_mask_size,7);
  Image filter_7_img("./images/ImageA.tif",filter_7_params);
  filter_7_img.write_tif("outFilter7.tif");

  *outStream << "creating gauss filtered image filter size 9: outFilter9.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_9_params = rcp(new Teuchos::ParameterList());
  filter_9_params->set(DICe::gauss_filter_image,true);
  filter_9_params->set(DICe::gauss_filter_mask_size,9);
  Image filter_9_img("./images/ImageA.tif",filter_9_params);
  filter_9_img.write_tif("outFilter9.tif");

  *outStream << "creating gauss filtered image filter size 11: outFilter11.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_11_params = rcp(new Teuchos::ParameterList());
  filter_11_params->set(DICe::gauss_filter_image,true);
  filter_11_params->set(DICe::gauss_filter_mask_size,11);
  Image filter_11_img("./images/ImageA.tif",filter_11_params);
  filter_11_img.write_tif("outFilter11.tif");

  *outStream << "creating gauss filtered image filter size 13: outFilter13.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_13_params = rcp(new Teuchos::ParameterList());
  filter_13_params->set(DICe::gauss_filter_image,true);
  filter_13_params->set(DICe::gauss_filter_mask_size,13);
  Image filter_13_img("./images/ImageA.tif",filter_13_params);
  filter_13_img.write_tif("outFilter13.tif");

  *outStream << "creating gauss filtered image filter size 13 (with hierarchical parallelism) to compare to flat filter functor" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_h_params = rcp(new Teuchos::ParameterList());
  filter_h_params->set(DICe::gauss_filter_image,true);
  filter_h_params->set(DICe::gauss_filter_use_hierarchical_parallelism,true);
  filter_h_params->set(DICe::gauss_filter_team_size,256);
  filter_h_params->set(DICe::gauss_filter_mask_size,13);
  Image filter_h_img("./images/ImageA.tif",filter_h_params);

  bool filter_error = false;
  for(size_t y=0;y<filter_h_img.height();++y){
    for(size_t x=0;x<filter_h_img.width();++x){
      if(std::abs(filter_h_img(x,y) - filter_13_img(x,y)) > grad_tol)
        filter_error = true;
    }
  }
  if(filter_error){
    *outStream << "Error, the filtered image using the flat functor does not match the hierarchical one" << std::endl;
    errorFlag++;
  }
  *outStream << "hierarchical image filter has been checked" << std::endl;


  *outStream << "--- End test ---" << std::endl;



  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

