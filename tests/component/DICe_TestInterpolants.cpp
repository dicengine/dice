// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Subset.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create a bi-linear image from an array
  *outStream << "creating an image from an array" << std::endl;
  const int_t array_w = 40;
  const int_t array_h = 30;
  Teuchos::ArrayRCP<intensity_t> intensities(array_w*array_h,0.0);
  // populate the intensities with a sin/cos function
  intensity_t x_val = 0.0, y_val = 0.0;
  for(int_t y=0;y<array_h;++y){
    for(int_t x=0;x<array_w;++x){
      x_val = 255*((scalar_t)x/(scalar_t)array_w);
      y_val = 255*((scalar_t)y/(scalar_t)array_h);
      intensities[y*array_w+x] = x_val*y_val;
    }
  }
  Teuchos::RCP<Image> array_img = Teuchos::rcp(new Image(array_w,array_h,intensities));
  array_img->write("interp_array_img.tif");

  const int_t cx = array_w/2;
  const int_t cy = array_h/2;

  *outStream << "creating a subset" << std::endl;
  Subset subset_bi(cx,cy,10,10);
  Subset subset_keys(cx,cy,10,10);
  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory();
  const scalar_t u = 0.3589;
  const scalar_t v = -2.89478;
  const scalar_t t = 0.262;
  shape_function->insert_motion(u,v,t);
  subset_bi.initialize(array_img,DEF_INTENSITIES,shape_function,BILINEAR);
  subset_keys.initialize(array_img,DEF_INTENSITIES,shape_function,KEYS_FOURTH);

  *outStream << "testing the intensity values" << std::endl;
  scalar_t error_bi = 0.0;
  scalar_t error_keys = 0.0;
  scalar_t px=0.0,py=0.0;
  for(int_t i=0;i<subset_bi.num_pixels();++i){
    shape_function->map(subset_bi.x(i),subset_bi.y(i),cx,cy,px,py);
    x_val = 255.0*px/(scalar_t)array_w;
    y_val = 255.0*py/(scalar_t)array_h;
    const scalar_t exact = x_val*y_val;
    error_bi += std::abs(subset_bi.def_intensities(i) - exact);
    error_keys += std::abs(subset_keys.def_intensities(i) - exact);
    //std::cout << " x " << subset_bi.x(i) << " y " << subset_bi.y(i) <<
    //    " exact " << exact << " bi " << subset_bi.def_intensities(i) <<
    //    " keys " << subset_keys.def_intensities(i) << std::endl;
    //std::cout << subset_keys.x(i) << " " << subset_keys.y(i) << " " << std::abs(subset_keys.def_intensities(i) - exact) << std::endl;
    //std::cout << subset_keys.x(i) << " " << subset_keys.y(i) << " " << std::abs(subset_bi.def_intensities(i) - exact) << std::endl;
  }
  *outStream << "bilinear interp error: " << error_bi << std::endl;
  *outStream << "keys interp error: " << error_keys << std::endl;
  if(error_bi > 1.0){
    *outStream << "Error, bilinear interpolation failed simple translation" << std::endl;
    errorFlag++;
  }
  if(error_keys > 2.0){ // high error in case float is used vs. double
    *outStream << "Error, keys fourth interpolation failed simple translation" << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

