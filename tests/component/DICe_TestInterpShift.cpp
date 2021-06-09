// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
#include <DICe_Schema.h>
#include <DICe_Image.h>

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

  Teuchos::ArrayRCP<scalar_t> coords_x(1,25.0);
  Teuchos::ArrayRCP<scalar_t> coords_y(1,20.0);
  const int_t subset_size = 21;
  const scalar_t sin_factor = 5.0;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  params->set(DICe::interpolation_method,DICe::KEYS_FOURTH);
  params->set(DICe::optimization_method,DICe::GRADIENT_BASED);
  params->set(DICe::max_solver_iterations_fast,100);
  // create a bi-linear image from an array
  *outStream << "creating an image from an array" << std::endl;
  const int_t array_w = 50;
  const int_t array_h = 40;
  Teuchos::ArrayRCP<storage_t> ref_intensities(array_w*array_h,0);
  // populate the intensities with a sin/cos function
  scalar_t x_val = 0.0, y_val = 0.0;
  for(int_t y=0;y<array_h;++y){
    for(int_t x=0;x<array_w;++x){
      x_val = std::abs(std::cos(((scalar_t)x/(scalar_t)array_w)*sin_factor*DICE_PI));
      y_val = std::abs(std::cos(((scalar_t)y/(scalar_t)array_h)*sin_factor*DICE_PI));
      ref_intensities[y*array_w+x] = static_cast<storage_t>(x_val*y_val*255.0);
    }
  }
  Teuchos::RCP<Image> ref_img = Teuchos::rcp(new Image(array_w,array_h,ref_intensities));
  //ref_img->write("shift_ref.png");

  std::vector<scalar_t> x_error;

  const scalar_t shift_step = 0.1;
  for(scalar_t shift = 0.0;shift<=1.0; shift+=shift_step){
    Teuchos::ArrayRCP<storage_t> def_intensities(array_w*array_h,0.0);
    // populate the intensities with a sin/cos function
    for(int_t y=0;y<array_h;++y){
      for(int_t x=0;x<array_w;++x){
        x_val = std::abs(std::cos((((scalar_t)x-shift)/(scalar_t)array_w)*sin_factor*DICE_PI));
        y_val = std::abs(std::cos(((scalar_t)y/(scalar_t)array_h)*sin_factor*DICE_PI));
        def_intensities[y*array_w+x] = static_cast<storage_t>(x_val*y_val*255.0);
      }
    }
    Teuchos::RCP<Image> def_img = Teuchos::rcp(new Image(array_w,array_h,def_intensities));
//    std::stringstream def_name;
//    def_name << "shift_def_" << shift << ".png";
//    def_img->write(def_name.str());

    DICe::Schema schema(coords_x,coords_y,subset_size,Teuchos::null,Teuchos::null,params);
    schema.set_ref_image(ref_img);
    schema.set_def_image(def_img);
    schema.execute_correlation();
    x_error.push_back(std::abs(schema.global_field_value(0,DICe::field_enums::SUBSET_DISPLACEMENT_X_FS) - shift));
    //schema.print_fields();
  }

  const scalar_t tol = 0.01;
  for (size_t i=0;i<x_error.size();++i){
    *outStream << "step " << i << " error " << x_error[i] << std::endl;
    if(x_error[i]>tol){
      *outStream << "failed step" << std::endl;
      errorFlag++;
    }
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

