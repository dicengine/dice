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

/*! \file  DICe_TestObjective.cpp
    \brief Testing of objective class construction and gamma methods
    actual correlation methods are tested in a separate test, not here
*/

#include <DICe_Schema.h>
#include <DICe_ObjectiveZNSSD.h>
#include <DICe.h>

#include <Teuchos_oblackholestream.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  int_t errorFlag  = 0;
  scalar_t errtol  = 5.0E-2;

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "creating stripes images " << std::endl;
  // create an image with black and white stripes:
  const int_t img_width = 199;
  const int_t img_height = 199;
  const int_t num_stripes = 10;
  const int_t stripe_width = 20;//img_width/num_stripes;
  Teuchos::ArrayRCP<intensity_t> intensities(img_width*img_height,0.0);
  for(int_t y=0;y<img_height;++y){
    for(int_t stripe=0;stripe<=num_stripes/2;++stripe){
      for(int_t x=stripe*(2*stripe_width);x<stripe*(2*stripe_width)+stripe_width;++x){
        if(x>=img_width)continue;
        intensities[y*img_width+x] = 255.0;
      }
    }
  }
  Teuchos::RCP<DICe::Image> refImg = Teuchos::rcp(new DICe::Image(img_width,img_height,intensities));
  *outStream << "reference striped image created successfuly\n";

  // testing objective gamma:

  // dummy deformation vector to pass to objective
  Teuchos::RCP<std::vector<scalar_t> > deformation = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));

  *outStream << "testing ZNSSD correlation" << std::endl;

  // track the correlation gamma for various pixel shifts:
  // no shift should result in gamma = 0.0 and when the images are opposite gamma should = 4.0
  for(int_t shift=0;shift<11;shift++){
    *outStream << "processing shift: " << shift*2 << "\n";
    Teuchos::ArrayRCP<intensity_t> intensitiesShift(img_width*img_height,0.0);
    for(int_t y=0;y<img_height;++y){
      for(int_t stripe=0;stripe<=num_stripes/2;++stripe){
        for(int_t x=stripe*(2*stripe_width) - shift*2;x<stripe*(2*stripe_width)+stripe_width - shift*2;++x){
          if(x>=img_width)continue;
          if(x<0)continue;
          intensitiesShift[y*img_width+x] = 255.0;
        }
      }
    }
    Teuchos::RCP<DICe::Image> defImg = Teuchos::rcp(new DICe::Image(img_width,img_height,intensitiesShift));
    // create a temp schema:
    Teuchos::ArrayRCP<scalar_t> coords_x(1,100);
    Teuchos::ArrayRCP<scalar_t> coords_y(1,100);
    DICe::Schema * schema = new DICe::Schema(coords_x,coords_y,99);
    schema->set_ref_image(img_width,img_height,intensities);
    schema->set_def_image(img_width,img_height,intensitiesShift);
    //schema->sync_fields_all_to_dist(); // distribute the fields across processors if necessary
    // create an objective:
    Teuchos::RCP<DICe::Objective_ZNSSD> obj = Teuchos::rcp(new DICe::Objective_ZNSSD(schema,0));
    // evaluate the correlation value:
    const scalar_t gamma =  obj->gamma(deformation);
    *outStream << "gamma value: " << gamma << std::endl;
    if(std::abs(gamma - shift*0.4)>errtol){
      *outStream << "Error, gamma is not " << shift*0.4 << " value=" << gamma << "\n";
      errorFlag++;
    }
    delete schema;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

