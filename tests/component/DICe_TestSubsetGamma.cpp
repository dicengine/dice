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
  work_t errtol  = 5.0E-2;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;


  *outStream << "---> CREATING STRIPES IMAGES " << std::endl;
  // create an image with black and white stripes:
  const int_t img_width = 199;
  const int_t img_height = 199;
  const int_t num_stripes = 10;
  const int_t stripe_width = 20;//img_width/num_stripes;
  Teuchos::ArrayRCP<storage_t> intensities(img_width*img_height,0);
  for(int_t y=0;y<img_height;++y){
    for(int_t stripe=0;stripe<=num_stripes/2;++stripe){
      for(int_t x=stripe*(2*stripe_width);x<stripe*(2*stripe_width)+stripe_width;++x){
        if(x>=img_width||x<0)continue;
        intensities[y*img_width+x] = 255;
      }
    }
  }
  *outStream << "creating the reference image" << std::endl;
  Teuchos::RCP<Image> refImg = Teuchos::rcp(new Image(img_width,img_height,intensities));
  Subset subset(100,100,99,99);
  subset.initialize(refImg);

  // track the correlation gamma for various pixel shifts:
  // no shift should result in gamma = 0.0 and when the images are opposite gamma should = 4.0
  for(int_t shift=0;shift<11;shift++){
    *outStream << "processing shift: " << shift*2 << "\n";
    Teuchos::ArrayRCP<storage_t> intensitiesShift(img_width*img_height,0.0);
    for(int_t y=0;y<img_height;++y){
      for(int_t stripe=0;stripe<=num_stripes/2;++stripe){
        for(int_t x=stripe*(2*stripe_width) - shift*2;x<stripe*(2*stripe_width)+stripe_width - shift*2;++x){
          if(x>=img_width||x<0)continue;
          intensitiesShift[y*img_width+x] = 255;
        }
      }
    }
    Teuchos::RCP<Image> defImg = Teuchos::rcp(new Image(img_width,img_height,intensitiesShift));

    *outStream << "initializing the subset" << std::endl;
    // initialize the reference and defored values for this subset
    subset.initialize(defImg,DEF_INTENSITIES);
    const work_t gamma = subset.gamma();
    *outStream << "gamma value: " << gamma << std::endl;
    if(std::abs(gamma - shift*0.4)>errtol){
      *outStream << "Error,  gamma is not " << shift*0.4 << " value=" << gamma << "\n";
      errorFlag++;
    }
  } // end shifts

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

