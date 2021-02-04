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

/*! \file  DICe_TestSchema.cpp
    \brief Testing of schema class
    NOTE: correlations are not tested here as they depend on DICe::Objective
*/

#include <DICe_Schema.h>
#include <DICe_Image.h>
#include <DICe.h>

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

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

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "creating images to use in schema creation" << std::endl;
  // read the ref and def images to create intensity profiles to use in schema construction:
  std::string refString = "./images/refSyntheticSpeckled.tif";
  std::string defString = "./images/defSyntheticSpeckled.tif";
  Teuchos::RCP<DICe::Image> imgRef = Teuchos::rcp( new DICe::Image(refString.c_str()));
  Teuchos::RCP<DICe::Image> imgDef = Teuchos::rcp( new DICe::Image(defString.c_str()));
  *outStream << "image dimensions: w = " << imgRef->width() << " h = " << imgRef->height() << std::endl;
  Teuchos::ArrayRCP<work_t> imgRefArray = imgRef->intensities();
  Teuchos::ArrayRCP<work_t> imgDefArray = imgDef->intensities();
  assert(imgRef->width() == imgDef->width());
  assert(imgRef->height() == imgDef->height());
  const int_t img_width = imgRef->width();
  const int_t img_height = imgRef->height();

  *outStream << "testing schema construction from image file names" << std::endl;
  Teuchos::RCP<DICe::Schema> schemaString = Teuchos::rcp(new DICe::Schema());
  schemaString->set_ref_image(refString);
  schemaString->set_def_image(defString);
  *outStream << "success" << std::endl;

  *outStream << "testing schema construction from array " << std::endl;
  Teuchos::RCP<DICe::Schema> schemaArray = Teuchos::rcp(new DICe::Schema());
  schemaArray->set_ref_image(img_width,img_height,imgRefArray);
  schemaArray->set_def_image(img_width,img_height,imgDefArray);
  *outStream << "success" << std::endl;

  *outStream << "testing schema construction from images " << std::endl;
  Teuchos::RCP<DICe::Schema> schemaImage = Teuchos::rcp(new DICe::Schema());
  schemaImage->set_ref_image(imgRef);
  schemaImage->set_def_image(imgDef);
  *outStream << "success" << std::endl;


  // check that the default values were set properly and that the images are of the right size:

  std::vector< Teuchos::RCP<DICe::Schema> > schemaVec;
  schemaVec.push_back(schemaString);  //0
  schemaVec.push_back(schemaArray);   //1
  schemaVec.push_back(schemaImage);   //2

  for(size_t i=0;i<schemaVec.size();++i){
    *outStream << "testing schema " << i << " params" << std::endl;
    if(schemaVec[i]->img_width()!=imgRef->width()){
      *outStream << "Error, image width is not right" << std::endl;
      errorFlag++;
    }
    if(schemaVec[i]->img_height()!=imgRef->height()){
      *outStream << "Error, image height is not right" << std::endl;
      errorFlag++;
    }
    if(schemaVec[i]->local_num_subsets()!=0){
      *outStream << "Error, num_points is not right" << std::endl;
      errorFlag++;
    }
    if(schemaVec[i]->interpolation_method()!=DICe::KEYS_FOURTH){
      *outStream << "Error, interpolation method is not right" << std::endl;
      errorFlag++;
    }
    if(schemaVec[i]->correlation_routine()!=DICe::GENERIC_ROUTINE){
      *outStream << "Error, correlation routine is not right" << std::endl;
      errorFlag++;
    }
    if(schemaVec[i]->projection_method()!=DICe::DISPLACEMENT_BASED){
      *outStream << "Error, projection method is not right" << std::endl;
      errorFlag++;
    }
    if(schemaVec[i]->optimization_method()!=DICe::GRADIENT_BASED_THEN_SIMPLEX){
      *outStream << "Error, optimization method is not right" << std::endl;
      errorFlag++;
    }
  }

  // try passing in an invalid parameter to make sure that it throws:
  Teuchos::RCP<Teuchos::ParameterList> badParams = rcp(new Teuchos::ParameterList());
  badParams->set("this_should_not_work",true);
  // this param should only be available internally and should also throw an error
  badParams->set(DICe::tolerance,0.01);
  bool exception_thrown_as_it_should = false;
  try{
    Teuchos::RCP<DICe::Schema> schemaTextBadParams = Teuchos::rcp(new DICe::Schema(badParams));
  }
  catch (std::logic_error &e) {
    exception_thrown_as_it_should = true;
    *outStream << e.what() << "\n";
  }; // end try
  if (!exception_thrown_as_it_should) {
    *outStream << "Error, An exception should have been thrown for invalid parameters but was not.\n";
    errorFlag++;
  };

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

