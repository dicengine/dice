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
// Questions?:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

/*! \file  DICe_Test_Objective.cpp
    \brief Testing of objective class construction and gamma methods
    actual correlation methods are tested in a separate test, not here
*/

#include <DICe_Schema.h>
#include <DICe_ObjectiveZNSSD.h>
#include <DICe_Types.h>

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <iostream>

int main(int argc, char *argv[]) {

  // only print output if args are given (for testing the output is quiet)
  SizeT iprint     = argc - 1;
  // for serial, the global MPI session is a no-op, but in parallel
  // ensures that MPI_Init is called (needed by the schema)
  Teuchos::GlobalMPISession mpi_session(&argc, &argv, NULL);
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  SizeT errorFlag  = 0;

  RealT errtol  = 5.0E-2;
  RealT errtolSoft = 1.0E7; // gamma values for SSD are large
  // TODO find a better way to check this

  try {

    *outStream << "---> CREATING STRIPES IMAGES " << std::endl;
    // create an image with black and white stripes:
    const SizeT img_width = 199;
    const SizeT img_height = 199;
    const SizeT num_stripes = 10;
    const SizeT stripe_width = 20;//img_width/num_stripes;
    const SizeT size = img_width * img_height;
    Teuchos::ArrayRCP<RealT> intensities(img_width*img_height,0.0);
    for(SizeT y=0;y<img_height;++y){
      for(SizeT stripe=0;stripe<=num_stripes/2;++stripe){
        for(SizeT x=stripe*(2*stripe_width);x<stripe*(2*stripe_width)+stripe_width;++x){
          if(x>=img_width)continue;
          intensities[y*img_width+x] = 255.0;
        }
      }
    }
    Teuchos::RCP<DICe::Image<RealT,SizeT> > refImg = Teuchos::rcp(new DICe::Image<RealT,SizeT> (img_width,img_height,intensities));
    *outStream << "---> REFERENCE IMAGE OF STRIPES HAS BEEN CREATED SUCCESSFULLY\n";

    // testing objective gamma:

    Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
    params->set(DICe::correlation_method, DICe::ZNSSD);
    // dummy deformation vector to pass to objective
    Teuchos::RCP<std::vector<RealT> > deformation = Teuchos::rcp(new std::vector<RealT>(DICE_DEFORMATION_SIZE,0.0));

    *outStream << "---> TESTING OBJECTIVE GAMMA WITH OBJECTIVE NORMALIZATION (using ZNSSD not SSD)" << std::endl;

    // track the correlation gamma for various pixel shifts:
    // no shift should result in gamma = 0.0 and when the images are opposite gamma should = 4.0
    for(SizeT shift=0;shift<11;shift++){
      *outStream << "---> PROCESSING SHIFT " << shift*2 << "\n";
      Teuchos::ArrayRCP<RealT> intensitiesShift(img_width*img_height,0.0);
      for(SizeT y=0;y<img_height;++y){
        for(SizeT stripe=0;stripe<=num_stripes/2;++stripe){
          for(SizeT x=stripe*(2*stripe_width) - shift*2;x<stripe*(2*stripe_width)+stripe_width - shift*2;++x){
            if(x>=img_width)continue;
            if(x<0)continue;
            intensitiesShift[y*img_width+x] = 255.0;
          }
        }
      }
      Teuchos::RCP<DICe::Image<RealT,SizeT> > defImg = Teuchos::rcp(new DICe::Image<RealT,SizeT> (img_width,img_height,intensitiesShift));
      // create a temp schema:
      DICe::Schema<RealT,SizeT> * schema = new DICe::Schema<RealT,SizeT>(img_width,img_height,intensities,intensitiesShift);
      schema->initialize(1,99);
      schema->field_value(0,DICe::COORDINATE_X) = 100;
      schema->field_value(0,DICe::COORDINATE_Y) = 100;
      schema->set_params(params);
      schema->sync_fields_all_to_dist(); // distribute the fields across processors if necessary
      if(!schema->use_objective_normalization()){
        *outStream << "---> POSSIBLE ERROR ABOVE!: Objective normalization should be active.\n";
        errorFlag++;
      }
      // create an objective:
      Teuchos::RCP<DICe::Objective_ZNSSD<RealT> > obj = Teuchos::rcp(new DICe::Objective_ZNSSD<RealT>(schema,0));
      // evaluate the correlation value:
      const RealT gamma =  obj->gamma(deformation);
      *outStream << "---> GAMMA VALUE: " << gamma << std::endl;
      if(std::abs(gamma - shift*0.4)>errtol){
        *outStream << "---> POSSIBLE ERROR ABOVE!  gamma is not " << shift*0.4 << " value=" << gamma << "\n";
        errorFlag++;
      }
      delete schema;
    }

    *outStream << "---> TESTING OBJECTIVE GAMMA WITHOUT OBJECTIVE NORMALIZATION (using SSD not ZNSSD)" << std::endl;

    params->set(DICe::correlation_method, DICe::SSD);

    const RealT gammaExactStep = 6.43748e+07;
    // track the correlation gamma for various pixel shifts:
    // no shift should result in gamma = 0.0 and when the images are opposite gamma should = 4.0
    for(SizeT shift=0;shift<11;shift++){
      *outStream << "---> PROCESSING SHIFT " << shift*2 << "\n";
      Teuchos::ArrayRCP<RealT> intensitiesShift(img_width*img_height,0.0);
      for(SizeT y=0;y<img_height;++y){
        for(SizeT stripe=0;stripe<=num_stripes/2;++stripe){
          for(SizeT x=stripe*(2*stripe_width) - shift*2;x<stripe*(2*stripe_width)+stripe_width - shift*2;++x){
            if(x>=img_width)continue;
            if(x<0)continue;
            intensitiesShift[y*img_width+x] = 255.0;
          }
        }
      }
      Teuchos::RCP<DICe::Image<RealT,SizeT> > defImg = Teuchos::rcp(new DICe::Image<RealT,SizeT> (img_width,img_height,intensitiesShift));
      // create a temp schema:
      DICe::Schema<RealT,SizeT> * schema = new DICe::Schema<RealT,SizeT>(img_width,img_height,intensities,intensitiesShift);
      schema->initialize(1,99);
      schema->field_value(0,DICe::COORDINATE_X) = 100;
      schema->field_value(0,DICe::COORDINATE_Y) = 100;
      schema->set_params(params);
      schema->sync_fields_all_to_dist(); // distribute the fields across processors if necessary
      if(schema->use_objective_normalization()){
        *outStream << "---> POSSIBLE ERROR ABOVE!: Objective normalization should not be active.\n";
        errorFlag++;
      }
      // create an objective:
      Teuchos::RCP<DICe::Objective_ZNSSD<RealT> > obj = Teuchos::rcp(new DICe::Objective_ZNSSD<RealT>(schema,0));
      // evaluate the correlation value:
      const RealT gamma =  obj->gamma(deformation);
      *outStream << "---> GAMMA VALUE: " << gamma << std::endl;
      if(std::abs(gamma - gammaExactStep*shift)>errtolSoft){
        *outStream << "---> POSSIBLE ERROR ABOVE!  gamma is not " << gammaExactStep*shift << " value=" << gamma << "\n";
        errorFlag++;
      }
      delete schema;
    }
  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

