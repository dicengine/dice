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
#include <DICe.h>
#include <DICe_NetCDF.h>

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
  scalar_t errorTol = 0.001;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create a netcdf reader:
  Teuchos::RCP<netcdf::NetCDF_Reader> netcdf = Teuchos::rcp(new netcdf::NetCDF_Reader());
  Teuchos::RCP<Image> img = netcdf->get_image("../images/goes14.2016.222.215844.BAND_01.nc");
  //img->write("netcdf_image.tif");
  //img->write("netcdf.rawi");

  // read the gold version of this image
  Teuchos::RCP<Image> gold_img = Teuchos::rcp(new Image("../images/netcdf.rawi"));
  const scalar_t diff = img->diff(gold_img);
  *outStream << "gold diff " << diff << std::endl;
  if(std::abs(diff) > errorTol){
    errorFlag++;
    *outStream << "Error, the NetCDF image was not read correctly" << std::endl;
  }

  Image subImg("../images/goes14.2016.222.215844.BAND_01.nc",1235,491,496,370);
  //subImg.write("netcdf_sub_image.tif");
  //subImg.write("netcdf_sub_image.rawi");
  Teuchos::RCP<Image> gold_sub_img = Teuchos::rcp(new Image("../images/netcdf_sub_image.rawi"));
  const scalar_t sub_diff = subImg.diff(gold_sub_img);
  *outStream << "sub gold diff " << sub_diff << std::endl;
  if(std::abs(sub_diff) > errorTol){
    errorFlag++;
    *outStream << "Error, the NetCDF sub image was not read correctly" << std::endl;
  }


  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

