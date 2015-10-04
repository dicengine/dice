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
#include <DICe_FFT.h>
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
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  scalar_t errorTol = 1.0E-5;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create an image and take the polar transform:
  *outStream << "testing polar transform of an image" << std::endl;
  Teuchos::RCP<Image> baboon = Teuchos::rcp(new Image("./images/baboon.rawi"));
  Teuchos::RCP<Image> polar_baboon = polar_transform(baboon);
  Teuchos::RCP<Image> polar_baboon_test = Teuchos::rcp(new Image("./images/baboon_polar.rawi"));
  const scalar_t diff_polar = polar_baboon->diff(polar_baboon_test);
  *outStream << "polar image diff: " << diff_polar << std::endl;
  if(diff_polar > errorTol){
    *outStream << "Error, the polar transform intensity values are not correct" << std::endl;
    errorFlag++;
  }

  *outStream << "testing the fft of an image" << std::endl;
  // compute the fft of the baboon image:
  Teuchos::RCP<Image> baboon_fft = image_fft(baboon);
  baboon_fft->write_tiff("baboon0.tif");
  Teuchos::RCP<Image> baboon_fft_test = Teuchos::rcp(new Image("./images/baboon_fft.rawi"));
  const scalar_t diff_fft = baboon_fft->diff(baboon_fft_test);
  *outStream << "fft image diff: " << diff_fft << std::endl;
  if(diff_fft > errorTol){
    *outStream << "Error, the fft of the intensity values is not correct" << std::endl;
    errorFlag++;
  }

  *outStream << "testing the fft cross correlation of a rotated image (counter-clockwise 30 deg)" << std::endl;
  Teuchos::RCP<Image> baboon_rot = Teuchos::rcp(new Image("./images/baboon_rotCC30.rawi"));
  // take the fft of the rotated image
  Teuchos::RCP<Image> baboon_rot_fft = image_fft(baboon_rot);
  // take the polar transform of the original and rotated images:
  Teuchos::RCP<Image> baboon_rot_pol = polar_transform(baboon_rot_fft);
  Teuchos::RCP<Image> baboon_pol = polar_transform(baboon_fft);
  // fft correlate the two polar fft images:
  scalar_t r = 0.0;
  scalar_t t = 0.0;
  phase_correlate_x_y(baboon_rot_pol,baboon_pol,t,r,true);
  *outStream << "radius: " << r << " theta: " << t << std::endl;
  if(std::abs(t - 0.5236) > 0.01){
    *outStream << "Error, the angle of rotation is not correct for the counter clockwise 30 deg case" << std::endl;
    errorFlag++;
  }
  if(std::abs(r) > 0.01){
    *outStream << "Error, the radius is not correct for the counter clockwise 30 deg case" << std::endl;
    errorFlag++;
  }

  *outStream << "testing the fft cross correlation of a rotated image (clockwise 45 deg)" << std::endl;
  Teuchos::RCP<Image> baboon_rot45 = Teuchos::rcp(new Image("./images/baboon_rotC45.rawi"));
  // take the fft of the rotated image
  Teuchos::RCP<Image> baboon_rot45_fft = image_fft(baboon_rot45);
  baboon_rot45_fft->write_tiff("baboon_45.tif");
 // take the polar transform of the original and rotated images:
  Teuchos::RCP<Image> baboon_rot45_pol = polar_transform(baboon_rot45_fft);
  // fft correlate the two polar fft images:
  scalar_t r45 = 0.0;
  scalar_t t45 = 0.0;
  phase_correlate_x_y(baboon_rot45_pol,baboon_pol,t45,r45,true);
  *outStream << "radius: " << r45 << " theta: " << t45 << std::endl;
  if(std::abs(t45 + 0.7854) > 0.01){
    *outStream << "Error, the angle of rotation is not correct for the clockwise 45 deg case" << std::endl;
    errorFlag++;
  }
  if(std::abs(r45) > 0.01){
    *outStream << "Error, the radius is not correct for the clockwise 45 deg case" << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

