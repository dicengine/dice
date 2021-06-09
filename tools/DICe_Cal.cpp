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
#include <DICe_Calibration.h>
#include <DICe_CameraSystem.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <cassert>

using namespace std;
using namespace DICe;

/// Initializes the cross correlation between two images from the left and right camera, respectively

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // read the parameters from the input file:
  if(argc!=2 && argc!=3){ // executable and input file
    std::cout << "Usage: DICe_Cal <cal_input_file.xml> [output_file=cal.xml]" << std::endl;
    exit(1);
  }

  std::string input_file = argv[1];
  std::string output_file = "cal.xml";
  if(argc==3)
    output_file = argv[2];
  print_banner();
  std::cout << "\nDICe_Cal begin\n" << std::endl;
  std::cout << "input file:  " << input_file << std::endl;
  std::cout << "output file: " << output_file << std::endl;

  DICe::Calibration cal(input_file);
  // create the intersection points and generate a calibrated camera system
  scalar_t rms = 0.0;
  Teuchos::RCP<Camera_System> cam_sys = cal.calibrate(rms);
  // write the calibration parameters to file which includes the interesection points
  cam_sys->write_camera_system_file(output_file);
//#ifdef DICE_DEBUG_MSG
  std::cout << *cam_sys.get() << std::endl;
//#endif
  std::cout << "\nDICe_Cal complete\n" << std::endl;

  DICe::finalize();

  if(rms < 0.0){
    std::cout << "\n*** Error, stereo calibration failed" << std::endl;
    return -1;
  }
  if(rms > 1.0){
    std::cout << "\n*** Warning, RMS error is large: " << rms << ". Should be under 0.5." << std::endl;
  }
  return 0;
}

