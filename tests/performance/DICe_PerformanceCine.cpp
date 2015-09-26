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
#include <DICe_Cine.h>
#include <DICe_Kokkos.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <boost/timer/timer.hpp>

#include <iostream>

using namespace DICe;
using namespace boost::timer;

int main(int argc, char *argv[]) {

  if(argc!=3){
    std::cerr << "Usage: DICe_PerformanceCine <cine_file> <1 or 0>, (1=verbose)" << std::endl;
    return 1;
  }

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  // only print output if args are given (for testing the output is quiet)
  if (argc > 2)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin performance test ---" << std::endl;

  // get the cine file name from the command line
  std::string cine_file = argv[1];
  *outStream << "reading cine file: " << cine_file << std::endl;

  // read 1000 images from a cine file
  cine::Cine_Reader cine(cine_file.c_str());

  // write out the first image to make sure that the cine read was successful
  //Teuchos::RCP<Image> frame_0 = cine.get_frame(0);
  //frame_0->write_tiff("cine0.tiff");

  const size_t num_frames = 1000;
  cpu_timer read_timer;
  {
    read_timer.start();
    for(size_t frame = 0; frame < num_frames; ++frame){
      Teuchos::RCP<Image> frame_0 = cine.get_frame(frame);
    }
    read_timer.stop();
  }
  *outStream << "** read time" << read_timer.format();


  *outStream << "--- End performance test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  return 0;
}

