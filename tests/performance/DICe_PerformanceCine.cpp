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
#include <DICe_Cine.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <boost/timer/timer.hpp>

#include <iostream>

using namespace DICe;
using namespace boost::timer;

int main(int argc, char *argv[]) {

  if(argc!=5){
    std::cerr << "Usage: DICe_PerformanceCine <cine_file> <start_frame> <end_frame> <1 or 0>, (1=verbose)" << std::endl;
    return 1;
  }

  DICe::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  // only print output if args are given (for testing the output is quiet)
  if (std::atoi(argv[4]) == 1)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin performance test ---" << std::endl;

  // get the cine file name from the command line
  std::string cine_file = argv[1];
  *outStream << "reading cine file: " << cine_file << std::endl;

  cine::Cine_Reader cine(cine_file.c_str());//,outStream.getRawPtr());

  int_t start_frame = std::atoi(argv[2]);
  int_t end_frame = std::atoi(argv[3]);
  *outStream << "start frame:       " << start_frame << std::endl;
  *outStream << "end frame:         " << end_frame << std::endl;

  Teuchos::RCP<Image> image = Teuchos::rcp(new Image(cine.width(),cine.height(),0.0));

  cpu_timer thread_timer;
  {
    thread_timer.start();
    for(int_t i = start_frame; i<end_frame; ++i){
      cine.get_frame(image,i,true);
      //std::stringstream name;
      //name << "./frame_images/frame_" << i << ".tif";
      //image->write(name.str());
    }
    thread_timer.stop();
  }
  *outStream << "** read time" << thread_timer.format();


  *outStream << "--- End performance test ---" << std::endl;

  DICe::finalize();

  return 0;
}

