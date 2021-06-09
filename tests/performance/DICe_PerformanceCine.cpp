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
#include <DICe_Image.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  if(argc!=5){
    std::cerr << "Usage: DICe_PerformanceCine <cine_file> <start_frame> <end_frame> <1 or 0>, (1=verbose)" << std::endl;
    return 1;
  }

  DICe::initialize(argc, argv);

  Teuchos::RCP<Teuchos::Time> read_time  = Teuchos::TimeMonitor::getNewCounter("read time");

  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  // only print output if args are given (for testing the output is quiet)
  if (std::strtol(argv[4],NULL,0) == 1)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin performance test ---" << std::endl;

  // get the cine file name from the command line
  std::string cine_file = argv[1];
  *outStream << "reading cine file: " << cine_file << std::endl;
  std::string stripped_fileName = cine_file;
  stripped_fileName.erase(stripped_fileName.length()-5,5);
  *outStream << "Cine file base name: " << stripped_fileName << std::endl;

  int_t start_frame = std::strtol(argv[2],NULL,0);
  int_t end_frame = std::strtol(argv[3],NULL,0);
  *outStream << "start frame:       " << start_frame << std::endl;
  *outStream << "end frame:         " << end_frame << std::endl;

  {
    Teuchos::TimeMonitor read_time_monitor(*read_time);
    for(int_t i = start_frame; i<end_frame; ++i){
      std::stringstream name;
      name << stripped_fileName << "_" << i << ".cine";
      //image->write(name.str());
      Teuchos::RCP<Image> image = Teuchos::rcp(new Image(name.str().c_str()));
      //std::stringstream name;
      //name << "./frame_images/frame_" << i << ".tif";
      //image->write(name.str());
    }
  }

  Teuchos::TimeMonitor::summarize(*outStream,false,true,false/*zero timers*/);

  *outStream << "--- End performance test ---" << std::endl;

  DICe::finalize();

  return 0;
}

