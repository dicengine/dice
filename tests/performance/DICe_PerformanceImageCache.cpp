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
#include <DICe_ImageUtils.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>
#include <random>

using namespace DICe;

int main(int argc, char *argv[]) {

  if(argc!=5){
    std::cerr << "Usage: DICe_PerformanceImageCache <cine_file> <start_frame> <end_frame> <1 or 0>, (1=verbose)" << std::endl;
    return 1;
  }

  DICe::initialize(argc, argv);

  Teuchos::RCP<Teuchos::Time> read_time  = Teuchos::TimeMonitor::getNewCounter("read time");
  Teuchos::RCP<Teuchos::Time> interp_time  = Teuchos::TimeMonitor::getNewCounter("interp time");
  Teuchos::RCP<Teuchos::Time> read_partial_time  = Teuchos::TimeMonitor::getNewCounter("read partial time");
  Teuchos::RCP<Teuchos::Time> interp_partial_time  = Teuchos::TimeMonitor::getNewCounter("interp partial time");

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
  //const int_t total_width = cine.width();
  //const int_t total_height = cine.height();
  //*outStream << "width:             " << total_width << std::endl;
  //*outStream << "height:            " << total_height << std::endl;
  // The sub-regions of the images are hard coded in terms of size and location so the cine size is constrained
  //TEUCHOS_TEST_FOR_EXCEPTION(total_width < 800,std::runtime_error,"Error the cine file frame width is too low");
  //TEUCHOS_TEST_FOR_EXCEPTION(total_height < 390,std::runtime_error,"Error the cine file frame height is too low");

  std::random_device rd; // obtain a random number from hardware
  std::mt19937 eng(rd()); // seed the generator
  std::uniform_int_distribution<> distr(0, 100); // define the range

  //const int_t num_interp_evals = 100000; // (subset size 30x50 with roughly 50 evaluations)
  const int_t window_size = 120;
  const int_t window_start_x = 250;
//  const int_t window_end_x = window_start_x + window_size;
  const int_t window_start_y = 130;
//  const int_t window_end_y = window_start_y + window_size;
  const int_t subset_start_x = 290;
  const int_t subset_start_y = 150;
  const int_t subset_end_x = 320;
  const int_t subset_end_y = 220;
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::subimage_width,window_size);
  imgParams->set(DICe::subimage_height,window_size);
  imgParams->set(DICe::subimage_offset_x,window_start_x);
  imgParams->set(DICe::subimage_offset_y,window_start_y);
  {
    for(int_t i = start_frame; i<=end_frame; ++i){

      std::stringstream name;
      name << stripped_fileName << "_" << i << ".cine";
      Teuchos::RCP<Image> image;
      {
        Teuchos::TimeMonitor read_time_monitor(*read_time);
        image = Teuchos::rcp(new Image(name.str().c_str()));//cine.get_frame(i,true,false);
      }

      work_t coord_x = 0.0;
      work_t coord_y = 0.0;
      {
        Teuchos::TimeMonitor interp_time_monitor(*interp_time);
        for(int_t y=subset_start_y;y<subset_end_y;++y){
          for(int_t x=subset_start_x;x<subset_end_x;++x){
            coord_x = x + distr(eng)/100.0;
            coord_y = y + distr(eng)/100.0;
            image->interpolate_keys_fourth(coord_x,coord_y);
            //std::cout << " x " << coord_x << " y " << coord_y << " " << value << std::endl;
          } // subset x loop
        } // subset y loop
      }
    } // frame loop
  }

  {
    for(int_t i = start_frame; i<=end_frame; ++i){
      //images = cine.get_frame(i,param_set,true,false,Teuchos::null);
      std::stringstream name;
      name << stripped_fileName << "_" << i << ".cine";

      Teuchos::RCP<Image> image;
      {
        Teuchos::TimeMonitor read_partial_time_monitor(*read_partial_time);
        image = Teuchos::rcp(new Image(name.str().c_str(),imgParams));
      }
      //cine.get_frame(i,window_start_x,window_start_y,window_end_x,window_end_y,true,false);

      work_t coord_x = 0.0;
      work_t coord_y = 0.0;
      {
        Teuchos::TimeMonitor interp_partial_time_monitor(*interp_partial_time);
        for(int_t y=subset_start_y;y<subset_end_y;++y){
          for(int_t x=subset_start_x;x<subset_end_x;++x){
            coord_x = x + distr(eng)/100.0;
            coord_y = y + distr(eng)/100.0;
            image->interpolate_bicubic_global(coord_x,coord_y);
          } // subset x loop
        } // subset y loop
      }
    } // frame loop
  }
  Teuchos::TimeMonitor::summarize(*outStream,false,true,false/*zero timers*/);

  *outStream << "--- End performance test ---" << std::endl;

  DICe::finalize();

  return 0;
}

