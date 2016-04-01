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
#include <DICe_ImageUtils.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <boost/timer/timer.hpp>

#include <iostream>
#include <random>

using namespace DICe;
using namespace boost::timer;

int main(int argc, char *argv[]) {

  if(argc!=5){
    std::cerr << "Usage: DICe_PerformanceImageCache <cine_file> <start_frame> <end_frame> <1 or 0>, (1=verbose)" << std::endl;
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
  const int_t total_width = cine.width();
  const int_t total_height = cine.height();
  *outStream << "width:             " << total_width << std::endl;
  *outStream << "height:            " << total_height << std::endl;
  // The sub-regions of the images are hard coded in terms of size and location so the cine size is constrained
  TEUCHOS_TEST_FOR_EXCEPTION(total_width < 800,std::runtime_error,"Error the cine file frame width is too low");
  TEUCHOS_TEST_FOR_EXCEPTION(total_height < 390,std::runtime_error,"Error the cine file frame height is too low");

  std::random_device rd; // obtain a random number from hardware
  std::mt19937 eng(rd()); // seed the generator
  std::uniform_int_distribution<> distr(0, 100); // define the range

  //const int_t num_interp_evals = 100000; // (subset size 30x50 with roughly 50 evaluations)
  const int_t window_size = 120;
  const int_t window_start_x = 250;
  const int_t window_end_x = window_start_x + window_size;
  const int_t window_start_y = 130;
  const int_t window_end_y = window_start_y + window_size;
  const int_t subset_start_x = 290;
  const int_t subset_start_y = 150;
  const int_t subset_end_x = 320;
  const int_t subset_end_y = 220;
  cpu_timer read_total_timer;
  cpu_timer interp_total_timer;
  {
    for(int_t i = start_frame; i<=end_frame; ++i){
      if(i==start_frame)
        read_total_timer.start();
      else
        read_total_timer.resume();
      Teuchos::RCP<Image> image = cine.get_frame(i,subset_start_x,subset_start_y,subset_end_x,subset_end_y,true,false);
      read_total_timer.stop();

      scalar_t coord_x = 0.0;
      scalar_t coord_y = 0.0;
      if(i==start_frame)
        interp_total_timer.start();
      else
        interp_total_timer.resume();
      for(int_t y=subset_start_y;y<subset_end_y;++y){
        for(int_t x=subset_start_x;x<subset_end_x;++x){
          coord_x = x + distr(eng)/100.0;
          coord_y = y + distr(eng)/100.0;
          interpolate_keys_fourth(coord_x,coord_y,image);
          //std::cout << " x " << coord_x << " y " << coord_y << " " << value << std::endl;
        } // subset x loop
      } // subset y loop
      interp_total_timer.stop();
    } // frame loop
  }

  //std::vector<Teuchos::RCP<Image> > images;
  Teuchos::RCP<std::map<int_t,Motion_Window_Params> > param_set = Teuchos::rcp(new std::map<int_t,Motion_Window_Params>());
  Motion_Window_Params params;
  params.start_x_ = window_start_x;
  params.start_y_ = window_start_y;
  params.end_x_ = window_end_x;
  params.end_y_ = window_end_y;
  param_set->insert(std::pair<int_t,Motion_Window_Params>(0,params));
  cpu_timer read_timer;
  cpu_timer interp_timer;
  {
    for(int_t i = start_frame; i<=end_frame; ++i){
      if(i==start_frame)
        read_timer.start();
      else
        read_timer.resume();
      //images = cine.get_frame(i,param_set,true,false,Teuchos::null);
      Teuchos::RCP<Image> image = cine.get_frame(i,window_start_x,window_start_y,window_end_x,window_end_y,true,false);
      image->write("Howdy-yall.tif");
      read_timer.stop();

      scalar_t coord_x = 0.0;
      scalar_t coord_y = 0.0;
      if(i==start_frame)
        interp_timer.start();
      else
        interp_timer.resume();
      for(int_t y=subset_start_y;y<subset_end_y;++y){
        for(int_t x=subset_start_x;x<subset_end_x;++x){
          coord_x = x + distr(eng)/100.0;
          coord_y = y + distr(eng)/100.0;
          image->interpolate_bicubic(coord_x,coord_y);
        } // subset x loop
      } // subset y loop
      interp_timer.stop();
    } // frame loop
  }
  *outStream << "** whole image read time" << read_total_timer.format();
  *outStream << "** whole image interp time" << interp_total_timer.format();
  *outStream << "** portion image read time" << read_timer.format();
  *outStream << "** portion image interp time" << interp_timer.format();

  *outStream << "--- End performance test ---" << std::endl;

  DICe::finalize();

  return 0;
}

