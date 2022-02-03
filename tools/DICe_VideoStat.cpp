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

/*! \file  DICe_VideoStat.cpp
    \brief Utility for exporting video files to tiff files
*/

#include <DICe.h>
#include <DICe_Parser.h>
#include <DICe_ImageIO.h>

#include <hypercine.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <cassert>

using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage ./DICe_CineToTiff <cine_file_name> <start_index> <end_index> <output_prefix>

  DICe::initialize(argc, argv);

  //Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  std::string delimiter = " ,\r";

  if(argc>=2){
    std::string help = argv[1];
    if(help=="-h"||argc>2){
      std::cout << " DICe_VideoStat (writes a file with the valid video index range) " << std::endl;
      std::cout << " Syntax: DICe_VideoStat <video_file_name>" << std::endl;
      exit(0);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(argc!=2,std::runtime_error,"Error, wrong number of input arguments");

  DEBUG_MSG("User specified " << argc << " arguments");
  for(int_t i=0;i<argc;++i){
    DEBUG_MSG(argv[i]);
  }
  std::string fileName = argv[1];
  *outStream << "Video file name: " << fileName << std::endl;

  int_t num_images, first_frame, last_frame, frame_rate;

  if(DICe::utils::is_cine_file(fileName)){
    Teuchos::RCP<hypercine::HyperCine> hc = DICe::utils::Video_Singleton::instance().hypercine(fileName,hypercine::HyperCine::TO_8_BIT);
    *outStream << "\nCine read successfully\n" << std::endl;
    num_images = hc->file_frame_count();
    first_frame = hc->file_first_frame_id();
    last_frame = first_frame + num_images - 1;
    frame_rate = hc->frame_rate();
  }else{
    Teuchos::RCP<cv::VideoCapture> vc = DICe::utils::Video_Singleton::instance().video_capture(fileName);
    *outStream << "\nVideo read successfully\n" << std::endl;
    num_images = vc->get(cv::CAP_PROP_FRAME_COUNT);
    first_frame = 0;
    last_frame = first_frame + num_images - 1;
    frame_rate = vc->get(cv::CAP_PROP_FPS);
  }

  *outStream << "Num frames:     " << num_images << std::endl;
  *outStream << "First frame:    " << first_frame << std::endl;
  *outStream << "Last frame:     " << last_frame << std::endl;
  *outStream << "Frame rate:     " << frame_rate << std::endl;

  // write stats to file
  create_directory(".dice");
  std::FILE * filePtr = fopen(".dice/.video_stats.dat","w");
  fprintf(filePtr,"%i %i %i %i\n",num_images,first_frame,last_frame,frame_rate);
  fclose(filePtr);

  DICe::finalize();

  return 0;
}

