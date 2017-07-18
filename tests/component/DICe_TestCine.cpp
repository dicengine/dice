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
#include <DICe_Image.h>
#include <DICe_Cine.h>
#include <DICe_ImageIO.h>

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
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  std::vector<std::string> cine_files;
  // the bool is whether ot not to convert the frame to 8 bit
  cine_files.push_back("packed_12bpp");
  cine_files.push_back("packed_raw_12bpp");
  cine_files.push_back("phantom_v12_raw_16bpp");
  cine_files.push_back("phantom_v1610");
  cine_files.push_back("phantom_v1610_16bpp");
  cine_files.push_back("phantom_v1610_raw");
  cine_files.push_back("phantom_v1610_raw_16bpp");
  cine_files.push_back("phantom_v2511_raw_16bpp");
  cine_files.push_back("phantom_v611_raw_16bpp");
  cine_files.push_back("phantom_v7_raw_16bpp");
  cine_files.push_back("phantom_v9_raw_16bpp");

  // all of these cine files should be dimensions 128 x 256 and have 6 frames each

  for(size_t i=0;i<cine_files.size();++i){
    std::stringstream full_name;
    full_name << "./images/" << cine_files[i] << ".cine";
    *outStream << "testing cine file: " << full_name.str() << std::endl;

    DICe::cine::Cine_Reader cine_reader(full_name.str(),outStream.getRawPtr());
    if(cine_reader.num_frames()!=6){
      *outStream << "Error, the number of frames is not correct, should be 6 is " << cine_reader.num_frames() << std::endl;
      errorFlag++;
    }
    if(cine_reader.height()!=128){
      *outStream << "Error, the image height is not correct. Should be 128 and is" << cine_reader.height() << std::endl;
      errorFlag++;
    }
    if(cine_reader.width()!=256){
      *outStream << "Error, the image width is not correct. Should be 256 and is " << cine_reader.width() << std::endl;
      errorFlag++;
    }
    *outStream << "image dimensions have been checked" << std::endl;

    for(int_t frame=0;frame<cine_reader.num_frames();++frame){
      std::stringstream markup_name;
      markup_name << "./images/" << cine_files[i] << "_" << frame + cine_reader.first_image_number() << ".cine";
      *outStream << "testing frame " << frame << std::endl;
       Teuchos::RCP<Image> cine_img = Teuchos::rcp(new Image(markup_name.str().c_str()));
      std::stringstream name;
      //std::stringstream outname;
      //std::stringstream tiffname;
      //outname << cine_files[i] << "_d_" << frame << ".rawi";
      //tiffname << cine_files[i] << "_" << frame << ".tiff";
      //cine_img->write(outname.str());
      //cine_img->write(tiffname.str());
#if DICE_USE_DOUBLE
      name << "./images/" << cine_files[i]<< "_d_" << frame << ".rawi";
#else
      name << "./images/" << cine_files[i]<< "_" << frame << ".rawi";
#endif
      Image cine_img_exact(name.str().c_str());
      bool intensity_value_error = false;
      for(int_t y=0;y<cine_reader.height();++y){
        for(int_t x=0;x<cine_reader.width();++x){
          if(std::abs((*cine_img)(x,y)-cine_img_exact(x,y)) > 0.05){
            //std::cout << x << " " << y << " actual " << (*cine_img)(x,y) << " exptected " << cine_img_exact(x,y) << std::endl;
            intensity_value_error=true;
          }
        }
      }
      if(intensity_value_error){
        *outStream << "Error, image " << i << ", the intensity values are not correct" << std::endl;
        errorFlag++;
      }
      *outStream << "intensity values have been checked" << std::endl;
    }
  }

  *outStream << "testing invalid cine file for an exception" << std::endl;

  bool exception_thrown = false;
  try{
    DICe::cine::Cine_Reader cine_reader("./images/invalid_color.cine",outStream.getRawPtr());
  }
  catch(const std::exception &e){
    exception_thrown = true;
    *outStream << "exception thrown as expected." << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for the compressed color cine file: invalid_color.cine" << std::endl;
    errorFlag++;
  }

  *outStream << "testing that invalid frame index throws an exception" << std::endl;
  exception_thrown = false;
  try{
    std::string file_name = "./images/packed_12bpp_1000.cine";
    Teuchos::RCP<Image> cine_img = Teuchos::rcp(new Image(file_name.c_str()));
  }
  catch(const std::exception &e){
    exception_thrown=true;
    *outStream << "exception thrown as expected." << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for an invalid frame index" << std::endl;
    errorFlag++;
  }

  *outStream << "testing reading a set of sub regions from a 10 bit cine " << std::endl;
  DICe::cine::Cine_Reader cine_reader("./images/packed_12bpp.cine",outStream.getRawPtr());
  std::stringstream win_frame;
  win_frame << "./images/packed_12bpp_" << cine_reader.first_image_number() << ".cine";
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::convert_cine_to_8_bit,false);
  Teuchos::RCP<Image> image_0_rcp = Teuchos::rcp(new Image(win_frame.str().c_str(),170,13,208-170+1,42-13+1,imgParams));
  //image_0_rcp->write("motion_window_12bpp.tif");
  Teuchos::RCP<Image> image_1_rcp = Teuchos::rcp(new Image(win_frame.str().c_str(),196,72,238-196+1,95-72+1,imgParams));
  //image_1_rcp->write("motion_window_12bpp.tif");
  std::vector<Teuchos::RCP<Image> > image_rcps;
  image_rcps.push_back(image_0_rcp);
  image_rcps.push_back(image_1_rcp);
  // output the images
  for(size_t i=0;i<image_rcps.size();++i){
    std::stringstream name;
#if DICE_USE_DOUBLE
    name << "./images/motion_window_d_" << i << ".rawi";
#else
    name << "./images/motion_window_" << i << ".rawi";
#endif
    Image cine_img_exact(name.str().c_str());
    //cine_img_exact.write("motion_window_12bpp_exact.tif");
    bool intensity_value_error = false;
    for(int_t y=0;y<cine_img_exact.height();++y){
      for(int_t x=0;x<cine_img_exact.width();++x){
        if(std::abs((* image_rcps[i])(x,y)-cine_img_exact(x,y)) > 0.05){
          //std::cout << x << " " << y << " actual " << (* image_rcps[i])(x,y) << " exptected " << cine_img_exact(x,y) << std::endl;
          intensity_value_error=true;
        }
      }
    }
    if(intensity_value_error){
      *outStream << "Error, image " << i << ", the intensity values are not correct" << std::endl;
      errorFlag++;
    }
  }
  *outStream << "motion window values have been checked" << std::endl;

  *outStream << "testing reading a set of sub regions from an 8 bit cine " << std::endl;
  DICe::cine::Cine_Reader cine_reader_8("./images/phantom_v1610.cine",outStream.getRawPtr());
  const int_t frame_5_index = 5 + cine_reader_8.first_image_number();
  std::stringstream file_name_8;
  file_name_8 << "./images/phantom_v1610_" << frame_5_index << ".cine";
  Teuchos::RCP<Image> image_8 = Teuchos::rcp(new Image(file_name_8.str().c_str(),158,15,196-158+1,45-15+1));
  //image_8->write("motion_window_8.tif");
  bool intensity_value_error = false;
#if DICE_USE_DOUBLE
    Image img_8_exact("./images/motion_window_d_8.rawi");
#else
    Image img_8_exact("./images/motion_window_8.rawi");
#endif
  //img_8_exact.write("motion_window_8_exact.tif");
  for(int_t y=0;y<image_8->height();++y){
    for(int_t x=0;x<image_8->width();++x){
      if(std::abs((*image_8)(x,y)-img_8_exact(x,y)) > 0.05){
        //std::cout << x << " " << y << " " << std::abs((*image_8)(x,y)-img_8_exact(x,y)) <<std::endl;
        intensity_value_error=true;
      }
    }
  }
  if(intensity_value_error){
    *outStream << "Error, the 8 bit intensity values are not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "8 bit motion window values have been checked" << std::endl;

  *outStream << "testing reading a set of sub regions from an 16 bit cine " << std::endl;
  DICe::cine::Cine_Reader cine_reader_16("./images/phantom_v1610_16bpp.cine",outStream.getRawPtr());
  std::stringstream file_name_16;
  file_name_16 << "./images/phantom_v1610_16bpp_" << 5 + cine_reader_16.first_image_number() << ".cine";
  Teuchos::RCP<Image> image_16 = Teuchos::rcp(new Image(file_name_16.str().c_str(),95,83,128-95+1,116-83+1,imgParams));
  //image_16->write("motion_window_d_16.rawi");
  intensity_value_error = false;
#if DICE_USE_DOUBLE
    Image img_16_exact("./images/motion_window_d_16.rawi");
#else
    Image img_16_exact("./images/motion_window_16.rawi");
#endif
  for(int_t y=0;y<image_16->height();++y){
    for(int_t x=0;x<image_16->width();++x){
      if(std::abs((*image_16)(x,y)-img_16_exact(x,y)) > 0.05){
        intensity_value_error=true;
      }
    }
  }
  if(intensity_value_error){
    *outStream << "Error, the 16 bit intensity values are not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "16 bit motion window values have been checked" << std::endl;


  int_t test_w = 0;
  int_t test_h = 0;
  DICe::utils::read_image_dimensions("./images/phantom_v1610_16bpp_0.cine",test_w,test_h);
  *outStream << "read cine via utils method w: " << test_w << " h: " << test_h << std::endl;
  if(test_w!=256||test_h!=128){
    *outStream << "Error, the image dimensions are wrong" << std::endl;
    errorFlag++;
  }

  // test the indexing from a cine file name
  int_t end_index = 0;
  int_t start_index = 0;
  bool is_avg = false;
  DICe::utils::cine_index("./images/phantom_v1610_16bpp_avg-345to-12.cine",start_index,end_index,is_avg);
  *outStream << "decypher cine index range start " << start_index << " end " << end_index << std::endl;
  if(start_index!=-345||end_index!=-12){
    *outStream << "Error, the image indices are not correct" << std::endl;
    errorFlag++;
  }
  if(!is_avg){
    *outStream << "Error, should have been flagged as an average image" << std::endl;
    errorFlag++;
  }

  DICe::utils::cine_index("./images/phantom_v1610_16bpp_23.cine",start_index,end_index,is_avg);
  *outStream << "decypher cine index range start " << start_index << " end " << end_index << std::endl;
  if(start_index!=23||end_index!=-1){
    *outStream << "Error, the image indices are not correct" << std::endl;
    errorFlag++;
  }
  if(is_avg){
    *outStream << "Error, should not have been flagged as an average image" << std::endl;
    errorFlag++;
  }

  // try creating a cine image using the standard image interface without a reader constructed manually:
  Teuchos::RCP<DICe::Image> img_cine_0 = Teuchos::rcp(new Image("./images/phantom_v1610_16bpp_-85.cine"));
  Teuchos::RCP<DICe::Image> img_cine_0_gold = Teuchos::rcp(new Image("./images/image_cine_-85.rawi"));
  const scalar_t diff = img_cine_0->diff(img_cine_0_gold);
  *outStream << "diff cine made without manual header creation vs gold: " << diff << std::endl;
  if(diff > 0.001){
    *outStream << "Error, the images do not match" << std::endl;
    errorFlag++;
  }
  //img_cine_0->write("image_cine_-85.rawi");


  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

