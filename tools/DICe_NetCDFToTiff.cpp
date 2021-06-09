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

/*! \file  DICe_NetCDFToTiff.cpp
    \brief Utility for exporting NetCDF files to tiff files
*/

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_NetCDF.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <cassert>
#include <string>
#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage 1: ./DICe_NetCDFToTiff <folder_name>
  /// converts all NetCDF files to images in this folder
  /// usage 2: ./DICe_NetCDFToTiff <file_name>
  /// converts a specific NetCDF file to a tif of the same base name

  DICe::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  std::string delimiter = " ,\r";

  // determine if the second argument is help, file name or folder name
  if(argc!=2){
    std::cout << " DICe_NetCDFToTiff (exports NetCDF satellite images to tiffs) " << std::endl;
    std::cout << " Syntax: DICe_NetCDFToTiff <file_name or folder_name>" << std::endl;
    exit(-1);
  }
  std::string name = argv[1];
  if(name=="-h"){
    std::cout << " DICe_NetCDFToTiff (exports NetCDF satellite images to tiffs) " << std::endl;
    std::cout << " Syntax: DICe_NetCDFToTiff <file_name or folder_name>" << std::endl;
    exit(0);
  }
  // determine if this is a file or folder
  const size_t len = name.length();
  bool is_file_name = false;
  if(len > 3){
    std::string substr = name.substr(len-3,len);
    std::cout << substr << std::endl;
    if(strcmp(substr.c_str(),".nc")==0){
      is_file_name = true;
    }
  }
  //Teuchos::RCP<DICe::netcdf::NetCDF_Reader> netcdf_reader = Teuchos::rcp(new netcdf::NetCDF_Reader());

  if(is_file_name){
    // determine if this is a block .nc with multiple images:
    netcdf::NetCDF_Reader netcdf_reader;
    int_t num_time_steps = 0;
    int_t width = 0;
    int_t height = 0;
    netcdf_reader.get_image_dimensions(name,width,height,num_time_steps);

    // strip the .nc part from the end of the file_name:
    std::string trimmed_name = name;
    const std::string ext(".nc");
    if(trimmed_name.size() > ext.size() && trimmed_name.substr(trimmed_name.size() - ext.size()) == ".nc" )
    {
       trimmed_name = trimmed_name.substr(0, trimmed_name.size() - ext.size());
    }else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid netcdf file: " << name);
    }
    if(num_time_steps > 1){
      // determine the total number of digits:
      int_t num_digits_total = 0;
      int_t decrement_total = num_time_steps;
      while (decrement_total){decrement_total /= 10; num_digits_total++;}

      for(int_t i=0;i<num_time_steps;++i){
        // decorate the file name
        int_t num_digits_image = 0;
        int_t decrement_image = i;
        if(decrement_image==0) num_digits_image = 1;
        else
          while (decrement_image){decrement_image /= 10; num_digits_image++;}
        int_t num_zeros = num_digits_total - num_digits_image;

        std::stringstream netcdf_ss;
        netcdf_ss << trimmed_name << "_frame_";
        for(int_t z=0;z<num_zeros;++z)
          netcdf_ss << "0";
        netcdf_ss << i << ".nc";
        Teuchos::RCP<Image> img = Teuchos::rcp(new Image(netcdf_ss.str().c_str()));//netcdf_reader->get_image(name);
        std::string new_name = netcdf_ss.str();
        const size_t new_len = new_name.length();
        new_name.replace(new_len-3,3,".tif");
        DEBUG_MSG("Converting file: " << netcdf_ss.str() << " to " << new_name);
        img->write(new_name);
      }
    }
    else{
      Teuchos::RCP<Image> img = Teuchos::rcp(new Image(name.c_str()));//netcdf_reader->get_image(name);
      std::string old_name = name;
      name.replace(len-3,3,".tif");
      DEBUG_MSG("Converting file: " << old_name << " to " << name);
      img->write(name);
    }
  }
  else{
    // removal of boost left this broken, will return to fix it later.
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"This method currently broken");
//#ifndef   DICE_DISABLE_BOOST_FILESYSTEM
//    DEBUG_MSG("Converting all NetCDF files in folder: " << name);
//    boost::filesystem::path dir_path(name);
//    if ( !boost::filesystem::exists( dir_path ) ){
//      std::cout << "Directory does not exist: " << name << std::endl;
//      exit(-1);
//    }
//    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
//    for ( boost::filesystem::directory_iterator itr( dir_path );itr != end_itr;++itr){
//      std::string file_name = itr->path().filename().string();
//      std::cout << "found file " << file_name << std::endl;
//      const size_t name_len = file_name.length();
//      std::string substr = file_name.substr(name_len-3,name_len);
//      if(strcmp(substr.c_str(),".nc")==0){
//        Teuchos::RCP<Image> img = Teuchos::rcp(new Image(file_name.c_str()));//netcdf_reader->get_image(file_name);
//        std::string old_name = file_name;
//        file_name.replace(name_len-3,3,".tif");
//        DEBUG_MSG("Converting file: " << old_name << " to " << file_name);
//        img->write(file_name);
//      }
//    }
//#else
//    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, boost filesystem required to convert all images in a directory");
//#endif
  }

  DICe::finalize();

  return 0;
}

