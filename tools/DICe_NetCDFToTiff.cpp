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

/*! \file  DICe_NetCDFToTiff.cpp
    \brief Utility for exporting NetCDF files to tiff files
*/

#include <DICe.h>
#include <DICe_Parser.h>
#include <DICe_Image.h>
#include <DICe_NetCDF.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <cassert>
#include <string>
#include <iostream>

#ifndef   DICE_DISABLE_BOOST_FILESYSTEM
#include    <boost/filesystem.hpp>
#endif

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
    Teuchos::RCP<Image> img = Teuchos::rcp(new Image(name.c_str()));//netcdf_reader->get_image(name);
    std::string old_name = name;
    name.replace(len-3,3,".tif");
    DEBUG_MSG("Converting file: " << old_name << " to " << name);
    img->write(name);
  }
  else{
#ifndef   DICE_DISABLE_BOOST_FILESYSTEM
    DEBUG_MSG("Converting all NetCDF files in folder: " << name);
    boost::filesystem::path dir_path(name);
    if ( !boost::filesystem::exists( dir_path ) ){
      std::cout << "Directory does not exist: " << name << std::endl;
      exit(-1);
    }
    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    for ( boost::filesystem::directory_iterator itr( dir_path );itr != end_itr;++itr){
      std::string file_name = itr->path().filename().string();
      std::cout << "found file " << file_name << std::endl;
      const size_t name_len = file_name.length();
      std::string substr = file_name.substr(name_len-3,name_len);
      if(strcmp(substr.c_str(),".nc")==0){
        Teuchos::RCP<Image> img = Teuchos::rcp(new Image(file_name.c_str()));//netcdf_reader->get_image(file_name);
        std::string old_name = file_name;
        file_name.replace(name_len-3,3,".tif");
        DEBUG_MSG("Converting file: " << old_name << " to " << file_name);
        img->write(file_name);
      }
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, boost filesystem required to convert all images in a directory");
#endif
  }

  DICe::finalize();

  return 0;
}

