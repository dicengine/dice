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

#ifndef DICE_NETCDF_H
#define DICE_NETCDF_H

#include <DICe.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <cassert>
#include <iostream>
#include <vector>

#if defined(WIN32)
  #include <cstdint>
#endif

namespace DICe {
/*!
 *  \namespace DICe::cine
 *  @{
 */
/// cine file format utilities
namespace netcdf{

/// \class DICe::netcdf::NetCDF
/// \brief A class to generate a DICe image from a netcdf file

class DICE_LIB_DLL_EXPORT
NetCDF_Reader
{
public:
  /// constructor
  NetCDF_Reader(){};
  /// default destructor
  virtual ~NetCDF_Reader(){};

  /// read the intensities from the netcdf file:
  /// \param file_name the name of the file to read
  /// \param time_index the time frame to retrieve
  /// \param intensities pointer to the intensity array (must be pre-allocated)
  /// \param width width of the sub frame
  /// \param height height of the subframe
  /// \param offset_x offset in x direction
  /// \param offset_y offset in y direction
  /// \param is_layout_right true if the arrays are oriented layout right in memory
  void read_netcdf_image(const char * file_name,
    const size_t time_index,
    scalar_t * intensities,
    const int_t width=0,
    const int_t height=0,
    const int_t offset_x=0,
    const int_t offset_y=0,
    const bool is_layout_right = true);

  /// retrive the image dimensions of a NetCDF file
  /// \param file_name the name of the file to get dims for
  /// \param width the width of the image
  /// \param height the height of the image
  /// \param num_time_steps the number of time steps in this file
  void get_image_dimensions(const std::string & file_name,
    int_t & width,
    int_t & height,
    int_t & num_time_steps);
};


/// class to write float arrays out a netcdf file
class DICE_LIB_DLL_EXPORT
NetCDF_Writer
{
public:
  /// constructor
  /// \param file_name the name of the file to create
  /// \param width the width of the data array
  /// \param height the height of the data array
  /// \param num_time_steps the number of time steps to include in the file
  /// \param var_names vector with the names of the varibles to save
  /// \param use_double forces output to be in double, not float
  NetCDF_Writer(const std::string & file_name,
    const int_t & width,
    const int_t & height,
    const int_t & num_time_steps,
    const std::vector<std::string> & var_names,
    const bool use_double = false);
  /// default destructor
  virtual ~NetCDF_Writer(){};

  /// write an array to the file
  /// if the file exists, overwrite
  /// \param var_name the name of the variable to write
  /// \param time_index the time frame to write
  /// \param vector of float values to write
  void write_float_array(const std::string var_name,
    const size_t time_index,
    const std::vector<float> & array);

  /// write an array to the file
  /// if the file exists, overwrite
  /// \param var_name the name of the variable to write
  /// \param time_index the time frame to write
  /// \param vector of double values to write
  void write_double_array(const std::string var_name,
    const size_t time_index,
    const std::vector<double> & array);

private:
  const std::string file_name_;
  /// x dimension of the array
  const int_t dim_x_;
  /// y dimension of the array
  const int_t dim_y_;
  /// the number of time steps
  const int_t num_time_steps_;
  /// collection of all the varible names
  std::vector<std::string> var_names_;
  /// collection of the variable ids
  std::vector<int_t> var_ids_;
};

// free function to take a .nc file (GOES formatted NetCDF4) and return a parameter list of important cal-related parameters
DICE_LIB_DLL_EXPORT
Teuchos::ParameterList netcdf_to_lat_long_projection_parameters(const std::string & left_file,
  const std::string & right_file);

// free function to convert left image pixel coordinates for full disk image to earth coordinate points
// and right image pixel coordinates
DICE_LIB_DLL_EXPORT
void netcdf_left_pixel_points_to_earth_and_right_pixel_coordinates(const Teuchos::ParameterList & params,
  const std::vector<float> & left_pixel_x,
  const std::vector<float> & left_pixel_y,
  std::vector<float> & earth_x,
  std::vector<float> & earth_y,
  std::vector<float> & earth_z,
  std::vector<float> & right_pixel_x,
  std::vector<float> & right_pixel_y);

}// end netcdf namespace
}// end DICe namespace

#endif

// example Goes satellite netcdf dump:

//netcdf goes12.2004.306.173144.BAND_02 {
//dimensions:
//  lines = 618 ;
//  elems = 904 ;
//  bands = 1 ;
//  auditCount = 2 ;
//  auditSize = 80 ;
//variables:
//  int version ;
//    version:long_name = "McIDAS area file version" ;
//  int sensorID ;
//    sensorID:long_name = "McIDAS sensor number" ;
//  int imageDate ;
//    imageDate:long_name = "image year and day of year" ;
//    imageDate:units = "ccyyddd" ;
//  int imageTime ;
//    imageTime:long_name = "image time in UTC" ;
//    imageTime:units = "hhmmss UTC" ;
//  int startLine ;
//    startLine:long_name = "starting image line" ;
//    startLine:units = "satellite coordinates" ;
//  int startElem ;
//    startElem:long_name = "starting image element" ;
//    startElem:units = "satellite coordinates" ;
//  int numLines ;
//    numLines:long_name = "number of lines" ;
//  int numElems ;
//    numElems:long_name = "number of elements" ;
//  int dataWidth ;
//    dataWidth:long_name = "number of bytes per source data point" ;
//    dataWidth:units = "bytes/data point" ;
//  int lineRes ;
//    lineRes:long_name = "resolution of each pixel in line direction" ;
//    lineRes:units = "km" ;
//  int elemRes ;
//    elemRes:long_name = "resolution of each pixel in element direction" ;
//    elemRes:units = "km" ;
//  int prefixSize ;
//    prefixSize:long_name = "line prefix size" ;
//    prefixSize:units = "bytes" ;
//  int crDate ;
//    crDate:long_name = "image creation year and day of year" ;
//    crDate:units = "ccyyddd" ;
//  int crTime ;
//    crTime:long_name = "image creation time in UTC" ;
//    crTime:units = "hhmmss UTC" ;
//  int bands(bands) ;
//    bands:long_name = "bands" ;
//  char auditTrail(auditCount, auditSize) ;
//    auditTrail:long_name = "audit trail" ;
//  float data(bands, lines, elems) ;
//    data:long_name = "data" ;
//    data:type = "GVAR" ;
//    data:units = "unitless" ;
//  float latitude(lines, elems) ;
//    latitude:long_name = "latitude" ;
//    latitude:units = "degrees" ;
//  float longitude(lines, elems) ;
//    longitude:long_name = "longitude" ;
//    longitude:units = "degrees" ;
//}
