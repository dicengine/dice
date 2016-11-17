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

#ifndef DICE_NETCDF_H
#define DICE_NETCDF_H

#include <DICe_Image.h>
#include <DICe_Parser.h>

#include <cassert>
#include <iostream>

#if defined(WIN32)
  #include <cstdint>
#endif

#include <Teuchos_RCP.hpp>

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

  /// retrieve a DICe image from a netcdf file
  /// \param file_name name of the netcdf file to read
  Teuchos::RCP<Image> get_image(const std::string & file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

private:

};

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
