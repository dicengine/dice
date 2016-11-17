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
#include <DICe_NetCDF.h>

#include <iostream>
#include <netcdf.h>

namespace DICe {
namespace netcdf {

Teuchos::RCP<Image>
NetCDF_Reader::get_image(const std::string & file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params){

  // *** Note: use native types for calls to netcdf, otherwise the array sizes and allocations may be off
  // nc_type 5 = float, 4 = int, 2 = char
  // assumes that there is only one time per file and one band per file

  int error_int = 0;

  // Open the file for read access
  int ncid;
  error_int = nc_open(file_name.c_str(), 0, &ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << file_name);

  int num_vars = 0;
  nc_inq_nvars(ncid, &num_vars);
  DEBUG_MSG("NetCDF_Reader::get_image(): number of variables in the file: " << num_vars);

  // acquire the dimensions of the file
  int num_data_dims = 0;
  int height = -1;
  int width = -1;
  nc_inq_ndims(ncid, &num_data_dims);
  DEBUG_MSG("NetCDF_Reader::get_image(): number of data dimensions: " <<  num_data_dims);
  for(int_t i=0;i<num_data_dims;++i){
    char var_name[100];
    size_t length = 0;
    nc_inq_dim(ncid,i,&var_name[0],&length);
    std::string var_name_str = var_name;
    DEBUG_MSG("NetCDF_Reader::get_image(): found dimension " << var_name << " of size " << length);
    if(strcmp(var_name, "xc") == 0){
      width = (int)length;
      assert(width > 0);
    }
    if(strcmp(var_name, "yc") == 0){
      height = (int)length;
      assert(height > 0);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(width <=0, std::runtime_error,"Error, could not find xc dimension in NetCDF file " << file_name);
  TEUCHOS_TEST_FOR_EXCEPTION(height <=0, std::runtime_error,"Error, could not find yc dimension in NetCDF file " << file_name);
  DEBUG_MSG("NetCDF_Reader::get_image(): image dimensions " << width << " x " << height);

  // get the variable names
  int data_var_index = -1;
  for(int_t i=0;i<num_vars;++i){
    char var_name[100];
    int nc_type;
    int num_dims = 0;
    int dim_ids[NC_MAX_VAR_DIMS]; // assume less than 100 ids
    int num_var_attr = 0;
    nc_inq_var(ncid,i, &var_name[0], &nc_type,&num_dims, dim_ids, &num_var_attr);
    nc_inq_varname(ncid, i, &var_name[0]);
    std::string var_name_str = var_name;
    DEBUG_MSG("NetCDF_Reader::get_image(): found variable " << var_name_str << " type " << nc_type << " num dims " << num_dims << " num attributes " << num_var_attr);
    if(strcmp(var_name, "data") == 0){
      data_var_index = i;
      assert(num_dims == 3);
      assert(nc_type == 5);
    }
    if(strcmp(var_name, "dataWidth") == 0){
      int data_width = 0;
      nc_get_var1_int(ncid,i,0,&data_width);
      DEBUG_MSG("NetCDF_Reader::get_image(): memory storage size per pixel " << data_width << " (bytes)");
      DEBUG_MSG("NetCDF_Reader::get_image(): total data memory storage " << data_width * width * height / 1000000.0 << " (Mb)");
      assert(num_dims == 0);
      assert(nc_type == 4);
    }
    if(strcmp(var_name, "lineRes") == 0){
      int line_res = 0;
      nc_get_var1_int(ncid,i,0,&line_res);
      DEBUG_MSG("NetCDF_Reader::get_image(): line resolution " << line_res << " (km)");
    }
    if(strcmp(var_name, "elemRes") == 0){
      int elem_res = 0;
      nc_get_var1_int(ncid,i,0,&elem_res);
      DEBUG_MSG("NetCDF_Reader::get_image(): element resolution " << elem_res << " (km)");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(data_var_index <=0, std::runtime_error,"Error, could not find data variable in NetCDF file " << file_name);

  // read the intensities
  Teuchos::ArrayRCP<intensity_t> intensities(width*height,0.0);
  float * data = new float[width*height];
  nc_get_var_float (ncid,data_var_index,data);
  float min_intensity = std::numeric_limits<float>::max();
  float max_intensity = std::numeric_limits<float>::min();
  for(int_t i=0;i<width*height;++i){
    intensities[i] = data[i]; // convert and store as intensity_t
    if(data[i] > max_intensity) max_intensity = data[i];
    if(data[i] < min_intensity) min_intensity = data[i];
  }
  DEBUG_MSG("NetCDF_Reader::get_image(): intensity range " << min_intensity << " to " << max_intensity);
  Teuchos::RCP<Image> img = Teuchos::rcp(new Image(width,height,intensities,params));
  delete [] data;
  // close the nc_file
  nc_close(ncid);

  return img;
}

} // end netcdf namespace
} // end DICe Namespace
