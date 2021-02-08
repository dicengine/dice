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
#include <DICe_NetCDF.h>

#include <Teuchos_RCP.hpp>

#include <iostream>
#include "netcdf.h"

namespace DICe {
namespace netcdf {

void
NetCDF_Reader::get_image_dimensions(const std::string & file_name,
  int_t & width,
  int_t & height,
  int_t & num_time_steps){

  int error_int = 0;

  // Open the file for read access
  int ncid;
  error_int = nc_open(file_name.c_str(), 0, &ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << file_name);
  // acquire the dimensions of the file
  int num_data_dims = 0;
  height = -1;
  width = -1;
  num_time_steps = 1;
  nc_inq_ndims(ncid, &num_data_dims);
  DEBUG_MSG("NetCDF_Reader::get_image(): number of data dimensions: " <<  num_data_dims);
  for(int_t i=0;i<num_data_dims;++i){
    char var_name[100];
    size_t length = 0;
    nc_inq_dim(ncid,i,&var_name[0],&length);
    std::string var_name_str = var_name;
    DEBUG_MSG("NetCDF_Reader::get_image(): found dimension " << var_name << " of size " << length);
    if(strcmp(var_name, "xc") == 0||strcmp(var_name, "imsize1") == 0||strcmp(var_name, "x") == 0){
      width = (int)length;
      assert(width > 0);
    }
    if(strcmp(var_name, "yc") == 0||strcmp(var_name, "imsize2") == 0||strcmp(var_name, "y") == 0){
      height = (int)length;
      assert(height > 0);
    }
    if(strcmp(var_name, "time") == 0||strcmp(var_name, "imsize3") == 0){
      num_time_steps = (int)length;
      assert(num_time_steps >= 0);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(width <=0, std::runtime_error,"Error, could not find xc dimension in NetCDF file " << file_name);
  TEUCHOS_TEST_FOR_EXCEPTION(height <=0, std::runtime_error,"Error, could not find yc dimension in NetCDF file " << file_name);
  DEBUG_MSG("NetCDF_Reader::get_image(): image dimensions " << width << " x " << height << " num time steps: " << num_time_steps);

  // close the nc_file
  nc_close(ncid);
}

void
NetCDF_Reader::read_netcdf_image(const char * file_name,
  const size_t time_index,
  scalar_t * intensities,
  const int_t subimage_width,
  const int_t subimage_height,
  const int_t offset_x,
  const int_t offset_y,
  const bool is_layout_right){
  TEUCHOS_TEST_FOR_EXCEPTION(offset_x!=0&&subimage_width==0,std::runtime_error,"offset_x cannot be nonzero if subimage_width is 0" << file_name);
  TEUCHOS_TEST_FOR_EXCEPTION(offset_y!=0&&subimage_height==0,std::runtime_error,"offset_y cannot be nonzero if subimage_height is 0" << file_name);

  int_t img_width = 0;
  int_t img_height = 0;
  int_t num_time_steps = 0;
  get_image_dimensions(file_name,img_width,img_height,num_time_steps);
  TEUCHOS_TEST_FOR_EXCEPTION(time_index < 0 || (int_t)time_index >= num_time_steps,std::runtime_error,"");
  const int_t width = subimage_width>0?subimage_width:img_width;
  const int_t height = subimage_height>0?subimage_height:img_height;
  DEBUG_MSG("NetCDF_Reader::read_netcdf_image(): width " << width << " height " << height << " offset_x " << offset_x << " offset_y " << offset_y);

  // Open the file for read access
  int ncid;
  int_t error_int = nc_open(file_name, 0, &ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << file_name);

  int num_vars = 0;
  nc_inq_nvars(ncid, &num_vars);
  DEBUG_MSG("NetCDF_Reader::get_image(): number of variables in the file: " << num_vars);

  // get the variable names
  int data_var_index = -1;
  int data_type = -1;
  for(int_t i=0;i<num_vars;++i){
    char var_name[100];
    int nc_type;
    int num_dims = 0;
    int dim_ids[NC_MAX_VAR_DIMS]; // assume less than 100 ids
    int num_var_attr = 0;
    nc_inq_var(ncid,i, &var_name[0], &nc_type,&num_dims, dim_ids, &num_var_attr);
    nc_inq_varname(ncid, i, &var_name[0]);
    std::string var_name_str = var_name;
    // netcdf data types: 1: BYTE 2: CHAR 3: SHORT 4: INT 5: FLOAT 6: DOUBLE 7: UBYTE 8: USHORT 9: UINT 10: INT64 12: STRING
    DEBUG_MSG("NetCDF_Reader::get_image(): found variable " << var_name_str << " type " << nc_type << " num dims " << num_dims << " num attributes " << num_var_attr);
    if(strcmp(var_name, "data") == 0){
      data_var_index = i;
      assert(num_dims == 3);
      assert(nc_type == 5 || nc_type == 6);
      data_type = nc_type;
    }else if(strcmp(var_name, "Rad") == 0){
      data_var_index = i;
      assert(num_dims == 2);
      assert(nc_type == 3);
      data_type = nc_type;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(data_var_index <0, std::runtime_error,"Error, could not find data or Rad variable in NetCDF file " << file_name);
  // initialize storage for the data in the netcdf file (read the whole image)
  //std::vector<storage_t> data(img_width*img_height);
  std::vector<size_t> starts(3,0); // not all elements are used (assumes max dimension of 3)
  std::vector<size_t> counts(3,0);
  if(data_type==3){
    starts[0] = offset_y;
    starts[1] = offset_x;
    starts[2] = 0;
    counts[0] = height;
    counts[1] = width;
    counts[2] = 0;
  }
  else if(data_type==5||data_type==6){
    starts[0] = time_index;
    starts[1] = offset_y;
    starts[2] = offset_x;
    counts[0] = 1;
    counts[1] = height;
    counts[2] = width;
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"invalid nc data_type " << data_type);
  }

  if(std::is_same<scalar_t,float>::value){
    // cast to avoid compiler error for the get var method that isn't used (it expects a pointer of type float)
    // if this if statement is true, this is a no-op, if not it never gets executed
    float * intens = (float *)intensities;
    nc_get_vara_float(ncid,data_var_index,&starts[0],&counts[0],intens);
  }else if(std::is_same<scalar_t,double>::value){
    // cast to avoid compiler error for the get var method that isn't used (it expects a pointer of type double)
    double * intens = (double *)intensities;
    nc_get_vara_double(ncid,data_var_index,&starts[0],&counts[0],intens);
  }else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"invalid intensity storage type for NetCDF (should be float or double)");
  }

  // close the nc_file
  nc_close(ncid);
}

NetCDF_Writer::NetCDF_Writer(const std::string & file_name,
  const int_t & width,
  const int_t & height,
  const int_t & num_time_steps,
  const std::vector<std::string> & var_names,
  const bool use_double):
  file_name_(file_name),
  dim_x_(width),
  dim_y_(height),
  num_time_steps_(num_time_steps),
  var_names_(var_names){

  // create a netcdf file:
  int_t retval = 0;
  int_t ncid = -1;
  const int_t num_dims = 3;
  retval = nc_create(file_name.c_str(),NC_CLOBBER, &ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
  DEBUG_MSG("created file " << file_name << " with id: " << ncid);
  int_t xdim_id = -1;
  int_t ydim_id = -1;
  int_t data_id = -1;
  int_t timedim_id = -1;
  retval = nc_def_dim(ncid, "xc", width, &xdim_id);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
  retval = nc_def_dim(ncid, "yc", height, &ydim_id);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
  retval = nc_def_dim(ncid, "time", NC_UNLIMITED, &timedim_id);//num_time_steps, &timedim_id);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
  std::vector<int_t> dim_ids(3);
  dim_ids[1] = ydim_id; dim_ids[2] = xdim_id; dim_ids[0] = timedim_id;
  var_ids_.resize(var_names_.size());
  for(size_t i=0;i<var_names.size();++i){
    /* Define the data variable. The type of the variable in this case is
     * NC_INT (4-byte integer). */
    std::string var_str = var_names[i];
    char * var_char = new char[var_str.size()+1];
    std::copy(var_str.begin(), var_str.end(), var_char);
    var_char[var_str.size()] = '\0';
    if(use_double)
      retval = nc_def_var(ncid, var_char, NC_DOUBLE, num_dims, &dim_ids[0], &data_id);
    else
      retval = nc_def_var(ncid, var_char, NC_FLOAT, num_dims, &dim_ids[0], &data_id);
    TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
    DEBUG_MSG("created variable " << var_char << " with id: " << data_id << " num dimensions " << num_dims << ": "  << dim_ids[0] << " " << dim_ids[1] << " " << dim_ids[2]);
    var_ids_[i] = data_id;
  }
  /* End define mode. This tells netCDF we are done defining
   * metadata. */
  retval = nc_enddef(ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,retval);
  nc_close(ncid);
}

void
NetCDF_Writer::write_float_array(const std::string var_name,
  const size_t time_index,
  const std::vector<float> & array){
  // search the list of names to ensure that one matches and get the id:
  int_t var_id = -1;
  for(size_t i=0;i<var_names_.size();++i){
    if(var_names_[i]==var_name)
      var_id = var_ids_[i];
  }
  DEBUG_MSG("writing variable " << var_name << " with id: " << var_id << " time step " << time_index << " to file " << file_name_);
  TEUCHOS_TEST_FOR_EXCEPTION(var_id<0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(time_index < 0 || (int_t) time_index >= num_time_steps_,std::runtime_error,"");
  // check that the dimensions are correct
  TEUCHOS_TEST_FOR_EXCEPTION(dim_x_*dim_y_!=(int_t)array.size(),std::runtime_error,"");
  int_t ncid = -1;
  int_t retval = 0;
  retval = nc_open(file_name_.c_str(),NC_WRITE,&ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(ncid<0,std::runtime_error,"error: " << retval);
  /* Write the data to the file.  */
  std::vector<size_t> starts(3);
  starts[0] = time_index;
  starts[1] = 0;
  starts[2] = 0;
  std::vector<size_t> counts(3);
  counts[0] = 1;
  counts[1] = dim_y_;
  counts[2] = dim_x_;
  retval = nc_put_vara_float(ncid,var_id,&starts[0],&counts[0],&array[0]);
  DEBUG_MSG("nc_put_vara_float() return value: " << retval);
  //retval = nc_put_var_float(ncid, var_id, &array[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
  nc_close(ncid);
}

void
NetCDF_Writer::write_double_array(const std::string var_name,
  const size_t time_index,
  const std::vector<double> & array){

  // search the list of names to ensure that one matches and get the id:
  int_t var_id = -1;
  for(size_t i=0;i<var_names_.size();++i){
    if(var_names_[i]==var_name)
      var_id = var_ids_[i];
  }
  DEBUG_MSG("writing variable " << var_name << " with id: " << var_id << " to file " << file_name_);
  TEUCHOS_TEST_FOR_EXCEPTION(var_id<0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(time_index < 0 || (int_t) time_index >= num_time_steps_,std::runtime_error,"");
  // check that the dimensions are correct
  TEUCHOS_TEST_FOR_EXCEPTION(dim_x_*dim_y_!=(int_t)array.size(),std::runtime_error,"");
  int_t ncid = -1;
  int_t retval = 0;
  retval = nc_open(file_name_.c_str(),NC_WRITE,&ncid);
  TEUCHOS_TEST_FOR_EXCEPTION(ncid<0,std::runtime_error,"");
  /* Write the data to the file.  */
  std::vector<size_t> starts(3);
  starts[0] = time_index;
  starts[1] = 0;
  starts[2] = 0;
  std::vector<size_t> counts(3);
  counts[0] = 1;
  counts[1] = dim_y_;
  counts[2] = dim_x_;
  retval = nc_put_vara_double(ncid,var_id,&starts[0],&counts[0],&array[0]);
  //retval = nc_put_var_double(ncid, var_id, &array[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(retval,std::runtime_error,"");
  nc_close(ncid);
}

DICE_LIB_DLL_EXPORT
Teuchos::ParameterList netcdf_to_lat_long_projection_parameters(const std::string & left_file,
  const std::string & right_file){

  // Open the file for read access
  int ncid_left = 0, ncid_right = 0;
  int error_int = nc_open(left_file.c_str(), 0, &ncid_left);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << left_file);
  error_int = nc_open(right_file.c_str(), 0, &ncid_right);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << right_file);
  int num_vars_left = 0, num_vars_right = 0;
  nc_inq_nvars(ncid_left, &num_vars_left);
  nc_inq_nvars(ncid_right, &num_vars_right);
  TEUCHOS_TEST_FOR_EXCEPTION(num_vars_left!=num_vars_right,std::runtime_error,"");

  std::string coord_x_str = "x";
  std::string coord_y_str = "y";
  std::string scale_str = "scale_factor";
  std::string offset_str = "add_offset";
  std::string proj_str = "goes_imager_projection";
  std::string proj_height = "perspective_point_height";
  std::string proj_major = "semi_major_axis";
  std::string proj_minor = "semi_minor_axis";
  std::string proj_long_origin = "longitude_of_projection_origin";
  float perspective_point_height_left = 0.0;
  float semi_major_axis_left = 0.0;
  float semi_minor_axis_left = 0.0;
  float long_of_proj_origin_left = 0.0;
  float coord_x_offset_left = 0.0;
  float coord_y_offset_left = 0.0;
  float coord_x_scale_factor_left = 0.0;
  float coord_y_scale_factor_left = 0.0;
  float perspective_point_height_right = 0.0;
  float semi_major_axis_right = 0.0;
  float semi_minor_axis_right = 0.0;
  float long_of_proj_origin_right = 0.0;
  float coord_x_offset_right = 0.0;
  float coord_y_offset_right = 0.0;
  float coord_x_scale_factor_right = 0.0;
  float coord_y_scale_factor_right = 0.0;
  float att_value = 0.0;
  // harvest background info for both imagers:
  for(int i=0;i<num_vars_left;++i){
    char var_name[100];
    int nc_type;
    int num_dims = 0;
    int dim_ids[NC_MAX_VAR_DIMS]; // assume less than 100 ids
    int num_var_attr = 0;
    nc_inq_var(ncid_left,i, &var_name[0], &nc_type,&num_dims, dim_ids, &num_var_attr);
    nc_inq_varname(ncid_left, i, &var_name[0]);
    std::string var_name_str = var_name;
    if(strcmp(var_name, proj_str.c_str()) == 0){
      nc_get_att_float(ncid_left,i,proj_height.c_str(),&att_value);
      perspective_point_height_left = att_value;
      nc_get_att_float(ncid_left,i,proj_major.c_str(),&att_value);
      semi_major_axis_left = att_value;
      nc_get_att_float(ncid_left,i,proj_minor.c_str(),&att_value);
      semi_minor_axis_left = att_value;
      nc_get_att_float(ncid_left,i,proj_long_origin.c_str(),&att_value);
      long_of_proj_origin_left = att_value;
    }else if(strcmp(var_name, coord_x_str.c_str()) == 0){
      nc_get_att_float(ncid_left,i,scale_str.c_str(),&att_value);
      coord_x_scale_factor_left = att_value;
      nc_get_att_float(ncid_left,i,offset_str.c_str(),&att_value);
      coord_x_offset_left = att_value;
    }else if(strcmp(var_name, coord_y_str.c_str()) == 0){
      nc_get_att_float(ncid_left,i,scale_str.c_str(),&att_value);
      coord_y_scale_factor_left = att_value;
      nc_get_att_float(ncid_left,i,offset_str.c_str(),&att_value);
      coord_y_offset_left = att_value;
    }
  }
  DEBUG_MSG("imager height left (m):           " << perspective_point_height_left);
  DEBUG_MSG("earth major axis left (m):        " << semi_major_axis_left);
  DEBUG_MSG("earth minor axis left (m):        " << semi_minor_axis_left);
  DEBUG_MSG("longitude of left imager (deg):   " << long_of_proj_origin_left);
  DEBUG_MSG("imager x scale factor left:       " << coord_x_scale_factor_left);
  DEBUG_MSG("imager x offset left (rad):       " << coord_x_offset_left);
  DEBUG_MSG("imager y scale factor left:       " << coord_y_scale_factor_left);
  DEBUG_MSG("imager y offset left (rad):       " << coord_y_offset_left);
  for(int i=0;i<num_vars_right;++i){
    char var_name[100];
    int nc_type;
    int num_dims = 0;
    int dim_ids[NC_MAX_VAR_DIMS]; // assume less than 100 ids
    int num_var_attr = 0;
    nc_inq_var(ncid_right,i, &var_name[0], &nc_type,&num_dims, dim_ids, &num_var_attr);
    nc_inq_varname(ncid_right, i, &var_name[0]);
    std::string var_name_str = var_name;
    if(strcmp(var_name, proj_str.c_str()) == 0){
      float att_value = 0.0;
      nc_get_att_float(ncid_right,i,proj_height.c_str(),&att_value);
      perspective_point_height_right = att_value;
      nc_get_att_float(ncid_right,i,proj_major.c_str(),&att_value);
      semi_major_axis_right = att_value;
      nc_get_att_float(ncid_right,i,proj_minor.c_str(),&att_value);
      semi_minor_axis_right = att_value;
      nc_get_att_float(ncid_right,i,proj_long_origin.c_str(),&att_value);
      long_of_proj_origin_right = att_value;
    }else if(strcmp(var_name, coord_x_str.c_str()) == 0){
      nc_get_att_float(ncid_right,i,scale_str.c_str(),&att_value);
      coord_x_scale_factor_right = att_value;
      nc_get_att_float(ncid_right,i,offset_str.c_str(),&att_value);
      coord_x_offset_right = att_value;
    }else if(strcmp(var_name, coord_y_str.c_str()) == 0){
      nc_get_att_float(ncid_right,i,scale_str.c_str(),&att_value);
      coord_y_scale_factor_right = att_value;
      nc_get_att_float(ncid_right,i,offset_str.c_str(),&att_value);
      coord_y_offset_right = att_value;
    }
  }
  // close the nc_files
  nc_close(ncid_left);
  nc_close(ncid_right);
  DEBUG_MSG("imager height right (m):          " << perspective_point_height_right);
  DEBUG_MSG("earth major axis right (m):       " << semi_major_axis_right);
  DEBUG_MSG("earth minor axis right (m):       " << semi_minor_axis_right);
  DEBUG_MSG("longitude of imager right (deg):  " << long_of_proj_origin_right);
  DEBUG_MSG("imager x scale factor right:      " << coord_x_scale_factor_right);
  DEBUG_MSG("imager x offset right (rad):      " << coord_x_offset_right);
  DEBUG_MSG("imager y scale factor right:      " << coord_y_scale_factor_right);
  DEBUG_MSG("imager y offset right (rad):      " << coord_y_offset_right);
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(perspective_point_height_right-perspective_point_height_left)>0.1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(semi_major_axis_right-semi_major_axis_left)>0.1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(semi_minor_axis_right-semi_minor_axis_left)>0.1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_x_scale_factor_right-coord_x_scale_factor_left)>0.001,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_y_scale_factor_right-coord_y_scale_factor_left)>0.001,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_x_offset_right-coord_x_offset_left)>0.001,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_y_offset_right-coord_y_offset_left)>0.001,std::runtime_error,"");
  DEBUG_MSG("converting image scan angle in x and y to lat and long for both images");

  // convert pixel coordinates to longitude and latitude:
  const float H = perspective_point_height_right + semi_major_axis_right;
  const float eccentricity = (semi_major_axis_right*semi_major_axis_right - semi_minor_axis_right*semi_minor_axis_right)
      /(semi_major_axis_right*semi_major_axis_right);

  Teuchos::ParameterList lat_long_params;
  lat_long_params.set("r_eq",semi_major_axis_right);
  lat_long_params.set("r_np",semi_minor_axis_right);
  lat_long_params.set("H",H);
  lat_long_params.set("scan_offset_x",coord_x_offset_left);
  lat_long_params.set("scan_offset_y",coord_y_offset_left);
  lat_long_params.set("scan_scale_x",coord_x_scale_factor_left);
  lat_long_params.set("scan_scale_y",coord_y_scale_factor_left);
  lat_long_params.set("earth_eccentricity",eccentricity);
  lat_long_params.set("long_of_proj_origin_left",long_of_proj_origin_left);
  lat_long_params.set("long_of_proj_origin_right",long_of_proj_origin_right);

  return lat_long_params;
}

DICE_LIB_DLL_EXPORT
void netcdf_left_pixel_points_to_earth_and_right_pixel_coordinates(const Teuchos::ParameterList & params,
  const std::vector<float> & left_pixel_x,
  const std::vector<float> & left_pixel_y,
  std::vector<float> & earth_x,
  std::vector<float> & earth_y,
  std::vector<float> & earth_z,
  std::vector<float> & right_pixel_x,
  std::vector<float> & right_pixel_y){
  assert(left_pixel_x.size()>0);
  assert(left_pixel_x.size()==left_pixel_y.size());
  const int num_pts = left_pixel_x.size();
  DEBUG_MSG("cal point grid num points:        " << num_pts);
  earth_x.clear();
  earth_x.resize(num_pts);
  earth_y.clear();
  earth_y.resize(num_pts);
  earth_z.clear();
  earth_z.resize(num_pts);
  right_pixel_x.clear();
  right_pixel_x.resize(num_pts);
  right_pixel_y.clear();
  right_pixel_y.resize(num_pts);

  const float r_eq = params.get<float>("r_eq");
  const float r_np = params.get<float>("r_np");
  const float H = params.get<float>("H");
  const float offset_x = params.get<float>("scan_offset_x");
  const float offset_y = params.get<float>("scan_offset_y");
  const float scale_x = params.get<float>("scan_scale_x");
  const float scale_y =  params.get<float>("scan_scale_y");
  const float eccentricity = params.get<float>("earth_eccentricity");
  const float long_of_proj_origin_left = params.get<float>("long_of_proj_origin_left");
  const float long_of_proj_origin_right = params.get<float>("long_of_proj_origin_right");

  for(int i=0;i<num_pts;++i){
    const float x_rad = left_pixel_x[i]*scale_x + offset_x;
    const float y_rad = left_pixel_y[i]*scale_y + offset_y;
    // convert from scan angle to satellite coords
    const float sinx = std::sin(x_rad);
    const float cosx = std::cos(x_rad);
    const float cosy = std::cos(y_rad);
    const float siny = std::sin(y_rad);
    const float a = sinx*sinx + cosx*cosx*(cosy*cosy + r_eq*r_eq*siny*siny/(r_np*r_np));
    const float b = -2.0*H*cosx*cosy;
    const float c = H*H - r_eq*r_eq;
    const float d = b*b - 4.0*a*c;
    const float r_s = d > 0.0? (-1.0*b - std::sqrt(d))/(2.0*a):1.0;
    const float sx = r_s*cosx*cosy;
    const float sy = -1.0*r_s*sinx;
    const float sz = r_s*cosx*siny;
    // convert the satellite coordinates to lat and long
    const float lat = d > 0.0? std::atan((r_eq*r_eq)/(r_np*r_np)*sz/std::sqrt((H-sx)*(H-sx)+sy*sy)) : -1.0;
    const float lon_left = (DICE_PI/180.0)*long_of_proj_origin_left - std::atan(sy/(H-sx));
    //const float lon_right = (DICE_PI/180.0)*long_of_proj_origin_right - std::atan(sy/(H-sx));
    // convert lat and long to x y z on the surface of the earth (with earth centered coords)
    const float sin_lat = std::sin(lat);
    const float sin_lon_left = std::sin(lon_left);
    const float cos_lat = std::cos(lat);
    const float cos_lon_left = std::cos(lon_left);
    const float N = r_eq/(std::sqrt(1.0 - eccentricity*eccentricity*sin_lat*sin_lat));
    const float px = N*cos_lat*cos_lon_left;
    const float py = N*cos_lat*sin_lon_left;
    const float pz = (1.0-eccentricity*eccentricity)*N*sin_lat;
    earth_x[i] = px;
    earth_y[i] = py;
    earth_z[i] = pz;
    //std::cout << left_pixel_x.size() << " " << px << " " << py << " " << pz << std::endl;
    // convert from earth-centered coords to right camera pixel
    const float plen = std::sqrt(px*px+py*py);
    const float cos_adiff = std::cos((DICE_PI/180.0)*long_of_proj_origin_right - lon_left);
    const float sin_adiff = std::sin((DICE_PI/180.0)*long_of_proj_origin_right - lon_left);
    const float psi = -1.0*std::tan(plen*sin_adiff/(H - plen*cos_adiff));
    right_pixel_x[i] = (psi - offset_x)/scale_x;
    right_pixel_y[i] = left_pixel_y[i]; // assume y pixel is the same for both imagers
    //      right_x.push_back(N*cos_lat*cos_lon_right);
    //      right_y.push_back(N*cos_lat*sin_lon_right);
    //      right_z.push_back((1.0-eccentricity*eccentricity)*N*sin_lat);
    //      latitude.push_back(lat);
    //      longitude_left.push_back(long_of_proj_origin_left - std::atan(sy/(H-sx))*180.0/DICE_PI);
    //      longitude_right.push_back(long_of_proj_origin_right - std::atan(sy/(H-sx))*180.0/DICE_PI);

    //std::cout << "point:  left image x " << left_pixel_x[i] << " y " << left_pixel_y[i] << " left lat " << lat << " long " << lon_left << " earth x " << earth_x[i] << " y " << earth_y[i] << " z " << earth_z[i] << std::endl;

    //      std::cout << " lpx " << left_pixel_x[left_pixel_x.size()-1] << " lpy " << left_pixel_y[left_pixel_y.size()-1] <<
    //          " rpx " << right_pixel_x[right_pixel_x.size()-1] << " rpy " << right_pixel_y[right_pixel_y.size()-1] << std::endl;
  }
}



} // end netcdf namespace
} // end DICe Namespace
