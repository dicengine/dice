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

#include <DICe_api.h>
#include <DICe_Schema.h>
#include <DICe_Parser.h>
#include <DICe_ParameterUtilities.h>

#ifdef __cplusplus
extern "C" {
#endif

DICE_LIB_DLL_EXPORT const int_t dice_correlate(scalar_t points[], int_t n_points,
                        int_t subset_size,
                        intensity_t ref_img[], int_t ref_w, int_t ref_h,
                        intensity_t def_img[], int_t def_w, int_t def_h,
                        Teuchos::ParameterList * input_params, const bool update_params){

  static bool initialized = false;

  DEBUG_MSG("dice_correlate() called with the following values:");
  DEBUG_MSG("n_points               " << n_points);
  DEBUG_MSG("subset_size            " << subset_size);
  DEBUG_MSG("ref_w                  " << ref_w);
  DEBUG_MSG("ref_h                  " << ref_h);
  DEBUG_MSG("def_w                  " << def_w);
  DEBUG_MSG("def_h                  " << def_h);
  DEBUG_MSG("update params          " << update_params);
  DEBUG_MSG("is initialized         " << initialized);

  // throw an error if the images are not of the same size
  if(ref_w!=def_w||ref_h!=def_h) return -1;
  // throw an error if the subset size is less than 1 pixel
  if(subset_size < 1) return -1;
  // throw an error if n_points is less than one
  if(n_points < 1) return -1;

  const int_t imgSize = ref_w*ref_h;

  // create an ArrayRCP to the image
  Teuchos::ArrayRCP<intensity_t> refRCP(ref_img,0,imgSize,false);
  Teuchos::ArrayRCP<intensity_t> defRCP(def_img,0,imgSize,false);

  static Teuchos::RCP<Teuchos::ParameterList> params;
  if(input_params!=0){
    DEBUG_MSG("Using user specified input parameters in dice_correlate()");
    params = rcp(input_params,false);
  }
  else{
    DEBUG_MSG("Using default parameters in dice_correlate()");
    static Teuchos::RCP<Teuchos::ParameterList> defaultParams = rcp(new Teuchos::ParameterList());
    if(!initialized){
      // grab the defualt values
      DICe::tracking_default_params(defaultParams.getRawPtr());
    }
    params = defaultParams;
  }

  // construct a schema:
  // make it static since we want the internal data to hang around
  static DICe::Schema schema(ref_w,ref_h,refRCP,defRCP,params);
  // call set_params on the schema in case the parameter have changed:
  if(!initialized||update_params)
    schema.set_params(params);

  // set the deformed image manually in case this is not the first image in the set:
  // the constructor won't get called on the static schema if this is not the first image since it's static
  schema.set_def_image(def_w,def_h,defRCP);

  if(!initialized){
    // initialize the schema
    // since the input data array is of a different size and in a different
    // order, we have to manually plug the values in
    Teuchos::ArrayRCP<scalar_t> coords_x(n_points,0.0);
    Teuchos::ArrayRCP<scalar_t> coords_y(n_points,0.0);
    for(int_t i=0;i<n_points;++i){
      coords_x[i] = points[i*DICE_API_STRIDE + 0];
      coords_y[i] = points[i*DICE_API_STRIDE + 1];
    }
    schema.initialize(coords_x,coords_y,subset_size);
  }
  initialized = true;

  // manually copy values into the schema's field values
  for(int_t i=0;i<n_points;++i){
    // leave the coordinates alone
    schema.field_value(i,DICe::COORDINATE_X)   = points[i*DICE_API_STRIDE + 0];
    schema.field_value(i,DICe::COORDINATE_Y)   = points[i*DICE_API_STRIDE + 1];
    schema.field_value(i,DICe::DISPLACEMENT_X) = points[i*DICE_API_STRIDE + 2];
    schema.field_value(i,DICe::DISPLACEMENT_Y) = points[i*DICE_API_STRIDE + 3];
    schema.field_value(i,DICe::ROTATION_Z)     = points[i*DICE_API_STRIDE + 4];
    schema.field_value(i,DICe::SIGMA)          = points[i*DICE_API_STRIDE + 5];
    schema.field_value(i,DICe::GAMMA)          = points[i*DICE_API_STRIDE + 6];
    schema.field_value(i,DICe::BETA)           = points[i*DICE_API_STRIDE + 7];
    schema.field_value(i,DICe::STATUS_FLAG)    = points[i*DICE_API_STRIDE + 8];
  }

  schema.execute_correlation();

  // extract the values from the correlation and put it back in the data array:
  for(int_t i=0;i<n_points;++i){
    // leave the coordinates alone
    points[i*DICE_API_STRIDE + 2] = schema.field_value(i,DICe::DISPLACEMENT_X);
    points[i*DICE_API_STRIDE + 3] = schema.field_value(i,DICe::DISPLACEMENT_Y);
    points[i*DICE_API_STRIDE + 4] = schema.field_value(i,DICe::ROTATION_Z);
    points[i*DICE_API_STRIDE + 5] = schema.field_value(i,DICe::SIGMA);
    points[i*DICE_API_STRIDE + 6] = schema.field_value(i,DICe::GAMMA);
    points[i*DICE_API_STRIDE + 7] = schema.field_value(i,DICe::BETA);
    points[i*DICE_API_STRIDE + 8] = schema.field_value(i,DICe::STATUS_FLAG);
  }

  return 0;
}

DICE_LIB_DLL_EXPORT const int_t dice_correlate_conformal(scalar_t points[],
                        intensity_t ref_img[], int_t ref_w, int_t ref_h,
                        intensity_t def_img[], int_t def_w, int_t def_h,
                        const char* subset_file, const char* param_file,
                        const bool write_output){

  static bool initialized = false;

  DEBUG_MSG("dice_correlate() called with the following values:");
  DEBUG_MSG("ref_w                  " << ref_w);
  DEBUG_MSG("ref_h                  " << ref_h);
  DEBUG_MSG("def_w                  " << def_w);
  DEBUG_MSG("def_h                  " << def_h);
  DEBUG_MSG("subset file            " << subset_file);
  DEBUG_MSG("write output           " << write_output);
  DEBUG_MSG("is initialized         " << initialized);

  // throw an error if the images are not of the same size
  if(ref_w!=def_w||ref_h!=def_h) return -1;

  static Teuchos::RCP<Teuchos::ParameterList> corr_params;

  if(!initialized){
    if(param_file!=0){
      corr_params = DICe::read_correlation_params(param_file);
      // make sure the use_sl_parameters option is set
      corr_params->set(DICe::use_tracking_default_params,true);
      DEBUG_MSG("User specified correlation Parameters:");
    }
    else{
      DEBUG_MSG("User did not specify correlation parameters (using all sl defaults).");
      corr_params = rcp(new Teuchos::ParameterList());
      DICe::tracking_default_params(corr_params.getRawPtr());
    }
#ifdef DICE_DEBUG_MSG
      corr_params->print(std::cout);
#endif
  }

  const int_t imgSize = ref_w*ref_h;

  // create an ArrayRCP to the image
  Teuchos::ArrayRCP<intensity_t> refRCP(ref_img,0,imgSize,false);
  Teuchos::ArrayRCP<intensity_t> defRCP(def_img,0,imgSize,false);

  // construct a schema:
  // make it static since we want the internal data to hang around
  static DICe::Schema schema(ref_w,ref_h,refRCP,defRCP,corr_params);

  // set the deformed image manually in case this is not the first image in the set:
  // the constructor won't get called on the static schema if this is not the first image since it's static
  schema.set_def_image(ref_w,ref_h,defRCP);

  // if a subset file is specified, copy over the subset centroids into the data array (replacing values)
  static Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs;
  static Teuchos::RCP<std::map<int_t,std::vector<int_t> > > blocking_subset_ids;
  static int_t n_points = -1;
  static int_t dim = 2; // assumes 2D
  TEUCHOS_TEST_FOR_EXCEPTION(!subset_file,std::runtime_error,"");
  if(!initialized){
    DEBUG_MSG("Reading subset information from file: " << subset_file);
    // get the conformal subset defs, blocking subset ids, and coordinates
    // TODO enable seed and sweep in dice_correlate_conformal
    Teuchos::RCP<DICe::Subset_File_Info> subset_info = DICe::read_subset_file(subset_file,ref_w,ref_h);
    Teuchos::RCP<std::vector<int_t> > subset_centroids = subset_info->coordinates_vector;
    conformal_area_defs = subset_info->conformal_area_defs;
    blocking_subset_ids = subset_info->id_sets_map;
    n_points = subset_info->coordinates_vector->size()/dim; // divide by three because the striding is x, y, neighbor_id
    // this library call requires that all subsets are defined with a conformal subset
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)conformal_area_defs->size()!=n_points,std::runtime_error,
      "Error there is a mismatch between the number of "
        "conformal subsets defined and the number of coordinate sets");
    // copy over the coordinates to the points array:
    for(int_t i=0;i<n_points;++i){
      points[i*DICE_API_STRIDE+0] = (*subset_centroids)[i*dim+0];
      points[i*DICE_API_STRIDE+1] = (*subset_centroids)[i*dim+1];
    }
  }

  if(!initialized){
    // initialize the schema
    // since the input data array is of a different size and in a different
    // order, we have to manually plug the values in
    // Passing -1 as subset size to require that all subsets are defined in the input file
    Teuchos::ArrayRCP<scalar_t> coords_x(n_points,0.0);
    Teuchos::ArrayRCP<scalar_t> coords_y(n_points,0.0);
    for(int_t i=0;i<n_points;++i){
      coords_x[i] = points[i*DICE_API_STRIDE + 0];
      coords_y[i] = points[i*DICE_API_STRIDE + 1];
    }
    schema.initialize(coords_x,coords_y,-1,conformal_area_defs);
    schema.set_obstructing_subset_ids(blocking_subset_ids);
  }

  initialized = true;

  // manually copy values into the schema's field values
  for(int_t i=0;i<n_points;++i){
    // leave the coordinates alone
    schema.field_value(i,DICe::COORDINATE_X)   = points[i*DICE_API_STRIDE + 0];
    schema.field_value(i,DICe::COORDINATE_Y)   = points[i*DICE_API_STRIDE + 1];
    schema.field_value(i,DICe::DISPLACEMENT_X) = points[i*DICE_API_STRIDE + 2];
    schema.field_value(i,DICe::DISPLACEMENT_Y) = points[i*DICE_API_STRIDE + 3];
    schema.field_value(i,DICe::ROTATION_Z)     = points[i*DICE_API_STRIDE + 4];
    schema.field_value(i,DICe::SIGMA)          = points[i*DICE_API_STRIDE + 5];
    schema.field_value(i,DICe::GAMMA)          = points[i*DICE_API_STRIDE + 6];
    schema.field_value(i,DICe::BETA)           = points[i*DICE_API_STRIDE + 7];
    schema.field_value(i,DICe::STATUS_FLAG)    = points[i*DICE_API_STRIDE + 8];
  }

  schema.execute_correlation();

  if(write_output)
    schema.write_output("./");

  // extract the values from the correlation and put it back in the data array:
  for(int_t i=0;i<n_points;++i){
    points[i*DICE_API_STRIDE + 0] = schema.field_value(i,DICe::COORDINATE_X);
    points[i*DICE_API_STRIDE + 1] = schema.field_value(i,DICe::COORDINATE_Y);
    points[i*DICE_API_STRIDE + 2] = schema.field_value(i,DICe::DISPLACEMENT_X);
    points[i*DICE_API_STRIDE + 3] = schema.field_value(i,DICe::DISPLACEMENT_Y);
    points[i*DICE_API_STRIDE + 4] = schema.field_value(i,DICe::ROTATION_Z);
    points[i*DICE_API_STRIDE + 5] = schema.field_value(i,DICe::SIGMA);
    points[i*DICE_API_STRIDE + 6] = schema.field_value(i,DICe::GAMMA);
    points[i*DICE_API_STRIDE + 7] = schema.field_value(i,DICe::BETA);
    points[i*DICE_API_STRIDE + 8] = schema.field_value(i,DICe::STATUS_FLAG);
  }

  return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif
