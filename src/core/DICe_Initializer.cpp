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

#include <DICe_Initializer.h>
#include <DICe_Schema.h>
#include <DICe_FieldEnums.h>
#include <DICe_FFT.h>
#include <DICe_Feature.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>

#include <boost/timer.hpp>

using namespace DICe::field_enums;

namespace DICe {

inline bool operator<(const def_triad& lhs, const def_triad& rhs){
  if(lhs.u_==rhs.u_){
    if(lhs.v_==rhs.v_){
      return lhs.t_ < lhs.t_;
    }
    else
      return lhs.v_<rhs.v_;
  }
  else
    return lhs.u_ < rhs.u_;
}
inline bool operator==(const def_triad& lhs, const def_triad& rhs){
  return lhs.u_ == rhs.u_ && lhs.v_ == rhs.v_ && lhs.t_ == rhs.t_;
}

Path_Initializer::Path_Initializer(Schema * schema,
  Teuchos::RCP<Subset> subset,
  const char * file_name,
  const size_t num_neighbors):
  Initializer(schema),
  subset_(subset),
  num_triads_(0),
  num_neighbors_(num_neighbors)
{
  DEBUG_MSG("Constructor for Path_Initializer with file: "  << file_name);
  TEUCHOS_TEST_FOR_EXCEPTION(num_neighbors_<=0,std::runtime_error,"");
  // read in the solution file:
  std::string line;
  std::fstream path_file(file_name,std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!path_file.is_open(),std::runtime_error,"Error, unable to load path file.");
  int_t num_lines = 0;
  // get the number of lines in the file:
  while (std::getline(path_file, line)){
    ++num_lines;
  }
  DEBUG_MSG("number of triads in the input path file: " << num_lines);
  path_file.clear();
  path_file.seekg(0,std::ios::beg);

  // there are 3 columns of data
  // id u v theta
  // TODO better checking of the input format
  // TODO enable other resolutions for the input files
  // currently the resolution is 0.5 pixels for u and v and 0.01 radians for theta
  scalar_t u_tmp=0.0,v_tmp=0.0,t_tmp=0.0;
  for(int_t line=0;line<num_lines;++line){
    path_file >> u_tmp >> v_tmp >> t_tmp;
    if(num_lines > 6){ // if the path file is small, don't filter it to the nearest half pixel, etc.
      u_tmp = floor((u_tmp*2)+0.5)/2;
      v_tmp = floor((v_tmp*2)+0.5)/2;
      t_tmp = floor(t_tmp*100+0.5)/100;
    }
    def_triad uvt(u_tmp,v_tmp,t_tmp);
    triads_.insert(uvt);
  }
  path_file.close();
  num_triads_ = triads_.size();
  DEBUG_MSG("number of triads in filtered set: " << num_triads_);
  TEUCHOS_TEST_FOR_EXCEPTION(num_triads_<=0,std::runtime_error,"");
  if(num_triads_<num_neighbors_) num_neighbors_ = num_triads_;

  DEBUG_MSG("creating the point cloud");
  point_cloud_ = Teuchos::rcp(new Point_Cloud_3D<scalar_t>());
  point_cloud_->pts.resize(num_triads_);
  std::set<def_triad>::iterator it = triads_.begin();
  int_t id = 0;
  for(;it!=triads_.end();++it){
    point_cloud_->pts[id].x = (*it).u_;
    point_cloud_->pts[id].y = (*it).v_;
    point_cloud_->pts[id].z = (*it).t_;
    id++;
  }
  DEBUG_MSG("building the kd-tree");
  kd_tree_ = Teuchos::rcp(new kd_tree_3d_t(3 /*dim*/, *point_cloud_.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree_->buildIndex();

  // now set up the neighbor list for each triad:
  neighbors_ = std::vector<size_t>(num_triads_*num_neighbors_,0);
  scalar_t query_pt[3];
  std::vector<size_t> ret_index(num_neighbors_);
  std::vector<scalar_t> out_dist_sqr(num_neighbors_);
  for(size_t id=0;id<num_triads_;++id){
    query_pt[0] = point_cloud_->pts[id].x;
    query_pt[1] = point_cloud_->pts[id].y;
    query_pt[2] = point_cloud_->pts[id].z;
    kd_tree_->knnSearch(&query_pt[0], num_neighbors_, &ret_index[0], &out_dist_sqr[0]);
    for(size_t i=0;i<num_neighbors_;++i){
      neighbors_[id*num_neighbors_ + i] = ret_index[i];
    }
  }
}

void
Path_Initializer::closest_triad(const scalar_t &u,
  const scalar_t &v,
  const scalar_t &t,
  size_t &id,
  scalar_t & distance_sqr)const{

  scalar_t query_pt[3];
  std::vector<size_t> ret_index(num_neighbors_,0.0);
  std::vector<scalar_t> out_dist_sqr(num_neighbors_,0.0);
  query_pt[0] = u;
  query_pt[1] = v;
  query_pt[2] = t;
  kd_tree_->knnSearch(&query_pt[0], 1, &ret_index[0], &out_dist_sqr[0]);
  id = ret_index[0];
  distance_sqr = out_dist_sqr[0];
}

void
Path_Initializer::write_to_text_file(const std::string & file_name)const{
  std::ofstream file;
  file.open(file_name.c_str());
  for(size_t id = 0;id<num_triads_;++id)
    file << point_cloud_->pts[id].x << " " << point_cloud_->pts[id].y << " " << point_cloud_->pts[id].z << "\n";
  file.close();
}

Status_Flag
Path_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  bool global_path_search_required = schema_->global_field_value(subset_gid,SIGMA_FS)==-1.0 || schema_->frame_id()==schema_->first_frame_id();
  if(global_path_search_required){
    initial_guess(schema_->def_img(),shape_function);
  }
  else{
    const scalar_t prev_u = schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS);
    const scalar_t prev_v = schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS);
    const scalar_t prev_t = schema_->global_field_value(subset_gid,ROTATION_Z_FS);
    initial_guess(schema_->def_img(),shape_function,prev_u,prev_v,prev_t);
  }
  return INITIALIZE_SUCCESSFUL;
}

scalar_t
Path_Initializer::initial_guess(Teuchos::RCP<Image> def_image,
  Teuchos::RCP<Local_Shape_Function> shape_function,
  const scalar_t & u,
  const scalar_t & v,
  const scalar_t & t){

  DEBUG_MSG("Path_Initializer::initial_guess(deformation,u,v,theta) called");
  TEUCHOS_TEST_FOR_EXCEPTION(def_image==Teuchos::null,std::runtime_error,"Error, pointer to deformed image must not be null here.");
  // find the closes triad in the set:
  size_t id = 0;
  scalar_t dist = 0.0;
  // iterate over the closest 6 triads to this one to see which one is best:
  // start with the given guess
  shape_function->insert_motion(u,v,t);
  // TODO what to do with the rest of the deformation entries (zero them)?
  subset_->initialize(def_image,DEF_INTENSITIES,shape_function);
  // assumes that the reference subset has already been initialized
  scalar_t gamma = subset_->gamma();
  DEBUG_MSG("input u: " << u << " v: " << v << " theta: " << t << " gamma: " << gamma);
  if(gamma<0.0) gamma = 4.0; // catch a failed gamma eval
  closest_triad(u,v,t,id,dist);
  DEBUG_MSG("closest triad id: " << id << " distance squared: " << dist);
  scalar_t best_u = u;
  scalar_t best_v = v;
  scalar_t best_t = t;
  scalar_t best_gamma = gamma;

  for(size_t neigh = 0;neigh<num_neighbors_;++neigh){
    const size_t neigh_id = neighbor(id,neigh);
    DEBUG_MSG("neigh id: " << neigh_id);
    shape_function->insert_motion(point_cloud_->pts[neigh_id].x,point_cloud_->pts[neigh_id].y,point_cloud_->pts[neigh_id].z);
    DEBUG_MSG("checking triad id: " << neigh_id << " " << point_cloud_->pts[neigh_id].x << " " <<
      point_cloud_->pts[neigh_id].y << " " << point_cloud_->pts[neigh_id].z);
    // TODO what to do with the rest of the deformation entries (zero them)?
    subset_->initialize(def_image,DEF_INTENSITIES,shape_function);
    // assumes that the reference subset has already been initialized
    gamma = subset_->gamma();
    DEBUG_MSG("gamma value " << gamma);
    if(gamma<0.0) gamma = 4.0; // catch a failed gamma eval
    if(gamma < best_gamma){
      best_gamma = gamma;
      shape_function->map_to_u_v_theta(subset_->centroid_x(),subset_->centroid_y(),best_u,best_v,best_t);
    }
  }
  shape_function->insert_motion(best_u,best_v,best_t);
  return best_gamma;
}

scalar_t
Path_Initializer::initial_guess(Teuchos::RCP<Image> def_image,
  Teuchos::RCP<Local_Shape_Function> shape_function){

  DEBUG_MSG("Path_Initializer::initial_guess(deformation) called");
  TEUCHOS_TEST_FOR_EXCEPTION(def_image==Teuchos::null,std::runtime_error,"Error, pointer to deformed image must not be null here.");

  scalar_t gamma = 0.0;
  scalar_t best_u = 0.0;
  scalar_t best_v = 0.0;
  scalar_t best_t = 0.0;
  scalar_t best_gamma = 100.0;

  DEBUG_MSG("Path_Initializer::initial_guess(deformation) point cloud has " << point_cloud_->pts.size() << " points");
  // iterate the entire set of triads:
  for(size_t id = 0;id<num_triads_;++id){
    shape_function->insert_motion(point_cloud_->pts[id].x,point_cloud_->pts[id].y,point_cloud_->pts[id].z);
    DEBUG_MSG("checking triad id: " << id << " " << point_cloud_->pts[id].x << " " <<
      point_cloud_->pts[id].y << " " << point_cloud_->pts[id].z);
    // TODO what to do with the rest of the deformation entries (zero them)?
    subset_->initialize(def_image,DEF_INTENSITIES,shape_function);
    // assumes that the reference subset has already been initialized
    gamma = subset_->gamma();
    DEBUG_MSG("gamma value " << std::setprecision(6) << gamma);
    if(gamma<0.0) gamma = 4.0; // catch a failed gamma eval
    if(gamma < best_gamma){
      best_gamma = gamma;
      shape_function->map_to_u_v_theta(subset_->centroid_x(),subset_->centroid_y(),best_u,best_v,best_t);
    }
  }
  shape_function->insert_motion(best_u,best_v,best_t);
  return best_gamma;
}

Status_Flag
Phase_Correlation_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  shape_function->insert_motion(phase_cor_u_x_ + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS),
    phase_cor_u_y_ + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS),
    schema_->global_field_value(subset_gid,ROTATION_Z_FS));
  return INITIALIZE_SUCCESSFUL;
};

void
Phase_Correlation_Initializer::pre_execution_tasks(){
  assert(schema_->prev_img()!=Teuchos::null);
  assert(schema_->def_img()!=Teuchos::null);
  DICe::phase_correlate_x_y(schema_->prev_img(),schema_->def_img(),phase_cor_u_x_,phase_cor_u_y_);
  DEBUG_MSG("Phase_Correlation_Initializer::pre_execution_tasks(): initial displacements ux: " << phase_cor_u_x_ << " uy: " << phase_cor_u_y_);
}

Status_Flag
Search_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){

  DEBUG_MSG("Search_Initializer::initial_guess(): called for subset " << subset_gid);

  TEUCHOS_TEST_FOR_EXCEPTION(step_size_u_==0.0,std::runtime_error,"Error, step x size must not be 0");
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_v_==0.0,std::runtime_error,"Error, step y size must not be 0");
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_theta_==0.0,std::runtime_error,"Error, step size theta must not be 0");

  // start with the input deformation
  scalar_t orig_u = 0.0,orig_v=0.0,orig_t=0.0;
  shape_function->map_to_u_v_theta(subset_->centroid_x(),subset_->centroid_y(),orig_u,orig_v,orig_t);
  const scalar_t start_u = step_size_u_ < 0.0 ? orig_u : orig_u - search_dim_u_;
  const scalar_t start_v = step_size_v_ < 0.0 ? orig_v : orig_v - search_dim_v_;
  const scalar_t start_t = step_size_theta_ < 0.0 ? orig_t : orig_t - search_dim_theta_;
  const scalar_t end_u = step_size_u_ < 0.0 ? orig_u : orig_u + search_dim_u_;
  const scalar_t end_v = step_size_v_ < 0.0 ? orig_v : orig_v + search_dim_v_;
  const scalar_t end_t = step_size_theta_ < 0.0 ? orig_t : orig_t + search_dim_theta_;
  if(step_size_u_ < 0.0) step_size_u_ = 1.0;
  if(step_size_v_ < 0.0) step_size_v_ = 1.0;
  if(step_size_theta_ < 0.0) step_size_theta_ = 1.0;
  const scalar_t gamma_good_enough = 1.0E-4;
  scalar_t min_gamma = 100.0;
  scalar_t min_u = 0.0;
  scalar_t min_v = 0.0;
  scalar_t min_theta = 0.0;
  // search in u, v, and theta
  DEBUG_MSG("Search ranges " << start_u << " to " << end_u << " " << start_v << " to " << end_v << " " << start_t << " to " << end_t);
  for(scalar_t trial_v = start_v;trial_v<=end_v;trial_v+=step_size_u_){
    for(scalar_t trial_u = start_u;trial_u<=end_u;trial_u+=step_size_v_){
      for(scalar_t trial_t = start_t;trial_t<=end_t;trial_t+=step_size_theta_){
        shape_function->insert_motion(trial_u,trial_v,trial_t);
        subset_->initialize(schema_->def_img(),DEF_INTENSITIES,shape_function);
        // assumes that the reference subset has already been initialized
        scalar_t gamma = 100.0;
        try{
          gamma = subset_->gamma();
          if(gamma<0.0) gamma = 4.0; // catch a failed gamma eval
        }
        catch(std::exception & e){
          gamma = 100.0;
        }
        //DEBUG_MSG("search pos " << trial_u << " " << trial_v << " " << trial_t << " gamma " << gamma);
        if(gamma < min_gamma){
          min_gamma = gamma;
          min_u = trial_u;
          min_v = trial_v;
          min_theta = trial_t;
        }
        if(gamma < gamma_good_enough){
          DEBUG_MSG("Found very small gamma: " << gamma << " skipping the rest of the search");
          DEBUG_MSG("Search initialization values: u, " << min_u << " v, "
            << min_v << " theta, " << min_theta);
          return INITIALIZE_SUCCESSFUL;
        }
      } // theta loop
    } // u loop
  } // v loop
  DEBUG_MSG("Search initialization values: " << min_u << " " << min_v << " " << min_theta << " gamma: " << min_gamma);
  shape_function->insert_motion(min_u,min_v,min_theta);
  if(min_gamma < 1.0)
    return INITIALIZE_SUCCESSFUL;
  else
    return INITIALIZE_FAILED;
};

Status_Flag
Field_Value_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  int_t sid = subset_gid;
  // logic for using neighbor values
  if(schema_->initialization_method()==DICe::USE_NEIGHBOR_VALUES ||
      (schema_->initialization_method()==DICe::USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY && schema_->frame_id()==schema_->first_frame_id())){
    sid = schema_->global_field_value(subset_gid,NEIGHBOR_ID_FS);
  }

  if(sid==-1) // catch case that subset does not have a neighbor
    sid=subset_gid;

  // make sure the data lives on this processor
  TEUCHOS_TEST_FOR_EXCEPTION(schema_->subset_local_id(sid)<0,std::runtime_error,
    "Error: Only subset ids on this processor can be used for initialization");

  const scalar_t sigma = schema_->global_field_value(sid,SIGMA_FS);
  if(sigma!=-1.0){
    shape_function->initialize_parameters_from_fields(schema_,sid);
    if(sid==subset_gid)
      return INITIALIZE_USING_PREVIOUS_FRAME_SUCCESSFUL;
    else{
      // if using a neighbor's value, the parameters have to be adjusted to account
      // for the change in centroids
      if(schema_->quadratic_shape_function_enabled()){
        scalar_t cx = schema_->global_field_value(subset_gid,SUBSET_COORDINATES_X_FS);
        scalar_t cy = schema_->global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS);
        scalar_t cx_neigh = schema_->global_field_value(sid,SUBSET_COORDINATES_X_FS);
        scalar_t cy_neigh = schema_->global_field_value(sid,SUBSET_COORDINATES_Y_FS);
        shape_function->update_params_for_centroid_change(cx - cx_neigh,cy - cy_neigh);
      }
      return INITIALIZE_USING_NEIGHBOR_VALUE_SUCCESSFUL;
    }
  }
  return INITIALIZE_FAILED;
};

void
Feature_Matching_Initializer::pre_execution_tasks(){
  assert(schema_->ref_img()!=Teuchos::null);
  assert(schema_->def_img()!=Teuchos::null);
  if(first_call_){
    prev_img_ = schema_->ref_img();
  }
  // read both images and match features between them
  std::vector<scalar_t> left_x;
  std::vector<scalar_t> left_y;
  std::vector<scalar_t> right_x;
  std::vector<scalar_t> right_y;
  {
    boost::timer t;
    const float tol = 0.005f;
    std::stringstream outname;
    outname << "fm_initializer_" << schema_->mesh()->get_comm()->get_rank() << ".png";
    match_features(prev_img_,schema_->def_img(0),left_x,left_y,right_x,right_y,tol,outname.str());
    const int_t num_matches = left_x.size();
    DEBUG_MSG("number of features matched: " << num_matches);
    DEBUG_MSG("time to compute features: "  << t.elapsed());
    TEUCHOS_TEST_FOR_EXCEPTION(num_matches < 10,std::runtime_error,"Error, not enough features matched for feature matching initializer");
  }
  // create a point cloud and find the nearest neighbor:
  point_cloud_ = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud_->pts.resize(left_x.size());
  for(size_t i=0;i<left_x.size();++i){
    point_cloud_->pts[i].x = left_x[i];
    point_cloud_->pts[i].y = left_y[i];
  }
  DEBUG_MSG("Feature_Matching_Initializer: building the kd-tree");
  kd_tree_ = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud_.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree_->buildIndex();
  u_.resize(left_x.size());
  v_.resize(left_x.size());
  for(size_t i=0;i<left_x.size();++i){
    u_[i] = right_x[i] - left_x[i];
    v_[i] = right_y[i] - left_y[i];
  }
  prev_img_ = schema_->def_img(0);
  first_call_ = false;
}

Status_Flag
Feature_Matching_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){

  if(schema_->global_field_value(subset_gid,SIGMA_FS)<0)
    return INITIALIZE_FAILED;
  const int_t num_neighbors = 1;
  // now set up the neighbor list for each triad:
  //std::vector<size_t>(subset_->num_pixels()*num_neighbors_,0);
  const scalar_t current_loc_x = schema_->global_field_value(subset_gid,SUBSET_COORDINATES_X_FS) + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS);
  const scalar_t current_loc_y = schema_->global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS) + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS);
  scalar_t query_pt[2];
  std::vector<size_t> ret_index(num_neighbors);
  std::vector<scalar_t> out_dist_sqr(num_neighbors);
  query_pt[0] = current_loc_x;
  query_pt[1] = current_loc_y;
  kd_tree_->knnSearch(&query_pt[0], num_neighbors, &ret_index[0], &out_dist_sqr[0]);
  const int_t nearest_feature_id = ret_index[0];
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)u_.size()<=nearest_feature_id,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)v_.size()<=nearest_feature_id,std::runtime_error,"");
  shape_function->insert_motion(u_[nearest_feature_id] + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS),
    v_[nearest_feature_id] + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS));
  DEBUG_MSG("Subset " << subset_gid << " init. with values: u " << u_[nearest_feature_id] + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS) <<
    " v " << v_[nearest_feature_id] + schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS) << " dist from nearest feature: " << std::sqrt(out_dist_sqr[0]) << " px");
  return INITIALIZE_SUCCESSFUL;
};


Status_Flag
Zero_Value_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  shape_function->clear();
  return INITIALIZE_SUCCESSFUL;
};


Optical_Flow_Initializer::Optical_Flow_Initializer(Schema * schema,
  Teuchos::RCP<Subset> subset):
  Initializer(schema),
  subset_(subset),
  num_neighbors_(20), // number of closest pixels
  window_size_(13),
  half_window_size_(7),
  ref_pt1_x_(0),
  ref_pt1_y_(0),
  ref_pt2_x_(0),
  ref_pt2_y_(0),
  current_pt1_x_(0.0),
  current_pt1_y_(0.0),
  current_pt2_x_(0.0),
  current_pt2_y_(0.0),
  reset_locations_(true),
  delta_1c_x_(0.0),
  delta_1c_y_(0.0),
  delta_12_x_(0.0),
  delta_12_y_(0.0),
  mag_ref_(0.0),
  ref_cx_(0.0),
  ref_cy_(0.0),
  initial_u_(0.0),
  initial_v_(0.0),
  initial_t_(0.0)
 {
  DEBUG_MSG("Optical_Flow_Initializer::Optical_Flow_Initializer()");
  DEBUG_MSG("Optical_Flow_Initializer: creating the point cloud");
  point_cloud_ = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud_->pts.resize(subset_->num_pixels());
  for(int_t i=0;i<subset_->num_pixels();++i){
    point_cloud_->pts[i].x = subset_->x(i);
    point_cloud_->pts[i].y = subset_->y(i);
  }
  DEBUG_MSG("Optical_Flow_Initializer: building the kd-tree");
  kd_tree_ = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud_.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree_->buildIndex();

  // now set up the neighbor list for each triad:
  neighbors_ = std::vector<size_t>(subset_->num_pixels()*num_neighbors_,0);
  scalar_t query_pt[2];
  std::vector<size_t> ret_index(num_neighbors_);
  std::vector<scalar_t> out_dist_sqr(num_neighbors_);
  for(int_t i=0;i<subset_->num_pixels();++i){
    query_pt[0] = point_cloud_->pts[i].x;
    query_pt[1] = point_cloud_->pts[i].y;
    kd_tree_->knnSearch(&query_pt[0], num_neighbors_, &ret_index[0], &out_dist_sqr[0]);
    for(size_t j=0;j<num_neighbors_;++j){
      neighbors_[i*num_neighbors_ + j] = ret_index[j];
    }
  }
  // create the window coefficients
  std::vector<scalar_t> coeffs(13,0.0);
//  coeffs[0] = 0.000; coeffs[1] = 0.0046; coeffs[2] = 0.03255; coeffs[3] = 0.1455; coeffs[4] = 0.42474; coeffs[5] = 0.80735;
//  coeffs[6] = 1.0;
//  coeffs[7] = 0.80735; coeffs[8] = 0.42474; coeffs[9] = 0.1455; coeffs[10] = 0.03255; coeffs[11] = 0.0046; coeffs[12] = 0.000;
  coeffs[0] = 0.51; coeffs[1] = 0.64; coeffs[2] = 0.84; coeffs[3] = 0.91; coeffs[4] = 0.96; coeffs[5] = 0.99;
  coeffs[6] = 1.0;
  coeffs[7] = 0.99; coeffs[8] = 0.96; coeffs[9] = 0.91; coeffs[10] = 0.84; coeffs[11] = 0.64; coeffs[12] = 0.51;

  for(int_t j=0;j<window_size_;++j){
    for(int_t i=0;i<window_size_;++i){
      window_coeffs_[i][j] = coeffs[i]*coeffs[j];
    }
  }
  ids_[0] = 0;
  ids_[1] = 0;
}

bool
Optical_Flow_Initializer::is_near_deactivated(const int_t pixel_id){
  // make sure the pixel itself is not deactivated
  if(subset_->is_deactivated_this_step(pixel_id)||!subset_->is_active(pixel_id)){
    return true;
  }
  // test all of this pixel's neighbors to see if any of them are deactivated:
  for(size_t neigh = 0;neigh<num_neighbors_;++neigh){
    const size_t neigh_id = neighbor(pixel_id,neigh);
    if(subset_->is_deactivated_this_step(neigh_id)||!subset_->is_active(neigh_id)){
      return true;
    }
  }
  return false;
}

int_t
Optical_Flow_Initializer::best_optical_flow_point(scalar_t & best_grad,
  Teuchos::ArrayRCP<int_t> & def_x,
  Teuchos::ArrayRCP<int_t> & def_y,
  Teuchos::ArrayRCP<scalar_t> & gx,
  Teuchos::ArrayRCP<scalar_t> & gy,
  std::set<std::pair<int_t,int_t> > & subset_pixels,
  Teuchos::RCP<std::vector<int_t> > existing_points,
  const bool allow_close_points){

  //Teuchos::ArrayRCP<intensity_t> colors(subset_->num_pixels(),0.0);
  // loop over the subset and find the places that are active, with all neighbors active
  // and highest gradient values
  int_t pt_id = 0;
  best_grad = 0.0;
  scalar_t grad_mag = 0.0;
  for(int_t i=0;i<subset_->num_pixels();++i){
    // make sure the pixel is not near a deactivated region
    if(is_near_deactivated(i)) {
      //colors[i] = 50.0;
      continue;
    }

    // make sure this pixel is not within 10 pixels of existing ids
    bool close_to_existing = false;
    if(existing_points!=Teuchos::null){
      // determine the number of active pixels
      int_t num_active = 0;
      for(int_t k=0;k<subset_->num_pixels();++k){
        if(!subset_->is_deactivated_this_step(k)&&subset_->is_active(k))
          num_active++;
      }
      // for most subsets, keep the distance between the optical flow points at least 10 pixels away
      // for small subsets make it 2 pixels away
      // the distance is measured in the reference frame
      const scalar_t dist_min = (num_active < 100||allow_close_points) ? 4.0 : 100.0;
      for(size_t k=0;k<existing_points->size();++k){
        const int_t loc_x = subset_->x((*existing_points)[k]);
        const int_t loc_y = subset_->y((*existing_points)[k]);
        const scalar_t dist_sq = (subset_->x(i)-loc_x)*(subset_->x(i)-loc_x) +
            (subset_->y(i)-loc_y)*(subset_->y(i)-loc_y);
        if(dist_sq < dist_min){
          close_to_existing = true;
          break;
        }
      } // existing points loop
    } // if has existing points
    if(close_to_existing){
      //colors[i] = 100.0;
      continue;
    }
    // check that all the neighbors are internal to the subset:
    bool neighbor_is_out_of_bounds = false;
    for(int_t m=-3;m<=3;++m){
      for(int_t n=-3;n<=3;++n){
        // need to check the deformed location for each pixel
        std::pair<int_t,int_t> pair(def_y[i]+m,def_x[i]+n);
        if(subset_pixels.find(pair)==subset_pixels.end()){
          neighbor_is_out_of_bounds = true;
          break;
        }
      }
      if(neighbor_is_out_of_bounds) break;
    }
    if(neighbor_is_out_of_bounds){
      //colors[i] = 255.0;
      continue;
    }
    //colors[i] = 200.0;
    // compare the rest to see which has the best gradient values:
    grad_mag = gx[i]*gx[i]*gy[i]*gy[i];
    if(grad_mag > best_grad){
      best_grad = grad_mag;
      pt_id = i;
    }
  }
  //colors[pt_id] = 0;
  //Image img(schema_->ref_img());
  //static int_t num_images = 0;
  //for(int_t i=0;i<subset_->num_pixels();++i)
  //  img.intensity_array()[subset_->y(i)*img.width()+subset_->x(i)] = colors[i];
  //std::stringstream filename;
  //filename << "locations_test_" << num_images << ".tif";
  //img.write(filename.str());
  //num_images++;

  // catch the case that no valid pixels exist
  if(pt_id==0&&best_grad==0.0){
    DEBUG_MSG("Optical_Flow_Initializer::best_optical_flow_point() FAILURE no valid points exist");
    return -1;
  }
  return pt_id;
}

Status_Flag
Optical_Flow_Initializer::set_locations(const int_t subset_gid){
  assert(schema_->prev_img()!=Teuchos::null);
  DEBUG_MSG("Optical_Flow_Initializer::set_locations() called");
  static scalar_t grad_coeffs[5] = {-1.0,8.0,0.0,-8.0,1.0};
  Teuchos::ArrayRCP<scalar_t> gx(subset_->num_pixels(),0.0);
  Teuchos::ArrayRCP<scalar_t> gy(subset_->num_pixels(),0.0);
  Teuchos::ArrayRCP<int_t> def_x(subset_->num_pixels(),0);
  Teuchos::ArrayRCP<int_t> def_y(subset_->num_pixels(),0);

  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(schema_);
  shape_function->initialize_parameters_from_fields(schema_,subset_gid);
  shape_function->map_to_u_v_theta(subset_->centroid_x(),subset_->centroid_y(),initial_u_,initial_v_,initial_t_);
  scalar_t mapped_x=0.0,mapped_y=0.0;
  int_t px=0,py=0;
  const int_t cx = subset_->centroid_x();
  const int_t cy = subset_->centroid_y();
  for(int_t i=0;i<subset_->num_pixels();++i){
    // compute the deformed shape:
    // mapped location
    shape_function->map(subset_->x(i),subset_->y(i),cx,cy,mapped_x,mapped_y);
    // get the nearest pixel location:
    px = (int_t)mapped_x;
    if(mapped_x - (int_t)mapped_x >= 0.5) px++;
    py = (int_t)mapped_y;
    if(mapped_y - (int_t)mapped_y >= 0.5) py++;
    def_x[i] = px;
    def_y[i] = py;
    // get the gradient information from the previous image at the deformed location:
    for(int_t j=0;j<5;++j){
      gx[i] += (-1.0/12.0)*grad_coeffs[j]*(*schema_->prev_img())(px-2+j,py);
      gy[i] += (-1.0/12.0)*grad_coeffs[j]*(*schema_->prev_img())(px,py-2+j);
    }
  }
  std::set<std::pair<int_t,int_t> > subset_pixels = subset_->deformed_shapes(shape_function,cx,cy,1.0);
  scalar_t best_grad = 0.0;
  ids_[0] = best_optical_flow_point(best_grad,def_x,def_y,gx,gy,subset_pixels);
  if(ids_[0] == -1){
    ref_pt1_x_ = 0; ref_pt1_y_ = 0;
    ref_pt2_x_ = 0; ref_pt2_y_ = 0;
    return INITIALIZE_FAILED;
  }
  ref_pt1_x_ = def_x[ids_[0]];//subset_->x(ids_[0]);
  ref_pt1_y_ = def_y[ids_[0]];//subset_->y(ids_[0]);
  DEBUG_MSG("Optical_Flow_Initializer::set_locations() point 1 id " << ids_[0] << " x " << ref_pt1_x_ << " y " << ref_pt1_y_);

  // find the second best id, but also can't be near the first one:
  Teuchos::RCP<std::vector<int_t> > existing_points = Teuchos::rcp(new std::vector<int_t>(1,ids_[0]));
  ids_[1] = best_optical_flow_point(best_grad,def_x,def_y,gx,gy,subset_pixels,existing_points);
  if(ids_[1] == -1){
    DEBUG_MSG("Optical_Flow_Initializer::set_locations() could not find good point 2, re-trying with allow_close_points");
    ids_[1] = best_optical_flow_point(best_grad,def_x,def_y,gx,gy,subset_pixels,existing_points,true);
  }
  if(ids_[1] == -1){
    ref_pt1_x_ = 0; ref_pt1_y_ = 0;
    ref_pt2_x_ = 0; ref_pt2_y_ = 0;
    return INITIALIZE_FAILED;
  }
  ref_pt2_x_ = def_x[ids_[1]];//subset_->x(ids_[1]);
  ref_pt2_y_ = def_y[ids_[1]];//subset_->y(ids_[1]);
  DEBUG_MSG("Optical_Flow_Initializer::set_locations() point 2 id " << ids_[1] << " x " << ref_pt2_x_ << " y " << ref_pt2_y_);
  reset_locations_ = false;
  current_pt1_x_ = ref_pt1_x_;
  current_pt1_y_ = ref_pt1_y_;
  current_pt2_x_ = ref_pt2_x_;
  current_pt2_y_ = ref_pt2_y_;
  ref_cx_ = cx + initial_u_;
  ref_cy_ = cy + initial_v_;
  // vector from point 1 to centroid in the ref config
  delta_1c_x_ = ref_cx_ - ref_pt1_x_;
  delta_1c_y_ = ref_cy_ - ref_pt1_y_;
  // vector from point 1 to point 2 in the ref config
  delta_12_x_ = ref_pt2_x_ - ref_pt1_x_;
  delta_12_y_ = ref_pt2_y_ - ref_pt1_y_;
  mag_ref_  = std::sqrt(delta_12_x_*delta_12_x_ + delta_12_y_*delta_12_y_);
  return INITIALIZE_SUCCESSFUL;
}

Status_Flag
Optical_Flow_Initializer::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  assert(schema_->prev_img()!=Teuchos::null);
  DEBUG_MSG("Optical_Flow_Initializer::initial_guess() Subset " << subset_gid);

  // determine if the locations need to be reset:
  DEBUG_MSG("Optical_Flow_Initializer::initial_guess() starting point locations: " << ref_pt1_x_ << " " <<
    ref_pt1_y_ << " and " << ref_pt2_x_ << " " << ref_pt2_y_ );
  if(reset_locations_==false){ // only check this if nothing else has requested a restart
    // check that all the optical flow locations are away from obstructions
    if(is_near_deactivated(ids_[0])||is_near_deactivated(ids_[1]))
      reset_locations_=true;
  }

  if(reset_locations_){
    Status_Flag location_flag = set_locations(subset_gid);
    if(location_flag!=INITIALIZE_SUCCESSFUL) {
      DEBUG_MSG("Optical_Flow_Initializer::initial_guess() set_locations FAILURE, using field values to initialize");
      shape_function->insert_motion(schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS),
        schema_->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS),
        schema_->global_field_value(subset_gid,ROTATION_Z_FS));
      return INITIALIZE_SUCCESSFUL;
    }
  }

  // check if the solve was skipped, if not use the last converged solution for the
  // new position of the optical flow points
  bool skip_solve = false;

  if(schema_->skip_solve_flags()->find(subset_gid)!=schema_->skip_solve_flags()->end()){
    skip_solve = frame_should_be_skipped(schema_->frame_id(),schema_->skip_solve_flags()->find(subset_gid)->second);
  }
  if(schema_->skip_all_solves())
    skip_solve = true;

  // reset the current locations of the optical flow points based on the last solution
  if(!skip_solve){
    const int_t cx = subset_->centroid_x();
    const int_t cy = subset_->centroid_y();
    // compute the deformed shape:
    // mapped location
    shape_function->map(subset_->x(ids_[0]),subset_->y(ids_[0]),cx,cy,current_pt1_x_,current_pt1_y_);
    // mapped location
    shape_function->map(subset_->x(ids_[1]),subset_->y(ids_[1]),cx,cy,current_pt2_x_,current_pt2_y_);
  }

  int_t pt_def_x[2] = {0,0};
  int_t pt_def_y[2] = {0,0};
  // find the nearest pixel for the updated location
  pt_def_x[0] = (int_t)current_pt1_x_;
  if(current_pt1_x_ - (int_t)(current_pt1_x_) >= 0.5) pt_def_x[0]++;
  pt_def_y[0] = (int_t)current_pt1_y_;
  if(current_pt1_y_ - (int_t)(current_pt1_y_) >= 0.5) pt_def_y[0]++;
  pt_def_x[1] = (int_t)current_pt2_x_;
  if(current_pt2_x_ - (int_t)(current_pt2_x_) >= 0.5) pt_def_x[1]++;
  pt_def_y[1] = (int_t)current_pt2_y_;
  if(current_pt2_y_ - (int_t)(current_pt2_y_) >= 0.5) pt_def_y[1]++;
  static scalar_t grad_coeffs[5] = {-1.0,8.0,0.0,-8.0,1.0};

  // do the optical flow about these points...
  const int N = 2;
  Teuchos::SerialDenseMatrix<int,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  int *IPIV = new int[N+1];
  double *EIGS = new double[N+1];
  int LWORK = N*N;
  int QWORK = 3*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *SWORK = new double[QWORK];
  Teuchos::LAPACK<int,double> lapack;

  scalar_t update_x[2] = {0.0,0.0};
  scalar_t update_y[2] = {0.0,0.0};
  for(int_t pt=0;pt<2;++pt){
    // clear the values of H and q
    H(0,0) = 0.0;
    H(1,0) = 0.0;
    H(0,1) = 0.0;
    H(1,1) = 0.0;
    q[0] = 0.0;
    q[1] = 0.0;
    scalar_t Ix = 0.0;
    scalar_t Iy = 0.0;
    scalar_t It = 0.0;
    scalar_t w_coeff = 0.0;
    int_t x=0,y=0;
    // loop over subset pixels in the deformed location
    for(int_t j=0;j<window_size_;++j){
      y = pt_def_y[pt] - half_window_size_ + j;
      for(int_t i=0;i<window_size_;++i){
        x = pt_def_x[pt] - half_window_size_ + i;
        // need to recompute the gradients here because the pixel may be outside the original subset:
        // get the gradient information from the previous image at the deformed location:
        Ix = 0.0;
        Iy = 0.0;
        for(int_t k=0;k<5;++k){
          Ix += (-1.0/12.0)*grad_coeffs[k]*(*schema_->prev_img())(x-2+k,y);
          Iy += (-1.0/12.0)*grad_coeffs[k]*(*schema_->prev_img())(x,y-2+k);
        }
        It = (*schema_->def_img())(x,y) - (*schema_->prev_img())(x,y);
        w_coeff = window_coeffs_[i][j];
        H(0,0) += Ix*Ix*w_coeff*w_coeff;
        H(1,0) += Ix*Iy*w_coeff*w_coeff;
        H(0,1) += Iy*Ix*w_coeff*w_coeff;
        H(1,1) += Iy*Iy*w_coeff*w_coeff;
        q[0] += Ix*It*w_coeff*w_coeff;
        q[1] += Iy*It*w_coeff*w_coeff;
      }
    }
    // do the inversion:
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
    lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
    lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
    // compute u and v
    update_x[pt] = -H(0,0)*q[0] - H(0,1)*q[1];
    update_y[pt] = -H(1,0)*q[0] - H(1,1)*q[1];
    DEBUG_MSG("Optical_Flow_Initializer::initial_guess() displacement for point " << pt << ": " << update_x[pt] << " " << update_y[pt]);
  }

  delete[] IPIV;
  delete[] EIGS;
  delete[] WORK;
  delete[] SWORK;

  // accumulate the displacements in case solve is skipped
  current_pt1_x_ += update_x[0];
  current_pt1_y_ += update_y[0];
  current_pt2_x_ += update_x[1];
  current_pt2_y_ += update_y[1];
  DEBUG_MSG("Optical_Flow_Initializer()::initial_guess() curren position point 1: " << current_pt1_x_ << " " << current_pt1_y_);
  DEBUG_MSG("Optical_Flow_Initializer()::initial_guess() curren position point 2: " << current_pt2_x_ << " " << current_pt2_y_);

  // work out the trig with the centroid of the subset:

  // vector from point 1 to point 2 in the def config
  const scalar_t delta_12_x_def = current_pt2_x_ - current_pt1_x_;
  const scalar_t delta_12_y_def = current_pt2_y_ - current_pt1_y_;
  const scalar_t mag_def  = std::sqrt(delta_12_x_def*delta_12_x_def + delta_12_y_def*delta_12_y_def);
  DEBUG_MSG("Optical_Flow_Initializer()::initial_guess() mag_ref " << mag_ref_ << " mag_def " << mag_def);

  // angle between these two vectors:
  const scalar_t a_dot_b = delta_12_x_*delta_12_x_def + delta_12_y_*delta_12_y_def;
  if(mag_ref_*mag_def==0.0) return INITIALIZE_FAILED; // FIXME use field_values here?
  // TODO check that the mag_ref and mag_def are about the same (should be rigid body motion?)
  scalar_t value = a_dot_b/(mag_ref_*mag_def);
  if (value < -1.0) value = -1.0 ;
  else if (value > 1.0) value = 1.0 ;
  scalar_t theta_12 = 1.0*std::acos(value); // DICe measures theta in the other direction
  // need to test theta + and theta - to see which one is right:
  // take the cross poduct, negative cross is negative theta
  scalar_t cross_prod = delta_12_x_*delta_12_y_def - (delta_12_x_def*delta_12_y_);
  if(cross_prod < 0.0) theta_12 = -1.0*theta_12;

  const scalar_t cos_t = std::cos(theta_12);
  const scalar_t sin_t = std::sin(theta_12);
  const scalar_t new_cx = cos_t*delta_1c_x_ - sin_t*delta_1c_y_ + current_pt1_x_;
  const scalar_t new_cy = sin_t*delta_1c_x_ + cos_t*delta_1c_y_ + current_pt1_y_;
  const scalar_t disp_x = new_cx - ref_cx_;
  const scalar_t disp_y = new_cy - ref_cy_;
  DEBUG_MSG("Optical_Flow_Initializer::initial_guess() computed u " << disp_x << " v " << disp_y << " theta " << theta_12);
  if(std::abs(disp_x)>schema_->prev_img()->width() || std::abs(disp_y)>schema_->prev_img()->height() || std::abs(theta_12) > DICE_TWOPI){
    DEBUG_MSG("Failed initialization due to invalid u, v, or theta.");
    return INITIALIZE_FAILED;
  }

  Teuchos::ArrayRCP<intensity_t> output_img(schema_->prev_img()->num_pixels(),0.0);
  for(int_t j=0;j<schema_->prev_img()->height();++j)
  {
    for(int_t i=0;i<schema_->prev_img()->width();++i)
    {
      if((i==pt_def_x[0]&&j==pt_def_y[0])||(i==pt_def_x[1]&&j==pt_def_y[1])){
      //if((i==pt1_x_&&j==pt1_y_)||(i==pt2_x_&&j==pt2_y_)){
        output_img[j*schema_->prev_img()->width()+i] = 255;
      }else{
        output_img[j*schema_->prev_img()->width()+i] = (*schema_->prev_img())(i,j);
      }
    }
  }

//  // uncomment to output images with dots where the optical flow points are
//  // turn on boost filesystem above...
//  std::stringstream dirname;
//  dirname << "./of_imgs_" << subset_gid;
//  DEBUG_MSG("Attempting to create directory : " << dirname.str());
//  boost::filesystem::path dir(dirname.str());
//  if(boost::filesystem::create_directory(dir)) {
//    DEBUG_MSG("Directory successfully created");
//  }
//  Image out(schema_->prev_img()->width(),schema_->prev_img()->height(),output_img);
//  std::stringstream filename;
//  filename << "./of_imgs_" << subset_gid << "/img_" << schema_->image_frame() << ".tif";
//  out.write(filename.str());

  // Note: these are additive displacements from the previous frame so add them to what's already in the array
  shape_function->insert_motion(initial_u_ + disp_x,initial_v_ + disp_y,initial_t_+ theta_12);
  // leave the rest alone...

  return INITIALIZE_SUCCESSFUL;
};

Motion_Test_Utility::Motion_Test_Utility(Schema * schema,
  const scalar_t & tol):
  schema_(schema),
  tol_(tol),
  motion_state_(MOTION_NOT_SET)
{
  DEBUG_MSG("Constructor for Motion_Test_Utility called, tol: " << tol_);
}

bool
Motion_Test_Utility::motion_detected(const int_t sub_image_id){
  assert(schema_->prev_img()!=Teuchos::null);
  // make sure the id is valid
  TEUCHOS_TEST_FOR_EXCEPTION(sub_image_id<0||sub_image_id>=(int_t)schema_->def_imgs()->size(),
    std::runtime_error,"Error, ivalid sub_image_id " << sub_image_id);
  // test if this is a repeat call for the same frame, but by another subset
  // if so, return the previous result.
  if(motion_state_!=MOTION_NOT_SET){
    DEBUG_MSG("Motion_Test_Utility::motion_detected() repeat call, return value: " << motion_state_);
    // return last result
    return motion_state_==MOTION_TRUE ? true: false;
  }
  else{
    // make sure that the images are gauss filtered:
    TEUCHOS_TEST_FOR_EXCEPTION(!schema_->def_img(sub_image_id)->has_gauss_filter(),std::runtime_error,
      "Error, Gauss filtering required for using motion windows, but gauss filtering is not enabled in the input.");
    const int_t half_mask = schema_->def_img(sub_image_id)->gauss_filter_mask_size()/2;
    const int_t w = schema_->def_img(sub_image_id)->width();
    const int_t h = schema_->def_img(sub_image_id)->height();
    DEBUG_MSG("Motion_Test_Utility::motion_detected(): motion window sub_image_id " << sub_image_id << " width " << w << " height " << h);
    //diff the two images and see if the difference is above the user requested tolerance
    scalar_t diff = 0.0;
    // skip the outer edges since they are not filtered
    for(int_t y=half_mask+1;y<h-(half_mask+1);++y){
      for(int_t x=half_mask+1;x<w-(half_mask+1);++x){
        diff += ((*schema_->def_img(sub_image_id))(x,y) - (*schema_->prev_img(sub_image_id))(x,y))*((*schema_->def_img(sub_image_id))(x,y) - (*schema_->prev_img(sub_image_id))(x,y));
      }
    }
    diff = std::sqrt(diff);
    DEBUG_MSG("Motion_Test_Utility::motion_detected() called, img diff: " << diff << " initial tol: " << tol_);
    if(tol_==-1.0&&diff!=0.0){ // user has not set a tolerance manually
      tol_ = diff + 5.0;
      DEBUG_MSG("Motion_Test_Utility::motion_detected() setting auto tolerance to: " << tol_);
    }
    motion_state_ = diff > tol_ ? MOTION_TRUE : MOTION_FALSE;
    return motion_state_==MOTION_TRUE ? true: false;
  }
}

}// End DICe Namespace
