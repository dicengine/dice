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

#include <DICe_Initializer.h>

#include <Teuchos_RCP.hpp>

#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>

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

Path_Initializer::Path_Initializer(Teuchos::RCP<Subset> subset,
  const char * file_name,
  const size_t num_neighbors):
  Initializer(subset),
  num_triads_(0),
  num_neighbors_(num_neighbors)
{
  DEBUG_MSG("Constructor for Path_Initializer with file: "  << file_name);
  assert(num_neighbors_>0);
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
  assert(num_triads_>0);
  if(num_triads_<num_neighbors_) num_neighbors_ = num_triads_;

  DEBUG_MSG("creating the point cloud");
  point_cloud_ = Teuchos::rcp(new Point_Cloud<scalar_t>());
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
  kd_tree_ = Teuchos::rcp(new my_kd_tree_t(3 /*dim*/, *point_cloud_.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
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

scalar_t
Path_Initializer::initial_guess(Teuchos::RCP<Image> def_image,
  Teuchos::RCP<std::vector<scalar_t> > deformation,
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
  (*deformation)[DISPLACEMENT_X] = u;
  (*deformation)[DISPLACEMENT_Y] = v;
  (*deformation)[ROTATION_Z] = t;
  // TODO what to do with the rest of the deformation entries (zero them)?
  subset_->initialize(def_image,DEF_INTENSITIES,deformation);
  // assumes that the reference subset has already been initialized
  scalar_t gamma = subset_->gamma();
  DEBUG_MSG("input u: " << u << " v: " << v << " theta: " << t << " gamma: " << gamma);
  closest_triad(u,v,t,id,dist);
  DEBUG_MSG("closest triad id: " << id << " distance squared: " << dist);
  scalar_t best_u = u;
  scalar_t best_v = v;
  scalar_t best_t = t;
  scalar_t best_gamma = gamma;

  for(size_t neigh = 0;neigh<num_neighbors_;++neigh){
    const size_t neigh_id = neighbor(id,neigh);
    DEBUG_MSG("neigh id: " << neigh_id);
    (*deformation)[DISPLACEMENT_X] = point_cloud_->pts[neigh_id].x;
    (*deformation)[DISPLACEMENT_Y] = point_cloud_->pts[neigh_id].y;
    (*deformation)[ROTATION_Z] = point_cloud_->pts[neigh_id].z;
    DEBUG_MSG("checking triad id: " << neigh_id << " " << (*deformation)[DISPLACEMENT_X] << " " <<
      (*deformation)[DISPLACEMENT_Y] << " " << (*deformation)[ROTATION_Z]);
    // TODO what to do with the rest of the deformation entries (zero them)?
    subset_->initialize(def_image,DEF_INTENSITIES,deformation);
    // assumes that the reference subset has already been initialized
    gamma = subset_->gamma();
    DEBUG_MSG("gamma value " << gamma);
    if(gamma < best_gamma){
      best_gamma = gamma;
      best_u = (*deformation)[DISPLACEMENT_X];
      best_v = (*deformation)[DISPLACEMENT_Y];
      best_t = (*deformation)[ROTATION_Z];
    }
  }
  (*deformation)[DISPLACEMENT_X] = best_u;
  (*deformation)[DISPLACEMENT_Y] = best_v;
  (*deformation)[ROTATION_Z] = best_t;
  return best_gamma;
}

scalar_t
Path_Initializer::initial_guess(Teuchos::RCP<Image> def_image,
  Teuchos::RCP<std::vector<scalar_t> > deformation){

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
    (*deformation)[DISPLACEMENT_X] = point_cloud_->pts[id].x;
    (*deformation)[DISPLACEMENT_Y] = point_cloud_->pts[id].y;
    (*deformation)[ROTATION_Z] = point_cloud_->pts[id].z;
    DEBUG_MSG("checking triad id: " << id << " " << (*deformation)[DISPLACEMENT_X] << " " <<
      (*deformation)[DISPLACEMENT_Y] << " " << (*deformation)[ROTATION_Z]);
    // TODO what to do with the rest of the deformation entries (zero them)?
    subset_->initialize(def_image,DEF_INTENSITIES,deformation);
    // assumes that the reference subset has already been initialized
    gamma = subset_->gamma();
    DEBUG_MSG("gamma value " << std::setprecision(6) << gamma);
    if(gamma < best_gamma){
      best_gamma = gamma;
      best_u = (*deformation)[DISPLACEMENT_X];
      best_v = (*deformation)[DISPLACEMENT_Y];
      best_t = (*deformation)[ROTATION_Z];
    }
  }
  (*deformation)[DISPLACEMENT_X] = best_u;
  (*deformation)[DISPLACEMENT_Y] = best_v;
  (*deformation)[ROTATION_Z] = best_t;
  return best_gamma;
}


Motion_Test_Initializer::Motion_Test_Initializer(const int_t origin_x,
  const int_t origin_y,
  const int_t width,
  const int_t height,
  const scalar_t & tol):
  Initializer(Teuchos::null),
  origin_x_(origin_x),
  origin_y_(origin_y),
  width_(width),
  height_(height),
  tol_(tol),
  prev_img_(Teuchos::null)
{
  DEBUG_MSG("Constructor for Motion_Test_Initializer called");
  DEBUG_MSG("origin_x: " << origin_x_ << " origin_y: " << origin_y_ <<
    " width: " << width_ << " height: " << height_ << " tol: " << tol_);
}

bool
Motion_Test_Initializer::motion_detected(Teuchos::RCP<Image> def_image){
  static bool motion = true;
  // test if this is a repeat call for the same frame, but by another subset
  // if so, return the previous result.
  static Teuchos::RCP<Image> save_image_ptr=Teuchos::null;
  if(save_image_ptr == def_image){
    DEBUG_MSG("Motion_Test_Initializer::motion_detected() repeat call, return value: " << motion);
    // return last result
    return motion;
  }
  // otherwise save off the last pointer
  save_image_ptr = def_image;

  // create a window of the deformed image according to the constructor parameters
  Teuchos::RCP<Image> window_img = Teuchos::rcp(new Image(def_image,origin_x_,origin_y_,width_,height_));
  // see if the previous image exists, if not return true as default
  if(prev_img_==Teuchos::null){
    DEBUG_MSG("Motion_Test_Initializer::motion_detected() first frame call, return value: 1 (automatically).");
    prev_img_ = window_img; // save off the deformed image as the previous one
    return true;
  }
  //diff the two images and see if the difference is above a threshold
  const scalar_t diff = window_img->diff(prev_img_);
  prev_img_ = window_img;
  DEBUG_MSG("Motion_Test_Initializer::motion_detected() called, result: " << diff << " tol: " << tol_);
  motion = diff > tol_ ? true : false;
//  std::fstream out;
//  out.open("diff_results.txt",std::fstream::app);
//  out << diff << "\n";
//  out.close();

  return motion;
}

}// End DICe Namespace
