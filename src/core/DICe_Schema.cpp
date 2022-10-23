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

#include <DICe.h>
#include <DICe_Schema.h>
#include <DICe_Objective.h>
#include <DICe_PostProcessor.h>
#include <DICe_ParameterUtilities.h>
#include <DICe_ImageUtils.h>
#include <DICe_ImageIO.h>
#include <DICe_FFT.h>
#include <DICe_Triangulation.h>
#include <DICe_Feature.h>
#include <DICe_Simplex.h>
#if DICE_ENABLE_NETCDF
  #include <DICe_NetCDF.h>
#endif
#ifdef DICE_ENABLE_GLOBAL
  #include <DICe_MeshIO.h>
  #include <DICe_MeshIOUtils.h>
#endif

#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>
#include <math.h>

#include <cassert>
#include <set>

namespace DICe {

using namespace field_enums;

Schema::Schema(const std::string & input_file_name,
  const std::string & params_file_name){
  // create a parameter list from the selected file
  Teuchos::RCP<Teuchos::ParameterList> corr_params = read_correlation_params(params_file_name);
  default_constructor_tasks(corr_params);
  // create a parameter list from the selected file
  Teuchos::RCP<Teuchos::ParameterList> input_params = read_input_params(input_file_name);
  initialize(input_params,corr_params);
}

Schema::Schema(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params){
  default_constructor_tasks(correlation_params);
  initialize(input_params,correlation_params);
}

Schema::Schema(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params,
  const Teuchos::RCP<Schema> & schema){
  default_constructor_tasks(correlation_params);
  initialize(input_params,schema);
}

Schema::Schema(const int_t roi_width,
  const int_t roi_height,
  const int_t step_size_x,
  const int_t step_size_y,
  const int_t subset_size,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  default_constructor_tasks(params);
  TEUCHOS_TEST_FOR_EXCEPTION(is_initialized_,std::runtime_error,"Error: this schema is already initialized.");
  TEUCHOS_TEST_FOR_EXCEPTION(subset_size<=0,std::runtime_error,"Error: width cannot be equal to or less than zero.");
  step_size_x_ = step_size_x;
  step_size_y_ = step_size_y;

  // create a buffer the size of one view along all edges
  const int_t trimmedWidth = roi_width - 2*subset_size;
  const int_t trimmedHeight = roi_height - 2*subset_size;
  // set up the control points
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_x<=0,std::runtime_error,"Error, step size x is <= 0");
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_y<=0,std::runtime_error,"Error, step size y is <= 0");
  const int_t numPointsX = trimmedWidth  / step_size_x + 1;
  const int_t numPointsY = trimmedHeight / step_size_y + 1;
  TEUCHOS_TEST_FOR_EXCEPTION(numPointsX<=0,std::runtime_error,"Error, numPointsX <= 0.");
  TEUCHOS_TEST_FOR_EXCEPTION(numPointsY<=0,std::runtime_error,"Error, numPointsY <= 0.");

  const int_t num_pts = numPointsX * numPointsY;

  Teuchos::ArrayRCP<scalar_t> coords_x(num_pts,0.0);
  Teuchos::ArrayRCP<scalar_t> coords_y(num_pts,0.0);
  int_t x_it=0, y_it=0;
  for (int_t i=0;i<num_pts;++i)
  {
     y_it = i / numPointsX;
     x_it = i - (y_it*numPointsX);
     coords_x[i] = subset_size + x_it * step_size_x_ -1;
     coords_y[i] = subset_size + y_it * step_size_y_ -1;
  }
  Teuchos::RCP<Decomp> decomp = Teuchos::rcp(new Decomp(coords_x,coords_y,Teuchos::null,Teuchos::null,params));
  initialize(decomp,subset_size);
  assert(global_num_subsets_==num_pts);
}

Schema::Schema(Teuchos::ArrayRCP<scalar_t> coords_x,
  Teuchos::ArrayRCP<scalar_t> coords_y,
  const int_t subset_size,
  Teuchos::RCP<std::map<int_t,Conformal_Area_Def> > conformal_subset_defs,
  Teuchos::RCP<std::vector<int_t> > neighbor_ids,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  default_constructor_tasks(params);
  Teuchos::RCP<Decomp> decomp = Teuchos::rcp(new Decomp(coords_x,coords_y,neighbor_ids,Teuchos::null,params));
  initialize(decomp,subset_size,conformal_subset_defs);
}

void
Schema::rotate_def_image(){
  if(def_image_rotation_!=ZERO_DEGREES){
    for(size_t i=0;i<def_imgs_.size();++i)
      def_imgs_[i] = def_imgs_[i]->apply_rotation(def_image_rotation_);
  }
}

// Note: this is meant to be called at the beginning of set_def_image, and not really anywhere else
void
Schema::swap_def_prev_images(){
  DEBUG_MSG("Schema::swap_def_prev_images() called");
  // update the previous image storage points
  assert(def_imgs_.size()==prev_imgs_.size());
  for(size_t i=0;i<def_imgs_.size();++i){
    if(def_imgs_[i]==Teuchos::null) continue;
    // ensure that the prev img storage is allocated and the right size
    if(prev_imgs_[i]==Teuchos::null||prev_imgs_[i]->width()!=def_imgs_[i]->width()||prev_imgs_[i]->height()!=def_imgs_[i]->height())
      prev_imgs_[i] = Teuchos::rcp(new DICe::Image(def_imgs_[i]));
    // swap the reference counted pointers on the image memory storage
    Teuchos::RCP<Image> hold_rcp = prev_imgs_[i]; // increase the reference count so the memory doesn't get deallocated
    prev_imgs_[i] = def_imgs_[i];
    def_imgs_[i] = hold_rcp;
    DEBUG_MSG("Schema::swap_def_prev_images() prev and def images " << i << " were swapped");
  }
}

void
Schema::set_def_image(const std::string & defName){
  DEBUG_MSG("Schema::set_def_image(): setting the deformed image using file name " << defName);
//  swap_def_prev_images();
  assert(def_imgs_.size()>0);
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::compute_image_gradients,compute_def_gradients_);
  imgParams->set(DICe::gauss_filter_images,gauss_filter_images_);
  imgParams->set(DICe::gauss_filter_mask_size,gauss_filter_mask_size_);
  imgParams->set(DICe::gradient_method,gradient_method_);
  imgParams->set(DICe::filter_failed_cine_pixels,filter_failed_cine_pixels_);
  imgParams->set(DICe::convert_cine_to_8_bit,convert_cine_to_8_bit_);
  imgParams->set(DICe::buffer_persistence_guaranteed,true);
  if(init_params_!=Teuchos::null){
    if(init_params_->isSublist(undistort_images)){
      imgParams->set(undistort_images,init_params_->sublist(undistort_images));
    }
  }
  const bool has_motion_window = motion_window_params_->size()>0;
  DEBUG_MSG("Schema::set_def_image(): has motion window " << has_motion_window);

  int_t w = 0, h = 0;
  utils::read_image_dimensions(defName.c_str(),w,h);
  bool force_reallocation = false;

  int_t sub_width = 0, sub_height = 0;
  int_t offset_x = 0, offset_y = 0;
  int_t end_x = 0, end_y = 0;
  if(has_extents_){
    const int_t buffer = 100; // if the extents are within 100 pixels of the image boundary use the whole image
    offset_x = def_extents_[0] > buffer && def_extents_[0] < w - buffer ? def_extents_[0] : 0;
    offset_y = def_extents_[2] > buffer && def_extents_[2] < h - buffer ? def_extents_[2] : 0;
    end_x = def_extents_[1] > buffer && def_extents_[1] < w - buffer ? def_extents_[1] : w;
    end_y = def_extents_[3] > buffer && def_extents_[3] < h - buffer ? def_extents_[3] : h;
    sub_width = end_x - offset_x;
    sub_height = end_y - offset_y;
  }

  // if the image is from a cine file, load a large chunk of frames into the memory buffer
  const int_t num_buffer_frames = !has_extents_ || has_motion_window ? CINE_BUFFER_NUM_FRAMES : 1; // the extents update after each frame so can only put one frame in the buffer
  if((frame_id_-first_frame_id_)%num_buffer_frames==0){
    if(DICe::utils::image_file_type(defName.c_str())==CINE){
      std::string undecorated_cine_file = DICe::utils::video_file_name(defName.c_str());
      // test if this is a cross-correlation (if so, don't use a buffer since only one frame is needed)
      // if it is a cross-correlation setting the def image, the ref and def images will come from different files
      // hence the check below on the cine file name
      assert(ref_img_!=Teuchos::null);
      std::string undecorated_ref_cine_file = DICe::utils::video_file_name(ref_img_->file_name().c_str());
      if(undecorated_cine_file==undecorated_ref_cine_file){ // assumes ref image has already been set
        // check if the window params are already set and just the frame needs to be updated
        const hypercine::HyperCine::Bit_Depth_Conversion_Type conversion_type = convert_cine_to_8_bit_ ? hypercine::HyperCine::TO_8_BIT :
            hypercine::HyperCine::QUAD_10_TO_12;
        // get last frame of cine file:
        const int_t frame_count = frame_id_ + num_buffer_frames >= first_frame_id_ + num_frames_ ?
            first_frame_id_+num_frames_-frame_id_ : num_buffer_frames;
        DEBUG_MSG("Schema::set_def_image(): *** reading cine buffer, frame id " << frame_id_ << " count " << frame_count);
        if((frame_id_-first_frame_id_==0&&has_motion_window)||has_extents_){
          hypercine::HyperCine::HyperFrame hf(frame_id_,frame_count);
          if(has_motion_window){
            for(std::map<int_t,Motion_Window_Params>::iterator it=motion_window_params_->begin();it!=motion_window_params_->end();++it){
              hf.add_window(it->second.start_x_,
                it->second.end_x_-it->second.start_x_,
                it->second.start_y_,
                it->second.end_y_ - it->second.start_y_);
            }
          }else{
            hf.add_window(offset_x,sub_width,offset_y,sub_height);
          }
          DICe::utils::cine_file_read_buffer(undecorated_cine_file,conversion_type,hf);
        }else{
          DICe::utils::cine_file_read_buffer(undecorated_cine_file,conversion_type,frame_id_,frame_count);
        }
      }else{
        DEBUG_MSG("Schema::set_def_image(): skipping reading cine buffer since the ref and def images come from different cine files (likely cross-correlation)");
        imgParams->set(DICe::buffer_persistence_guaranteed,false);
      }
    }
  }

  for(size_t id=0;id<def_imgs_.size();++id){
    if(has_extents_||has_motion_window){
      if(has_motion_window){
        bool found_id = false;
        for(std::map<int_t,Motion_Window_Params>::iterator it=motion_window_params_->begin();it!=motion_window_params_->end();++it){
          if(it->second.sub_image_id_==(int_t)id){
            offset_x = it->second.start_x_;
            offset_y = it->second.start_y_;
            end_x = it->second.end_x_;
            end_y = it->second.end_y_;
            sub_width = end_x - offset_x;
            sub_height = end_y - offset_y;
            found_id = true;
            break;
          }
        }
        if(!found_id){ // no motion window exists for this sub image id
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
            "No motion window found for this sub image id, if motion windows are used, one must be set for each subset");
        }
      }
      imgParams->set(DICe::subimage_width,sub_width);
      imgParams->set(DICe::subimage_height,sub_height);
      imgParams->set(DICe::subimage_offset_x,offset_x);
      imgParams->set(DICe::subimage_offset_y,offset_y);
      DEBUG_MSG("Setting the deformed image using extents x: " << offset_x << " to " << end_x <<
        " y: " << offset_y << " to " << end_y << " width " << sub_width << " height " << sub_height);
      if(def_imgs_[id]!=Teuchos::null)
        if(def_imgs_[id]->width()!=sub_width||def_imgs_[id]->height()!=sub_height)
          force_reallocation = true;
    }else{
      if(def_imgs_[id]!=Teuchos::null) // detect if the overall image dimensions changed
        if(def_imgs_[id]->width()!=w || def_imgs_[id]->height()!=h)
          force_reallocation = true;
    }
    // see if the image has already been allocated:
    if(def_imgs_[id]==Teuchos::null||force_reallocation)
      def_imgs_[id] = Teuchos::rcp( new Image(defName.c_str(),imgParams));
    else
      def_imgs_[id]->update(defName.c_str(),imgParams);
    if(def_image_rotation_!=ZERO_DEGREES){
      def_imgs_[id] = def_imgs_[id]->apply_rotation(def_image_rotation_);
    }
    def_imgs_[id]->set_file_name(defName);
  }
}

void
Schema::set_def_image(Teuchos::RCP<Image> img,
  const int_t id){
  DEBUG_MSG("Schema::set_def_image(): resetting the deformed image (using an Image object) for sub image id " << id);
//  swap_def_prev_images();
  assert(def_imgs_.size()>0);
  assert(id<(int_t)def_imgs_.size());
  def_imgs_[id] = img;
  if(gauss_filter_images_&&!def_imgs_[id]->has_gauss_filter()){ // the filter may have alread been applied to the image
      def_imgs_[id]->gauss_filter(gauss_filter_mask_size_);
  }
  if(compute_def_gradients_&&!def_imgs_[id]->has_gradients()){
    def_imgs_[id]->compute_gradients();
  }
  if(def_image_rotation_!=ZERO_DEGREES){
    Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
    imgParams->set(DICe::compute_image_gradients,true); // automatically compute the gradients
    imgParams->set(DICe::gradient_method,gradient_method_);
    def_imgs_[id] = def_imgs_[id]->apply_rotation(def_image_rotation_,imgParams);
  }
}

void
Schema::set_def_image(const int_t img_width,
  const int_t img_height,
  const Teuchos::ArrayRCP<storage_t> defRCP,
  const int_t id){
  DEBUG_MSG("Schema::set_def_image(): setting the deformed image using an array of intensity values");
//  swap_def_prev_images();
  assert(def_imgs_.size()>0);
  assert(id<(int_t)def_imgs_.size());
  TEUCHOS_TEST_FOR_EXCEPTION(img_width<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(img_height<=0,std::runtime_error,"");
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::compute_image_gradients,compute_ref_gradients_);
  imgParams->set(DICe::gradient_method,gradient_method_);
  imgParams->set(DICe::gauss_filter_mask_size,gauss_filter_mask_size_);
  imgParams->set(DICe::gradient_method,gradient_method_);
  imgParams->set(DICe::filter_failed_cine_pixels,filter_failed_cine_pixels_);
  def_imgs_[id] = Teuchos::rcp( new Image(img_width,img_height,defRCP,imgParams));
  if(def_image_rotation_!=ZERO_DEGREES){
    def_imgs_[id] = def_imgs_[id]->apply_rotation(def_image_rotation_);
  }
}

void
Schema::set_ref_image(const std::string & refName){
  DEBUG_MSG("Schema: setting the reference image to " << refName);
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::compute_image_gradients,compute_ref_gradients_); // automatically compute the gradients if the ref image is changed
  imgParams->set(DICe::gauss_filter_images,gauss_filter_images_);
  imgParams->set(DICe::gauss_filter_mask_size,gauss_filter_mask_size_);
  imgParams->set(DICe::gradient_method,gradient_method_);
  imgParams->set(DICe::compute_laplacian_image,compute_laplacian_image_);
  imgParams->set(DICe::filter_failed_cine_pixels,filter_failed_cine_pixels_);
  imgParams->set(DICe::convert_cine_to_8_bit,convert_cine_to_8_bit_);
  if(init_params_!=Teuchos::null){
    if(init_params_->isSublist(undistort_images)){
      imgParams->set(undistort_images,init_params_->sublist(undistort_images));
    }
  }
  if(has_extents_){
    utils::read_image_dimensions(refName.c_str(),full_ref_img_width_,full_ref_img_height_);
    const int_t buffer = 100; // if the extents are within 100 pixels of the image boundary use the whole image
    const int_t offset_x = ref_extents_[0] > buffer && ref_extents_[0] < full_ref_img_width_ - buffer ? ref_extents_[0] : 0;
    const int_t offset_y = ref_extents_[2] > buffer && ref_extents_[2] < full_ref_img_height_ - buffer ? ref_extents_[2] : 0;
    const int_t end_x = ref_extents_[1] > buffer && ref_extents_[1] < full_ref_img_width_ - buffer ? ref_extents_[1] : full_ref_img_width_;
    const int_t end_y = ref_extents_[3] > buffer && ref_extents_[3] < full_ref_img_height_ - buffer ? ref_extents_[3] : full_ref_img_height_;
    const int_t sub_width = end_x - offset_x;
    const int_t sub_height = end_y - offset_y;
    imgParams->set(DICe::subimage_width,sub_width);
    imgParams->set(DICe::subimage_height,sub_height);
    imgParams->set(DICe::subimage_offset_x,offset_x);
    imgParams->set(DICe::subimage_offset_y,offset_y);
    DEBUG_MSG("Setting the reference image using extents x: " << offset_x << " to " << end_x << " y: " << offset_y << " to " << end_y);
    ref_img_ = Teuchos::rcp( new Image(refName.c_str(),imgParams));
  }
  else
    ref_img_ = Teuchos::rcp( new Image(refName.c_str(),imgParams));
  if(ref_image_rotation_!=ZERO_DEGREES){
    ref_img_ = ref_img_->apply_rotation(ref_image_rotation_,imgParams);
  }
  if(prev_imgs_[0]==Teuchos::null){
    prev_imgs_[0] = Teuchos::rcp( new Image(refName.c_str(),imgParams));
    if(ref_image_rotation_!=ZERO_DEGREES){
      prev_imgs_[0] = prev_imgs_[0]->apply_rotation(ref_image_rotation_,imgParams);
    }
  }// end prev img is null
}

void
Schema::set_ref_image(const int_t img_width,
  const int_t img_height,
  const Teuchos::ArrayRCP<storage_t> refRCP){
  DEBUG_MSG("Schema: setting the reference image using width, height, and intensity array");
  TEUCHOS_TEST_FOR_EXCEPTION(img_width<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(img_height<=0,std::runtime_error,"");
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::compute_image_gradients,compute_ref_gradients_);
  imgParams->set(DICe::gauss_filter_images,gauss_filter_images_);
  imgParams->set(DICe::gauss_filter_mask_size,gauss_filter_mask_size_);
  imgParams->set(DICe::gradient_method,gradient_method_);
  imgParams->set(DICe::filter_failed_cine_pixels,filter_failed_cine_pixels_);
  ref_img_ = Teuchos::rcp( new Image(img_width,img_height,refRCP,imgParams));
  if(ref_image_rotation_!=ZERO_DEGREES){
    ref_img_ = ref_img_->apply_rotation(ref_image_rotation_,imgParams);
  }
  if(prev_imgs_[0]==Teuchos::null){
    prev_imgs_[0] = Teuchos::rcp(new Image(ref_img_,imgParams));//Teuchos::rcp( new Image(img_width,img_height,refRCP,imgParams));
    // dont apply the rotation because the pointer is set to the ref image which has already been rotated
//    if(ref_image_rotation_!=ZERO_DEGREES){
//      prev_imgs_[0] = prev_imgs_[0]->apply_rotation(ref_image_rotation_,imgParams);
//    }
  }
}

void
Schema::set_ref_image(Teuchos::RCP<Image> img){
  if(ref_img_==Teuchos::null)
    DEBUG_MSG("Schema::set_ref_image() setting the reference image to " << img->file_name());
  else
    DEBUG_MSG("Schema::set_ref_image() resetting the reference image from " << ref_img_->file_name() << " to " << img->file_name());
  ref_img_ = Teuchos::rcp( new Image(img)); // always reallocate the ref image as a deep copy
  //ref_img_ = img;
  if(gauss_filter_images_){
    if(!ref_img_->has_gauss_filter()) // the filter may have alread been applied to the image
      ref_img_->gauss_filter(gauss_filter_mask_size_);
  }
  if(compute_ref_gradients_&&!ref_img_->has_gradients()){
    ref_img_->compute_gradients();
  }
  if(ref_image_rotation_!=ZERO_DEGREES){
    Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
    imgParams->set(DICe::compute_image_gradients,true); // automatically compute the gradients for rotations
    imgParams->set(DICe::gradient_method,gradient_method_);
    ref_img_ = ref_img_->apply_rotation(ref_image_rotation_,imgParams);
  }
  if(prev_imgs_[0]==Teuchos::null){
    prev_imgs_[0] = Teuchos::rcp(new DICe::Image(ref_img_));
  }
}

void
Schema::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  global_num_subsets_ = 0;
  local_num_subsets_ = 0;
  subset_dim_ = -1;
  step_size_x_ = -1;
  step_size_y_ = -1;
  frame_id_ = 0;
  frame_skip_ = 1;
  first_frame_id_ = 0;
  num_frames_ = -1;
  has_output_spec_ = false;
  is_initialized_ = false;
  analysis_type_ = LOCAL_DIC;
  has_post_processor_ = false;
  normalize_gamma_with_active_pixels_ = false;
  gauss_filter_images_ = false;
  gauss_filter_mask_size_ = 7;
  init_params_ = params==Teuchos::null ? Teuchos::rcp(new Teuchos::ParameterList()):
    Teuchos::rcp(new Teuchos::ParameterList(*params));
  comm_ = Teuchos::rcp(new MultiField_Comm());
  path_file_names_ = Teuchos::rcp(new std::map<int_t,std::string>());
  optical_flow_flags_ = Teuchos::rcp(new std::map<int_t,bool>());
  skip_solve_flags_ = Teuchos::rcp(new std::map<int_t,std::vector<int_t> >());
  motion_window_params_ = Teuchos::rcp(new std::map<int_t,Motion_Window_Params>());
  initial_gamma_threshold_ = -1.0;
  final_gamma_threshold_ = -1.0;
  path_distance_threshold_ = -1.0;
  stat_container_ = Teuchos::rcp(new Stat_Container());
  use_incremental_formulation_ = false;
  use_nonlinear_projection_ = false;
  read_full_images_ = false;
  sort_txt_output_ = false;
  threshold_block_size_ = -1;
  set_params(params);
  prev_imgs_.push_back(Teuchos::null);
  def_imgs_.push_back(Teuchos::null);
  has_extents_ = false;
  ref_extents_.resize(4,-1);
  def_extents_.resize(4,-1);
  full_ref_img_width_ = -1;
  full_ref_img_height_ = -1;
}

void
Schema::set_params(const std::string & params_file_name){
  // create a parameter list from the selected file
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> paramsPtr(params.get());
  Teuchos::updateParametersFromXmlFile(params_file_name,paramsPtr);
  set_params(params);
}

void
Schema::set_params(const Teuchos::RCP<Teuchos::ParameterList> & params){

  const int_t proc_rank = comm_->get_rank();

  if(params!=Teuchos::null){
    if(params->get<bool>(DICe::use_global_dic,false)){
#ifdef DICE_ENABLE_GLOBAL
      analysis_type_=GLOBAL_DIC;
#else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, code was not compiled with global DIC enabled.");
#endif
    }
  }

  // start with the default params and add any that are specified by the input params
  Teuchos::RCP<Teuchos::ParameterList> diceParams = Teuchos::rcp( new Teuchos::ParameterList("Schema_Correlation_Parameters") );

  if(analysis_type_==GLOBAL_DIC){
    dice_default_params(diceParams.getRawPtr());
    if(proc_rank == 0) DEBUG_MSG("Initializing schema params with full-field default global parameters");
    // Overwrite any params that are specified by the params argument
    if(params!=Teuchos::null){
      // check that all the parameters are valid:
      // this should catch the case that the user misspelled one of the parameters:
      bool allParamsValid = true;
      for(Teuchos::ParameterList::ConstIterator it=params->begin();it!=params->end();++it){
        bool paramValid = false;
        for(int_t j=0;j<DICe::num_valid_global_correlation_params;++j){
          if(it->first==valid_global_correlation_params[j].name_){
            diceParams->setEntry(it->first,it->second); // overwrite the default value with argument param specified values
            paramValid = true;
          }
        }
        // catch post processor entries
        for(int_t j=0;j<DICe::num_valid_post_processor_params;++j){
          if(it->first==valid_post_processor_params[j]){
            diceParams->setEntry(it->first,it->second); // overwrite the default value with argument param specified values
            paramValid = true;
          }
        }
        if(!paramValid){
          allParamsValid = false;
          if(proc_rank == 0) std::cout << "Error: Invalid parameter: " << it->first << std::endl;
        }
      }
      if(!allParamsValid){
        if(proc_rank == 0) std::cout << "NOTE: valid parameters include: " << std::endl;
        for(int_t j=0;j<DICe::num_valid_global_correlation_params;++j){
          if(proc_rank == 0) std::cout << valid_global_correlation_params[j].name_ << std::endl;
        }
        for(int_t j=0;j<DICe::num_valid_post_processor_params;++j){
          if(proc_rank == 0) std::cout << valid_post_processor_params[j] << std::endl;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!allParamsValid,std::invalid_argument,"Invalid parameter");
    }
  }
  else if(analysis_type_==LOCAL_DIC){
    bool use_tracking_defaults = false;
    if(params!=Teuchos::null){
      use_tracking_defaults = params->get<bool>(DICe::use_tracking_default_params,false);
    }
    // First set all of the params to their defaults in case the user does not specify them:
    if(use_tracking_defaults){
      tracking_default_params(diceParams.getRawPtr());
      if(proc_rank == 0) DEBUG_MSG("Initializing schema params with tracking default parameters");
    }
    else{
      dice_default_params(diceParams.getRawPtr());
      if(proc_rank == 0) DEBUG_MSG("Initializing schema params with full-field default parameters");
    }
    // Overwrite any params that are specified by the params argument
    if(params!=Teuchos::null){
      // check that all the parameters are valid:
      // this should catch the case that the user misspelled one of the parameters:
      bool allParamsValid = true;
      for(Teuchos::ParameterList::ConstIterator it=params->begin();it!=params->end();++it){
        bool paramValid = false;
        for(int_t j=0;j<DICe::num_valid_correlation_params;++j){
          if(it->first==valid_correlation_params[j].name_){
            diceParams->setEntry(it->first,it->second); // overwrite the default value with argument param specified values
            paramValid = true;
          }
        }
        // catch post processor entries
        for(int_t j=0;j<DICe::num_valid_post_processor_params;++j){
          if(it->first==valid_post_processor_params[j]){
            diceParams->setEntry(it->first,it->second); // overwrite the default value with argument param specified values
            paramValid = true;
          }
        }
        // catch the output folder and output prefix sent to schema for exodus output
        if(it->first==output_folder||it->first==output_prefix){
          diceParams->setEntry(it->first,it->second); // overwrite the default value with argument param specified values
          paramValid = true;
        }
        // catch the camera system file sent to schema for camera-based shape functions like the rigid body one
        if(it->first==camera_system_file){
          diceParams->setEntry(it->first,it->second); // overwrite the default value with argument param specified values
          paramValid = true;
        }
        if(!paramValid){
          allParamsValid = false;
          if(proc_rank == 0) std::cout << "Error: Invalid parameter: " << it->first << std::endl;
        }
      }
      if(!allParamsValid){
        if(proc_rank == 0) std::cout << "NOTE: valid parameters include: " << std::endl;
        for(int_t j=0;j<DICe::num_valid_correlation_params;++j){
          if(proc_rank == 0) std::cout << valid_correlation_params[j].name_ << std::endl;
        }
        for(int_t j=0;j<DICe::num_valid_post_processor_params;++j){
          if(proc_rank == 0) std::cout << valid_post_processor_params[j] << std::endl;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!allParamsValid,std::invalid_argument,"Invalid parameter");
    }
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized analysis_type");
  }
#ifdef DICE_DEBUG_MSG
  if(proc_rank == 0) {
    std::cout << "Full set of correlation parameters: " << std::endl;
    diceParams->print(std::cout);
  }
#endif

  initial_condition_file_ = diceParams->get<std::string>(DICe::initial_condition_file,"");
  use_incremental_formulation_ = diceParams->get<bool>(DICe::use_incremental_formulation,false);
  use_nonlinear_projection_ = diceParams->get<bool>(DICe::use_nonlinear_projection,false);
  read_full_images_ = diceParams->get<bool>(DICe::read_full_images,false);
  sort_txt_output_ = diceParams->get<bool>(DICe::sort_txt_output,false);
  gauss_filter_images_ = diceParams->get<bool>(DICe::gauss_filter_images,false);
  filter_failed_cine_pixels_ = diceParams->get<bool>(DICe::filter_failed_cine_pixels,false);
  convert_cine_to_8_bit_ = diceParams->get<bool>(DICe::convert_cine_to_8_bit,true);
  gauss_filter_mask_size_ = diceParams->get<int_t>(DICe::gauss_filter_mask_size,7);
  compute_ref_gradients_ = diceParams->get<bool>(DICe::compute_ref_gradients,true);
  compute_def_gradients_ = diceParams->get<bool>(DICe::compute_def_gradients,false);
  compute_laplacian_image_ = diceParams->get<bool>(DICe::compute_laplacian_image,false);
  if(diceParams->get<bool>(DICe::compute_image_gradients,false)) { // this flag turns them both on
    compute_ref_gradients_ = true;
    compute_def_gradients_ = true;
  }
//  if(use_incremental_formulation_){ // force gradient calcs to be on for incremental
//    compute_ref_gradients_ = true;
//    compute_def_gradients_ = true;
//  }
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::projection_method),std::runtime_error,"");
  projection_method_ = diceParams->get<Projection_Method>(DICe::projection_method);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::interpolation_method),std::runtime_error,"");
  interpolation_method_ = diceParams->get<Interpolation_Method>(DICe::interpolation_method);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::gradient_method),std::runtime_error,"");
  gradient_method_ = diceParams->get<Gradient_Method>(DICe::gradient_method);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::max_evolution_iterations),std::runtime_error,"");
  max_evolution_iterations_ = diceParams->get<int_t>(DICe::max_evolution_iterations);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::max_solver_iterations_fast),std::runtime_error,"");
  max_solver_iterations_fast_ = diceParams->get<int_t>(DICe::max_solver_iterations_fast);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::fast_solver_tolerance),std::runtime_error,"");
  fast_solver_tolerance_ = diceParams->get<double>(DICe::fast_solver_tolerance);
  // make sure image gradients are on at least for the reference image for any gradient based optimization routine
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::optimization_method),std::runtime_error,"");
  optimization_method_ = diceParams->get<Optimization_Method>(DICe::optimization_method);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::correlation_routine),std::runtime_error,"");
  correlation_routine_ = diceParams->get<Correlation_Routine>(DICe::correlation_routine);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::initialization_method),std::runtime_error,"");
  initialization_method_ = diceParams->get<Initialization_Method>(DICe::initialization_method);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::cross_initialization_method),std::runtime_error,"");
  cross_initialization_method_ = diceParams->get<Initialization_Method>(DICe::cross_initialization_method);
  TEUCHOS_TEST_FOR_EXCEPTION(use_nonlinear_projection_&&cross_initialization_method_==USE_SPACE_FILLING_ITERATIONS,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::max_solver_iterations_robust),std::runtime_error,"");
  max_solver_iterations_robust_ = diceParams->get<int_t>(DICe::max_solver_iterations_robust);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::robust_solver_tolerance),std::runtime_error,"");
  robust_solver_tolerance_ = diceParams->get<double>(DICe::robust_solver_tolerance);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::skip_solve_gamma_threshold),std::runtime_error,"");
  skip_solve_gamma_threshold_ = diceParams->get<double>(DICe::skip_solve_gamma_threshold);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::skip_all_solves),std::runtime_error,"");
  skip_all_solves_ = diceParams->get<bool>(DICe::skip_all_solves);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::initial_gamma_threshold),std::runtime_error,"");
  initial_gamma_threshold_ = diceParams->get<double>(DICe::initial_gamma_threshold);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::final_gamma_threshold),std::runtime_error,"");
  final_gamma_threshold_ = diceParams->get<double>(DICe::final_gamma_threshold);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::path_distance_threshold),std::runtime_error,"");
  path_distance_threshold_ = diceParams->get<double>(DICe::path_distance_threshold);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::disp_jump_tol),std::runtime_error,"");
  disp_jump_tol_ = diceParams->get<double>(DICe::disp_jump_tol);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::theta_jump_tol),std::runtime_error,"");
  theta_jump_tol_ = diceParams->get<double>(DICe::theta_jump_tol);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::robust_delta_disp),std::runtime_error,"");
  robust_delta_disp_ = diceParams->get<double>(DICe::robust_delta_disp);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::robust_delta_theta),std::runtime_error,"");
  robust_delta_theta_ = diceParams->get<double>(DICe::robust_delta_theta);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::enable_translation),std::runtime_error,"");
  enable_translation_ = diceParams->get<bool>(DICe::enable_translation);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::enable_rotation),std::runtime_error,"");
  enable_rotation_ = diceParams->get<bool>(DICe::enable_rotation);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::enable_normal_strain),std::runtime_error,"");
  enable_normal_strain_ = diceParams->get<bool>(DICe::enable_normal_strain);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::enable_shear_strain),std::runtime_error,"");
  enable_shear_strain_ = diceParams->get<bool>(DICe::enable_shear_strain);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::shape_function_type),std::runtime_error,"");
  shape_function_type_ = diceParams->get<Shape_Function_Type>(DICe::shape_function_type);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::output_deformed_subset_images),std::runtime_error,"");
  output_deformed_subset_images_ = diceParams->get<bool>(DICe::output_deformed_subset_images);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::output_deformed_subset_intensity_images),std::runtime_error,"");
  output_deformed_subset_intensity_images_ = diceParams->get<bool>(DICe::output_deformed_subset_intensity_images);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::output_evolved_subset_images),std::runtime_error,"");
  output_evolved_subset_images_ = diceParams->get<bool>(DICe::output_evolved_subset_images);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::use_subset_evolution),std::runtime_error,"");
  use_subset_evolution_ = diceParams->get<bool>(DICe::use_subset_evolution);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::override_force_simplex),std::runtime_error,"");
  override_force_simplex_ = diceParams->get<bool>(DICe::override_force_simplex);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::pixel_integration_order),std::runtime_error,"");
  pixel_integration_order_ = diceParams->get<int_t>(DICe::pixel_integration_order);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::obstruction_skin_factor),std::runtime_error,"");
  obstruction_skin_factor_ = diceParams->get<double>(DICe::obstruction_skin_factor);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::levenberg_marquardt_regularization_factor),std::runtime_error,"");
  levenberg_marquardt_regularization_factor_ = diceParams->get<double>(DICe::levenberg_marquardt_regularization_factor);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::momentum_factor),std::runtime_error,"");
  momentum_factor_ = diceParams->get<double>(DICe::momentum_factor);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::output_beta),std::runtime_error,"");
  output_beta_ = diceParams->get<bool>(DICe::output_beta);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::write_exodus_output),std::runtime_error,"");
  write_exodus_output_ = diceParams->get<bool>(DICe::write_exodus_output);
  threshold_block_size_ = diceParams->get<int>(DICe::threshold_block_size,-1);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::use_search_initialization_for_failed_steps),std::runtime_error,"");
  use_search_initialization_for_failed_steps_ = diceParams->get<bool>(DICe::use_search_initialization_for_failed_steps);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::normalize_gamma_with_active_pixels),std::runtime_error,"");
  normalize_gamma_with_active_pixels_ = diceParams->get<bool>(DICe::normalize_gamma_with_active_pixels);
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::rotate_ref_image_90),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::rotate_ref_image_180),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::rotate_ref_image_270),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::rotate_def_image_90),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::rotate_def_image_180),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!diceParams->isParameter(DICe::rotate_def_image_270),std::runtime_error,"");
  // last one read wins here:
  ref_image_rotation_ = ZERO_DEGREES;
  def_image_rotation_ = ZERO_DEGREES;
  if(diceParams->get<bool>(DICe::rotate_ref_image_90)) ref_image_rotation_ = NINTY_DEGREES;
  if(diceParams->get<bool>(DICe::rotate_ref_image_180)) ref_image_rotation_ = ONE_HUNDRED_EIGHTY_DEGREES;
  if(diceParams->get<bool>(DICe::rotate_ref_image_270)) ref_image_rotation_ = TWO_HUNDRED_SEVENTY_DEGREES;
  if(diceParams->get<bool>(DICe::rotate_def_image_90)) def_image_rotation_ = NINTY_DEGREES;
  if(diceParams->get<bool>(DICe::rotate_def_image_180)) def_image_rotation_ = ONE_HUNDRED_EIGHTY_DEGREES;
  if(diceParams->get<bool>(DICe::rotate_def_image_270)) def_image_rotation_ = TWO_HUNDRED_SEVENTY_DEGREES;
  if(normalize_gamma_with_active_pixels_)
    DEBUG_MSG("Gamma values will be normalized by the number of active pixels.");
  if(analysis_type_==GLOBAL_DIC){
    compute_ref_gradients_ = true;
  }
  else{
    if(optimization_method_!=DICe::SIMPLEX) {
      compute_ref_gradients_ = true;
    }
  }
  // create all the necessary post processors
  // create any of the post processors that may have been requested
  if(diceParams->isParameter(DICe::post_process_vsg_strain)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_vsg_strain);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    if(!ppParams->isParameter(displacement_x_field_name)&&analysis_type_==LOCAL_DIC){
      ppParams->set<std::string>(displacement_x_field_name,SUBSET_DISPLACEMENT_X_FS.get_name_label());
      ppParams->set<std::string>(displacement_y_field_name,SUBSET_DISPLACEMENT_Y_FS.get_name_label());
    }
    if(!ppParams->isParameter(coordinates_x_field_name)&&analysis_type_==LOCAL_DIC){
      ppParams->set<std::string>(coordinates_x_field_name,SUBSET_COORDINATES_X_FS.get_name_label());
      ppParams->set<std::string>(coordinates_y_field_name,SUBSET_COORDINATES_Y_FS.get_name_label());
    }
    Teuchos::RCP<VSG_Strain_Post_Processor> vsg_ptr = Teuchos::rcp (new VSG_Strain_Post_Processor(ppParams));
    post_processors_.push_back(vsg_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_plotly_contour)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_plotly_contour);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    Teuchos::RCP<Plotly_Contour_Post_Processor> pc_ptr = Teuchos::rcp (new Plotly_Contour_Post_Processor(ppParams));
    post_processors_.push_back(pc_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_crack_locator)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_crack_locator);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    Teuchos::RCP<Crack_Locator_Post_Processor> pc_ptr = Teuchos::rcp (new Crack_Locator_Post_Processor(ppParams));
    post_processors_.push_back(pc_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_nlvc_strain)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_nlvc_strain);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    if(!ppParams->isParameter(displacement_x_field_name)&&analysis_type_==LOCAL_DIC){
      ppParams->set<std::string>(displacement_x_field_name,SUBSET_DISPLACEMENT_X_FS.get_name_label());
      ppParams->set<std::string>(displacement_y_field_name,SUBSET_DISPLACEMENT_Y_FS.get_name_label());
    }
    if(!ppParams->isParameter(coordinates_x_field_name)&&analysis_type_==LOCAL_DIC){
      ppParams->set<std::string>(coordinates_x_field_name,SUBSET_COORDINATES_X_FS.get_name_label());
      ppParams->set<std::string>(coordinates_y_field_name,SUBSET_COORDINATES_Y_FS.get_name_label());
    }
    Teuchos::RCP<NLVC_Strain_Post_Processor> nlvc_ptr = Teuchos::rcp (new NLVC_Strain_Post_Processor(ppParams));
    post_processors_.push_back(nlvc_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_altitude)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_altitude);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    Teuchos::RCP<Altitude_Post_Processor> alt_ptr = Teuchos::rcp (new Altitude_Post_Processor(ppParams));
    post_processors_.push_back(alt_ptr);
  }
  // automatically add the uncertainty post processor if not global:
  if(analysis_type_!=GLOBAL_DIC){
    Teuchos::RCP<Uncertainty_Post_Processor> uncertainty_ptr = Teuchos::rcp(new Uncertainty_Post_Processor(Teuchos::null));
    post_processors_.push_back(uncertainty_ptr);
  }
  // check for live_plot file and add them to the end of the PP list:
  std::fstream livePlotDataFile("live_plot.dat", std::ios_base::in);
  if(livePlotDataFile.good()){
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    if(analysis_type_==LOCAL_DIC){
      ppParams->set<std::string>(displacement_x_field_name,SUBSET_DISPLACEMENT_X_FS.get_name_label());
      ppParams->set<std::string>(displacement_y_field_name,SUBSET_DISPLACEMENT_Y_FS.get_name_label());
      ppParams->set<std::string>(coordinates_x_field_name,SUBSET_COORDINATES_X_FS.get_name_label());
      ppParams->set<std::string>(coordinates_y_field_name,SUBSET_COORDINATES_Y_FS.get_name_label());
    }
    Teuchos::RCP<Live_Plot_Post_Processor> lp_ptr = Teuchos::rcp (new Live_Plot_Post_Processor(ppParams));
    post_processors_.push_back(lp_ptr);
  }
  if(post_processors_.size()>0) has_post_processor_ = true;

  Teuchos::RCP<Teuchos::ParameterList> outputParams;
  if(diceParams->isParameter(DICe::output_spec)){
    if(proc_rank == 0) DEBUG_MSG("Output spec was provided by user");
    // Strip output params sublist out of params
    Teuchos::ParameterList output_sublist = diceParams->sublist(DICe::output_spec);
    outputParams = Teuchos::rcp( new Teuchos::ParameterList());
    // iterate the sublist and add the params to the output params:
    for(Teuchos::ParameterList::ConstIterator it=output_sublist.begin();it!=output_sublist.end();++it){
      outputParams->setEntry(it->first,it->second);
    }
  }
  // create the output spec:
  const std::string delimiter = diceParams->get<std::string>(DICe::output_delimiter," ");
  const bool omit_row_id = diceParams->get<bool>(DICe::omit_output_row_id,false);
  output_spec_ = Teuchos::rcp(new DICe::Output_Spec(this,omit_row_id,outputParams,delimiter));
  has_output_spec_ = true;

  if(diceParams->isParameter(DICe::exact_solution_constant_value_x)||diceParams->isParameter(DICe::exact_solution_constant_value_y)){
    TEUCHOS_TEST_FOR_EXCEPTION(diceParams->get<bool>(DICe::estimate_resolution_error,false),std::runtime_error,"");
    const scalar_t value_x = diceParams->get<double>(DICe::exact_solution_constant_value_x,0.0);
    const scalar_t value_y = diceParams->get<double>(DICe::exact_solution_constant_value_y,0.0);
    compute_laplacian_image_ = true;
    image_deformer_ = Teuchos::rcp(new Image_Deformer(value_x,value_y,Image_Deformer::CONSTANT_VALUE));
  }
  if(diceParams->isParameter(DICe::exact_solution_dic_challenge_14)){
    TEUCHOS_TEST_FOR_EXCEPTION(diceParams->get<bool>(DICe::estimate_resolution_error,false),std::runtime_error,"");
    const scalar_t value = diceParams->get<double>(DICe::exact_solution_dic_challenge_14,0.0);
    compute_laplacian_image_ = true;
    image_deformer_ = Teuchos::rcp(new Image_Deformer(value,0.0,Image_Deformer::DIC_CHALLENGE_14));
  }
}

void
Schema::update_extents(const bool use_transformation_augmentation){
  // don't use image extents when conformal subsets are used
  if(conformal_subset_defs_!=Teuchos::null){
    if(conformal_subset_defs_->size()>0){
      has_extents_ = false;
      return;
    }
  }
  if(read_full_images_){
    has_extents_ = false;
    return;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(motion_window_params_->size()>0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(mesh_==Teuchos::null,std::runtime_error,"");
  Teuchos::RCP<MultiField> coords = mesh_->get_field(INITIAL_COORDINATES_FS);
  Teuchos::RCP<MultiField> disp;
  Teuchos::RCP<MultiField_Map> map = mesh_->get_vector_node_dist_map();
  Teuchos::RCP<MultiField> proj_aug_x;
  Teuchos::RCP<MultiField> proj_aug_y;
  if(use_nonlinear_projection_){
    proj_aug_x = mesh_->get_field(PROJECTION_AUG_X_FS);
    proj_aug_y = mesh_->get_field(PROJECTION_AUG_Y_FS);
  }
  Teuchos::RCP<MultiField> aug = Teuchos::rcp( new MultiField(map,1,true));
  if(analysis_type_ == LOCAL_DIC){
    const int_t spa_dim = 2;
    Teuchos::RCP<MultiField> disp_x = mesh_->get_field(SUBSET_DISPLACEMENT_X_FS);
    Teuchos::RCP<MultiField> disp_y = mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS);
    disp = Teuchos::rcp( new MultiField(map,1,true));
    for(int_t i=0;i<local_num_subsets_;++i){
      disp->local_value(i*spa_dim+0) = disp_x->local_value(i);
      disp->local_value(i*spa_dim+1) = disp_y->local_value(i);
      if(use_nonlinear_projection_){
        aug->local_value(i*spa_dim+0) = proj_aug_x->local_value(i);
        aug->local_value(i*spa_dim+1) = proj_aug_y->local_value(i);
      }
    }
  }else{
    disp = mesh_->get_field(DISPLACEMENT_FS);
  }
  Teuchos::RCP<MultiField> current_coords = Teuchos::rcp( new MultiField(map,1,true));
  current_coords->update(1.0,*coords,1.0);
  current_coords->update(1.0,*disp,1.0);
  if(use_transformation_augmentation&&use_nonlinear_projection_)
    current_coords->update(1.0,*aug,1.0);
  //std::cout << " COORDS " << std::endl;
  //coords->describe();

  scalar_t min_x_ref = std::numeric_limits<int_t>::max();
  scalar_t max_x_ref = 0.0;
  scalar_t min_y_ref = std::numeric_limits<int_t>::max();
  scalar_t max_y_ref = 0.0;
  scalar_t min_x_def = std::numeric_limits<int_t>::max();
  scalar_t max_x_def = 0.0;
  scalar_t min_y_def = std::numeric_limits<int_t>::max();
  scalar_t max_y_def = 0.0;

  const int_t num_pts = coords->get_map()->get_num_local_elements()/2;
  assert(num_pts>0);
  assert(current_coords->get_map()->get_num_local_elements()==coords->get_map()->get_num_local_elements());

  for(int_t i=0;i<num_pts;++i){
    if(coords->local_value(i*2+0) < min_x_ref) min_x_ref = coords->local_value(i*2+0);
    if(coords->local_value(i*2+0) > max_x_ref) max_x_ref = coords->local_value(i*2+0);
    if(coords->local_value(i*2+1) < min_y_ref) min_y_ref = coords->local_value(i*2+1);
    if(coords->local_value(i*2+1) > max_y_ref) max_y_ref = coords->local_value(i*2+1);
    if(current_coords->local_value(i*2+0) < min_x_def) min_x_def = current_coords->local_value(i*2+0);
    if(current_coords->local_value(i*2+0) > max_x_def) max_x_def = current_coords->local_value(i*2+0);
    if(current_coords->local_value(i*2+1) < min_y_def) min_y_def = current_coords->local_value(i*2+1);
    if(current_coords->local_value(i*2+1) > max_y_def) max_y_def = current_coords->local_value(i*2+1);
  }
  const int_t buffer = use_transformation_augmentation&&use_nonlinear_projection_ ? 500 : 100;
  min_x_ref -= buffer; min_y_ref -= buffer;
  max_x_ref += buffer; max_y_ref += buffer;
  min_x_ref = min_x_ref < buffer ? 0 : std::round(min_x_ref);
  min_y_ref = min_y_ref < buffer ? 0 : std::round(min_y_ref);
  min_x_def -= buffer; min_y_def -= buffer;
  max_x_def += buffer; max_y_def += buffer;
  min_x_def = min_x_def < buffer ? 0 : std::round(min_x_def);
  min_y_def = min_y_def < buffer ? 0 : std::round(min_y_def);
  has_extents_ = true;
  ref_extents_[0] = min_x_ref;
  ref_extents_[1] = max_x_ref;
  ref_extents_[2] = min_y_ref;
  ref_extents_[3] = max_y_ref;
  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] Setting REFERENCE domain extents:");
  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] x " << ref_extents_[0] << " to " << ref_extents_[1]);
  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] y " << ref_extents_[2] << " to " << ref_extents_[3]);
  def_extents_[0] = min_x_def;
  def_extents_[1] = max_x_def;
  def_extents_[2] = min_y_def;
  def_extents_[3] = max_y_def;
  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] Setting DEFORMED domain extents:");
  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] x " << def_extents_[0] << " to " << def_extents_[1]);
  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] y " << def_extents_[2] << " to " << def_extents_[3]);
  assert(min_x_ref<max_x_ref);
  assert(min_y_ref<max_y_ref);
  assert(min_x_def<max_x_def);
  assert(min_y_def<max_y_def);
}

void
Schema::initialize(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params){

  const std::string output_folder = input_params->get<std::string>(DICe::output_folder,"");
  const std::string output_prefix = input_params->get<std::string>(DICe::output_prefix,"DICe_solution");
  init_params_->set(DICe::output_prefix,output_prefix);
  init_params_->set(DICe::output_folder,output_folder);
  if(input_params->isParameter(DICe::camera_system_file)){
    const std::string camera_sys_file = input_params->get<std::string>(DICe::camera_system_file);
    init_params_->set(DICe::camera_system_file,camera_sys_file);
  }

  if(analysis_type_==GLOBAL_DIC){
    // create the computational mesh:
    //TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::mesh_size),std::runtime_error,
    //  "Error, missing required input parameter: mesh_size");
    if(input_params->isParameter(DICe::mesh_size)){
      const scalar_t mesh_size = input_params->get<double>(DICe::mesh_size);
      init_params_->set(DICe::mesh_size,(double)mesh_size); // pass the mesh size to the stored parameters for this schema (used by global method)
    }
    //TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_file),std::runtime_error,
    //  "Error, missing required input parameter: subset_file");
    if(input_params->isParameter(DICe::subset_file)){
      const std::string subset_file = input_params->get<std::string>(DICe::subset_file);
      init_params_->set(DICe::subset_file,subset_file);
    }
    if(input_params->isParameter(DICe::mesh_file)){
      const std::string mesh_file = input_params->get<std::string>(DICe::mesh_file);
      init_params_->set(DICe::mesh_file,mesh_file);
    }
    //init_params_->set(DICe::global_formulation,input_params->get<Global_Formulation>(DICe::global_formulation,HORN_SCHUNCK));
#ifdef DICE_ENABLE_GLOBAL
    global_algorithm_ = Teuchos::rcp(new DICe::global::Global_Algorithm(this,init_params_));
#endif
    // initialize the post processors
    for(size_t i=0;i<post_processors_.size();++i)
      post_processors_[i]->initialize(mesh_);
    is_initialized_ = true;
    return;
  }

  Teuchos::RCP<Decomp> decomp = Teuchos::rcp(new Decomp(input_params,correlation_params));
  const int_t proc_rank = comm_->get_rank();

  // if the subset locations are specified in an input file, read them in (else they will be defined later)
  Teuchos::RCP<DICe::Subset_File_Info> subset_info;
  int_t step_size = -1;
  int_t subset_size = -1;
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs;
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > blocking_subset_ids;
  Teuchos::RCP<std::set<int_t> > force_simplex;
  const bool has_subset_file = input_params->isParameter(DICe::subset_file);
  DICe::Subset_File_Info_Type subset_info_type = DICe::SUBSET_INFO;
  if(has_subset_file){
    std::string fileName = input_params->get<std::string>(DICe::subset_file);
    subset_info = decomp->subset_info();//sDICe::read_subset_file(fileName,decomp->image_width(),decomp->image_height());
    subset_info_type = subset_info->type;
  }
  if(!has_subset_file || subset_info_type==DICe::REGION_OF_INTEREST_INFO){
    TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::step_size),std::runtime_error,
      "Error, step size has not been specified");
    step_size = input_params->get<int_t>(DICe::step_size);
    DEBUG_MSG("Correlation point centroids were not specified by the user. \nThey will be evenly distrubed in the region"
        " of interest with separation (step_size) of " << step_size << " pixels.");
    TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_size),std::runtime_error,
      "Error, the subset size has not been specified"); // required for all square subsets case
    subset_size = input_params->get<int_t>(DICe::subset_size);
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(subset_info==Teuchos::null,std::runtime_error,"");
    conformal_area_defs = subset_info->conformal_area_defs;
    blocking_subset_ids = subset_info->id_sets_map;
    force_simplex = subset_info->force_simplex;
    if((int_t)subset_info->conformal_area_defs->size()<decomp->num_global_subsets()){
      // Only require this if not all subsets are conformal:
      TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_size),std::runtime_error,
        "Error, the subset size has not been specified");
      subset_size = input_params->get<int_t>(DICe::subset_size);
    }
  }
  set_step_size(step_size); // this is done just so the step_size appears in the output file header (it's not actually used)
  // set the blocking subset ids if they exist
  set_obstructing_subset_ids(blocking_subset_ids);
  // set the subsets that should force the simplex method
  set_force_simplex(force_simplex);
  // initialize the schema
  initialize(decomp,subset_size,conformal_area_defs);

  // set the seed value if they exist
  if(subset_info!=Teuchos::null){
    if(subset_info->path_file_names->size()>0){
      set_path_file_names(subset_info->path_file_names);
    }
    if(subset_info->skip_solve_flags->size()>0){
      set_skip_solve_flags(subset_info->skip_solve_flags);
#ifdef DICE_DEBUG_MSG
      std::cout << "[DICe_DEBUG]: Schema::initialize(): skip solve flags" << std::endl;
      std::string on = "ON";
      std::string off = "OFF";
      std::string state = "";
      std::map<int_t,std::vector<int_t> >::const_iterator it=skip_solve_flags_->begin();
      for(;it!=skip_solve_flags_->end();++it){
        bool skip_on = false;
        std::cout << "[DICe_DEBUG]: Schema::initialize(): subset " << it->first << " has the following flags" << std::endl;
        for(size_t id=0;id<it->second.size();++id){
          skip_on = !skip_on;
          state = skip_on ? on : off;
          std::cout << "[DICe_DEBUG]: Schema::initialize(): at frame " << it->second[id] << " skip solve is " << state << std::endl;
        }
      }
#endif
    }
    if(subset_info->optical_flow_flags->size()>0){
      set_optical_flow_flags(subset_info->optical_flow_flags);
    }
    if(subset_info->motion_window_params->size()>0){
      // make sure not running in parallel (motion window use_subset_id may be off processor) TODO fix this, ex

      set_motion_window_params(subset_info->motion_window_params);
      // change the def image storage to be a vector of motion windows rather than one large image
      def_imgs_.resize(subset_info->num_motion_windows);
      prev_imgs_.resize(subset_info->num_motion_windows);
      for(int_t i=0;i<subset_info->num_motion_windows;++i){
        def_imgs_[i] = Teuchos::null;
        prev_imgs_[i] = Teuchos::null;
      }
    }
    if(subset_info->seed_subset_ids->size()>0){
      //has_seed(true);
      TEUCHOS_TEST_FOR_EXCEPTION(subset_info->displacement_map->size()<=0,std::runtime_error,"");
      std::map<int_t,int_t>::iterator it=subset_info->seed_subset_ids->begin();
      for(;it!=subset_info->seed_subset_ids->end();++it){
        const int_t subset_id = it->first;
        if(subset_local_id(subset_id)<0) continue;
        const int_t roi_id = it->second;
        TEUCHOS_TEST_FOR_EXCEPTION(subset_info->displacement_map->find(roi_id)==subset_info->displacement_map->end(),
          std::runtime_error,"");
        global_field_value(subset_id,SUBSET_DISPLACEMENT_X_FS) = subset_info->displacement_map->find(roi_id)->second.first;
        global_field_value(subset_id,SUBSET_DISPLACEMENT_Y_FS) = subset_info->displacement_map->find(roi_id)->second.second;
        if(proc_rank==0) DEBUG_MSG("Seeding the displacement solution for subset " << subset_id << " with ux: " <<
          global_field_value(subset_id,SUBSET_DISPLACEMENT_X_FS) << " uy: " << global_field_value(subset_id,SUBSET_DISPLACEMENT_Y_FS));
        if(subset_info->normal_strain_map->find(roi_id)!=subset_info->normal_strain_map->end()){
          TEUCHOS_TEST_FOR_EXCEPTION(shape_function_type_!=DICe::AFFINE_SF,std::runtime_error,"Error, seeds can only be used with the affine shape function");
          global_field_value(subset_id,NORMAL_STRETCH_XX_FS) = subset_info->normal_strain_map->find(roi_id)->second.first;
          global_field_value(subset_id,NORMAL_STRETCH_YY_FS) = subset_info->normal_strain_map->find(roi_id)->second.second;
          if(proc_rank==0) DEBUG_MSG("Seeding the normal strain solution for subset " << subset_id << " with ex: " <<
            global_field_value(subset_id,NORMAL_STRETCH_XX_FS) << " ey: " << global_field_value(subset_id,NORMAL_STRETCH_YY_FS));
        }
        if(subset_info->shear_strain_map->find(roi_id)!=subset_info->shear_strain_map->end()){
          TEUCHOS_TEST_FOR_EXCEPTION(shape_function_type_!=DICe::AFFINE_SF,std::runtime_error,"Error, seeds can only be used with the affine shape function");
          global_field_value(subset_id,SHEAR_STRETCH_XY_FS) = subset_info->shear_strain_map->find(roi_id)->second;
          if(proc_rank==0) DEBUG_MSG("Seeding the shear strain solution for subset " << subset_id << " with gamma_xy: " <<
            global_field_value(subset_id,SHEAR_STRETCH_XY_FS));
        }
        if(subset_info->rotation_map->find(roi_id)!=subset_info->rotation_map->end()){
          global_field_value(subset_id,ROTATION_Z_FS) = subset_info->rotation_map->find(roi_id)->second;
          if(proc_rank==0) DEBUG_MSG("Seeding the rotation solution for subset " << subset_id << " with theta_z: " <<
            global_field_value(subset_id,ROTATION_Z_FS));
        }
      }
    }
  }
}

void
Schema::initialize(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
  const Teuchos::RCP<Schema> schema){
  const std::string output_folder = input_params->get<std::string>(DICe::output_folder,"");
  init_params_->set(DICe::output_folder,output_folder);
  std::string output_prefix = input_params->get<std::string>(DICe::output_prefix,"DICe_solution");
  output_prefix += "_stereo";
  init_params_->set(DICe::output_prefix,output_prefix);

  if(analysis_type_==GLOBAL_DIC){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, global stereo has not been implemented yet");
//    // create the computational mesh:
//    TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::mesh_size),std::runtime_error,
//      "Error, missing required input parameter: mesh_size");
//    const scalar_t mesh_size = input_params->get<double>(DICe::mesh_size);
//    init_params_->set(DICe::mesh_size,mesh_size); // pass the mesh size to the stored parameters for this schema (used by global method)
//    TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_file),std::runtime_error,
//      "Error, missing required input parameter: subset_file");
//    const std::string subset_file = input_params->get<std::string>(DICe::subset_file);
//    init_params_->set(DICe::subset_file,subset_file);
//    //init_params_->set(DICe::global_formulation,input_params->get<Global_Formulation>(DICe::global_formulation,HORN_SCHUNCK));
//#ifdef DICE_ENABLE_GLOBAL
//    global_algorithm_ = Teuchos::rcp(new DICe::global::Global_Algorithm(this,init_params_));
//#endif
//    return;
  }

  set_step_size(schema->step_size_x()); // this is done just so the step_size appears in the output file header (it's not actually used)
  // let the schema know how many images there are in the sequence:
  // initialize the schema
  // collect the overlap coordinates vector from the other schema because the local fields will only have this proc's
  // the size of the coords vector needs to be the global number of elements, but only the locally used elements need values
  TEUCHOS_TEST_FOR_EXCEPTION(schema->skip_solve_flags()->size()>0,std::runtime_error,"Error skip solves cannot be used in stereo");
  TEUCHOS_TEST_FOR_EXCEPTION(schema->motion_window_params()->size()>0,std::runtime_error,"Error motion windows cannot be used in stereo");
  Teuchos::RCP<MultiField> stereo_coords_x = schema->mesh()->get_overlap_field(STEREO_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> stereo_coords_y = schema->mesh()->get_overlap_field(STEREO_COORDINATES_Y_FS);
  global_num_subsets_ = schema->mesh()->get_scalar_node_dist_map()->get_num_global_elements();
  TEUCHOS_TEST_FOR_EXCEPTION(global_num_subsets_<=0,std::runtime_error,"");
  subset_dim_ = schema->subset_dim();
  this_proc_gid_order_ = schema->this_proc_gid_order();

  const int_t num_overlap_coords = schema->mesh()->get_scalar_node_overlap_map()->get_num_local_elements();
  Teuchos::ArrayRCP<scalar_t> overlap_coords_x(num_overlap_coords,0.0);
  Teuchos::ArrayRCP<scalar_t> overlap_coords_y(num_overlap_coords,0.0);
  Teuchos::ArrayRCP<int_t> node_map(num_overlap_coords,0);
  for(int_t i=0;i<num_overlap_coords;++i){
    overlap_coords_x[i] = std::round(stereo_coords_x->local_value(i));
    overlap_coords_y[i] = std::round(stereo_coords_y->local_value(i));
    node_map[i] = schema->mesh()->get_scalar_node_overlap_map()->get_global_element(i);
  }
  const int_t num_elem = schema->mesh()->get_scalar_node_dist_map()->get_num_local_elements();
  Teuchos::ArrayRCP<int_t> connectivity(num_elem,0);
  Teuchos::ArrayRCP<int_t> elem_map(num_elem,0);
  // note: this assumes the elements are contiguous in terms of ids and that the overlap ids
  // are in ascending order
  int_t overlap_offset = 0;
  for(int_t i=0;i<node_map.size();++i){
    if(schema->mesh()->get_scalar_node_dist_map()->get_global_element_list()[0] <= node_map[i]) break;
    overlap_offset++;
  }
  for(int_t i=0;i<num_elem;++i){
    connectivity[i] = overlap_offset + i + 1; // + 1 because exodus elem ids are 1-based
    elem_map[i] = schema->mesh()->get_scalar_node_dist_map()->get_global_element(i);
  }
  // filename for output
  std::stringstream exo_name;
  if(init_params_!=Teuchos::null)
    exo_name << init_params_->get<std::string>(DICe::output_prefix,"DICe_solution_stereo") << ".e";
  else
    exo_name << "DICe_solution_stereo.e";
  // dummy arrays
  std::vector<std::pair<int_t,int_t> > dirichlet_boundary_nodes;
  std::set<int_t> neumann_boundary_nodes;
  std::set<int_t> lagrange_boundary_nodes;

  mesh_ = DICe::mesh::create_point_or_tri_mesh(DICe::mesh::MESHLESS,
    overlap_coords_x,
    overlap_coords_y,
    connectivity,
    node_map,
    elem_map,
    dirichlet_boundary_nodes,
    neumann_boundary_nodes,
    lagrange_boundary_nodes,
    exo_name.str());

  conformal_subset_defs_ = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def>);

  // initialize the post processors
  for(size_t i=0;i<post_processors_.size();++i)
    post_processors_[i]->initialize(mesh_);

  is_initialized_ = true;

  TEUCHOS_TEST_FOR_EXCEPTION(mesh_==Teuchos::null,std::runtime_error,"Error: mesh should not be null here");
  local_num_subsets_ = mesh_->get_scalar_node_dist_map()->get_num_local_elements();
  create_mesh_fields(true);

  mesh_->get_field(NEIGHBOR_ID_FS)->update(1.0,*schema->mesh()->get_field(NEIGHBOR_ID_FS),0.0);

  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] right schema initialized");
}


void
Schema::initialize(Teuchos::RCP<Decomp> decomp,
  const int_t subset_size,
  Teuchos::RCP<std::map<int_t,Conformal_Area_Def> > conformal_subset_defs){
  if(is_initialized_){
    assert(global_num_subsets_>0);
    assert(local_num_subsets_>0);
    return;  // no need to initialize if already done
  }
  TEUCHOS_TEST_FOR_EXCEPTION(decomp==Teuchos::null,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(decomp->overlap_coords_x().size() <= 0,std::runtime_error,"Error, invalid x coordinates");
  global_num_subsets_ = decomp->num_global_subsets();
  subset_dim_ = subset_size;

  // create an evenly split map to start:
  this_proc_gid_order_ = decomp->this_proc_gid_order();

  // create an exodus mesh for output
  create_mesh(decomp);

  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] num local subsets: " << local_num_subsets_);
  // initialize the conformal subset map to avoid havng to check if its null always
  if(conformal_subset_defs==Teuchos::null)
    conformal_subset_defs_ = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def>);
  else
    conformal_subset_defs_ = conformal_subset_defs;

  TEUCHOS_TEST_FOR_EXCEPTION(global_num_subsets_<(int_t)conformal_subset_defs_->size(),std::runtime_error,
    "Error, data is not the right size, conformal_subset_defs_.size() is too large for the data array");
  // ensure that the ids in conformal subset defs are valid:
  std::map<int_t,Conformal_Area_Def>::iterator it = conformal_subset_defs_->begin();
  for( ;it!=conformal_subset_defs_->end();++it){
    assert(it->first >= 0);
    assert(it->first < global_num_subsets_);
  }
  // ensure that a subset size was specified if not all subsets are conformal:
  if(analysis_type_==LOCAL_DIC&&(int_t)conformal_subset_defs_->size()<global_num_subsets_){
    TEUCHOS_TEST_FOR_EXCEPTION(subset_size<=0,std::runtime_error,"");
  }

  // initialize the post processors
  for(size_t i=0;i<post_processors_.size();++i)
    post_processors_[i]->initialize(mesh_);

  is_initialized_ = true;

  if(decomp->neighbor_ids()!=Teuchos::null)
    for(int_t i=0;i<local_num_subsets_;++i){
      const int_t gid = subset_global_id(i);
      const int_t olid = mesh_->get_scalar_node_overlap_map()->get_local_element(gid);
      local_field_value(i,NEIGHBOR_ID_FS) = (*decomp->neighbor_ids())[olid];
    }

  DEBUG_MSG("[PROC " << mesh_->get_comm()->get_rank() << "] schema initialized");
}

void
Schema::create_mesh(Teuchos::RCP<Decomp> decomp){

  const int_t num_overlap_coords = decomp->id_decomp_overlap_map()->get_num_local_elements();
  const int_t num_coords = decomp->id_decomp_map()->get_num_local_elements();
  Teuchos::ArrayRCP<int_t> node_map(num_overlap_coords,0);
  for(int_t i=0;i<num_overlap_coords;++i){
    node_map[i] = decomp->id_decomp_overlap_map()->get_global_element(i);
  }

  // create a  DICe::mesh::Mesh that has a connectivity with only one node per elem
  //const int_t num_points = coords_x.size();
  // the subset ownership is dictated by the dist_map
  // the overlap map for is dictated by which neighbors are needed to access
  Teuchos::ArrayRCP<int_t> connectivity(num_coords,0);
  Teuchos::ArrayRCP<int_t> elem_map(num_coords,0);
  // note: this assumes the elements are contiguous in terms of ids and that the overlap ids
  // are in ascending order
  int_t overlap_offset = 0;
  for(int_t i=0;i<node_map.size();++i){
    if(decomp->id_decomp_map()->get_global_element_list()[0] <= node_map[i]) break;
    overlap_offset++;
  }
  for(int_t i=0;i<num_coords;++i){
    connectivity[i] = overlap_offset + i + 1; // + 1 because exodus elem ids are 1-based
    elem_map[i] = decomp->id_decomp_map()->get_global_element_list()[i];
  }
  // filename for output
  std::stringstream exo_name;
  if(init_params_!=Teuchos::null)
    exo_name << init_params_->get<std::string>(output_prefix,"DICe_solution") << ".e";
  else
    exo_name << "DICe_solution.e";

  // dummy arrays
  std::vector<std::pair<int_t,int_t> > dirichlet_boundary_nodes;
  std::set<int_t> neumann_boundary_nodes;
  std::set<int_t> lagrange_boundary_nodes;
  mesh_ = DICe::mesh::create_point_or_tri_mesh(DICe::mesh::MESHLESS,
    decomp->overlap_coords_x(),
    decomp->overlap_coords_y(),
    connectivity,
    node_map,
    elem_map,
    dirichlet_boundary_nodes,
    neumann_boundary_nodes,
    lagrange_boundary_nodes,
    exo_name.str());

  TEUCHOS_TEST_FOR_EXCEPTION(mesh_==Teuchos::null,std::runtime_error,"Error: mesh should not be null here");
  local_num_subsets_ = mesh_->get_scalar_node_dist_map()->get_num_local_elements();
  create_mesh_fields();
}

void
Schema::create_mesh_fields(const bool is_stereo){
  mesh_->create_field(field_enums::SUBSET_COORDINATES_X_FS);
  mesh_->create_field(field_enums::SUBSET_COORDINATES_Y_FS);
  mesh_->create_field(field_enums::SUBSET_DISPLACEMENT_X_FS);
  mesh_->create_field(field_enums::SUBSET_DISPLACEMENT_X_NM1_FS);
  mesh_->create_field(field_enums::SUBSET_DISPLACEMENT_Y_FS);
  mesh_->create_field(field_enums::SUBSET_DISPLACEMENT_Y_NM1_FS);
  mesh_->create_field(field_enums::ROTATION_Z_FS);
  mesh_->create_field(field_enums::ROTATION_Z_NM1_FS);
  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
  std::map<Field_Spec,size_t>::const_iterator it = shape_function->spec_map()->begin();
  const std::map<Field_Spec,size_t>::const_iterator it_end = shape_function->spec_map()->end();
  for(;it!=it_end;++it){
    mesh_->create_field(it->first);
    // creat the nm1 field as well
    DICe::field_enums::Field_Spec fs_nm1(it->first.get_field_type(),it->first.get_name(),it->first.get_rank(),DICe::field_enums::STATE_N_MINUS_ONE,false,true);
    mesh_->create_field(fs_nm1);
  }
  shape_function->reset_fields(this);
  mesh_->create_field(field_enums::SSSIG_FS);
  mesh_->create_field(field_enums::SIGMA_FS);
  mesh_->create_field(field_enums::GAMMA_FS);
  mesh_->create_field(field_enums::BETA_FS);
  mesh_->create_field(field_enums::OMEGA_FS);
  mesh_->create_field(field_enums::NOISE_LEVEL_FS);
  mesh_->create_field(field_enums::CONTRAST_LEVEL_FS);
  mesh_->create_field(field_enums::ACTIVE_PIXELS_FS);
  mesh_->create_field(field_enums::MATCH_FS);
  mesh_->create_field(field_enums::ITERATIONS_FS);
  mesh_->create_field(field_enums::STATUS_FLAG_FS);
  mesh_->create_field(field_enums::NEIGHBOR_ID_FS);
  mesh_->create_field(field_enums::CONDITION_NUMBER_FS);
  if(use_nonlinear_projection_){
    mesh_->create_field(field_enums::PROJECTION_AUG_X_FS);
    mesh_->create_field(field_enums::PROJECTION_AUG_Y_FS);
  }
  if(!is_stereo){
    mesh_->create_field(field_enums::CROSS_CORR_Q_FS);
    mesh_->create_field(field_enums::CROSS_CORR_R_FS);
    mesh_->create_field(field_enums::STEREO_SUBSET_DISPLACEMENT_X_FS);
    mesh_->create_field(field_enums::STEREO_SUBSET_DISPLACEMENT_Y_FS);
    mesh_->create_field(field_enums::MODEL_DISPLACEMENT_X_FS);
    mesh_->create_field(field_enums::MODEL_DISPLACEMENT_Y_FS);
    mesh_->create_field(field_enums::MODEL_DISPLACEMENT_Z_FS);
    mesh_->create_field(field_enums::STEREO_COORDINATES_X_FS);
    mesh_->create_field(field_enums::STEREO_COORDINATES_Y_FS);
    mesh_->create_field(field_enums::MODEL_COORDINATES_X_FS);
    mesh_->create_field(field_enums::MODEL_COORDINATES_Y_FS);
    mesh_->create_field(field_enums::MODEL_COORDINATES_Z_FS);
    mesh_->create_field(field_enums::EPI_A_FS);
    mesh_->create_field(field_enums::EPI_B_FS);
    mesh_->create_field(field_enums::EPI_C_FS);
    mesh_->create_field(field_enums::CROSS_EPI_ERROR_FS);
    mesh_->create_field(field_enums::FIELD_1_FS);
    mesh_->create_field(field_enums::FIELD_2_FS);
    mesh_->create_field(field_enums::FIELD_3_FS);
    mesh_->create_field(field_enums::FIELD_4_FS);
    mesh_->create_field(field_enums::FIELD_5_FS);
    mesh_->create_field(field_enums::FIELD_6_FS);
    mesh_->create_field(field_enums::FIELD_7_FS);
    mesh_->create_field(field_enums::FIELD_8_FS);
    mesh_->create_field(field_enums::FIELD_9_FS);
    mesh_->create_field(field_enums::FIELD_10_FS);
    mesh_->create_field(field_enums::STEREO_M_MAX_FS);
  }
  if(use_incremental_formulation_){
    mesh_->create_field(field_enums::ACCUMULATED_DISP_FS);
    mesh_->get_field(field_enums::ACCUMULATED_DISP_FS)->put_scalar(0.0);
  }

  // fill the subset coordinates field:
  Teuchos::RCP<MultiField> coords = mesh_->get_field(field_enums::INITIAL_COORDINATES_FS);
  for(int_t i=0;i<local_num_subsets_;++i){
    local_field_value(i,SUBSET_COORDINATES_X_FS) = coords->local_value(i*2+0);
    local_field_value(i,SUBSET_COORDINATES_Y_FS) = coords->local_value(i*2+1);
  }
}

void
Schema::post_execution_tasks(){
  if(analysis_type_==GLOBAL_DIC){
#ifdef DICE_ENABLE_GLOBAL
    global_algorithm_->post_execution_tasks(frame_id_);
#endif
  }
  swap_def_prev_images();
}

void
Schema::project_right_image_into_left_frame(Teuchos::RCP<Triangulation> tri,
  const bool reference){
  DEBUG_MSG("Schema::exectute_cross_correlation(): projecting the right image onto the left frame of reference");
  const int_t w = ref_img_->width();
  const int_t h = ref_img_->height();
  Teuchos::RCP<Image> img = reference ? ref_img_ : def_imgs_[0];
  const int_t olx = ref_img_->offset_x();
  const int_t oly = ref_img_->offset_y();
  const int_t orx = reference ? ref_img_->offset_x() : def_imgs_[0]->offset_x();
  const int_t ory = reference ? ref_img_->offset_y() : def_imgs_[0]->offset_y();
  Teuchos::RCP<Image> proj_img = reference ? Teuchos::rcp(new Image(ref_img_)) : Teuchos::rcp(new Image(def_imgs_[0]));
  Teuchos::ArrayRCP<storage_t> intens = proj_img->intensities();
  scalar_t xr = 0.0;
  scalar_t yr = 0.0;
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      tri->project_left_to_right_sensor_coords(i+olx,j+oly,xr,yr);
      intens[j*w+i] = img->interpolate_keys_fourth(xr-orx,yr-ory);
    }
  }
  if(reference){
    // automatically compute the derivatives of the new reference image...
    proj_img->compute_gradients();
    set_ref_image(proj_img);
  }else{
    set_def_image(proj_img);
  }
}

int_t
Schema::execute_cross_correlation(){
  if(analysis_type_==GLOBAL_DIC){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, cross-correlation has not been implemented for global DIC");
    return 0;
  }
  if(cross_initialization_method_==USE_SPACE_FILLING_ITERATIONS){
    DEBUG_MSG("Schema::execute_cross_correlation(): skipping cross-correlation since initialization method is USE_SPACE_FILLING_ITERATIONS");
    return 0;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(cross_initialization_method_!=USE_PLANAR_PROJECTION,std::runtime_error,
    "at this point USE_PLANAR_PROJECTION is the only valid cross init method");

  // make sure the data is ready to go since it may have been initialized externally by an api
  assert(is_initialized_);
  assert(global_num_subsets_>0);
  assert(local_num_subsets_>0);
  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();
  DEBUG_MSG("********************");
  std::stringstream progress;
  progress << "[PROC " << proc_id << " of " << num_procs << "] processing cross correlation";
  DEBUG_MSG(progress.str());
  DEBUG_MSG("********************");

  // keep track of the original parameters
  const Initialization_Method orig_init_method = initialization_method_;
  const Optimization_Method orig_opt_method = optimization_method_;
  const double orig_jump_tol = disp_jump_tol_;
  disp_jump_tol_ = 100.0;

  // change the parameters for cross-correlation
  initialization_method_ = USE_FIELD_VALUES; //USE_FEATURE_MATCHING;
  if(optimization_method_==GRADIENT_BASED||optimization_method_==GRADIENT_BASED_THEN_SIMPLEX)
    optimization_method_=GRADIENT_THEN_SEARCH;

  // project the right image onto the left if requested
  if(use_nonlinear_projection_){
    TEUCHOS_TEST_FOR_EXCEPTION(orig_init_method==USE_SATELLITE_GEOMETRY,std::runtime_error,"");
    // the nonlinear projection may not be good enough to initialize so start with a search window
    Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
    for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
      bool init_success = true;
      Teuchos::RCP<Objective> obj;
      try{
        obj = Teuchos::rcp(new Objective_ZNSSD(this,subset_global_id(subset_index)));
      }
      catch(...){
        init_success = false;
      }
      if(init_success){
        // search farther in x because we assume that the cross correlation is between two cameras with the same y
        const scalar_t search_step_u = 1.0; // pixels
        const scalar_t search_dim_u = 50.0; // pixels
        const int_t subset_gid = subset_global_id(subset_index);
        Search_Initializer searcher(this,obj->subset(),search_step_u,search_dim_u,-1.0,0.0,-1.0,0.0);
        searcher.initial_guess(subset_gid,shape_function);
        scalar_t min_u = 0.0,min_v = 0.0, min_t = 0.0;
        shape_function->map_to_u_v_theta(global_field_value(subset_gid,SUBSET_COORDINATES_X_FS),global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS),
          min_u,min_v,min_t);
        local_field_value(subset_index,SUBSET_DISPLACEMENT_X_FS) = min_u;
        //local_field_value(subset_index,DISPLACEMENT_Y) = min_v;
        //DEBUG_MSG("Schema::execute_cross_correlation(): subest gid " << subset_global_id(subset_index) << " search min u " << min_u << " v " << min_v << " gamma " << best_gamma);
        DEBUG_MSG("Schema::execute_cross_correlation(): subest gid " << subset_global_id(subset_index) << " search min u " << min_u << " gamma " << obj->gamma(shape_function));
      } // end subset loop
    }
  }

#ifdef DICE_DEBUG_MSG
  std::stringstream message;
  message << std::endl;
  for(int_t i=0;i<local_num_subsets_;++i){
    message << "[PROC " << proc_id << "] Owns subset global id (in cross-correlation order): " << this_proc_gid_order_[i] << " neighbor id: " << global_field_value(this_proc_gid_order_[i],NEIGHBOR_ID_FS) << std::endl;
  }
  DEBUG_MSG(message.str());
#endif

  prepare_optimization_initializers();
  for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
    try{
      Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(this,this_proc_gid_order_[subset_index]));
      generic_correlation_routine(obj);
    }
    catch(...){
      DEBUG_MSG("Schema::execute_cross_correlation(): subset " << this_proc_gid_order_[subset_index] << " failed");
      record_failed_step(this_proc_gid_order_[subset_index],static_cast<int_t>(CORRELATION_FAILED_BY_EXCEPTION),-1);
    }
  }
  // check the percentage of successful subsets:
  int_t num_successful = 0;
  for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
    if(local_field_value(subset_index,SIGMA_FS) > 0.0)
      num_successful++;
    else
      local_field_value(subset_index,NEIGHBOR_ID_FS) = -2; // set the neigh id to -1 to denote that the cross-correlation failed
  }
  DEBUG_MSG("[PROC " << proc_id << "]: success rate: " << (scalar_t)num_successful/(scalar_t)local_num_subsets_);
  for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
    DEBUG_MSG("[PROC " << proc_id << "] global subset id " << subset_global_id(subset_index) << " post execute_cross_correlation() field values, u: " <<
      local_field_value(subset_index,SUBSET_DISPLACEMENT_X_FS) << " v: " << local_field_value(subset_index,SUBSET_DISPLACEMENT_Y_FS)
      << " theta: " << local_field_value(subset_index,ROTATION_Z_FS) << " sigma: " << local_field_value(subset_index,SIGMA_FS) << " gamma: " <<
      local_field_value(subset_index,GAMMA_FS) << " beta: " << local_field_value(subset_index,BETA_FS) << " omega: " << local_field_value(subset_index,OMEGA_FS));
  }

  // add the projection fields back to the output for the total u/v, etc.
  if(use_nonlinear_projection_){
    // the original projective transform has to be added to the displacements
    Teuchos::RCP<MultiField> proj_aug_x = mesh_->get_field(PROJECTION_AUG_X_FS);
    Teuchos::RCP<MultiField> proj_aug_y = mesh_->get_field(PROJECTION_AUG_Y_FS);
    mesh_->get_field(SUBSET_DISPLACEMENT_X_FS)->update(1.0,*proj_aug_x,1.0);
    mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS)->update(1.0,*proj_aug_y,1.0);
  }

  // find any points that weren't reached by the space filling and turn them off
  for(int_t i=0;i<local_num_subsets_;++i){
    float a = local_field_value(i,EPI_A_FS);
    float b = local_field_value(i,EPI_B_FS);
    float c = local_field_value(i,EPI_C_FS);
    float stereo_x = local_field_value(i,SUBSET_COORDINATES_X_FS) + local_field_value(i,SUBSET_DISPLACEMENT_X_FS);
    float stereo_y = local_field_value(i,SUBSET_COORDINATES_Y_FS) + local_field_value(i,SUBSET_DISPLACEMENT_Y_FS);
    const float dist = (std::abs(a*stereo_x+b*stereo_y+c)/std::sqrt(a*a+b*b));
    local_field_value(i,CROSS_EPI_ERROR_FS) = dist;
//    if(dist>epi_dist_tol){
//      local_field_value(i,SIGMA_FS) = -1.0;
//    }
  }

  // reset the neighbor ids fiel and initialization method
  //neigh_ids->update(1.0,*original_neigh_ids,0.0);
  initialization_method_ = orig_init_method;
  optimization_method_ = orig_opt_method;
  disp_jump_tol_ = orig_jump_tol;
  // clear the optimization initializers so that the correlation in time uses the user requested one
  opt_initializers_.clear();

  return 0;
};

int_t
Schema::execute_correlation(){
  if(analysis_type_==GLOBAL_DIC){
#ifdef DICE_ENABLE_GLOBAL
    Status_Flag global_status = CORRELATION_FAILED;
    try{
      global_status = global_algorithm_->execute();
      update_frame_id();
    }
    catch(std::exception & e){
      std::cout << "Error, global correlation failed: " << e.what() << std::endl;
    }
    if(global_status!=CORRELATION_SUCCESSFUL){
      std::cout << "********* Error, global correlation failed **************** " << std::endl;
      return 1;
    }
#endif
    return 0;
  }
  // make sure the data is ready to go since it may have been initialized externally by an api
  assert(is_initialized_);
  assert(global_num_subsets_>0);
  assert(local_num_subsets_>0);
  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  DEBUG_MSG("********************");
  std::stringstream progress;
  progress << "[PROC " << proc_id << " of " << num_procs << "] IMAGE " << (frame_id_ - first_frame_id_)/frame_skip_ + 1;
  if(num_frames_>0)
    progress << " of " << num_frames_;
  DEBUG_MSG(progress.str());
  DEBUG_MSG("********************");

  // reset the motion detectors for each subset if used
  for(std::map<int_t,Teuchos::RCP<Motion_Test_Utility> >::iterator it = motion_detectors_.begin();
      it != motion_detectors_.end();++it){
    DEBUG_MSG("Resetting motion detector: " << it->first);
    it->second->reset();
  }

#ifdef DICE_DEBUG_MSG
  std::stringstream message;
  message << std::endl;
  for(int_t i=0;i<local_num_subsets_;++i){
    message << "[PROC " << proc_id << "] Owns subset global id (in order): " << this_proc_gid_order_[i] << " neighbor id: " << global_field_value(this_proc_gid_order_[i],NEIGHBOR_ID_FS) << std::endl;
  }
  DEBUG_MSG(message.str());
#endif

  // if the formulation is incremental, clear the displacement fields
  if(use_incremental_formulation_&&frame_id_>first_frame_id_){
    mesh_->get_field(SUBSET_DISPLACEMENT_X_FS)->put_scalar(0.0);
    mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS)->put_scalar(0.0);
  }
  // reset the sigma field for feature matching initializer
  if(initialization_method_==USE_FEATURE_MATCHING){
    for(int_t i=0;i<local_num_subsets_;++i){
      // only turn the point back on if the point didn't fail the cross-correlation sgnified by NEIGH_ID = -2
      if(local_field_value(i,NEIGHBOR_ID_FS)>-2.0) local_field_value(i,SIGMA_FS) = 0.0;
    }
  }
#ifdef DICE_ENABLE_GLOBAL
  if(has_initial_condition_file()&&frame_id_==first_frame_id_){
    TEUCHOS_TEST_FOR_EXCEPTION(initialization_method_!=USE_FIELD_VALUES,std::runtime_error,
      "Initialization method must be USE_FIELD_VALUES if an initial condition file is specified");
    Teuchos::RCP<DICe::mesh::Importer_Projector> importer = Teuchos::rcp(new DICe::mesh::Importer_Projector(initial_condition_file_,mesh_));
    TEUCHOS_TEST_FOR_EXCEPTION(importer->num_target_pts()!=local_num_subsets_,std::runtime_error,"");
    std::vector<scalar_t> disp_x;
    std::vector<scalar_t> disp_y;
    if(importer->is_valid_vector_source_field(initial_condition_file_,"SUBSET_DISPLACEMENT")){
      importer->import_vector_field(initial_condition_file_,"SUBSET_DISPLACEMENT",disp_x,disp_y);
    }
    else{
      importer->import_vector_field(initial_condition_file_,"DISPLACEMENT",disp_x,disp_y);
    }
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)disp_x.size()!=local_num_subsets_||(int_t)disp_y.size()!=local_num_subsets_,std::runtime_error,"");
    Teuchos::RCP<MultiField> ux = mesh_->get_field(SUBSET_DISPLACEMENT_X_FS);
    Teuchos::RCP<MultiField> uy = mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS);
    for(int_t i=0;i<local_num_subsets_;++i){
      ux->local_value(i) = disp_x[i];
      uy->local_value(i) = disp_y[i];
    }
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)this_proc_gid_order_.size()!=local_num_subsets_,std::runtime_error,
    "Error, the subset gid order vector is the wrong size");
  // The generic routine is typically used when the dataset involves numerous subsets,
  // but only a small number of images. In this case it's more efficient to re-allocate the
  // objectives at every step, since making them static would consume a lot of memory
  if(correlation_routine_==GENERIC_ROUTINE){
    // make sure that motion windows are not used
    TEUCHOS_TEST_FOR_EXCEPTION(motion_window_params_->size()!=0,std::runtime_error,
      "Error, motion windows are intended only for the TRACKING_ROUTINE");
    prepare_optimization_initializers();
    for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
      DEBUG_MSG("Schema::execute_correlation(): creating Objective for subset " << this_proc_gid_order_[subset_index]);
      try{
        Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(this,this_proc_gid_order_[subset_index]));
        DEBUG_MSG("Schema::execute_correlation(): Objective creation successful");
        generic_correlation_routine(obj);
      }
      catch(...){
        DEBUG_MSG("Schema::execute_correlation(): subset " << this_proc_gid_order_[subset_index] << " failed");
        record_failed_step(this_proc_gid_order_[subset_index],static_cast<int_t>(INITIALIZE_FAILED_BY_EXCEPTION),-1);
      }
    }
  }
  // In this routine there are usually only a handful of subsets, but thousands of images.
  // In this case it is a lot more efficient to make the objectives static since there won't
  // be very many of them, and we can avoid the allocation cost at every step
  else if(correlation_routine_==TRACKING_ROUTINE){
    // construct the static objectives if they haven't already been constructed
    if(obj_vec_.empty()){
      for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
        const int_t subset_gid = subset_global_id(subset_index);
        //const int_t subset_gid = this_proc_gid_order_[subset_index];
        DEBUG_MSG("[PROC " << proc_id << "] Adding objective to obj_vec_ " << subset_gid);
        obj_vec_.push_back(Teuchos::rcp(new Objective_ZNSSD(this,subset_gid)));
        // set the sub_image id for each subset:
        if(motion_window_params_->find(subset_gid)!=motion_window_params_->end()){
          const int_t use_subset_id = motion_window_params_->find(subset_gid)->second.use_subset_id_;
          const int_t sub_image_id = use_subset_id ==-1 ? motion_window_params_->find(subset_gid)->second.sub_image_id_:
              motion_window_params_->find(use_subset_id)->second.sub_image_id_;
          DEBUG_MSG("[PROC " << proc_id << "] setting the sub_image id for subset " << subset_gid << " to " << sub_image_id);
          obj_vec_[subset_index]->subset()->set_sub_image_id(sub_image_id);
        }
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)obj_vec_.size()!=local_num_subsets_,std::runtime_error,"");
    prepare_optimization_initializers();
    // execute the subsets in order
    for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
      const int_t subset_gid = this_proc_gid_order_[subset_index];
      const int_t subset_lid = subset_local_id(subset_gid);
      check_for_blocking_subsets(subset_gid);
      if(use_incremental_formulation_&&frame_id_>first_frame_id_){
        obj_vec_[subset_lid]->subset()->initialize(ref_img_,REF_INTENSITIES);
      }
      generic_correlation_routine(obj_vec_[subset_lid]);
    }
    if(output_deformed_subset_images_)
      write_deformed_subsets_image();
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"ERROR: unknown correlation routine.");

  for(int_t subset_index=0;subset_index<local_num_subsets_;++subset_index){
    DEBUG_MSG("[PROC " << proc_id << "] global subset id " << subset_global_id(subset_index) << " post execute_correlation() field values, u: " <<
      local_field_value(subset_index,SUBSET_DISPLACEMENT_X_FS) << " v: " << local_field_value(subset_index,SUBSET_DISPLACEMENT_Y_FS)
      << " theta: " << local_field_value(subset_index,ROTATION_Z_FS) << " sigma: " << local_field_value(subset_index,SIGMA_FS) << " gamma: " <<
      local_field_value(subset_index,GAMMA_FS) << " beta: " << local_field_value(subset_index,BETA_FS) << " omega: " << local_field_value(subset_index,OMEGA_FS));
  }

  // accumulate the displacements
  // for incremental, the displacement field gets zeroed out each frame and
  // another vector is needed to store the total displacement. Once execute correlation is completed, the
  // accumulated displacements are temporarily put back in the displacement field for writing the output
  // before it gets zeroed again for the next frame
  if(use_incremental_formulation_&&frame_id_>first_frame_id_){
    const int_t spa_dim = mesh_->spatial_dimension();
    Teuchos::RCP<DICe::MultiField> accumulated_disp = mesh_->get_field(ACCUMULATED_DISP_FS);
    for(int_t i=0;i<local_num_subsets_;++i){
      accumulated_disp->local_value(i*spa_dim+0) += local_field_value(i,SUBSET_DISPLACEMENT_X_FS);
      accumulated_disp->local_value(i*spa_dim+1) += local_field_value(i,SUBSET_DISPLACEMENT_Y_FS);
      local_field_value(i,SUBSET_DISPLACEMENT_X_FS) = accumulated_disp->local_value(i*spa_dim+0);
      local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) = accumulated_disp->local_value(i*spa_dim+1);
    }
  }
  update_frame_id();
  return 0;
};

void
Schema::save_cross_correlation_fields(){
  Teuchos::RCP<MultiField> ux = mesh_->get_field(SUBSET_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> uy = mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS);
  Teuchos::RCP<MultiField> cross_q = mesh_->get_field(CROSS_CORR_Q_FS);
  Teuchos::RCP<MultiField> cross_r = mesh_->get_field(CROSS_CORR_R_FS);
  cross_q->update(1.0,*ux,0.0);
  cross_r->update(1.0,*uy,0.0);
  // clear the deformation fields and update the initial positions of the cross corr subsets
  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
  shape_function->reset_fields(this);
  int_t num_failures = 0;
  scalar_t worst_gamma = 0.0;
  for(int_t i=0;i<local_num_subsets_;++i){
    local_field_value(i,STEREO_COORDINATES_X_FS) = local_field_value(i,SUBSET_COORDINATES_X_FS) + cross_q->local_value(i);
    local_field_value(i,STEREO_COORDINATES_Y_FS) = local_field_value(i,SUBSET_COORDINATES_Y_FS) + cross_r->local_value(i);
    if(local_field_value(i,SIGMA_FS) < 0.0)
      num_failures++;
    if(local_field_value(i,GAMMA_FS) > worst_gamma)
      worst_gamma = local_field_value(i,GAMMA_FS);
  }
  std::FILE * filePtr = fopen("projection_out.dat","a");
  fprintf(filePtr,"# Number of failed cross-correlation points: %i\n",num_failures);
  fprintf(filePtr,"# Worst cross-correlation gamma: %e\n",worst_gamma);
  fclose(filePtr);
}


void
Schema::execute_post_processors(){
  // compute post-processed quantities
  for(size_t i=0;i<post_processors_.size();++i){
    post_processors_[i]->update_current_frame_id(frame_id_-frame_skip_); // decrement frame skip since the frame id got updated by execute_correlation
    post_processors_[i]->execute(ref_img_,def_imgs_[0]);
  }
  DEBUG_MSG("[PROC " << comm_->get_rank() << "] post processing complete");
}

void
Schema::prepare_optimization_initializers(){
  // method only needs to be called once, return if the pointers are alread addressed
  if(opt_initializers_.size()>0){
    DEBUG_MSG("Repeat call to prepare_optimization_initializers(), calling pre_execution_tasks");
    for(std::map<int_t,Teuchos::RCP<Initializer> >::iterator opt_it = opt_initializers_.begin();
        opt_it != opt_initializers_.end();++opt_it){
      //assert(*opt_it!=Teuchos::null);
      opt_it->second->pre_execution_tasks();
    }
    return;
  }

  // set up the default initializer
  Teuchos::RCP<Initializer> default_initializer;
  if(initialization_method_==USE_PHASE_CORRELATION){
    DEBUG_MSG("Default initializer is phase correlation initializer");
    default_initializer = Teuchos::rcp(new Phase_Correlation_Initializer(this));
  }
  else if(initialization_method_==USE_ZEROS){
    DEBUG_MSG("Default initializer is zero value initializer");
    default_initializer = Teuchos::rcp(new Zero_Value_Initializer(this));
  }
  else if(initialization_method_==USE_FEATURE_MATCHING){
    DEBUG_MSG("Default initializer is feature matching initializer");
    default_initializer = Teuchos::rcp(new Feature_Matching_Initializer(this,threshold_block_size_));
  }
  else if(initialization_method_==USE_SATELLITE_GEOMETRY){
    DEBUG_MSG("Default initializer is satelite geometry initializer");
    default_initializer = Teuchos::rcp(new Satellite_Geometry_Initializer(this));
  }
  else if(initialization_method_==USE_IMAGE_REGISTRATION){
    DEBUG_MSG("Default initializer is image registration initializer");
    default_initializer = Teuchos::rcp(new Image_Registration_Initializer(this));
  }
  else if(initialization_method_==USE_OPTICAL_FLOW){
    // make syre tga the correlation routine is tracking routine
    TEUCHOS_TEST_FOR_EXCEPTION(correlation_routine_!=TRACKING_ROUTINE,std::invalid_argument,"Error, USE_OPTICAL_FLOW "
        "initialization method is only available for the TRACKING_ROUTINE correlation_routine.");
  }
  else{ // use field values
    DEBUG_MSG("Default initializer is field value initializer");
    default_initializer = Teuchos::rcp(new Field_Value_Initializer(this));
  }

  // create the optimization initializers (one for each subset for tracking)
  if(correlation_routine_==TRACKING_ROUTINE){
    // iterate all the subetsets to see if any have path files associated
    for(size_t i=0;i<obj_vec_.size();++i){
      const int_t subset_gid = obj_vec_[i]->correlation_point_global_id();
      const bool has_path_file = path_file_names_->find(subset_gid)!=path_file_names_->end();
      if(has_path_file){ // path file will trump optical flow if it exists
        const int_t num_neighbors = 6; // number of path neighbors to search while initializing
        std::string path_file_name = path_file_names_->find(subset_gid)->second;
        DEBUG_MSG("Subset " << subset_gid << " using path file " << path_file_name << " as initializer");
        opt_initializers_.insert(std::pair<int_t,Teuchos::RCP<Initializer> >(subset_gid,
          Teuchos::rcp(new Path_Initializer(this,obj_vec_[i]->subset(),path_file_name.c_str(),num_neighbors))));
      }
      // optical flow was requested for all subsets
      else if(initialization_method_==USE_OPTICAL_FLOW){
        DEBUG_MSG("Subset " << subset_gid << " using optical flow initializer (as requested by general "
            "initialization_method in correlation params file)");
        opt_initializers_.insert(std::pair<int_t,Teuchos::RCP<Initializer> >(subset_gid,
          Teuchos::rcp(new Optical_Flow_Initializer(this,obj_vec_[i]->subset()))));
      }
      // optical flow was requested for this subset specfically
      else if(optical_flow_flags_->find(subset_gid)!=optical_flow_flags_->end()){
        if(optical_flow_flags_->find(subset_gid)->second ==true){
          TEUCHOS_TEST_FOR_EXCEPTION(has_path_file,std::runtime_error,"Error, cannot set USE_PATH_FILE and USE_OPTICAL_FLOW "
              "for the same subset in the subset file");
          DEBUG_MSG("Subset " << subset_gid << " using optical flow initializer "
              "(as requested in subset file specifically for this subset)");
          opt_initializers_.insert(std::pair<int_t,Teuchos::RCP<Initializer> >(subset_gid,
            Teuchos::rcp(new Optical_Flow_Initializer(this,obj_vec_[i]->subset()))));
        }
      }
      else{
        DEBUG_MSG("Subset " << subset_gid << " using default initializer");
        opt_initializers_.insert(std::pair<int_t,Teuchos::RCP<Initializer> >(subset_gid,default_initializer));
      }
    } // end obj_vec
  }
  // if not tracking routine use one master initializer for all subsets
  else{
    // make sure that path files were not requested
    if(path_file_names_->size()>0){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Error, path files cannot be used with the GENERIC_ROUTINE correlation routine");
    }
    opt_initializers_.insert(std::pair<int_t,Teuchos::RCP<Initializer> >(0,default_initializer));
  }

  // call pre-correlation tasks for initializers
  for(std::map<int_t,Teuchos::RCP<Initializer> >::iterator opt_it = opt_initializers_.begin();
      opt_it != opt_initializers_.end();++opt_it){
    opt_it->second->pre_execution_tasks();
  }
}

bool
Schema::motion_detected(const int_t subset_gid){
  DEBUG_MSG("Schema::motion_detected() called");
  if(motion_window_params_->find(subset_gid)!=motion_window_params_->end()){
    // if there is no motion detection turned on, always return true (that motion is detected)
    if(motion_window_params_->find(subset_gid)->second.use_motion_detection_==false) return true;
    const int_t use_subset_id = motion_window_params_->find(subset_gid)->second.use_subset_id_==-1 ? subset_gid:
        motion_window_params_->find(subset_gid)->second.use_subset_id_;
    const int_t sub_image_id = motion_window_params_->find(subset_gid)->second.sub_image_id_;
    if(motion_detectors_.find(use_subset_id)==motion_detectors_.end()){
      // create the motion detector because it doesn't exist
      DEBUG_MSG("Creating a motion test utility for subset " << subset_gid << " using id " << use_subset_id);
      Motion_Window_Params mwp = motion_window_params_->find(use_subset_id)->second;
      motion_detectors_.insert(std::pair<int_t,Teuchos::RCP<Motion_Test_Utility> >(use_subset_id,Teuchos::rcp(new Motion_Test_Utility(this,mwp.tol_))));
    }
    TEUCHOS_TEST_FOR_EXCEPTION(motion_detectors_.find(use_subset_id)==motion_detectors_.end(),std::runtime_error,
      "Error, the motion detector should exist here, but it doesn't.");
    bool motion_det = motion_detectors_.find(use_subset_id)->second->motion_detected(sub_image_id);
    DEBUG_MSG("Subset " << subset_gid << " TEST_FOR_MOTION using window defined for subset " << use_subset_id <<
      " result " << motion_det);
    return motion_det;
  }
  else{
    DEBUG_MSG("Subset " << subset_gid << " will not test for motion");
    return true;
  }
}

void
Schema::record_failed_step(const int_t subset_gid,
  const int_t status,
  const int_t num_iterations){
  DEBUG_MSG("Subset " << subset_gid << " record failed step, status: " << status);
  // initialize the subset again to update the displacement fields, etc. in case this subset
  // gets turned back on in a subsequent frame
  if(initialization_method_==USE_FEATURE_MATCHING){
    Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
    try{
      initial_guess(subset_gid,shape_function);
    }
    catch (...) { // a non-graceful exception occurred in initialization
    };
    shape_function->save_fields(this,subset_gid);
  }
  global_field_value(subset_gid,SIGMA_FS) = -1.0;
  global_field_value(subset_gid,MATCH_FS) = -1.0;
  global_field_value(subset_gid,GAMMA_FS) = -1.0;
  global_field_value(subset_gid,BETA_FS) = -1.0;
  global_field_value(subset_gid,OMEGA_FS) = -1.0;
  global_field_value(subset_gid,NOISE_LEVEL_FS) = -1.0;
  global_field_value(subset_gid,CONTRAST_LEVEL_FS) = -1.0;
  global_field_value(subset_gid,ACTIVE_PIXELS_FS) = -1.0;
  global_field_value(subset_gid,STATUS_FLAG_FS) = status;
  global_field_value(subset_gid,ITERATIONS_FS) = num_iterations;
}

void
Schema::record_step(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function,
  const scalar_t & sigma,
  const scalar_t & match,
  const scalar_t & gamma,
  const scalar_t & beta,
  const scalar_t & noise,
  const scalar_t & contrast,
  const int_t active_pixels,
  const int_t status,
  const int_t num_iterations){
  DEBUG_MSG("Subset " << subset_gid << " record step");
  shape_function->save_fields(this,subset_gid);
  global_field_value(subset_gid,SIGMA_FS) = sigma;
  global_field_value(subset_gid,MATCH_FS) = match; // 0 means data is successful
  global_field_value(subset_gid,GAMMA_FS) = gamma;
  global_field_value(subset_gid,BETA_FS) = beta;
  global_field_value(subset_gid,NOISE_LEVEL_FS) = noise;
  global_field_value(subset_gid,CONTRAST_LEVEL_FS) = contrast;
  global_field_value(subset_gid,ACTIVE_PIXELS_FS) = active_pixels;
  global_field_value(subset_gid,STATUS_FLAG_FS) = status;
  global_field_value(subset_gid,ITERATIONS_FS) = num_iterations;
}

Status_Flag
Schema::initial_guess(const int_t subset_gid,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  // for non-tracking routines, there is only the zero-th entry in the initializers map
  int_t sid = 0;
  // tracking routine has a different initializer for each subset
  if(correlation_routine_==TRACKING_ROUTINE){
    sid = subset_gid;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(opt_initializers_.find(sid)==opt_initializers_.end(),std::runtime_error,
    "Initializer does not exist, but should here");
  return opt_initializers_.find(sid)->second->initial_guess(subset_gid,shape_function);
}

void
Schema::generic_correlation_routine(Teuchos::RCP<Objective> obj){

  const int_t subset_gid = obj->correlation_point_global_id();
  TEUCHOS_TEST_FOR_EXCEPTION(subset_local_id(subset_gid)==-1,std::runtime_error,
    "Error: subset id is not local to this process.");
  DEBUG_MSG("[PROC " << comm_->get_rank() << "] SUBSET " << subset_gid << " (" << global_field_value(subset_gid,SUBSET_COORDINATES_X_FS) <<
    "," << global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS) << ")");

  // if for some reason the coordinates of this subset are outside the image domain, record a failed step,
  // this may have occurred for a subset in the left image projected to the right that is not in the right image
  if(subset_dim_ > 0 && frame_id_==first_frame_id_){
    const scalar_t current_pos_x = global_field_value(subset_gid,SUBSET_COORDINATES_X_FS) + global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS);
    const scalar_t current_pos_y = global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS) + global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS);
    if(current_pos_x < def_imgs_[0]->offset_x()+subset_dim_/2 || current_pos_x > def_imgs_[0]->width()+def_imgs_[0]->offset_x() - subset_dim_/2 ||
        current_pos_y < def_imgs_[0]->offset_y()+subset_dim_/2 || current_pos_y > def_imgs_[0]->height()+def_imgs_[0]->offset_y() - subset_dim_/2){
      std::cout << "***WARNING***: subset origin location out of field of view at initialization" <<
        " current pos " << current_pos_x << " " << current_pos_y << " limits x " << def_imgs_[0]->offset_x()+subset_dim_/2 << " to " <<
        def_imgs_[0]->width()+def_imgs_[0]->offset_x() - subset_dim_/2 << " limits y " << def_imgs_[0]->offset_y()+subset_dim_/2 << " to " <<
        def_imgs_[0]->height()+def_imgs_[0]->offset_y() - subset_dim_/2 << std::endl;
      record_failed_step(subset_gid,static_cast<int_t>(INITIALIZE_FAILED_BY_EXCEPTION),-1);
      return;
    }
  }

  // check if the solve should be skipped
  bool skip_frame = false;
  if(skip_solve_flags_->find(subset_gid)!=skip_solve_flags_->end()||skip_all_solves_){
    if(skip_solve_flags_->find(subset_gid)!=skip_solve_flags_->end()){
      // determine for this subset id if it should be skipped:
      DEBUG_MSG("Subset " << subset_gid << " checking skip solve for trigger based frame id " << frame_id_);
      skip_frame = frame_should_be_skipped(frame_id_,skip_solve_flags_->find(subset_gid)->second);
      DEBUG_MSG("Subset " << subset_gid << " frame_should_be_skipped return value: " << skip_frame);
    }
  }
  //
  //  test for motion if requested by the user in the subsets.txt file
  //
  bool motion = true;
  if(!skip_frame&&!skip_all_solves_)
    motion = motion_detected(subset_gid);
  if(!motion){
    DEBUG_MSG("Subset " << subset_gid << " skipping frame due to no motion");
    // only change the match value and the status flag
    global_field_value(subset_gid,MATCH_FS) = 0.0;
    global_field_value(subset_gid,STATUS_FLAG_FS) = static_cast<int_t>(FRAME_SKIPPED_DUE_TO_NO_MOTION);
    global_field_value(subset_gid,ITERATIONS_FS) = 0;
    return;
  }
  //
  //  initial guess for the subset's solution parameters
  //
  Status_Flag init_status = INITIALIZE_SUCCESSFUL;
  Status_Flag corr_status = CORRELATION_FAILED;
  int_t num_iterations = -1;
  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
  try{
    init_status = initial_guess(subset_gid,shape_function);
  }
  catch (...) { // a non-graceful exception occurred in initialization
    record_failed_step(subset_gid,static_cast<int_t>(INITIALIZE_FAILED_BY_EXCEPTION),num_iterations);
    return;
  };
  //
  //  check if initialization was successful
  //
  if(init_status==INITIALIZE_FAILED){
    // try again with a search initializer
    if(correlation_routine_==TRACKING_ROUTINE && use_search_initialization_for_failed_steps_){
      TEUCHOS_TEST_FOR_EXCEPTION(shape_function_type_==DICe::RIGID_BODY_SF,std::runtime_error,
        "error, cannot use search initialization with rigid body shape function");
      stat_container_->register_search_call(subset_gid,frame_id_);
      // before giving up, try a search initialization, then simplex, then give up if it still can't track:
      const scalar_t search_step_xy = 1.0; // pixels
      const scalar_t search_dim_xy = 10.0; // pixels
      const scalar_t search_step_theta = 0.01; // radians (keep theta the same)
      const scalar_t search_dim_theta = 0.0;
      // reset the deformation position to the previous step's value
      shape_function->clear();
      shape_function->insert_motion(global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS),global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS),
        global_field_value(subset_gid,ROTATION_Z_FS));
      Search_Initializer searcher(this,obj->subset(),search_step_xy,search_dim_xy,search_step_xy,search_dim_xy,search_step_theta,search_dim_theta);
      init_status = searcher.initial_guess(subset_gid,shape_function);
    }
    if(init_status==INITIALIZE_FAILED){
      if(correlation_routine_==TRACKING_ROUTINE) stat_container_->register_failed_init(subset_gid,frame_id_);
      record_failed_step(subset_gid,static_cast<int_t>(init_status),num_iterations);
      return;
    }
  }
  //
  //  check if the user requested to skip the solve and only initialize (param set in subset file)
  //
  if(skip_frame||skip_all_solves_){
    if(skip_all_solves_){
      DEBUG_MSG("Subset " << subset_gid << " skip solve (skip_all_solves parameter was set)");
    }else{
      DEBUG_MSG("Subset " << subset_gid << " skip solve (as requested in the subset file via SKIP_SOLVE keyword)");
    }
    scalar_t noise_std_dev = 0.0;
    const scalar_t initial_sigma = obj->sigma(shape_function,noise_std_dev);
    const scalar_t initial_gamma = obj->gamma(shape_function);
    const scalar_t initial_beta = output_beta_ ? obj->beta(shape_function) : 0.0;
    const scalar_t contrast = obj->subset()->contrast_std_dev();
    const int_t active_pixels = obj->subset()->num_active_pixels();
    record_step(obj->correlation_point_global_id(),
      shape_function,initial_sigma,0.0,initial_gamma,initial_beta,
      noise_std_dev,contrast,active_pixels,static_cast<int_t>(FRAME_SKIPPED),num_iterations);
    // evolve the subsets and output the images requested as well
    // turn on pixels that at the beginning were hidden behind an obstruction
    if(use_subset_evolution_&&frame_id_>first_frame_id_+frame_skip_){
      DEBUG_MSG("[PROC " << comm_->get_rank() << "] Evolving subset " << subset_gid << " using newly exposed pixels for intensity values");
      obj->subset()->turn_on_previously_obstructed_pixels();
    }
    //  Write debugging images if requested
    if(output_deformed_subset_intensity_images_)
      write_deformed_subset_intensity_image(obj);
    if(output_evolved_subset_images_)
      write_reference_subset_intensity_image(obj);
    return;
  }

  //
  //  if user requested testing the initial value of gamma, do that here
  //
  if(initial_gamma_threshold_!=-1.0){
    const scalar_t initial_gamma = obj->gamma(shape_function);
    DEBUG_MSG("Subset " << subset_gid << " initial gamma value: " << initial_gamma);
    if(initial_gamma > initial_gamma_threshold_ || initial_gamma < 0.0){
      DEBUG_MSG("Subset " << subset_gid << " initial gamma value FAILS threshold test, gamma: " <<
        initial_gamma << " (threshold: " << initial_gamma_threshold_ << ")");
      record_failed_step(subset_gid,static_cast<int_t>(INITIALIZE_FAILED),num_iterations);
      return;
    }
  }
  // determine if the subset is a blocker and if so, force it to use simplex method:
  // also force simplex if it is a blocked subset (not enough speckles to use grad-based method)
  bool force_simplex = false;
  if(!override_force_simplex_){
    if(force_simplex_!=Teuchos::null)
      if(force_simplex_->find(subset_gid)!=force_simplex_->end()) force_simplex=true;
    if(obstructing_subset_ids_!=Teuchos::null){
      if(obstructing_subset_ids_->find(subset_gid)!=obstructing_subset_ids_->end()){
        if(obstructing_subset_ids_->find(subset_gid)->second.size()>0){
          force_simplex = true;
          DEBUG_MSG("[PROC " << comm_->get_rank() << "] SUBSET " << subset_gid << " is a blocker or blocked subset, forcing simplex method for this subset.");
        }
      }
      std::map<int_t,std::vector<int_t> >::iterator blk_it = obstructing_subset_ids_->begin();
      std::map<int_t,std::vector<int_t> >::iterator blk_end = obstructing_subset_ids_->end();
      for(;blk_it!=blk_end;++blk_it){
        std::vector<int_t> * obst_ids = &blk_it->second;
        for(size_t i=0;i<obst_ids->size();++i){
          if((*obst_ids)[i]==subset_gid){
            force_simplex = true;
            DEBUG_MSG("[PROC " << comm_->get_rank() << "] SUBSET " << subset_gid << " is a blocking subset, forcing simplex method for this subset.");
          }
        }
      }
    } // loop over obstructing subset ids
  } // end !override force simplex
  const scalar_t prev_u = global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS);
  const scalar_t prev_v = global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS);
  const scalar_t prev_t = global_field_value(subset_gid,ROTATION_Z_FS);
  //
  // perform the correlation
  //
  if(optimization_method_==DICe::SIMPLEX||optimization_method_==DICe::SIMPLEX_THEN_GRADIENT_BASED||force_simplex){
    try{
      corr_status = obj->computeUpdateRobust(shape_function,num_iterations);
    }
    catch (...) { //a non-graceful exception occurred
      corr_status = CORRELATION_FAILED_BY_EXCEPTION;
    };
  }
  else if(optimization_method_==DICe::GRADIENT_BASED||optimization_method_==DICe::GRADIENT_BASED_THEN_SIMPLEX||
      optimization_method_==DICe::GRADIENT_THEN_SEARCH){
    try{
      corr_status = obj->computeUpdateFast(shape_function,num_iterations);
    }
    catch (...) { //a non-graceful exception occurred
      corr_status = CORRELATION_FAILED_BY_EXCEPTION;
    };
  }
  //
  //  test for the jump tolerances here:
  //
  // test for jump failure (too high displacement or rotation from last step due to subset getting lost)
  bool jump_pass = true;
  scalar_t new_u = 0.0,new_v = 0.0, new_t = 0.0;
  shape_function->map_to_u_v_theta(global_field_value(subset_gid,SUBSET_COORDINATES_X_FS),global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS),
    new_u,new_v,new_t);
  scalar_t diffU = new_u - prev_u;
  scalar_t diffV = new_v - prev_v;
  scalar_t diffT = new_t - prev_t;
  DEBUG_MSG("Subset " << subset_gid << " U jump: " << diffU << " V jump: " << diffV << " T jump: " << diffT);
  if(std::abs(diffU) > disp_jump_tol_ || std::abs(diffV) > disp_jump_tol_ || std::abs(diffT) > theta_jump_tol_){
    if(correlation_routine_==TRACKING_ROUTINE) stat_container_->register_jump_exceeded(subset_gid,frame_id_);
    jump_pass = false;
  }
  DEBUG_MSG("Subset " << subset_gid << " jump pass: " << jump_pass);
  if(corr_status!=CORRELATION_SUCCESSFUL||!jump_pass){
    bool second_attempt_failed = false;
    if(optimization_method_==DICe::SIMPLEX||optimization_method_==DICe::GRADIENT_BASED||force_simplex){
      second_attempt_failed = true;
    }
    else if(optimization_method_==DICe::GRADIENT_BASED_THEN_SIMPLEX||optimization_method_==DICe::GRADIENT_THEN_SEARCH){
      if(correlation_routine_==TRACKING_ROUTINE) stat_container_->register_backup_opt_call(subset_gid,frame_id_);
      // try again using simplex
      init_status = initial_guess(subset_gid,shape_function);
      if(optimization_method_==DICe::GRADIENT_BASED_THEN_SIMPLEX){
        try{
          corr_status = obj->computeUpdateRobust(shape_function,num_iterations);
        }
        catch (...) { //a non-graceful exception occurred
          corr_status = CORRELATION_FAILED_BY_EXCEPTION;
        };
      }
      else if(optimization_method_==DICe::GRADIENT_THEN_SEARCH){
        // search farther in x because we assume that the cross correlation is between two cameras with the same y
        const scalar_t search_step_u = 1.0; // pixels
        const scalar_t search_dim_u = 50.0; // pixels
        Search_Initializer searcher(this,obj->subset(),search_step_u,search_dim_u,-1.0,0.0,-1.0,0.0);
        init_status = searcher.initial_guess(subset_gid,shape_function);
        scalar_t min_u = 0.0,min_v = 0.0, min_t = 0.0;
        shape_function->map_to_u_v_theta(global_field_value(subset_gid,SUBSET_COORDINATES_X_FS),global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS),
          min_u,min_v,min_t);
        DEBUG_MSG("Subset " << subset_gid << " GRADIENT_THEN_SEARCH method used, search-based initial u: " << min_u);// << " v: " << min_v);
        try{
          corr_status = obj->computeUpdateFast(shape_function,num_iterations);
        }
        catch (...) { //a non-graceful exception occurred
          corr_status = CORRELATION_FAILED_BY_EXCEPTION;
        };
      } // end gradient then search
      if(corr_status!=CORRELATION_SUCCESSFUL)
        second_attempt_failed = true;
    }
    else if(optimization_method_==DICe::SIMPLEX_THEN_GRADIENT_BASED){
      if(correlation_routine_==TRACKING_ROUTINE) stat_container_->register_backup_opt_call(subset_gid,frame_id_);
      // try again using gradient based
      init_status = initial_guess(subset_gid,shape_function);
      try{
          corr_status = obj->computeUpdateFast(shape_function,num_iterations);
      }
      catch (...) { //a non-graceful exception occurred
        corr_status = CORRELATION_FAILED_BY_EXCEPTION;
      };
      if(corr_status!=CORRELATION_SUCCESSFUL){
        second_attempt_failed = true;
      }
    }
    if(second_attempt_failed){
      if(corr_status!=CORRELATION_SUCCESSFUL){
        record_failed_step(subset_gid,static_cast<int_t>(corr_status),num_iterations);
        return;
      }
    }
  }
  //
  //  test final gamma if user requested
  //
  scalar_t noise_std_dev = 0.0;
  const scalar_t sigma = obj->sigma(shape_function,noise_std_dev);
  if(sigma < 0.0){
    DEBUG_MSG("Subset " << subset_gid << " final sigma value FAILS threshold test, sigma: " <<
      sigma << " (threshold: " << 0.0 << ")");
    // TODO for the phase correlation initialization method, the initial guess needs to be stored
    record_failed_step(subset_gid,static_cast<int_t>(FRAME_FAILED_DUE_TO_NEGATIVE_SIGMA),num_iterations);
    return;
  }
  const scalar_t gamma = obj->gamma(shape_function);
  const scalar_t beta = output_beta_ ? obj->beta(shape_function) : 0.0;
  if((final_gamma_threshold_!=-1.0 && gamma > final_gamma_threshold_)||gamma < 0.0){
    DEBUG_MSG("Subset " << subset_gid << " final gamma value " << gamma << " FAILS threshold test or is negative, gamma: " <<
      gamma << " (threshold: " << final_gamma_threshold_ << ")");
    // TODO for the phase correlation initialization method, the initial guess needs to be stored
    record_failed_step(subset_gid,static_cast<int_t>(FRAME_FAILED_DUE_TO_HIGH_GAMMA),num_iterations);
    return;
  }
  //
  //  test path distance if user requested
  //
  const bool has_path_file = path_file_names_->find(subset_gid)!=path_file_names_->end();
  if(path_distance_threshold_!=-1.0&&has_path_file){
    scalar_t path_distance = 0.0;
    size_t id = 0;
    // dynamic cast the pointer to get access to the derived class methods
    Teuchos::RCP<Path_Initializer> path_initializer =
        Teuchos::rcp_dynamic_cast<Path_Initializer>(opt_initializers_.find(subset_gid)->second);
    scalar_t pt_u = 0.0,pt_v = 0.0,pt_t=0.0;
    shape_function->map_to_u_v_theta(global_field_value(subset_gid,SUBSET_COORDINATES_X_FS),global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS),
      pt_u,pt_v,pt_t);
    path_initializer->closest_triad(pt_u,pt_v,pt_t,id,path_distance);
    DEBUG_MSG("Subset " << subset_gid << " path distance: " << path_distance);
    if(path_distance > path_distance_threshold_)
    {
      DEBUG_MSG("Subset " << subset_gid << " path distance value FAILS threshold test, distance from path: " <<
        path_distance << " (threshold: " << path_distance_threshold_ << ")");
      record_failed_step(subset_gid,static_cast<int_t>(FRAME_FAILED_DUE_TO_HIGH_PATH_DISTANCE),num_iterations);
      return;
    }
  }
  //
  //  Test jumps again
  //
  new_u = 0.0;new_v = 0.0;new_t = 0.0;
  shape_function->map_to_u_v_theta(global_field_value(subset_gid,SUBSET_COORDINATES_X_FS),global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS),
    new_u,new_v,new_t);
  diffU = new_u - prev_u;
  diffV = new_v - prev_v;
  diffT = new_t - prev_t;
  DEBUG_MSG("Subset " << subset_gid << " U jump: " << diffU << " V jump: " << diffV << " T jump: " << diffT);
  if(std::abs(diffU) > disp_jump_tol_ || std::abs(diffV) > disp_jump_tol_ || std::abs(diffT) > theta_jump_tol_){
    DEBUG_MSG("Subset " << subset_gid << " FAILS jump test: ");
    // TODO for the phase correlation initialization method, the initial guess needs to be stored
    record_failed_step(subset_gid,static_cast<int_t>(JUMP_TOLERANCE_EXCEEDED),num_iterations);
    return;
  }
  // TODO how to respond to failure here? or for initialization?
  //
  // SUCCESS
  //
  if(projection_method_==VELOCITY_BASED) save_off_fields(subset_gid);
  const scalar_t contrast = obj->subset()->contrast_std_dev();
  const int_t active_pixels = obj->subset()->num_active_pixels();
  record_step(obj->correlation_point_global_id(),
    shape_function,sigma,0.0,gamma,beta,noise_std_dev,contrast,active_pixels,
    static_cast<int_t>(init_status),num_iterations);
  //
  //  turn on pixels that at the beginning were hidden behind an obstruction
  //
  if(use_subset_evolution_&&frame_id_>first_frame_id_+frame_skip_){
    DEBUG_MSG("[PROC " << comm_->get_rank() << "] Evolving subset " << subset_gid << " using newly exposed pixels for intensity values");
    obj->subset()->turn_on_previously_obstructed_pixels();
  }
  //
  //  Write debugging images if requested
  //
  if(output_deformed_subset_intensity_images_)
    write_deformed_subset_intensity_image(obj);
  if(output_evolved_subset_images_)
    write_reference_subset_intensity_image(obj);
}

void
Schema::write_deformed_subset_intensity_image(Teuchos::RCP<Objective> obj){
  DEBUG_MSG("[PROC " << comm_->get_rank() << "] Attempting to create directory : ./deformed_subset_intensities/");
  std::string dirStr = "./deformed_subset_intensities/";
  create_directory(dirStr);
  int_t num_zeros = 0;
  if(num_frames_>0){
    int_t num_digits_total = 0;
    int_t num_digits_image = 0;
    int_t decrement_total = first_frame_id_ + num_frames_*frame_skip_;
    int_t decrement_image = frame_id_;
    while (decrement_total){decrement_total /= 10; num_digits_total++;}
    if(decrement_image==0) num_digits_image = 1;
    else
      while (decrement_image){decrement_image /= 10; num_digits_image++;}
    num_zeros = num_digits_total - num_digits_image;
  }
  std::stringstream ss;
  ss << dirStr << "deformedSubset_" << obj->correlation_point_global_id() << "_";
  for(int_t i=0;i<num_zeros;++i)
    ss << "0";
  ss << frame_id_ << ".tif";
  obj->subset()->write_image(ss.str(),true);
}

void
Schema::write_reference_subset_intensity_image(Teuchos::RCP<Objective> obj){
  DEBUG_MSG("[PROC " << comm_->get_rank() << "] Attempting to create directory : ./evolved_subsets/");
  std::string dirStr = "./evolved_subsets/";
  create_directory(dirStr);
  int_t num_zeros = 0;
  if(num_frames_>0){
    int_t num_digits_total = 0;
    int_t num_digits_image = 0;
    int_t decrement_total = first_frame_id_ + num_frames_*frame_skip_;
    int_t decrement_image = frame_id_;
    while (decrement_total){decrement_total /= 10; num_digits_total++;}
    if(decrement_image==0) num_digits_image = 1;
    else
      while (decrement_image){decrement_image /= 10; num_digits_image++;}
    num_zeros = num_digits_total - num_digits_image;
  }
  std::stringstream ss;
  ss << dirStr << "evolvedSubset_" << obj->correlation_point_global_id() << "_";
  for(int_t i=0;i<num_zeros;++i)
    ss << "0";
  ss << frame_id_ << ".tif";
  obj->subset()->write_image(ss.str());
}

void
Schema::estimate_resolution_error(const Teuchos::RCP<Teuchos::ParameterList> & correlation_params,
  std::string & output_folder,
  std::string & resolution_output_folder,
  std::string & prefix,
  Teuchos::RCP<std::ostream> & outStream){
  const int_t proc_id = comm_->get_rank();
  assert(ref_img_->width()>0);
  assert(ref_img_->height()>0);
  const scalar_t min_dim = ref_img_->width() < ref_img_->height() ? ref_img_->width() : ref_img_->height();
  const scalar_t min_period = correlation_params->get<double>(DICe::estimate_resolution_error_min_period,25);
  const scalar_t max_period = correlation_params->get<double>(DICe::estimate_resolution_error_max_period,min_dim/3.0);
  TEUCHOS_TEST_FOR_EXCEPTION(min_period > max_period,std::runtime_error," min period " << min_period << " max period: " << max_period);
  const scalar_t period_factor = correlation_params->get<double>(DICe::estimate_resolution_error_period_factor,0.5);
  const scalar_t min_amp = correlation_params->get<double>(DICe::estimate_resolution_error_min_amplitude,0.5);
  const scalar_t max_amp = correlation_params->get<double>(DICe::estimate_resolution_error_max_amplitude,4.0);
  const scalar_t amp_step = correlation_params->get<double>(DICe::estimate_resolution_error_amplitude_step,0.5);
  const scalar_t speckle_size = correlation_params->get<double>(DICe::estimate_resolution_error_speckle_size,-1.0);
  const scalar_t noise_percent = correlation_params->get<double>(DICe::estimate_resolution_error_noise_percent,-1.0);

  // the full image width and height must be set
  TEUCHOS_TEST_FOR_EXCEPTION(full_ref_img_width_<=0||full_ref_img_height_<=0,std::runtime_error,"");

  // search the mesh fields to see if vsg or nlvc strain exist:
  const bool has_vsg = mesh_->has_field(VSG_STRAIN_XX);
  const bool has_nlvc = mesh_->has_field(NLVC_STRAIN_XX);
  const bool is_subset_based = analysis_type_ == LOCAL_DIC;

  scalar_t subset_elem_size = 0.0;
  scalar_t step_size = -1.0;
  scalar_t vsg_size = -1.0;
  scalar_t nlvc_size = -1.0;
  if(analysis_type_==GLOBAL_DIC){
#ifdef DICE_ENABLE_GLOBAL
    subset_elem_size = global_algorithm()->mesh_size();
#endif
  }else{
    subset_elem_size = subset_dim_;
    step_size = step_size_x_;
  }
  if(has_vsg){
    Teuchos::ParameterList sublist = init_params_->sublist(DICe::post_process_vsg_strain);
    vsg_size = sublist.get<int_t>(DICe::strain_window_size_in_pixels,-1);
  }
  if(has_nlvc){
    Teuchos::ParameterList sublist = init_params_->sublist(DICe::post_process_nlvc_strain);
    nlvc_size = sublist.get<int_t>(DICe::horizon_diameter_in_pixels,-1);
  }
  if(proc_id==0){
    DEBUG_MSG("****************************************************************");
    DEBUG_MSG("Schema::estimate_resolution_error(): ");
    DEBUG_MSG("****************************************************************");
    DEBUG_MSG("subset or element size:    " << subset_elem_size << " pixels");
    DEBUG_MSG("step size:                 " << step_size << " pixels");
    DEBUG_MSG("strain window size:        " << vsg_size << " pixels");
    DEBUG_MSG("nonlocal horizon diameter: " << nlvc_size << " pixels");
    DEBUG_MSG("minumum motion period:     " << min_period << " pixels");
    DEBUG_MSG("maximum motion period:     " << max_period << " pixels");
    DEBUG_MSG("period factor:             " << period_factor);
    DEBUG_MSG("minimum motion amplitude:  " << min_amp << " pixels");
    DEBUG_MSG("maximum motion amplitude:  " << max_amp << " pixels");
    DEBUG_MSG("amplitude step:            " << amp_step << " pixels");
    DEBUG_MSG("speckle size:              " << speckle_size << " pixels (negative means not specified)");
    DEBUG_MSG("noise level:               " << noise_percent << "% of 255 counts (negative means not specified)");
    DEBUG_MSG("****************************************************************");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(min_period <= 0.0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(min_amp <= 0.0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(max_period < min_period,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(period_factor <= 0.0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(max_amp < min_amp,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(amp_step <= 0.0,std::runtime_error,"");

  // create the images folder if it doesn't exist
#if defined(WIN32)
  std::string image_dir_str = resolution_output_folder + "synthetic_images\\";
#else
  std::string image_dir_str = resolution_output_folder + "synthetic_images/";
#endif
  create_directory(image_dir_str);
  // create the results folder if it doesn't exist
#if defined(WIN32)
  std::string data_dir_str = resolution_output_folder + "synthetic_results\\";
#else
  std::string data_dir_str = resolution_output_folder + "synthetic_results/";
#endif
  DEBUG_MSG("Attempting to create directory : " << data_dir_str);
  create_directory(data_dir_str);

  std::stringstream result_stream;

  // populate the exact sol, etc
  mesh_->create_field(EXACT_SOL_VECTOR_FS);
  Teuchos::RCP<MultiField> exact_disp = mesh_->get_field(EXACT_SOL_VECTOR_FS);
  mesh_->create_field(EXACT_STRAIN_XX_FS);
  Teuchos::RCP<MultiField> exact_strain_xx = mesh_->get_field(EXACT_STRAIN_XX_FS);
  mesh_->create_field(EXACT_STRAIN_YY_FS);
  Teuchos::RCP<MultiField> exact_strain_yy = mesh_->get_field(EXACT_STRAIN_YY_FS);
  mesh_->create_field(EXACT_STRAIN_XY_FS);
  Teuchos::RCP<MultiField> exact_strain_xy = mesh_->get_field(EXACT_STRAIN_XY_FS);
  // create the error fields
  mesh_->create_field(DISP_ERROR_FS);
  Teuchos::RCP<MultiField> disp_error = mesh_->get_field(DISP_ERROR_FS);
  Teuchos::RCP<MultiField> vsg_error_xx, vsg_error_xy, vsg_error_yy, nlvc_error_xx, nlvc_error_xy, nlvc_error_yy;
  if(has_vsg){
    mesh_->create_field(VSG_STRAIN_XX_ERROR_FS);
    vsg_error_xx = mesh_->get_field(VSG_STRAIN_XX_ERROR_FS);
    mesh_->create_field(VSG_STRAIN_XY_ERROR_FS);
    vsg_error_xy = mesh_->get_field(VSG_STRAIN_XY_ERROR_FS);
    mesh_->create_field(VSG_STRAIN_YY_ERROR_FS);
    vsg_error_yy = mesh_->get_field(VSG_STRAIN_YY_ERROR_FS);
  }
  if(has_nlvc){
    mesh_->create_field(NLVC_STRAIN_XX_ERROR_FS);
    nlvc_error_xx = mesh_->get_field(NLVC_STRAIN_XX_ERROR_FS);
    mesh_->create_field(NLVC_STRAIN_XY_ERROR_FS);
    nlvc_error_xy = mesh_->get_field(NLVC_STRAIN_XY_ERROR_FS);
    mesh_->create_field(NLVC_STRAIN_YY_ERROR_FS);
    nlvc_error_yy = mesh_->get_field(NLVC_STRAIN_YY_ERROR_FS);
  }

  // generate synthetic speckled image instead of using the reference image
  if(speckle_size >= 1.0){
    TEUCHOS_TEST_FOR_EXCEPTION(speckle_size > 500.0,std::runtime_error,"Error, unreasonable speckle size: " << speckle_size);
    DEBUG_MSG("Creating a reference image with synthetic speckles of size " << speckle_size);
    Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
    imgParams->set(DICe::compute_image_gradients,true);
    imgParams->set(DICe::gradient_method,gradient_method_);
    Teuchos::RCP<Image> speckled_ref = create_synthetic_speckle_image<storage_t>(ref_img_->width(),ref_img_->height(),
      ref_img_->offset_x(),ref_img_->offset_y(),speckle_size,imgParams);
    set_ref_image(speckled_ref);
  }

  // TODO fix this to communicate the speckle stats across all processors not just each individual proc
  int_t avg_speckle_size = -1;
  if(speckle_size>=1.0)
    avg_speckle_size = speckle_size;
  else{
    if(proc_id==0){ // other procs don't need to report speckle size
      // have to build full images for this step on process zero
      TEUCHOS_TEST_FOR_EXCEPTION(!ref_img_->has_file_name(),std::runtime_error,"");
      Teuchos::RCP<Image> full_ref = Teuchos::rcp(new Image(ref_img_->file_name().c_str()));
      avg_speckle_size = compute_speckle_stats(data_dir_str,full_ref);
    }
  }

  std::stringstream data_name;
  data_name << data_dir_str << "spatial_resolution.txt";
  // see if the file exists already:
  std::ifstream output_file(data_name.str().c_str());
  if(proc_id==0&&!output_file.good()){
    std::FILE * infoFilePtr = fopen(data_name.str().c_str(),"w");
    fprintf(infoFilePtr,"subset_elem_size(px) step_size(px) speckle_size(px) noise_percent(%%) vsg_size(px) nlvc_size(px) period(px) amp(px) u_err_min(rel%%) u_err_max(rel%%) u_err_avg(rel%%) u_err_std_dev(rel%%) "
        "v_err_min(rel%%) v_err_max(rel%%) v_err_avg(rel%%) v_err_std_dev(rel%%)");
    fprintf(infoFilePtr," u_peaks_avg_err(rel%%) u_peaks_std_dev_err(rel%%) v_peaks_avg_err(rel%%) v_peaks_std_dev_err(rel%%)");
    if(has_vsg){
      fprintf(infoFilePtr," vsg_xx_err_min(rel%%) vsg_xx_err_max(rel%%) vsg_xx_err_avg(rel%%) vsg_xx_std_dev(rel%%) "
          "vsg_xy_err_min(rel%%) vsg_xy_err_max(rel%%) vsg_xy_err_avg(rel%%) vsg_xy_err_std_dev(rel%%) "
          "vsg_yy_err_min(rel%%) vsg_yy_err_max(rel%%) vsg_yy_err_avg(rel%%) vsg_yy_err_std_dev(rel%%)");
      fprintf(infoFilePtr," vsg_xx_peaks_avg_err(rel%%) vsg_xx_peaks_std_dev_err(rel%%) vsg_yy_peaks_avg_err(rel%%) vsg_yy_peaks_std_dev_err(rel%%)");
    }
    if(has_nlvc){
      fprintf(infoFilePtr," nlvc_xx_err_min(rel%%) nlvc_xx_err_max(rel%%) nlvc_xx_err_avg(rel%%) nlvc_xx_std_dev(rel%%) "
          "nlvc_xy_err_min(rel%%) nlvc_xy_err_max(rel%%) nlvc_xy_err_avg(rel%%) nlvc_xy_err_std_dev(rel%%) "
          "nlvc_yy_err_min(rel%%) nlvc_yy_err_max(rel%%) nlvc_yy_err_avg(rel%%) nlvc_yy_err_std_dev(rel%%)");
      fprintf(infoFilePtr," nlvc_xx_peaks_avg_err(rel%%) nlvc_xx_peaks_std_dev_err(rel%%) nlvc_yy_peaks_avg_err(rel%%) nlvc_yy_peaks_std_dev_err(rel%%)");
    }
    fprintf(infoFilePtr,"\n");
    fclose(infoFilePtr);
  }
  const int_t spa_dim = mesh_->spatial_dimension();
  for(scalar_t period=max_period;period>=min_period;period*=period_factor){
    // reset the displacements between frequency updates, otherwise the existing solution makes a nice initial guess
    if(is_subset_based){
      mesh_->get_field(SUBSET_DISPLACEMENT_X_FS)->put_scalar(0.0);
      mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS)->put_scalar(0.0);
    }else{
      mesh_->get_field(DISPLACEMENT_FS)->put_scalar(0.0);
    }
    mesh_->get_field(SIGMA_FS)->put_scalar(0.0);
    for(scalar_t amplitude=min_amp;amplitude<=max_amp;amplitude+=amp_step){
      if(proc_id==0)
        std::cout << "processing resolution error for period " << period << " amplitude " << amplitude << std::endl;
      // create an image deformer class
      image_deformer_ = Teuchos::rcp(new Image_Deformer(period,amplitude,Image_Deformer::SIN_COS));
      std::stringstream sincos_name;
      Teuchos::RCP<Image> def_img;
      std::stringstream amp_ss;
      std::stringstream per_ss;
      amp_ss << amplitude;
      std::string amp_s = amp_ss.str();
      std::replace( amp_s.begin(), amp_s.end(), '.', 'p'); // replace dots with p for file name
      per_ss << period;
      std::string per_s = per_ss.str();
      std::replace( per_s.begin(), per_s.end(), '.', 'p'); // replace dots with p for file name
      sincos_name << image_dir_str << "amp_" << std::setprecision(4) << amp_s << "_period_" << std::setprecision(4) << per_s << "_proc_" << proc_id << ".tif";

      // check to see if the deformed image already exists:
      std::ifstream f(sincos_name.str().c_str());
      //if(f.good()){
      //  DEBUG_MSG("using previously saved image");
      //  def_img = Teuchos::rcp(new DICe::Image(sincos_name.str().c_str()));
      //}else{
        DEBUG_MSG("generating new synthetic image");
        def_img = image_deformer_->deform_image(ref_img());
        if(noise_percent > 0.0){
          add_noise_to_image(def_img,noise_percent);
        }
        def_img->write(sincos_name.str());
      //}

      // set the deformed image for the schema
      set_def_image(def_img);
      int_t corr_error = execute_correlation();
      TEUCHOS_TEST_FOR_EXCEPTION(corr_error,std::runtime_error,"Error, correlation unsuccesssful");
      DEBUG_MSG("Error prediction step correlation return value " << corr_error);
      execute_post_processors();
      post_execution_tasks();

      // gather all owned fields here
      Teuchos::RCP<MultiField> coords = mesh_->get_field(INITIAL_COORDINATES_FS);
      Teuchos::RCP<MultiField> disp;
      if(is_subset_based){
        Teuchos::RCP<MultiField> disp_x = mesh_->get_field(SUBSET_DISPLACEMENT_X_FS);
        Teuchos::RCP<MultiField> disp_y = mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS);
        Teuchos::RCP<MultiField_Map> map = mesh_->get_vector_node_dist_map();
        disp = Teuchos::rcp( new MultiField(map,1,true));
        for(int_t i=0;i<local_num_subsets_;++i){
          disp->local_value(i*spa_dim+0) = disp_x->local_value(i);
          disp->local_value(i*spa_dim+1) = disp_y->local_value(i);
        }
      }else{
        disp = mesh_->get_field(DISPLACEMENT_FS);
      }
      Teuchos::RCP<MultiField> vsg_xx;
      Teuchos::RCP<MultiField> vsg_xy;
      Teuchos::RCP<MultiField> vsg_yy;
      Teuchos::RCP<MultiField> nlvc_xx;
      Teuchos::RCP<MultiField> nlvc_xy;
      Teuchos::RCP<MultiField> nlvc_yy;
      if(has_vsg){
        vsg_xx = mesh_->get_field(VSG_STRAIN_XX_FS);
        vsg_xy = mesh_->get_field(VSG_STRAIN_XY_FS);
        vsg_yy = mesh_->get_field(VSG_STRAIN_YY_FS);
      }
      if(has_nlvc){
        nlvc_xx = mesh_->get_field(NLVC_STRAIN_XX_FS);
        nlvc_xy = mesh_->get_field(NLVC_STRAIN_XY_FS);
        nlvc_yy = mesh_->get_field(NLVC_STRAIN_YY_FS);
      }
      // compute the error fields
      for(int_t i=0;i<local_num_subsets_;++i){
        const scalar_t x = coords->local_value(i*spa_dim+0);
        const scalar_t y = coords->local_value(i*spa_dim+1);
        const scalar_t u = disp->local_value(i*spa_dim+0);
        const scalar_t v = disp->local_value(i*spa_dim+1);
        scalar_t exact_u = 0.0;
        scalar_t exact_v = 0.0;
        image_deformer_->compute_deformation(x,y,exact_u,exact_v);
        exact_disp->local_value(i*spa_dim+0) = exact_u;
        exact_disp->local_value(i*spa_dim+1) = exact_v;
        scalar_t error_v = 0.0;
        scalar_t error_u = 0.0;
        image_deformer_->compute_displacement_error(x,y,u,v,error_u,error_v);
        disp_error->local_value(i*spa_dim+0) = std::abs(error_u);
        disp_error->local_value(i*spa_dim+1) = std::abs(error_v);
        scalar_t strain_xx = 0.0;
        scalar_t strain_xy = 0.0;
        scalar_t strain_yy = 0.0;
        image_deformer_->compute_lagrange_strain(x,y,strain_xx,strain_xy,strain_yy);
        exact_strain_xx->local_value(i) = strain_xx;
        exact_strain_xy->local_value(i) = strain_xy;
        exact_strain_yy->local_value(i) = strain_yy;
        if(has_vsg){
          const scalar_t e_xx = vsg_xx->local_value(i);
          const scalar_t e_xy = vsg_xy->local_value(i);
          const scalar_t e_yy = vsg_yy->local_value(i);
          scalar_t error_xx = 0.0;
          scalar_t error_xy = 0.0;
          scalar_t error_yy = 0.0;
          image_deformer_->compute_lagrange_strain_error(x,y,e_xx,e_xy,e_yy,error_xx,error_xy,error_yy);
          vsg_error_xx->local_value(i) = std::abs(error_xx);
          vsg_error_xy->local_value(i) = std::abs(error_xy);
          vsg_error_yy->local_value(i) = std::abs(error_yy);
        }
        if(has_nlvc){
          const scalar_t e_xx = nlvc_xx->local_value(i);
          const scalar_t e_xy = nlvc_xy->local_value(i);
          const scalar_t e_yy = nlvc_yy->local_value(i);
          scalar_t error_xx = 0.0;
          scalar_t error_xy = 0.0;
          scalar_t error_yy = 0.0;
          image_deformer_->compute_lagrange_strain_error(x,y,e_xx,e_xy,e_yy,error_xx,error_xy,error_yy);
          nlvc_error_xx->local_value(i) = std::abs(error_xx);
          nlvc_error_xy->local_value(i) = std::abs(error_xy);
          nlvc_error_yy->local_value(i) = std::abs(error_yy);
        }
      } // end local subsets loop

      result_stream << subset_elem_size << " " << step_size << " " << avg_speckle_size << " " << noise_percent << " " << vsg_size << " " << nlvc_size;

      // collect the global stats based on the field info above:
      scalar_t min_error_u = 0.0;
      scalar_t max_error_u = 0.0;
      scalar_t avg_error_u = 0.0;
      scalar_t std_dev_error_u = 0.0;
      scalar_t min_error_v = 0.0;
      scalar_t max_error_v = 0.0;
      scalar_t avg_error_v = 0.0;
      scalar_t std_dev_error_v = 0.0;
      mesh_->field_stats(DISP_ERROR_FS,min_error_u,max_error_u,avg_error_u,std_dev_error_u,0,SIGMA_FS,-1.0);
      mesh_->field_stats(DISP_ERROR_FS,min_error_v,max_error_v,avg_error_v,std_dev_error_v,1,SIGMA_FS,-1.0);
      result_stream << " " << std::setprecision(4) << period << " "<< std::setprecision(4) << amplitude
          << " " << min_error_u << " " << max_error_u << " " << avg_error_u << " " << std_dev_error_u << " " << min_error_v << " " << max_error_v << " " << avg_error_v << " " << std_dev_error_v;

      scalar_t peaks_avg_error_x = 0.0;
      scalar_t peaks_std_dev_error_x = 0.0;
      scalar_t peaks_avg_error_y = 0.0;
      scalar_t peaks_std_dev_error_y = 0.0;
      // analyze the peaks of the output to evaluate the roll off
      compute_roll_off_stats(period,full_ref_img_width_,full_ref_img_height_,coords,disp,exact_disp,disp_error,
        peaks_avg_error_x,peaks_std_dev_error_x,peaks_avg_error_y,peaks_std_dev_error_y);
      result_stream << " " << peaks_avg_error_x << " " << peaks_std_dev_error_x << " " << peaks_avg_error_y << " " << peaks_std_dev_error_y;

      if(has_vsg){
        scalar_t min_vsg_xx = 0.0;
        scalar_t max_vsg_xx = 0.0;
        scalar_t avg_vsg_xx = 0.0;
        scalar_t std_dev_vsg_xx = 0.0;
        scalar_t min_vsg_xy = 0.0;
        scalar_t max_vsg_xy = 0.0;
        scalar_t avg_vsg_xy = 0.0;
        scalar_t std_dev_vsg_xy = 0.0;
        scalar_t min_vsg_yy = 0.0;
        scalar_t max_vsg_yy = 0.0;
        scalar_t avg_vsg_yy = 0.0;
        scalar_t std_dev_vsg_yy = 0.0;
        mesh_->field_stats(VSG_STRAIN_XX_ERROR_FS,min_vsg_xx,max_vsg_xx,avg_vsg_xx,std_dev_vsg_xx,0,SIGMA_FS,-1.0);
        mesh_->field_stats(VSG_STRAIN_XY_ERROR_FS,min_vsg_xy,max_vsg_xy,avg_vsg_xy,std_dev_vsg_xy,0,SIGMA_FS,-1.0);
        mesh_->field_stats(VSG_STRAIN_YY_ERROR_FS,min_vsg_yy,max_vsg_yy,avg_vsg_yy,std_dev_vsg_yy,0,SIGMA_FS,-1.0);
        result_stream << " " << min_vsg_xx << " " << max_vsg_xx << " " << avg_vsg_xx << " " << std_dev_vsg_xx;
        result_stream << " " << min_vsg_xy << " " << max_vsg_xy << " " << avg_vsg_xy << " " << std_dev_vsg_xy;
        result_stream << " " << min_vsg_yy << " " << max_vsg_yy << " " << avg_vsg_yy << " " << std_dev_vsg_yy;
        scalar_t strain_peaks_avg_error_x = 0.0;
        scalar_t strain_peaks_std_dev_error_x = 0.0;
        scalar_t strain_peaks_avg_error_y = 0.0;
        scalar_t strain_peaks_std_dev_error_y = 0.0;
        // assemble the strains into a vector
        Teuchos::RCP<MultiField_Map> map = mesh_->get_vector_node_dist_map();
        Teuchos::RCP<MultiField> strain = Teuchos::rcp( new MultiField(map,1,true));
        Teuchos::RCP<MultiField> exact_strain = Teuchos::rcp( new MultiField(map,1,true));
        Teuchos::RCP<MultiField> strain_error = Teuchos::rcp( new MultiField(map,1,true));
        for(int_t i=0;i<local_num_subsets_;++i){
          strain->local_value(i*spa_dim+0) = vsg_xx->local_value(i);
          strain->local_value(i*spa_dim+1) = vsg_yy->local_value(i);
          exact_strain->local_value(i*spa_dim+0) = exact_strain_xx->local_value(i);
          exact_strain->local_value(i*spa_dim+1) = exact_strain_yy->local_value(i);
          strain_error->local_value(i*spa_dim+0) = vsg_error_xx->local_value(i);
          strain_error->local_value(i*spa_dim+1) = vsg_error_yy->local_value(i);
        }
        // analyze the peaks of the output to evaluate the roll off
        compute_roll_off_stats(period,full_ref_img_width_,full_ref_img_height_,coords,strain,exact_strain,strain_error,
          strain_peaks_avg_error_x,strain_peaks_std_dev_error_x,strain_peaks_avg_error_y,strain_peaks_std_dev_error_y);
        result_stream << " " << strain_peaks_avg_error_x << " " << strain_peaks_std_dev_error_x << " " << strain_peaks_avg_error_y << " " << strain_peaks_std_dev_error_y;
      }
      if(has_nlvc){
        scalar_t min_nlvc_xx = 0.0;
        scalar_t max_nlvc_xx = 0.0;
        scalar_t avg_nlvc_xx = 0.0;
        scalar_t std_dev_nlvc_xx = 0.0;
        scalar_t min_nlvc_xy = 0.0;
        scalar_t max_nlvc_xy = 0.0;
        scalar_t avg_nlvc_xy = 0.0;
        scalar_t std_dev_nlvc_xy = 0.0;
        scalar_t min_nlvc_yy = 0.0;
        scalar_t max_nlvc_yy = 0.0;
        scalar_t avg_nlvc_yy = 0.0;
        scalar_t std_dev_nlvc_yy = 0.0;
        mesh_->field_stats(NLVC_STRAIN_XX_ERROR_FS,min_nlvc_xx,max_nlvc_xx,avg_nlvc_xx,std_dev_nlvc_xx,0,SIGMA_FS,-1.0);
        mesh_->field_stats(NLVC_STRAIN_XY_ERROR_FS,min_nlvc_xy,max_nlvc_xy,avg_nlvc_xy,std_dev_nlvc_xy,0,SIGMA_FS,-1.0);
        mesh_->field_stats(NLVC_STRAIN_YY_ERROR_FS,min_nlvc_yy,max_nlvc_yy,avg_nlvc_yy,std_dev_nlvc_yy,0,SIGMA_FS,-1.0);
        result_stream << " " << min_nlvc_xx << " " << max_nlvc_xx << " " << avg_nlvc_xx << " " << std_dev_nlvc_xx;
        result_stream << " " << min_nlvc_xy << " " << max_nlvc_xy << " " << avg_nlvc_xy << " " << std_dev_nlvc_xy;
        result_stream << " " << min_nlvc_yy << " " << max_nlvc_yy << " " << avg_nlvc_yy << " " << std_dev_nlvc_yy;
        scalar_t strain_peaks_avg_error_x = 0.0;
        scalar_t strain_peaks_std_dev_error_x = 0.0;
        scalar_t strain_peaks_avg_error_y = 0.0;
        scalar_t strain_peaks_std_dev_error_y = 0.0;
        // assemble the strains into a vector
        Teuchos::RCP<MultiField_Map> map = mesh_->get_vector_node_dist_map();
        Teuchos::RCP<MultiField> strain = Teuchos::rcp( new MultiField(map,1,true));
        Teuchos::RCP<MultiField> exact_strain = Teuchos::rcp( new MultiField(map,1,true));
        Teuchos::RCP<MultiField> strain_error = Teuchos::rcp( new MultiField(map,1,true));
        for(int_t i=0;i<local_num_subsets_;++i){
          strain->local_value(i*spa_dim+0) = vsg_xx->local_value(i);
          strain->local_value(i*spa_dim+1) = vsg_yy->local_value(i);
          exact_strain->local_value(i*spa_dim+0) = exact_strain_xx->local_value(i);
          exact_strain->local_value(i*spa_dim+1) = exact_strain_yy->local_value(i);
          strain_error->local_value(i*spa_dim+0) = nlvc_error_xx->local_value(i);
          strain_error->local_value(i*spa_dim+1) = nlvc_error_yy->local_value(i);
        }
        // analyze the peaks of the output to evaluate the roll off
        compute_roll_off_stats(period,full_ref_img_width_,full_ref_img_height_,coords,strain,exact_strain,strain_error,
          strain_peaks_avg_error_x,strain_peaks_std_dev_error_x,strain_peaks_avg_error_y,strain_peaks_std_dev_error_y);
        result_stream << " " << strain_peaks_avg_error_x << " " << strain_peaks_std_dev_error_x << " " << strain_peaks_avg_error_y << " " << strain_peaks_std_dev_error_y;
      }

      result_stream << std::endl;
      write_output(output_folder,prefix,false,true);
      // write the results to the .info file
      if(proc_id==0){
        std::FILE * infoFilePtr = fopen(data_name.str().c_str(),"a");
        fprintf(infoFilePtr,"%s",result_stream.str().c_str());
        fclose(infoFilePtr);
        *outStream << result_stream.str();
      }
      result_stream.clear();
      result_stream.str("");
    } // end step loop
  } // end mag loop
}

/// correlate point by point branching out by neighbors
void
Schema::space_fill_correlate(const int_t seed_gid,
  const std::vector<int_t> & in_gids,
  std::vector<int_t> & out_gids,
  const int_t num_neigh,
  Teuchos::RCP<kd_tree_2d_t> kd_tree,
  const scalar_t & epi_error_tol,
  Teuchos::RCP<Triangulation> tri){

  TEUCHOS_TEST_FOR_EXCEPTION(in_gids.size()==0,std::runtime_error,"");
  out_gids.clear();

  for(size_t s=0;s<in_gids.size();++s){
    const int_t subset_gid = in_gids[s];
    DEBUG_MSG("Schema::space_fill_correlate: spreading to neighbors of SUBSET " << subset_gid);
    // get the u and v values for this subset
    const scalar_t subset_u = global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS);
    const scalar_t subset_v = global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS);
    // find neighbors of this subset
    std::vector<size_t> ret_index(num_neigh);
    std::vector<scalar_t> out_dist_sqr(num_neigh);
    scalar_t query_pt[2];
    query_pt[0] = global_field_value(subset_gid,SUBSET_COORDINATES_X_FS);
    query_pt[1] = global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS);
    kd_tree->knnSearch(&query_pt[0],num_neigh,&ret_index[0],&out_dist_sqr[0]);
    for(int_t i=0;i<num_neigh;++i){
      const int_t neigh_id = ret_index[i];
      const int_t neigh_gid = subset_global_id(neigh_id);
      if(local_field_value(neigh_id,NEIGHBOR_ID_FS)!=0) continue;
      DEBUG_MSG("Schema::space_fill_correlate: neigh global id " << neigh_gid <<
        " cx " << local_field_value(neigh_id,SUBSET_COORDINATES_X_FS) << " cy " << local_field_value(neigh_id,SUBSET_COORDINATES_Y_FS) <<
        " dist " << std::sqrt(out_dist_sqr[i]));
      local_field_value(neigh_id,FIELD_10_FS) = seed_gid;
      local_field_value(neigh_id,NEIGHBOR_ID_FS) = -1;

      // correlate
      int_t num_iterations = -1;
      Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(this,this_proc_gid_order_[neigh_id]));
      Teuchos::RCP<Local_Shape_Function> neigh_shape_function = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
      neigh_shape_function->insert_motion(subset_u,subset_v);
      Status_Flag corr_status = obj->computeUpdateFast(neigh_shape_function,num_iterations);

      // check the sigma values and status flag
      scalar_t noise_std_dev = 0.0;
      const scalar_t cross_sigma = obj->sigma(neigh_shape_function,noise_std_dev);
      local_field_value(neigh_id,SIGMA_FS) = cross_sigma;
      if(cross_sigma<0.0){
        DEBUG_MSG("Schema::space_fill_correlate(): failed cross init sigma");
        continue;
      }
      // check correlation was successful
      if(corr_status!=CORRELATION_SUCCESSFUL){
        local_field_value(neigh_id,SIGMA_FS) = -1.0;
        DEBUG_MSG("Schema::space_fill_correlate(): failed correlation");
        continue;
      }
      // check epipolar error
      scalar_t cross_t = 0.0, cross_u = 0.0, cross_v = 0.0;
      neigh_shape_function->map_to_u_v_theta(local_field_value(neigh_id,SUBSET_COORDINATES_X_FS),
        local_field_value(neigh_id,SUBSET_COORDINATES_Y_FS),cross_u,cross_v,cross_t);
      const float a = local_field_value(neigh_id,EPI_A_FS);
      const float b = local_field_value(neigh_id,EPI_B_FS);
      const float c = local_field_value(neigh_id,EPI_C_FS);
      const float stereo_x = local_field_value(neigh_id,SUBSET_COORDINATES_X_FS) + cross_u;
      const float stereo_y = local_field_value(neigh_id,SUBSET_COORDINATES_Y_FS) + cross_v;
      std::vector<scalar_t> sx(1,stereo_x);
      std::vector<scalar_t> sy(1,stereo_y);
      tri->undistort_points(sx,sy,1); // right camera
      const float dist = (std::abs(a*sx[0]+b*sy[0]+c)/std::sqrt(a*a+b*b));
      DEBUG_MSG("Schema::space_fill_correlate(): epipolar error for neighbor " << i << " gid " << neigh_gid << ": " << dist);
      if(dist>epi_error_tol){
        local_field_value(neigh_id,SIGMA_FS) = -1.0;
        DEBUG_MSG("Schema::space_fill_correlate(): subset " << neigh_gid << " failed epipolar distance threshold, epipolar dist: " << dist);
        continue;
      }
      // assume success at this point, set the field values
      local_field_value(neigh_id,SUBSET_DISPLACEMENT_X_FS) = cross_u;
      local_field_value(neigh_id,SUBSET_DISPLACEMENT_Y_FS) = cross_v;
      local_field_value(neigh_id,NEIGHBOR_ID_FS) = subset_gid;

      out_gids.push_back(neigh_gid);
    }
  }
}

bool
Schema::shape_function_params_valid(Teuchos::RCP<Local_Shape_Function> shape_function,
    Teuchos::RCP<Objective> obj,
    const scalar_t & gamma_tol,
    const scalar_t & u_tol,
    const scalar_t & v_tol,
    const scalar_t & t_tol,
    const Status_Flag corr_status,
    scalar_t & return_gamma,
    scalar_t & return_sigma)const{

    // check the sigma values and status flag
    scalar_t noise_std_dev = 0.0;
    return_sigma = obj->sigma(shape_function,noise_std_dev);
    if(return_sigma<0.0){
      DEBUG_MSG("Schema::shape_function_params_valid(): fails sigma check");
      return false;
    }
    if(corr_status!=CORRELATION_SUCCESSFUL){
      DEBUG_MSG("Schema::shape_function_params_valid(): fails correlation status");
      return false;
    }
    return_gamma = obj->gamma(shape_function);
    if(std::abs(return_gamma)>gamma_tol){ // fails the similarity constraint
      DEBUG_MSG("Schema::shape_function_params_valid(): fails gamma check");
      return false;
    }
    scalar_t cross_u = 0.0, cross_v = 0.0, cross_t = 0.0;
    shape_function->map_to_u_v_theta(obj->subset()->centroid_x(),obj->subset()->centroid_y(),cross_u,cross_v,cross_t);
    if(u_tol>0.0){
      if(std::abs(cross_u)>u_tol){
        DEBUG_MSG("Schema::shape_function_params_valid(): fails u magnitude check");
        return false;
      }
    }
    if(v_tol>0.0){
      if(std::abs(cross_v)>v_tol){
        DEBUG_MSG("Schema::shape_function_params_valid(): fails v magnitude check");
        return false;
      }
    }
    if(t_tol>0.0){
      if(std::abs(cross_t)>t_tol){
        DEBUG_MSG("Schema::shape_function_params_valid(): fails theta magnitude check");
        return false;
      }
    }
    // assume success if you got here
    return true;
}

void
Schema::march_from_neighbor_init(const Direction & neighbor_direction,
    const scalar_t & step,
    Teuchos::RCP<Schema> schema,
    const scalar_t & gamma_tol,
    const scalar_t & v_tol,
    cv::Mat & debug_img,
    const std::string & debug_img_name,
    const int_t color,
    const int_t dot_size){

  const int_t num_cols = (schema->ref_img()->width() - 2*schema->subset_dim())/schema->step_size_x() + 1;
  const int_t num_rows = (schema->ref_img()->height() - 2*schema->subset_dim())/schema->step_size_y() + 1;
  assert(step>0.0);
  const scalar_t sssig_threshold = 150.0; // fixed for now, could add to input
  int_t num_iterations = -1;
  int_t dir = 1; // -1 based on direction DOWN and RIGHT reverse iterate
  int_t id_start = 0;
  if(neighbor_direction==DOWN||neighbor_direction==RIGHT){
    dir = -1;
    id_start = schema->local_num_subsets() - 1;
  }
  int_t neigh_increment = -1; // based on direction LEFT
  if(neighbor_direction==RIGHT)
    neigh_increment = 1;
  else if(neighbor_direction==UP)
    neigh_increment = -num_cols;
  else if(neighbor_direction==DOWN)
    neigh_increment = num_cols;

  for(int_t local_id = id_start; 0 <= local_id && local_id < schema->local_num_subsets(); local_id+= dir){
    const int_t row = local_id/num_cols;
    const int_t col = local_id%num_cols;
    if(row==0||row>=num_rows-1||col==0||col>=num_cols-1)continue; // skip the boarders
    const int_t neigh_id = local_id + neigh_increment;
    if(schema->local_field_value(local_id,SSSIG_FS) < sssig_threshold) continue; // not enough gradient information
    if(schema->local_field_value(local_id,SIGMA_FS) > 0.0) continue; // positive sigma means it converged in a previous step
    if(schema->local_field_value(neigh_id,SIGMA_FS) < 0.0) continue; // bad neigh
    const int_t my_x = schema->local_field_value(local_id,SUBSET_COORDINATES_X_FS);
    const int_t my_y = schema->local_field_value(local_id,SUBSET_COORDINATES_Y_FS);
    const int_t neigh_x = schema->local_field_value(neigh_id,SUBSET_COORDINATES_X_FS);
    const int_t neigh_y = schema->local_field_value(neigh_id,SUBSET_COORDINATES_Y_FS);
    const scalar_t step_x = neigh_x > my_x ? -1.0 * step : step;
    const scalar_t step_y = neigh_y > my_y ? -1.0 * step : step;
    const int_t num_steps_x = std::abs(neigh_x - my_x)/step;
    const int_t num_steps_y = std::abs(neigh_y - my_y)/step;
    DEBUG_MSG("Schema::march_from_neighbor_init(): initializing subset " << local_id << " at "  << my_x <<
        " " << my_y << " based on neighbor " << neigh_id << " at " << neigh_x << " " << neigh_y);

    // use the neighbor's displacement as a starting point
    Teuchos::RCP<Local_Shape_Function> sf = shape_function_factory(schema.get());
    for(int_t i=0;i<sf->num_params();++i)
      (*sf)(i) = schema->local_field_value(neigh_id,sf->field_spec(i));

    sf->print_parameters();

    // try incremental correlations to get to the subset location
    scalar_t subset_x = neigh_x;
    scalar_t subset_y = neigh_y;
    scalar_t cross_gamma = 0.0, cross_sigma = 0.0;
    Status_Flag corr_status;
    Teuchos::RCP<Objective> obj;
    std::vector<scalar_t> save_params(sf->num_params(),0.0);
    for(int_t ix=0;ix<=num_steps_x;++ix){
      subset_x = neigh_x + ix*step_x;
      for(int_t iy=0;iy<=num_steps_y;++iy){
        subset_y = neigh_y + iy*step_y;
        // save off the current params in case this step fails
        for(int_t i=0;i<sf->num_params();++i)
          save_params[i] = (*sf)(i);
        // needs a new objective every time since the reference intensities need to be updated
        num_iterations = -1;
        obj = Teuchos::rcp(new Objective_ZNSSD(schema.get(),subset_x,subset_y));
        corr_status = obj->computeUpdateFast(sf,num_iterations);
        // catch the case that this increment failed and reset the initial guess
        if(!shape_function_params_valid(sf,obj,gamma_tol,-1.0,v_tol,-1.0,corr_status,cross_gamma,cross_sigma)){
          DEBUG_MSG("Schema::march_from_neighbor_init(): failed marching step skipping this intermediate step ");
          for(int_t i=0;i<sf->num_params();++i)
            (*sf)(i) = save_params[i];
        }
      }
    }
    if(my_x!=subset_x||my_y!=subset_y){ // in case the step size isn't an even split of the gap from the neighbor
      num_iterations = -1;
      obj = Teuchos::rcp(new Objective_ZNSSD(schema.get(),my_x,my_y));
      corr_status = obj->computeUpdateFast(sf,num_iterations);
    }
    if(shape_function_params_valid(sf,obj,gamma_tol,-1.0,v_tol,-1.0,corr_status,cross_gamma,cross_sigma)){
      schema->record_step(schema->subset_global_id(local_id),sf,cross_sigma,1,cross_gamma,0.0,0.0,0.0,0,corr_status,num_iterations);
      cv::Point pt1(my_x,my_y);
      cv::circle(debug_img, pt1, dot_size, cv::Scalar(color),3);
    }else{
      DEBUG_MSG("Schema::march_from_neighbor_init(): failed correlation at local id");
      schema->record_failed_step(schema->subset_global_id(local_id),corr_status,num_iterations);
    }
  }
  cv::imwrite(debug_img_name,debug_img);
}

int_t
Schema::initialize_cross_correlation(Teuchos::RCP<Triangulation> tri,
  const Teuchos::RCP<Teuchos::ParameterList> & input_params){
  DEBUG_MSG("Schema::initialize_cross_correlation(): cross init type " << to_string(cross_initialization_method_));

  const int_t proc_rank = comm_->get_rank();

  // compute the epipolar coeffs for each subset
  DEBUG_MSG("Schema::initialize_cross_correlation(): computing subset epipolar coefficients");
  cv::Mat F = tri->fundamental_matrix();
  cv::Mat lines;

  std::vector<scalar_t> pointsx(local_num_subsets_,0.0);
  std::vector<scalar_t> pointsy(local_num_subsets_,0.0);
  for(int_t i=0;i<local_num_subsets_;++i){
    pointsx[i] = local_field_value(i,SUBSET_COORDINATES_X_FS);
    pointsy[i] = local_field_value(i,SUBSET_COORDINATES_Y_FS);
  } // get rid of lens distortions for the epipolar constraint
  tri->undistort_points(pointsx,pointsy,0);// camera 0 since this is for the left camera
  std::vector<cv::Point2f> points(local_num_subsets_);
  for(int_t i=0;i<local_num_subsets_;++i)
    points[i] = cv::Point2f(pointsx[i],pointsy[i]);
  cv::computeCorrespondEpilines(points,1,F,lines); // epipolar lines for the subset centroids
  assert((int)points.size()==lines.rows);
  for(int_t i=0;i<local_num_subsets_;++i){
    local_field_value(i,EPI_A_FS) = lines.at<float>(i,0);
    local_field_value(i,EPI_B_FS) = lines.at<float>(i,1);
    local_field_value(i,EPI_C_FS) = lines.at<float>(i,2);
  }

  // if you are processor 0 load the ref and def images and call the estimate routine
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::gauss_filter_images,gauss_filter_images_);
  imgParams->set(DICe::gauss_filter_mask_size,gauss_filter_mask_size_);
  std::string left_image_string;
  std::string right_image_string;
  if(proc_rank==0){
    std::vector<std::string> image_files;
    std::vector<std::string> stereo_image_files;
    // decypher the image names from the input files
    int_t frame_id_start=0,num_frames=1,frame_skip=1;
    DICe::decipher_image_file_names(input_params,image_files,stereo_image_files,frame_id_start,num_frames,frame_skip);
    left_image_string = image_files[0];
    right_image_string = stereo_image_files[0];
    DEBUG_MSG("Schema::initialize_cross_correlation(): left file " << left_image_string);
    DEBUG_MSG("Schema::initialize_cross_correlation(): right file " << right_image_string);
  }

  if(cross_initialization_method_==USE_RECTIFIED_CORRESPONDENCES){
	  DEBUG_MSG("Schema::initialize_cross_correlation(): using correspondences from rectified images to initialize");

	  // create rectified images and store them in the .dice folder
	  cv::Mat left_img = cv::imread(left_image_string, cv::ImreadModes::IMREAD_GRAYSCALE);
	  cv::Mat right_img = cv::imread(right_image_string, cv::ImreadModes::IMREAD_GRAYSCALE);
	  cv::Mat M1 = tri->camera_matrix(0);
	  cv::Mat M2 = tri->camera_matrix(1);
	  cv::Mat D1 = tri->distortion_matrix(0);
	  cv::Mat D2 = tri->distortion_matrix(1);
	  cv::Mat R  = tri->rotation_matrix();
	  cv::Mat T  = tri->translation_matrix();
	  cv::Mat R1, R2, P1, P2, map11, map12, map21, map22;
	  cv::stereoRectify(M1, D1, M2, D2, left_img.size(), R, T, R1, R2, P1, P2, cv::noArray(), 0);
	  cv::initUndistortRectifyMap(M1, D1, R1, P1, left_img.size(),  CV_16SC2, map11, map12);
	  cv::initUndistortRectifyMap(M2, D2, R2, P2, right_img.size(), CV_16SC2, map21, map22);
	  cv::Mat left_img_rmp, right_img_rmp;
	  cv::remap(left_img, left_img_rmp, map11, map12, cv::INTER_LINEAR);
	  cv::remap(right_img, right_img_rmp, map21, map22, cv::INTER_LINEAR);

    create_directory(".dice");
	  std::string outname_left = ".dice/left_rectified.png";
	  std::string outname_right = ".dice/right_rectified.png";
    cv::imwrite(outname_left,left_img_rmp);
    cv::imwrite(outname_right,right_img_rmp);
    // update the left and right images for stereo cross-correlation
    set_ref_image(outname_left);
    set_def_image(outname_right);

	  // do a feature matching on the rectified images

    float feature_epi_dist_tol = 0.5f;
    float epi_dist_tol = 0.5f;
    if(tri->avg_epipolar_error()!=0.0){
      feature_epi_dist_tol = 3.0*tri->avg_epipolar_error();
      epi_dist_tol = 3.0*tri->avg_epipolar_error();
    }
    DEBUG_MSG("Schema::initialize_cross_correlation(): feature epi dist tol: " << feature_epi_dist_tol);
    DEBUG_MSG("Schema::initialize_cross_correlation(): epi dist tol: " << epi_dist_tol);
    Teuchos::RCP<DICe::Image> left_image = Teuchos::rcp(new Image(outname_left.c_str(),imgParams));
    Teuchos::RCP<DICe::Image> right_image = Teuchos::rcp(new Image(outname_right.c_str(),imgParams));

    DEBUG_MSG("Schema::initialize_cross_correlation(): begin matching features on processor 0");
    float feature_tol = 0.005f;
    std::vector<scalar_t> left_x, right_x, left_y, right_y;
    std::vector<scalar_t> good_left_x, good_right_x, good_left_y, good_right_y, good_u;
    match_features(left_image,right_image,left_x,left_y,right_x,right_y,feature_tol,".dice/fm_rectified_corr_0p005.png");
    if(left_x.size() < 10){
      DEBUG_MSG("Schema::initialize_cross_correlation(): initial attempt failed with tol = 0.005f, setting to 0.001f and trying again.");
      feature_tol = 0.001f;
      match_features(left_image,right_image,left_x,left_y,right_x,right_y,feature_tol,".dice/fm_rectified_corr_0p001.png");
      if(left_x.size()<10){
        DEBUG_MSG("Schema::initialize_cross_correlation(): initial attempt failed with tol = 0.001f, equalizing histogram and trying again.");
        match_features(left_image,right_image,left_x,left_y,right_x,right_y,feature_tol,".dice/fm_rectified_corr_0p001_eq.png",7);
      }
    }
    if(left_x.size()<1){
      std::cout << "Schema::initialize_cross_correlation(): error: feature matching failed" << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }
    DEBUG_MSG("Schema::initialize_cross_correlation(): matching features complete");
    // Weed out any matched features that aren't on the epipolar line
    good_left_x.reserve(left_x.size());
    good_left_y.reserve(left_x.size());
    good_right_x.reserve(left_x.size());
    good_right_y.reserve(left_x.size());
    good_u.reserve(left_x.size());
    for(size_t i=0;i<left_x.size();++i){
      // epipolar lines should be straight so any slanted pairs are bad ones, only need to check y diff
      const float dist_sq = (right_y[i] - left_y[i])*(right_y[i] - left_y[i]);
      DEBUG_MSG("fm point " << i << " xl (" << left_x[i] << ") yl (" << left_y[i] << ") xr (" << right_x[i] << ") yr (" << right_y[i] << ") dist from epiline " << std::sqrt(dist_sq));
      if(dist_sq<feature_epi_dist_tol){
        good_left_x.push_back(left_x[i]);
        good_left_y.push_back(left_y[i]);
        good_right_x.push_back(right_x[i]);
        good_right_y.push_back(right_y[i]);
        good_u.push_back(right_x[i] - left_x[i]);
      }
    }
    DEBUG_MSG("Schema::initialize_cross_correlation(): num good matches: " << good_left_x.size());
    if(good_left_x.size()<1){
      std::cout << "Schema::initialize_cross_correlation(): error: feature matching failed (not enough matches)" << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }

    // set up a point cloud
    DEBUG_MSG("creating the point cloud using nanoflann");
    Point_Cloud_2D<scalar_t> point_cloud(good_left_x,good_left_y);//_ = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
    const int_t num_neigh = 7;

    // create a schema for the course grid that will be used to initialize the subsets
    Teuchos::RCP<Teuchos::ParameterList> schema_params = rcp(new Teuchos::ParameterList());
    schema_params->set(DICe::compute_def_gradients,true);
    schema_params->set(DICe::shape_function_type,DICe::AFFINE_SF);
    schema_params->set(DICe::enable_translation,true);
    schema_params->set(DICe::enable_rotation,true);
    schema_params->set(DICe::enable_shear_strain,true);
    schema_params->set(DICe::enable_normal_strain,true);
    schema_params->set(DICe::max_solver_iterations_fast,250);
    const scalar_t sssig_threshold = 150.0; // fixed for now, could add to input
    const int_t step_size = 31; // also could add as input param
    const int_t subset_size = 31;
    const scalar_t v_tol = 1.0; // if the y displacement is greater than v_tol it violates the epipolar constraint so it can't be a good match
    const scalar_t gamma_tol = 0.5; // test of how similar the converged solution makes the ref and def subset intensity values
    Teuchos::RCP<DICe::Schema> rect_schema = Teuchos::rcp(new DICe::Schema(left_image->width(),left_image->height(),step_size,step_size,subset_size,schema_params));
    rect_schema->set_ref_image(outname_left);
    rect_schema->set_def_image(outname_right);
    // use sigma to track which subsets successfully correlated
    rect_schema->mesh()->get_field(SIGMA_FS)->put_scalar(0.0);
    Teuchos::RCP<Local_Shape_Function> sf = shape_function_factory(rect_schema.get());//Teuchos::rcp(new Affine_Shape_Function(true,true,true));
    scalar_t cross_sigma = 0.0;
    scalar_t cross_gamma = 0.0;

    // do a check of the sssig level for each subset

    TEUCHOS_TEST_FOR_EXCEPTION(!rect_schema->ref_img()->has_gradients(),std::runtime_error,
      "Error, testing valid points for SSSIG tol, but image gradients have not been computed");
    for(int_t local_id=0;local_id<rect_schema->local_num_subsets();++local_id){
      const int_t cx = rect_schema->local_field_value(local_id,SUBSET_COORDINATES_X_FS);
      const int_t cy = rect_schema->local_field_value(local_id,SUBSET_COORDINATES_Y_FS);
      Subset sssig_subset(cx,cy,subset_size,subset_size);
      sssig_subset.initialize(rect_schema->ref_img());
      const scalar_t SSSIG = sssig_subset.sssig();
      rect_schema->local_field_value(local_id,SSSIG_FS) = SSSIG;
      if(SSSIG < sssig_threshold){
        rect_schema->local_field_value(local_id,SIGMA_FS) = -1.0;
        cv::Point pt1(cx,cy);
        cv::circle(left_img_rmp, pt1, 10, cv::Scalar(100),3);
        std::stringstream text_label;
        text_label << local_id;
      }
    } // end subset sssig check loop

    // loop over the whole domain attempting to initialize the subsets based on a least squares fit of feature matched points

    const int_t num_loops = 3;
    for(int_t loop=0;loop<num_loops; ++loop){

      for(int_t local_id=0;local_id<rect_schema->local_num_subsets();++local_id){
        if(rect_schema->local_field_value(local_id,SSSIG_FS) < sssig_threshold) continue; // not enough gradient information
        if(rect_schema->local_field_value(local_id,SIGMA_FS) > 0.0) continue; // positive sigma means it converged in a previous step
        const scalar_t x = rect_schema->local_field_value(local_id,SUBSET_COORDINATES_X_FS);
        const scalar_t y = rect_schema->local_field_value(local_id,SUBSET_COORDINATES_Y_FS);
        const scalar_t init_u = point_cloud.knn_least_squares(x,y,num_neigh, good_u);
        DEBUG_MSG("Schema::initialize_cross_correlation(): initial guess for subset " << local_id << " at "  << x <<  " " << y << " is " << init_u);
        sf->clear();
        sf->insert_motion(init_u,0.0);
        Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(rect_schema.get(),rect_schema->subset_global_id(local_id)));
        int_t num_iterations = -1;
        Status_Flag corr_status = obj->computeUpdateFast(sf,num_iterations);
        if(shape_function_params_valid(sf,obj,gamma_tol,-1.0,v_tol,-1.0,corr_status,cross_gamma,cross_sigma)){          // assume success at this point
          rect_schema->record_step(obj->correlation_point_global_id(),sf,cross_sigma,1,cross_gamma,0.0,0.0,0.0,0,corr_status,num_iterations);
          point_cloud.add_point(x, y);
          good_u.push_back(rect_schema->local_field_value(local_id,SUBSET_DISPLACEMENT_X_FS));
          cv::Point pt1(x,y);
          cv::circle(left_img_rmp, pt1, 10, cv::Scalar(0),3);
        }else{
          rect_schema->record_failed_step(obj->correlation_point_global_id(),corr_status,num_iterations);
          continue;
        }
      }
      // output image for debugging with iteration number in filename
      std::stringstream debug_name;
      debug_name << "initial_pass_"<< loop << ".png";
      cv::imwrite(debug_name.str(),left_img_rmp);
    } // end loop over feature matching attempts

    // top down marching pixel by pixel from neighbor

    const scalar_t marching_step_size = 2.0;
    march_from_neighbor_init(UP,marching_step_size,rect_schema,gamma_tol,v_tol,left_img_rmp,"neigh_march_top.png",255,10);
    march_from_neighbor_init(LEFT,marching_step_size,rect_schema,gamma_tol,v_tol,left_img_rmp,"neigh_march_left.png",225,10);
    march_from_neighbor_init(RIGHT,marching_step_size,rect_schema,gamma_tol,v_tol,left_img_rmp,"neigh_march_right.png",200,10);
    march_from_neighbor_init(DOWN,marching_step_size,rect_schema,gamma_tol,v_tol,left_img_rmp,"neigh_march_bottom.png",175,10);
    march_from_neighbor_init(UP,marching_step_size,rect_schema,gamma_tol,v_tol,left_img_rmp,"neigh_march_top_2.png",255,8);
    march_from_neighbor_init(RIGHT,marching_step_size,rect_schema,gamma_tol,v_tol,left_img_rmp,"neigh_march_right_2.png",225,8);

    std::FILE * disparityFilePtr = fopen("final_disparity.txt","w");
    for(int_t local_id = 0; local_id < rect_schema->local_num_subsets(); ++local_id){
      if(rect_schema->local_field_value(local_id, SIGMA_FS)<0.0) continue;
      fprintf(disparityFilePtr,"%f %f %f %f %f\n",
          rect_schema->local_field_value(local_id,SUBSET_COORDINATES_X_FS),
          rect_schema->local_field_value(local_id,SUBSET_COORDINATES_Y_FS),
          rect_schema->local_field_value(local_id,SUBSET_DISPLACEMENT_X_FS),
          rect_schema->local_field_value(local_id,ITERATIONS_FS),
          rect_schema->local_field_value(local_id,GAMMA_FS));
    }
    fclose(disparityFilePtr);

    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Implementation not completed yet");

    // TODO fill in the gaps between subsets

    // TODO compute the un-rectified q and r fields

    DEBUG_MSG("Schema::initialize_cross_correlation(): rectified correspondences successful");
    return 0;
  }

  if(cross_initialization_method_==USE_SPACE_FILLING_ITERATIONS){
    DEBUG_MSG("Schema::initialize_cross_correlation(): using feature matching and space filling iterations to initialize");
    float feature_epi_dist_tol = 0.5f;
    float epi_dist_tol = 0.5f;
    if(tri->avg_epipolar_error()!=0.0){
      feature_epi_dist_tol = 3.0*tri->avg_epipolar_error();
      epi_dist_tol = 3.0*tri->avg_epipolar_error();
    }
    DEBUG_MSG("Schema::initialize_cross_correlation(): feature epi dist tol: " << feature_epi_dist_tol);
    DEBUG_MSG("Schema::initialize_cross_correlation(): epi dist tol: " << epi_dist_tol);
    DEBUG_MSG("Schema::initialize_cross_correlation(): done computing subset epipolar coefficients");
    Teuchos::RCP<DICe::Image> left_img = Teuchos::rcp(new Image(left_image_string.c_str(),imgParams));
    Teuchos::RCP<DICe::Image> right_img = Teuchos::rcp(new Image(right_image_string.c_str(),imgParams));

    // use feature matching to get candidates for seed points
    // Try several things to get some features to match in the cross correlation:
    // First do a normal feature matching with low tolerance
    // Then try with a larger tol
    // Then try equalizing the histogram and decreasing the tolerance again

    DEBUG_MSG("Schema::initialize_cross_correlation(): begin matching features on processor 0");
    float feature_tol = 0.005f;
    std::vector<scalar_t> left_x, right_x, left_y, right_y;
    std::vector<scalar_t> good_left_x, good_right_x, good_left_y, good_right_y;
    match_features(left_img,right_img,left_x,left_y,right_x,right_y,feature_tol,".dice/fm_space_filling.png");
//    if(left_x.size() < 5){
    if(left_x.size() < 10){
      DEBUG_MSG("Schema::initialize_cross_correlation(): initial attempt failed with tol = 0.005f, setting to 0.001f and trying again.");
      feature_tol = 0.001f;
      match_features(left_img,right_img,left_x,left_y,right_x,right_y,feature_tol,".dice/fm_space_filling.png");
      if(left_x.size()<10){
        DEBUG_MSG("Schema::initialize_cross_correlation(): initial attempt failed with tol = 0.001f, equalizing histogram and trying again.");
        match_features(left_img,right_img,left_x,left_y,right_x,right_y,feature_tol,".dice/fm_space_filling.png",7);
      }
    }
    if(left_x.size()<1){
      std::cout << "Schema::initialize_cross_correlation(): error: feature matching failed" << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }
    DEBUG_MSG("Schema::initialize_cross_correlation(): matching features complete");
    // Weed out any matched features that aren't on the epipolar line

    cv::Mat featureLines;
    std::vector<cv::Point2f> featurePoints(left_x.size());
    good_left_x.reserve(left_x.size());
    good_left_y.reserve(left_x.size());
    good_right_x.reserve(left_x.size());
    good_right_y.reserve(left_x.size());
    std::vector<scalar_t> left_x_undist = left_x;
    std::vector<scalar_t> right_x_undist = right_x;
    std::vector<scalar_t> left_y_undist = left_y;
    std::vector<scalar_t> right_y_undist = right_y;

    // get rid of lens distortions for the epipolar constraint
    tri->undistort_points(left_x_undist,left_y_undist,0);// camera 0 since this is for the left camera
    tri->undistort_points(right_x_undist,right_y_undist,1);
    for(size_t i=0;i<left_x.size();++i)
      featurePoints[i] = cv::Point2f(left_x_undist[i],left_y_undist[i]);
    cv::computeCorrespondEpilines(featurePoints,1,F,featureLines); // epipolar lines for the feature points
//    cv::Mat epi_img = cv::imread(".dice/fm_space_filling.png",cv::IMREAD_COLOR);
    assert((int)featurePoints.size()==featureLines.rows);
    std::vector<cv::Point2f> good_points;
    for(size_t i=0;i<featurePoints.size();++i){
      const float a = featureLines.at<float>(i,0);
      const float b = featureLines.at<float>(i,1);
      const float c = featureLines.at<float>(i,2);
      const float dist = (std::abs(a*right_x_undist[i]+b*right_y_undist[i]+c)/std::sqrt(a*a+b*b));
      DEBUG_MSG("fm point " << i << " xl (" << left_x[i] << ") " << left_x_undist[i] << " yl (" << left_y[i] << ") " << left_y_undist[i] <<
        " xr (" << right_x[i] << ") " << right_x_undist[i]  <<  " yr (" << right_y[i] << ") " << right_y_undist[i] << " dist from epiline " << dist);
      if(dist<feature_epi_dist_tol){
        good_left_x.push_back(left_x[i]);
        good_left_y.push_back(left_y[i]);
        good_right_x.push_back(right_x[i]);
        good_right_y.push_back(right_y[i]);
      }
    }
    // TODO Try removing this to see if it works now with the lens distortions accounted for in the epiline tol
    DEBUG_MSG("Schema::initialize_cross_correlation(): num good matches: " << good_left_x.size());
    if(good_left_x.size()<1){ // see if relaxing the tolerance helps
      feature_epi_dist_tol *= 5.0;
      epi_dist_tol *= 5.0;
      DEBUG_MSG("Schema::initialize_cross_correlation(): increasing epipolar tolerances" << good_left_x.size());
      DEBUG_MSG("Schema::initialize_cross_correlation(): feature epi dist tol: " << feature_epi_dist_tol);
      DEBUG_MSG("Schema::initialize_cross_correlation(): epi dist tol: " << epi_dist_tol);
      for(size_t i=0;i<featurePoints.size();++i){
        const float a = featureLines.at<float>(i,0);
        const float b = featureLines.at<float>(i,1);
        const float c = featureLines.at<float>(i,2);
        const float dist = (std::abs(a*right_x_undist[i]+b*right_y_undist[i]+c)/std::sqrt(a*a+b*b));
        DEBUG_MSG("fm point " << i << " xl (" << left_x[i] << ") " << left_x_undist[i] << " yl (" << left_y[i] << ") " << left_y_undist[i] <<
          " xr (" << right_x[i] << ") " << right_x_undist[i]  <<  " yr (" << right_y[i] << ") " << right_y_undist[i] << " dist from epiline " << dist);
        if(dist<feature_epi_dist_tol){
          good_left_x.push_back(left_x[i]);
          good_left_y.push_back(left_y[i]);
          good_right_x.push_back(right_x[i]);
          good_right_y.push_back(right_y[i]);
        }
      }
    }
    if(good_left_x.size()<1){
      std::cout << "Schema::initialize_cross_correlation(): error: feature matching failed (not enough matches)" << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }

    // build a point cloud
    // create neighborhood lists using nanoflann:
    DEBUG_MSG("Schema::initialize_cross_correlation(): creating the point cloud using nanoflann");
    Teuchos::RCP<Point_Cloud_2D<scalar_t> > pc = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
    pc->pts.resize(local_num_subsets_);
    for(int_t i=0;i<local_num_subsets_;++i){
      pc->pts[i].x = local_field_value(i,SUBSET_COORDINATES_X_FS);
      pc->pts[i].y = local_field_value(i,SUBSET_COORDINATES_Y_FS);
    }
    DEBUG_MSG("building the kd-tree");
    Teuchos::RCP<kd_tree_2d_t> kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/,*pc.get(),nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
    kd_tree->buildIndex();
    DEBUG_MSG("Schema::initialize_cross_correlation(): kd-tree completed");

    // turn on compute_def_gradients here
    bool orig_compute_def_gradients = compute_def_gradients_;
    compute_def_gradients_ = true;
    // update the left and right images for stereo cross-correlation
    set_ref_image(left_image_string);
    set_def_image(right_image_string);

    mesh_->get_field(NEIGHBOR_ID_FS)->put_scalar(0);

    // set the seed u and v values for the nearest subsets to each feature matched point and do a cross-correlation for each
    std::vector<size_t> ret_index(1);
    std::vector<scalar_t> out_dist_sqr(1);
    const scalar_t max_dist_from_fm_point = 50.0; // a seed subset needs to be within 200 px of this point to be used
    scalar_t query_pt[2];
    for(size_t i=0;i<good_left_x.size();++i){
      query_pt[0] = good_left_x[i];
      query_pt[1] = good_left_y[i];
      kd_tree->knnSearch(&query_pt[0],1,&ret_index[0],&out_dist_sqr[0]);
      if(out_dist_sqr[0]>max_dist_from_fm_point*max_dist_from_fm_point){
        DEBUG_MSG("fm matched point " << i << " x " << good_left_x[i] << " y " << good_left_y[i] << " was too far from any subsets to be used");
        continue;
      }
      const int_t local_id = ret_index[0];
      if(local_field_value(local_id,NEIGHBOR_ID_FS) != 0) continue; // seed subset can't be one that already has been correlated
      DEBUG_MSG("Schema::initialize_cross_correlation(): closest subset to fm match point " << good_left_x[i] << " " << good_left_y[i] <<
        " is local subset " << local_id << " at " << local_field_value(local_id,SUBSET_COORDINATES_X_FS) << " " << local_field_value(local_id,SUBSET_COORDINATES_Y_FS) );
      local_field_value(local_id,NEIGHBOR_ID_FS) = -1; // mark this seed as having been visited

      // set the u and v values for this point and correlate it
      const scalar_t u = good_right_x[i] - good_left_x[i];
      const scalar_t v = good_right_y[i] - good_left_y[i];
      DEBUG_MSG("Schema::initialize_cross_correlation(): seed SUBSET_ID " << subset_global_id(local_id) << " seed u " << u << " seed v " << v);

      // correlate
      int_t num_iterations = -1;
      Teuchos::RCP<Local_Shape_Function> shape_function = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
      Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(this,this_proc_gid_order_[local_id]));
      shape_function->insert_motion(u,v);
      Status_Flag corr_status = obj->computeUpdateFast(shape_function,num_iterations);

      // check the sigma values and status flag
      scalar_t noise_std_dev = 0.0;
      const scalar_t cross_sigma = obj->sigma(shape_function,noise_std_dev);
      local_field_value(local_id,SIGMA_FS) = cross_sigma;
      if(cross_sigma<0.0){
        DEBUG_MSG("Schema::initialize_cross_correlation(): failed cross init sigma");
        continue;
      }
      if(corr_status!=CORRELATION_SUCCESSFUL){
        local_field_value(local_id,SIGMA_FS) = -1.0;
        DEBUG_MSG("Schema::initialize_cross_correlation(): failed correlation");
        continue;
      }
      scalar_t cross_u = 0.0, cross_v = 0.0, cross_t = 0.0;
      shape_function->map_to_u_v_theta(local_field_value(local_id,SUBSET_COORDINATES_X_FS),
        local_field_value(local_id,SUBSET_COORDINATES_Y_FS),
        cross_u,cross_v,cross_t);
      // check the epipolar distance
      const float a = local_field_value(local_id,EPI_A_FS);
      const float b = local_field_value(local_id,EPI_B_FS);
      const float c = local_field_value(local_id,EPI_C_FS);
      const float stereo_x = local_field_value(local_id,SUBSET_COORDINATES_X_FS) + cross_u;
      const float stereo_y = local_field_value(local_id,SUBSET_COORDINATES_Y_FS) + cross_v;
      std::vector<scalar_t> sx(1,stereo_x);
      std::vector<scalar_t> sy(1,stereo_y);
      tri->undistort_points(sx,sy,1); // right camera
      const float dist = (std::abs(a*sx[0]+b*sy[0]+c)/std::sqrt(a*a+b*b));
      DEBUG_MSG("Schema::initialize_cross_correlation(): epipolar error for seed subset: " << dist);
      if(dist>epi_dist_tol){
        local_field_value(local_id,SIGMA_FS) = -1.0;
        DEBUG_MSG("Schema::initialize_cross_correlation(): failed epipolar distance threshold");
        continue;
      }
      local_field_value(local_id,SUBSET_DISPLACEMENT_X_FS) = cross_u;
      local_field_value(local_id,SUBSET_DISPLACEMENT_Y_FS) = cross_v;
      local_field_value(local_id,NEIGHBOR_ID_FS) = subset_global_id(local_id); // seeds get themselves as neighbors
      local_field_value(local_id,FIELD_10_FS) = subset_global_id(local_id);

      const int_t num_neigh = 7;
      const int_t max_iterations = 1000000;
      int_t it_count = 0;
      std::vector<int_t> in_gids, out_gids;
      in_gids.reserve(local_num_subsets_);
      out_gids.reserve(local_num_subsets_);
      in_gids.push_back(subset_global_id(local_id));
      while(in_gids.size()>0){
        TEUCHOS_TEST_FOR_EXCEPTION(it_count > max_iterations,std::runtime_error,
          "error: inifinte loop detected in space filling initializer"); // guard against infinite loop
        space_fill_correlate(subset_global_id(local_id),in_gids,out_gids,num_neigh,kd_tree,epi_dist_tol,tri);
        in_gids.clear();
        for(size_t i=0;i<out_gids.size();++i)
          in_gids.push_back(out_gids[i]);
        it_count++;
      } // end space filling loop for each seed
    } // end good candidate seed points

    // find any points that weren't reached by the space filling and turn them off
    for(int_t i=0;i<local_num_subsets_;++i){
      const float a = local_field_value(i,EPI_A_FS);
      const float b = local_field_value(i,EPI_B_FS);
      const float c = local_field_value(i,EPI_C_FS);
      const float stereo_x = local_field_value(i,SUBSET_COORDINATES_X_FS) + local_field_value(i,SUBSET_DISPLACEMENT_X_FS);
      const float stereo_y = local_field_value(i,SUBSET_COORDINATES_Y_FS) + local_field_value(i,SUBSET_DISPLACEMENT_Y_FS);
      std::vector<scalar_t> sx(1,stereo_x);
      std::vector<scalar_t> sy(1,stereo_y);
      tri->undistort_points(sx,sy,1); // right camera
      const float dist = (std::abs(a*sx[0]+b*sy[0]+c)/std::sqrt(a*a+b*b));
      local_field_value(i,CROSS_EPI_ERROR_FS) = dist;
//      DEBUG_MSG("Schema::initialize_cross_correlation(): epipolar error for subset: " << dist);
      if(dist>epi_dist_tol){
        local_field_value(i,SIGMA_FS) = -1.0;
//        DEBUG_MSG("Schema::initialize_cross_correlation(): failed epipolar distance threshold");
      }
    }


    // try to correlate the points that the space filling curve didn't get to by using the intersection
    // of a line between the nearest neighbors and the epipolar line as an initial guess
    for(int_t i=0;i<local_num_subsets_;++i){
      if(local_field_value(i,NEIGHBOR_ID_FS)>0) continue; // only iterate the points that weren't reached by the space filling
      DEBUG_MSG("Schema::initialize_cross_correlation(): looking for left and right neighbors of subset " << subset_global_id(i));
      int_t left_neigh_id = -1;
      int_t right_neigh_id = -1;
      // find the closes point to the left and to the right
      assert(step_size_x_>0.0);
      query_pt[0] = local_field_value(i,SUBSET_COORDINATES_X_FS) - step_size_x_;
      query_pt[1] = local_field_value(i,SUBSET_COORDINATES_Y_FS);
      while(query_pt[0]>5.0){
        // find the nearest neighbor to this point:
        kd_tree->knnSearch(&query_pt[0],1,&ret_index[0],&out_dist_sqr[0]);
        const int_t neigh_id  = ret_index[0];
        assert(neigh_id>=0&&neigh_id<local_num_subsets_);
        if(local_field_value(neigh_id,SIGMA_FS)>=0.0){
          left_neigh_id = neigh_id;
          break;
        }
        query_pt[0] = query_pt[0] - step_size_x_;
      }
      query_pt[0] = local_field_value(i,SUBSET_COORDINATES_X_FS) + step_size_x_;
      query_pt[1] = local_field_value(i,SUBSET_COORDINATES_Y_FS);
      while(query_pt[0]<=ref_img_->width()-5.0){
        // find the nearest neighbor to this point:
        kd_tree->knnSearch(&query_pt[0],1,&ret_index[0],&out_dist_sqr[0]);
        const int_t neigh_id  = ret_index[0];
        assert(neigh_id>=0&&neigh_id<local_num_subsets_);
        if(local_field_value(neigh_id,SIGMA_FS)>=0.0){
          right_neigh_id = neigh_id;
          break;
        }
        query_pt[0] = query_pt[0] + step_size_x_;
      }
      DEBUG_MSG("Schema::initialize_cross_correlation(): left neigh id " << left_neigh_id << " right neigh id " << right_neigh_id);

      if(left_neigh_id>=0&&right_neigh_id>=0&&left_neigh_id<local_num_subsets_&&right_neigh_id<local_num_subsets_){
        const float a = local_field_value(i,EPI_A_FS);
        const float b = local_field_value(i,EPI_B_FS);
        const float c = local_field_value(i,EPI_C_FS);
        if(b==0.0)continue;
        // compute the coeffs for the epipolar line
        const scalar_t A = -1.0*a/b;
        const scalar_t C = -1.0*c/b;
        // compute the coeffs for the line between the good points in right image space

        std::vector<scalar_t> neighx(2,0.0);
        std::vector<scalar_t> neighy(2,0.0);
        neighx[0] = local_field_value(left_neigh_id,SUBSET_COORDINATES_X_FS) + local_field_value(left_neigh_id,SUBSET_DISPLACEMENT_X_FS); // at this point the displacement is between the left and right image location for this subset centroid
        neighy[0] = local_field_value(left_neigh_id,SUBSET_COORDINATES_Y_FS) + local_field_value(left_neigh_id,SUBSET_DISPLACEMENT_Y_FS);
        neighx[1] = local_field_value(right_neigh_id,SUBSET_COORDINATES_X_FS) + local_field_value(right_neigh_id,SUBSET_DISPLACEMENT_X_FS);
        neighy[1] = local_field_value(right_neigh_id,SUBSET_COORDINATES_Y_FS) + local_field_value(right_neigh_id,SUBSET_DISPLACEMENT_Y_FS);
        tri->undistort_points(neighx,neighy,1);
        const scalar_t B = (neighy[1]-neighy[0])/(neighx[1]-neighx[0]);
        const scalar_t D = neighy[0] - B*neighx[0];
        if(A-B==0.0) continue;
        const scalar_t px = (D-C)/(A-B);
        const scalar_t py = A*(D-C)/(A-B) + C;
        const scalar_t u = px - local_field_value(i,SUBSET_COORDINATES_X_FS);
        const scalar_t v = py - local_field_value(i,SUBSET_COORDINATES_Y_FS);
        DEBUG_MSG("Schema::initialize_cross_correlation(): initial guess based on epipolar nearest neighbor line intersection: " << px << " " << py << " u " << u << " v " << v);
        // check bounds on this point
        if(px<5.0||px>ref_img_->width()-5.0) continue;
        if(py<5.0||py>ref_img_->height()-5.0) continue;

        // Try correlating based on this point
        // TODO consolidate this into a function call:
        int_t num_iterations = -1;
        Teuchos::RCP<Local_Shape_Function> shape_function = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
        Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(this,this_proc_gid_order_[i]));
        shape_function->insert_motion(u,v);
        Status_Flag corr_status = obj->computeUpdateFast(shape_function,num_iterations);

        // check the sigma values and status flag
        scalar_t noise_std_dev = 0.0;
        const scalar_t cross_sigma = obj->sigma(shape_function,noise_std_dev);
        local_field_value(i,SIGMA_FS) = cross_sigma;
        if(cross_sigma<0.0){
          DEBUG_MSG("Schema::initialize_cross_correlation(): failed cross init sigma (even after line intersection method applied)");
          continue;
        }
        if(corr_status!=CORRELATION_SUCCESSFUL){
          local_field_value(i,SIGMA_FS) = -1.0;
          DEBUG_MSG("Schema::initialize_cross_correlation(): failed correlation (even after line intersection method applied)");
          continue;
        }
        scalar_t cross_u = 0.0, cross_v = 0.0, cross_t = 0.0;
        shape_function->map_to_u_v_theta(local_field_value(i,SUBSET_COORDINATES_X_FS),
          local_field_value(i,SUBSET_COORDINATES_Y_FS),
          cross_u,cross_v,cross_t);
        // check the epipolar distance
        const float stereo_x = local_field_value(i,SUBSET_COORDINATES_X_FS) + cross_u;
        const float stereo_y = local_field_value(i,SUBSET_COORDINATES_Y_FS) + cross_v;
        std::vector<scalar_t> sx(1,stereo_x);
        std::vector<scalar_t> sy(1,stereo_y);
        tri->undistort_points(sx,sy,1); // right camera
        const float dist = (std::abs(a*sx[0]+b*sy[0]+c)/std::sqrt(a*a+b*b));
        DEBUG_MSG("Schema::initialize_cross_correlation(): epipolar error for second pass subset: " << dist);
        if(dist>epi_dist_tol){
          local_field_value(i,SIGMA_FS) = -1.0;
          DEBUG_MSG("Schema::initialize_cross_correlation(): failed epipolar distance threshold");
          continue;
        }
        local_field_value(i,SUBSET_DISPLACEMENT_X_FS) = cross_u;
        local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) = cross_v;
        local_field_value(i,NEIGHBOR_ID_FS) = subset_global_id(i); // seeds get themselves as neighbors
        local_field_value(i,FIELD_10_FS) = subset_global_id(i);
//        const int_t num_neigh = 7;
//        const int_t max_iterations = 1000000;
//        int_t it_count = 0;
//        std::vector<int_t> in_gids, out_gids;
//        in_gids.reserve(local_num_subsets_);
//        out_gids.reserve(local_num_subsets_);
//        in_gids.push_back(subset_global_id(i));
//        while(in_gids.size()>0){
//          TEUCHOS_TEST_FOR_EXCEPTION(it_count > max_iterations,std::runtime_error,
//            "error: inifinte loop detected in space filling initializer"); // guard against infinite loop
//          space_fill_correlate(subset_global_id(i),in_gids,out_gids,num_neigh,kd_tree,epi_dist_tol,tri);
//          in_gids.clear();
//          for(size_t i=0;i<out_gids.size();++i)
//            in_gids.push_back(out_gids[i]);
//          it_count++;
//        } // end space filling loop for each seed
      }
    } // end subset loop for neighbor initializers

    // reset the compute def gradients flag
    compute_def_gradients_ = orig_compute_def_gradients;
    for(int_t i=0;i<local_num_subsets_;++i){
      if(local_field_value(i,SIGMA_FS) <0.0) local_field_value(i,NEIGHBOR_ID_FS) = -2; // use this as an indicator that this point failed cross correlation
    }
    DEBUG_MSG("Schema::initialize_cross_correlation(): space filling iterations successful");
    return 0;
  }

  if(cross_initialization_method_==USE_SATELLITE_GEOMETRY){
#if DICE_ENABLE_NETCDF
    mesh_->create_field(field_enums::EARTH_SURFACE_X_FS);
    mesh_->create_field(field_enums::EARTH_SURFACE_Y_FS);
    mesh_->create_field(field_enums::EARTH_SURFACE_Z_FS);
    Teuchos::ParameterList lat_long_params = DICe::netcdf::netcdf_to_lat_long_projection_parameters(left_image_string,right_image_string);
    std::vector<float> left_x(local_num_subsets_);
    std::vector<float> left_y(local_num_subsets_);
    std::vector<float> right_x;
    std::vector<float> right_y;
    std::vector<float> earth_x;
    std::vector<float> earth_y;
    std::vector<float> earth_z;
    for(int_t i=0;i<local_num_subsets_;++i){
      left_x[i] = local_field_value(i,SUBSET_COORDINATES_X_FS);
      left_y[i] = local_field_value(i,SUBSET_COORDINATES_Y_FS);
    }
    DICe::netcdf::netcdf_left_pixel_points_to_earth_and_right_pixel_coordinates(lat_long_params,left_x,left_y,earth_x,earth_y,earth_z,right_x,right_y);
    for(int_t i=0;i<local_num_subsets_;++i){
      local_field_value(i,SUBSET_DISPLACEMENT_X_FS) = right_x[i] - left_x[i];
      local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) = right_y[i] - left_y[i];
      local_field_value(i,EARTH_SURFACE_X_FS) = earth_x[i];
      local_field_value(i,EARTH_SURFACE_Y_FS) = earth_y[i];
      local_field_value(i,EARTH_SURFACE_Z_FS) = earth_z[i];
    }
    return 0;
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"USE_SATELLITE_GEOMETRY should only be used if NetCDF is on");
#endif
  }

  DEBUG_MSG("Schema::initialize_cross_correlation(): estimating the projective transform from left to right camera");
  // if you are processor 0 load the ref and def images and call the estimate routine
  if(proc_rank==0){
    Teuchos::RCP<DICe::Image> left_image = Teuchos::rcp(new Image(left_image_string.c_str(),imgParams));
    Teuchos::RCP<DICe::Image> right_image = Teuchos::rcp(new Image(right_image_string.c_str(),imgParams));
    const int_t success = tri->estimate_projective_transform(left_image,right_image,true,use_nonlinear_projection_,proc_rank);
    TEUCHOS_TEST_FOR_EXCEPTION(success!=0,std::runtime_error,"Error, Schema::initialize_cross_correlation(): estimate_projective_transform failed");
  }

  // communicate the triangulation params to all other procs from 0 (9 projective parameters followed by 12 warp parameters in one vector)
  Teuchos::Array<int_t> zero_owned_ids;
  const int_t num_proj_params = 9;
  const int_t num_warp_params = 12;
  const int_t num_params = num_proj_params + num_warp_params;
  if(proc_rank==0){
    for(int_t i=0;i<num_params;++i)
      zero_owned_ids.push_back(i);
  }
  Teuchos::Array<int_t> all_owned_ids;
  for(int_t i=0;i<num_params;++i)
    all_owned_ids.push_back(i);
  Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField_Map> all_map = Teuchos::rcp (new MultiField_Map(-1, all_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField> zero_data = Teuchos::rcp(new MultiField(zero_map,1,true));
  Teuchos::RCP<MultiField> all_data = Teuchos::rcp(new MultiField(all_map,1,true));
  if(proc_rank==0){
    for(int_t i=0;i<num_proj_params;++i){
      zero_data->local_value(i) = (*tri->projective_params())[i];
      DEBUG_MSG("[PROC " << proc_rank << "] projective_param " << i << " " << (*tri->projective_params())[i]);
    }
    for(int_t i=0;i<num_warp_params;++i){
      DEBUG_MSG("[PROC " << proc_rank << "] warp_param " << i << " " << (*tri->warp_params())[i]);
      zero_data->local_value(i+num_proj_params) = (*tri->warp_params())[i];
    }
  }
  // now export the zero owned values to all
  MultiField_Exporter exporter(*all_map,*zero_data->get_map());
  all_data->do_import(zero_data,exporter,INSERT);
  // each processor sets its parameters given the ones determined on processor 0
  if(proc_rank!=0){
    Teuchos::RCP<std::vector<scalar_t> > proj_params = Teuchos::rcp(new std::vector<scalar_t>(num_proj_params,0.0));
    Teuchos::RCP<std::vector<scalar_t> > warp_params = Teuchos::rcp(new std::vector<scalar_t>(num_warp_params,0.0));
    for(int_t i=0;i<num_proj_params;++i){
      (*proj_params)[i] = all_data->local_value(i);
    }
    for(int_t i=0;i<num_warp_params;++i){
      (*warp_params)[i] = all_data->local_value(i+num_proj_params);
    }
    tri->set_projective_params(proj_params);
    tri->set_warp_params(warp_params);
    for(int_t i=0;i<num_proj_params;++i){
      DEBUG_MSG("[PROC " << proc_rank << "] projective_param " << i << " " << (*tri->projective_params())[i]);
    }
    for(int_t i=0;i<num_warp_params;++i){
      DEBUG_MSG("[PROC " << proc_rank << "] warp_param " << i << " " << (*tri->warp_params())[i]);
    }
  }

  // the rest of this is done on all processors:

  // only do this if the analysis does not ask for right to left projection
  // otherwise, leave these field uninitialized
  Teuchos::RCP<MultiField> proj_aug_x;
  Teuchos::RCP<MultiField> proj_aug_y;
  if(use_nonlinear_projection_){
    proj_aug_x = mesh_->get_field(PROJECTION_AUG_X_FS);
    proj_aug_y = mesh_->get_field(PROJECTION_AUG_Y_FS);
  }
  for(int_t i=0;i<local_num_subsets_;++i){
    scalar_t cx = local_field_value(i,SUBSET_COORDINATES_X_FS);
    scalar_t cy = local_field_value(i,SUBSET_COORDINATES_Y_FS);
    scalar_t px = 0.0;
    scalar_t py = 0.0;
    tri->project_left_to_right_sensor_coords(cx,cy,px,py);
    if(use_nonlinear_projection_){
      proj_aug_x->local_value(i) = px - cx;
      proj_aug_y->local_value(i) = py - cy;
    }else{
      local_field_value(i,SUBSET_DISPLACEMENT_X_FS) = px - cx;
      local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) = py - cy;
    }
  }

  DEBUG_MSG("Schema::initialize_cross_correlation(): projective transform estimation successful");
  return 0;
}

int_t
Schema::execute_triangulation(Teuchos::RCP<Triangulation> tri){
  Teuchos::RCP<MultiField> disp_x = mesh_->get_field(SUBSET_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> disp_y = mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS);
  // for all the failed points, average the nearest neighbor result so that the plotting
  // doesn't get ruined, but still record a -1 for sigma and match fields
  Teuchos::RCP<MultiField> sigma = mesh_->get_field(SIGMA_FS);
  Teuchos::RCP<MultiField> match = mesh_->get_field(MATCH_FS);
  // make sure the stereo coords fields are populated
  Teuchos::RCP<MultiField> coords_x = mesh_->get_field(SUBSET_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> coords_y = mesh_->get_field(SUBSET_COORDINATES_Y_FS);
  Teuchos::RCP<MultiField> model_x = mesh_->get_field(MODEL_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> model_y = mesh_->get_field(MODEL_COORDINATES_Y_FS);
  Teuchos::RCP<MultiField> model_z = mesh_->get_field(MODEL_COORDINATES_Z_FS);
  Teuchos::RCP<MultiField> model_disp_x = mesh_->get_field(MODEL_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> model_disp_y = mesh_->get_field(MODEL_DISPLACEMENT_Y_FS);
  Teuchos::RCP<MultiField> model_disp_z = mesh_->get_field(MODEL_DISPLACEMENT_Z_FS);
  std::vector<scalar_t> world_x(local_num_subsets_,0.0);
  std::vector<scalar_t> world_y(local_num_subsets_,0.0);
  std::vector<scalar_t> world_z(local_num_subsets_,0.0);
  std::vector<scalar_t> img_x(local_num_subsets_,0.0);
  std::vector<scalar_t> img_y(local_num_subsets_,0.0);
  for(int_t i=0;i<local_num_subsets_;++i){
    img_x[i] = coords_x->local_value(i) + disp_x->local_value(i);
    img_y[i] = coords_y->local_value(i) + disp_y->local_value(i);
  }
  tri->triangulate(img_x,img_y,world_x,world_y,world_z);
  for(int_t i=0;i<local_num_subsets_;++i){
    if(frame_id_==first_frame_id_){
      model_x->local_value(i) = world_x[i];
      model_y->local_value(i) = world_y[i];
      model_z->local_value(i) = world_z[i];
    }
    else{
      model_disp_x->local_value(i) = world_x[i] - model_x->local_value(i);
      model_disp_y->local_value(i) = world_y[i] - model_y->local_value(i);
      model_disp_z->local_value(i) = world_z[i] - model_z->local_value(i);
    }
  }
  return 0;
}

int_t
Schema::execute_triangulation(Teuchos::RCP<Triangulation> tri,
  Teuchos::RCP<Schema> right_schema){
  if(tri==Teuchos::null) return 0;
  if(right_schema==Teuchos::null) return execute_triangulation(tri);
  TEUCHOS_TEST_FOR_EXCEPTION(right_schema==Teuchos::null,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(right_schema->local_num_subsets()!=local_num_subsets_,std::runtime_error,
    "Error, incompatible schemas: left number of subsets " << local_num_subsets_ << " right " << right_schema->local_num_subsets());

  Teuchos::RCP<MultiField> disp_x = mesh_->get_field(SUBSET_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> disp_y = mesh_->get_field(SUBSET_DISPLACEMENT_Y_FS);
  Teuchos::RCP<MultiField> stereo_disp_x = right_schema->mesh()->get_field(SUBSET_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> stereo_disp_y = right_schema->mesh()->get_field(SUBSET_DISPLACEMENT_Y_FS);

  // for all the failed points, average the nearest neighbor result so that the plotting
  // doesn't get ruined, but still record a -1 for sigma and match fields
  Teuchos::RCP<MultiField> sigma_left = mesh_->get_field(SIGMA_FS);
  Teuchos::RCP<MultiField> match_left = mesh_->get_field(MATCH_FS);
  Teuchos::RCP<MultiField> sigma_right = right_schema->mesh()->get_field(SIGMA_FS);

  Teuchos::RCP<MultiField> max_m = mesh_->get_field(STEREO_M_MAX_FS);

//  // set up a neighbor search tree:
//  Teuchos::RCP<Point_Cloud<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud<scalar_t>());
//  point_cloud->pts.resize(local_num_subsets_);
//  for(int_t i=0;i<local_num_subsets_;++i){
//    point_cloud->pts[i].x = local_field_value(i,COORDINATE_X);
//    point_cloud->pts[i].y = local_field_value(i,COORDINATE_Y);
//    point_cloud->pts[i].z = 0.0;
//  }
//  DEBUG_MSG("Schema::execute_triangulation(): building the kd-tree");
//  Teuchos::RCP<my_kd_tree_t> kd_tree = Teuchos::rcp(new my_kd_tree_t(3 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
//  kd_tree->buildIndex();
//  DEBUG_MSG("Schema::execute_triangulation(): kd-tree completed");
//
//  // start with the best local id and expand from there
//  scalar_t query_pt[3];
//  const int_t num_neigh = 5;
//  std::vector<size_t> ret_index(num_neigh);
//  std::vector<scalar_t> out_dist_sqr(num_neigh);

  for(int_t i=0;i<local_num_subsets_;++i){
//    if(sigma_left->local_value(i) < 0 || sigma_right->local_value(i) < 0){
//      // find 5 nearest neighbors in the left schema:
//      query_pt[0] = point_cloud->pts[i].x;
//      query_pt[1] = point_cloud->pts[i].y;
//      query_pt[2] = point_cloud->pts[i].z;
//      kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
//      scalar_t avg_u = 0.0;
//      scalar_t avg_v = 0.0;
//      int_t num_neigh_valid = 0;
//      if(sigma_left->local_value(i) < 0){  // it was the left subset that failed
//        for(size_t i=0;i<num_neigh;++i){
//          const int_t neigh_id = ret_index[i]; // local id
//          if(sigma_left->local_value(neigh_id) > 0){
//            avg_u += disp_x->local_value(neigh_id);
//            avg_v += disp_y->local_value(neigh_id);
//            num_neigh_valid++;
//          }
//        }
//        if(num_neigh_valid!=0){
//          avg_u /= num_neigh_valid;
//          avg_v /= num_neigh_valid;
//        }
//        DEBUG_MSG("Schema::execute_triangulation(): failed left subset " << subset_global_id(i) <<
//          " replacing disp value " << disp_x->local_value(i) << "," << disp_y->local_value(i) <<
//          " with neighbor average for traingulation " << avg_u << "," << avg_v);
//        disp_x->local_value(i) = avg_u;
//        disp_y->local_value(i) = avg_v;
//      }
//      else{ // it was the right subset that failed
//        for(size_t i=0;i<num_neigh;++i){
//          const int_t neigh_id = ret_index[i]; // local id
//          if(sigma_right->local_value(neigh_id) > 0){
//            avg_u += stereo_disp_x->local_value(neigh_id);
//            avg_v += stereo_disp_y->local_value(neigh_id);
//            num_neigh_valid++;
//          }
//        }
//        if(num_neigh_valid!=0){
//          avg_u /= num_neigh_valid;
//          avg_v /= num_neigh_valid;
//        }
//        DEBUG_MSG("Schema::execute_triangulation(): failed right subset " << subset_global_id(i) <<
//          " replacing disp value " << stereo_disp_x->local_value(i) << "," << stereo_disp_y->local_value(i) <<
//          " with neighbor average for traingulation " << avg_u << "," << avg_v);
//        stereo_disp_x->local_value(i) = avg_u;
//        stereo_disp_y->local_value(i) = avg_v;
//      }
//    } // end left of right sigma < 0
    if(sigma_right->local_value(i) < 0){
      DEBUG_MSG("Schema::execute_triangulation(): setting left subset gid " << sigma_left->get_map()->get_global_element(i) <<
        " sigma value to -1 since right sigma = -1 " << " (right subset gid " << sigma_right->get_map()->get_global_element(i) << ")");
      sigma_left->local_value(i) = -1;
      match_left->local_value(i) = -1;
    }
  } // end subset loop

  // make sure the stereo coords fields are populated
  Teuchos::RCP<MultiField> coords_x = mesh_->get_field(SUBSET_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> coords_y = mesh_->get_field(SUBSET_COORDINATES_Y_FS);
  // the coordinates of the stereo subsets already got copied over at the end of the cross corr execution
  Teuchos::RCP<MultiField> stereo_coords_x = mesh_->get_field(STEREO_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> stereo_coords_y = mesh_->get_field(STEREO_COORDINATES_Y_FS);
  TEUCHOS_TEST_FOR_EXCEPTION(stereo_coords_x->norm()==0.0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(stereo_coords_y->norm()==0.0,std::runtime_error,"");

  // copy the stereo displacement field into place
  Teuchos::RCP<MultiField> my_stereo_disp_x = mesh_->get_field(STEREO_SUBSET_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> my_stereo_disp_y = mesh_->get_field(STEREO_SUBSET_DISPLACEMENT_Y_FS);
  my_stereo_disp_x->update(1.0,*stereo_disp_x,0.0);
  my_stereo_disp_y->update(1.0,*stereo_disp_y,0.0);
  Teuchos::RCP<MultiField> model_x = mesh_->get_field(MODEL_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> model_y = mesh_->get_field(MODEL_COORDINATES_Y_FS);
  Teuchos::RCP<MultiField> model_z = mesh_->get_field(MODEL_COORDINATES_Z_FS);
  Teuchos::RCP<MultiField> model_disp_x = mesh_->get_field(MODEL_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> model_disp_y = mesh_->get_field(MODEL_DISPLACEMENT_Y_FS);
  Teuchos::RCP<MultiField> model_disp_z = mesh_->get_field(MODEL_DISPLACEMENT_Z_FS);
  scalar_t X=0.0,Y=0.0,Z=0.0;
  scalar_t Xw=0.0,Yw=0.0,Zw=0.0;
  scalar_t xl=0.0,yl=0.0;
  scalar_t xr=0.0,yr=0.0;
  scalar_t max_m_value = 0.0;
  // if this is the first frame and a best fit plane is being used, clear the transform entries in case they have already been specified by the user

  bool best_fit = false;
  if(frame_id_==first_frame_id_){
    std::ifstream f("best_fit_plane.dat");
    // if there is a best fit plane file, determine the best fit plane
    // this has to be done after the inital 3d coords are established
    if(f.good()){
      best_fit = true;
      DEBUG_MSG("Schema::execute_triangulation(): clearing the trans_extrinsics_ values");
      tri->reset_cam_0_to_world();
    }
  }
  for(int_t i=0;i<local_num_subsets_;++i){
    xl = coords_x->local_value(i) + disp_x->local_value(i);
    yl = coords_y->local_value(i) + disp_y->local_value(i);
    xr = stereo_coords_x->local_value(i) + my_stereo_disp_x->local_value(i);
    yr = stereo_coords_y->local_value(i) + my_stereo_disp_y->local_value(i);
    max_m_value = tri->triangulate(xl,yl,xr,yr,X,Y,Z,Xw,Yw,Zw);
    max_m->local_value(i) = max_m_value;
    if(frame_id_==first_frame_id_){
      model_x->local_value(i) = Xw; // w-coordinates have been transformed by a user defined transform to world or model coords
      model_y->local_value(i) = Yw;
      model_z->local_value(i) = Zw;
    }
    else{
      model_disp_x->local_value(i) = Xw - model_x->local_value(i); // w-coordinates have been transformed by a user defined transform to world or model coords
      model_disp_y->local_value(i) = Yw - model_y->local_value(i);
      model_disp_z->local_value(i) = Zw - model_z->local_value(i);
    }
    //std::cout << " xl " << xl << " yl " << yl << " xr " << xr << " yr " << yr << " X " << Xw << " Y " << Yw << " Z " << Zw << std::endl;
  }
  if(frame_id_==first_frame_id_ && best_fit){
    Teuchos::RCP<MultiField> sigma = mesh_->get_field(SIGMA_FS);
    tri->best_fit_plane(model_x,model_y,model_z,sigma);
    // retriangulate the coordinate in the first frame
    for(int_t i=0;i<local_num_subsets_;++i){
      xl = coords_x->local_value(i) + disp_x->local_value(i);
      yl = coords_y->local_value(i) + disp_y->local_value(i);
      xr = stereo_coords_x->local_value(i) + my_stereo_disp_x->local_value(i);
      yr = stereo_coords_y->local_value(i) + my_stereo_disp_y->local_value(i);
      tri->triangulate(xl,yl,xr,yr,X,Y,Z,Xw,Yw,Zw);
      model_x->local_value(i) = Xw; // w-coordinates have been transformed by a user defined transform to world or model coords
      model_y->local_value(i) = Yw;
      model_z->local_value(i) = Zw;
    }
  }
  return 0;
}

// TODO fix this up so that it works with conformal subsets:
void
Schema::write_control_points_image(const std::string & fileName,
  const bool use_def_image,
  const bool use_one_point){

  assert(subset_dim_>0);
  Teuchos::RCP<Image> img = (use_def_image) ? def_imgs_[0] : ref_img_;

  const int_t width = img->width();
  const int_t height = img->height();
  const int_t ox = img->offset_x();
  const int_t oy = img->offset_y();

  // first, create new intensities based on the old
  Teuchos::ArrayRCP<storage_t> intensities(width*height,0);
  for (int_t i=0;i<width*height;++i)
    intensities[i] = (*img)(i);

  int_t x=0,y=0,xAlt=0,yAlt=0;
  const int_t numLocalControlPts = local_num_subsets_;
  // put a black box around the subset
  int_t i_start = 0;
  if(use_one_point) i_start = numLocalControlPts/2;
  const int_t i_end = use_one_point ? i_start + 1 : numLocalControlPts;
  const int_t color = use_one_point ? 255 : 0;
  for (int_t i=i_start;i<i_end;++i){
    x = local_field_value(i,SUBSET_COORDINATES_X_FS) - ox;
    y = local_field_value(i,SUBSET_COORDINATES_Y_FS) - oy;
    for(int_t j=0;j<subset_dim_;++j){
      xAlt = x - subset_dim_/2 + j;
      intensities[(y+subset_dim_/2)*width+xAlt] = color;
      intensities[(y-subset_dim_/2)*width+xAlt] = color;
    }
    for(int_t j=0;j<subset_dim_;++j){
      yAlt = y - subset_dim_/2 + j;
      intensities[(x+subset_dim_/2)+width*yAlt] = color;
      intensities[(x-subset_dim_/2)+width*yAlt] = color;
    }
  }
  // place white plus signs at the control points
  for (int_t i=0;i<numLocalControlPts;++i){
    x = local_field_value(i,SUBSET_COORDINATES_X_FS) - ox;
    y = local_field_value(i,SUBSET_COORDINATES_Y_FS) - oy;
    intensities[y*width+x] = 255;
    for(int_t j=0;j<3;++j){
      intensities[y*width+(x+j)] = 255;
      intensities[y*width+(x-j)] = 255;
      intensities[(y+j)*width + x] = 255;
      intensities[(y-j)*width + x] = 255;
    }
  }
  // place black plus signs at the control points that were successful
  for (int_t i=0;i<numLocalControlPts;++i){
    if(local_field_value(i,SIGMA_FS)<=0) return;
    x = local_field_value(i,SUBSET_COORDINATES_X_FS) - ox;
    y = local_field_value(i,SUBSET_COORDINATES_Y_FS) - oy;
    intensities[y*width+x] = 0;
    for(int_t j=0;j<2;++j){
      intensities[y*width+(x+j)] = 0;
      intensities[y*width+(x-j)] = 0;
      intensities[(y+j)*width + x] = 0;
      intensities[(y-j)*width + x] = 0;
    }
  }
  // create a new image based on the info above:
  Teuchos::RCP<Image> new_img = Teuchos::rcp(new Image(width,height,intensities));

  // write the image:
  new_img->write(fileName);
}

void
Schema::write_mat_output(const std::string & output_folder,
  const std::string & prefix){
  if(analysis_type_==GLOBAL_DIC){
    return; // no subset_coordinates for global (should we address this?)
  }

  Teuchos::RCP<MultiField> sigma = mesh_->get_field(SIGMA_FS);
  Teuchos::RCP<MultiField> ximg = mesh_->get_field(SUBSET_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> yimg = mesh_->get_field(SUBSET_COORDINATES_Y_FS);
  Teuchos::RCP<MultiField> model_x = mesh_->get_field(MODEL_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> model_y = mesh_->get_field(MODEL_COORDINATES_Y_FS);
  Teuchos::RCP<MultiField> model_z = mesh_->get_field(MODEL_COORDINATES_Z_FS);
  Teuchos::RCP<MultiField> model_u = mesh_->get_field(MODEL_DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> model_v = mesh_->get_field(MODEL_DISPLACEMENT_Y_FS);
  Teuchos::RCP<MultiField> model_w = mesh_->get_field(MODEL_DISPLACEMENT_Z_FS);

  std::stringstream file_name_ss;
  int_t frame_id = frame_id_-frame_skip_; // frame was already updated when results output is called
  file_name_ss << output_folder << prefix << "_" << frame_id << ".mat";


  // determine the number of valid points:
  int_t num_good = 0;
  for(int_t i=0;i<local_num_subsets_;++i)
    if(sigma->local_value(i)>=0.0)
      num_good++;

  char const *pname = "DICe";
  double *pr;
  Mat_Info info;
  int mn;
  FILE *fp;
  fp=fopen(file_name_ss.str().c_str(),"wb");
  if(fp==NULL){
    std::cout << "Error: could not write .mat file " << file_name_ss.str() << std::endl;
    return;
  }
  else{
    info.type = 0000;
    info.mrows = num_good;
    info.ncols = 8;
    info.imagf = 0;
    info.namelen = 5;
    std::vector<double> data(num_good*info.ncols,0.0);
    pr = &data[0];

    // copy the data from the fields into the column format requested by the DIC Challenge Spec
    int_t current = 0;
    for(int_t i=0;i<local_num_subsets_;++i){
      if(sigma->local_value(i)<0.0) continue;
      data[0*num_good + current] = ximg->local_value(i);
      data[1*num_good + current] = yimg->local_value(i);
      data[2*num_good + current] = model_x->local_value(i);
      data[3*num_good + current] = model_y->local_value(i);
      data[4*num_good + current] = model_z->local_value(i);
      data[5*num_good + current] = model_u->local_value(i);
      data[6*num_good + current] = model_v->local_value(i);
      data[7*num_good + current] = model_w->local_value(i);
      current++;
    }

    fwrite(&info,sizeof(Mat_Info),1,fp);
    mn = info.mrows*info.ncols;
    fwrite(pname, sizeof(char), info.namelen,fp);
    fwrite(pr,sizeof(double),mn,fp);
  }
  fclose(fp);
}

void
Schema::write_output(const std::string & output_folder,
  const std::string & prefix,
  const bool separate_files_per_subset,
  const bool separate_header_file,
  const bool no_text_output,
  const bool write_mat_file){
  if(analysis_type_==GLOBAL_DIC){
    return;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(output_spec_==Teuchos::null,std::runtime_error,"");
  int_t my_proc = comm_->get_rank();
  int_t proc_size = comm_->get_size();

#ifdef DICE_ENABLE_GLOBAL // global is enabled doesn't mean the analysis is global DIC it just means exodus is available as an output format
  if (write_exodus_output_) {
    if(frame_id_==first_frame_id_+frame_skip_){
      std::string output_dir= "";
      if(init_params_!=Teuchos::null)
        output_dir = init_params_->get<std::string>(DICe::output_folder,"");
      DICe::mesh::create_output_exodus_file(mesh_,output_dir);
      DICe::mesh::create_exodus_output_variable_names(mesh_);
    }
    DICe::mesh::exodus_output_dump(mesh_,(frame_id_-first_frame_id_)/frame_skip_,(frame_id_-first_frame_id_)/frame_skip_);
  }
#endif

  if(write_mat_file) write_mat_output(output_folder,prefix);


  if(no_text_output) return;

  // populate the RCP vector of fields in the output spec
  output_spec_->gather_fields();

  // only process 0 actually writes the output
  //if(my_proc!=0) return;

  std::stringstream infoName;
  infoName << output_folder << prefix << ".info";

  if(separate_files_per_subset){
    for(int_t subset=0;subset<local_num_subsets_;++subset){
      // determine the number of digits to append:
      int_t num_digits_total = 0;
      int_t num_digits_subset = 0;
      int_t decrement_total = global_num_subsets_;
      int_t decrement_subset = subset_global_id(subset);
      while (decrement_total){decrement_total /= 10; num_digits_total++;}
      if(subset_global_id(subset)==0) num_digits_subset = 1;
      else
        while (decrement_subset){decrement_subset /= 10; num_digits_subset++;}
      int_t num_zeros = num_digits_total - num_digits_subset;

      // determine the file name for this subset
      std::stringstream fName;
      fName << output_folder << prefix << "_";
      for(int_t i=0;i<num_zeros;++i)
        fName << "0";
      fName << subset_global_id(subset);
      if(proc_size>1)
        fName << "." << proc_size << "." << my_proc;
      fName << ".txt";
      if(frame_id_==first_frame_id_+frame_skip_){
        std::FILE * filePtr = fopen(fName.str().c_str(),"w"); // overwrite the file if it exists
        if(separate_header_file&&my_proc==0){
          std::FILE * infoFilePtr = fopen(infoName.str().c_str(),"w"); // overwrite the file if it exists
          output_spec_->write_info(infoFilePtr,true);
          fclose(infoFilePtr);
         }
        else{
          output_spec_->write_info(filePtr,false);
        }
        output_spec_->write_header(filePtr,"FRAME");
        fclose (filePtr);
      }
      // append the latest result to the file
      std::FILE * filePtr = fopen(fName.str().c_str(),"a");
      output_spec_->write_frame(filePtr,frame_id_-frame_skip_,subset_global_id(subset)); // frame is decremented because write gets called after update_frame
      fclose (filePtr);
    } // subset loop
  }
  else{
    std::stringstream fName;
    std::stringstream infofName;
    // determine the number of digits to append:
    int_t num_zeros = 0;
    if(num_frames_>0){
      int_t num_digits_total = 0;
      int_t num_digits_image = 0;
      int_t decrement_total = first_frame_id_+num_frames_*frame_skip_;
      int_t decrement_image = frame_id_-frame_skip_; // decremented because the frame was updated before write was called
      while (decrement_total){decrement_total /= 10; num_digits_total++;}
      if(decrement_image==0)
        num_digits_image = 1;
      else
        while (decrement_image){decrement_image /= 10; num_digits_image++;}
      num_zeros = num_digits_total - num_digits_image;
    }
    fName << output_folder << prefix << "_";
    infofName << output_folder << prefix << ".txt";
    for(int_t i=0;i<num_zeros;++i)
      fName << "0";
    fName << frame_id_-frame_skip_;
    if(proc_size >1)
      fName << "." << proc_size << "." << my_proc;
    fName << ".txt";
    std::FILE * filePtr = fopen(fName.str().c_str(),"w");

    if(separate_header_file && frame_id_<= first_frame_id_+frame_skip_ && my_proc==0){
      std::FILE * infoFilePtr = fopen(infoName.str().c_str(),"w"); // overwrite the file if it exists
      output_spec_->write_info(infoFilePtr,true);
      fclose(infoFilePtr);
    }
    else if(!separate_header_file){
      output_spec_->write_info(filePtr,false);
    }
    output_spec_->write_header(filePtr,"SUBSET_ID");

    // determine the sort order
    if(sort_txt_output_){
      // gather the coordinates fields
      Teuchos::RCP<MultiField> subset_coords_x = mesh_->get_field(SUBSET_COORDINATES_X_FS);
      Teuchos::RCP<MultiField> subset_coords_y = mesh_->get_field(SUBSET_COORDINATES_Y_FS);
      std::vector<std::tuple<int_t,scalar_t,scalar_t> >
        data(local_num_subsets_,std::tuple<int_t,scalar_t,scalar_t>(0,0.0,0.0));
      for(int_t i=0;i<local_num_subsets_;++i){
        std::get<0>(data[i]) = subset_global_id(i);
        std::get<1>(data[i]) = subset_coords_x->local_value(i);
        std::get<2>(data[i]) = subset_coords_y->local_value(i);
      }
      // sort the vector of tuples by y coordinate
      std::sort(std::begin(data), std::end(data), [](const std::tuple<int_t,scalar_t,scalar_t> & a, const std::tuple<int_t,scalar_t,scalar_t>& b)
      {return std::get<1>(a) < std::get<1>(b) || ((std::get<1>(a) == std::get<1>(b))&&(std::get<2>(a) < std::get<2>(b)));});
      // sort the vector of tuples by x coordinate
      //std::sort(std::begin(data), std::end(data), [](const std::tuple<int_t,scalar_t,scalar_t> & a, const std::tuple<int_t,scalar_t,scalar_t>& b)
      //{return std::get<1>(a) < std::get<1>(b);});
      // write the output
      for(int_t i=0;i<local_num_subsets_;++i){
        const int_t sorted_index = std::get<0>(data[i]);
        output_spec_->write_frame(filePtr,sorted_index,sorted_index);
      }
    }
    else{
      for(int_t i=0;i<local_num_subsets_;++i){
        output_spec_->write_frame(filePtr,subset_global_id(i),i);
      }
    }
    fclose (filePtr);
  }
}

void
Schema::write_stats(const std::string & output_folder,
  const std::string & prefix){
  if(analysis_type_==GLOBAL_DIC)
    return;
  TEUCHOS_TEST_FOR_EXCEPTION(output_spec_==Teuchos::null,std::runtime_error,"");
  std::stringstream infoName;
  infoName << output_folder << prefix << ".info";
  // the info file must exist for the stats to be written, otherwise no op
  std::ifstream f(infoName.str().c_str());
  if(!f.good()) return;
  std::FILE * infoFilePtr = fopen(infoName.str().c_str(),"a");
  output_spec_->write_stats(infoFilePtr);
  fclose(infoFilePtr);
}

// NOTE: only prints scalar fields
void
Schema::print_fields(const std::string & fileName){

  if(global_num_subsets_==0){
    std::cout << " Schema has 0 control points." << std::endl;
    return;
  }
  const int_t proc_id = comm_->get_rank();
  std::vector<std::string> field_names = mesh_->get_field_names(DICe::NODE_RANK,DICe::SCALAR_FIELD_TYPE,false);
  if(fileName==""){
    std::cout << "[PROC " << proc_id << "] DICE::Schema Fields and Values: " << std::endl;
    for(int_t i=0;i<local_num_subsets_;++i){
      std::cout << "[PROC " << proc_id << "] subset global id: " << subset_global_id(i) << std::endl;
      for(size_t j=0;j<field_names.size();++j){
        std::cout << "[PROC " << proc_id << "]   " << field_names[j] <<  " " <<
            local_field_value(i,mesh_->get_field_spec(field_names[j])) << std::endl;
      }
    }
  }
  else{
    std::FILE * outFile;
    outFile = fopen(fileName.c_str(),"a");
    for(int_t i=0;i<local_num_subsets_;++i){
      fprintf(outFile,"%i ",subset_global_id(i));
      for(size_t j=0;j<field_names.size();++j){
        fprintf(outFile,"%s %4.4E ",field_names[j].c_str(),local_field_value(i,mesh_->get_field_spec(field_names[j])));
      }
      fprintf(outFile,"\n");
    }
    fclose(outFile);
  }
}

void
Schema::check_for_blocking_subsets(const int_t subset_global_id){
  if(obstructing_subset_ids_==Teuchos::null) return;
  if(obstructing_subset_ids_->find(subset_global_id)==obstructing_subset_ids_->end()) return;
  if(obstructing_subset_ids_->find(subset_global_id)->second.size()==0) return;

  const int_t subset_lid = subset_local_id(subset_global_id);

  // turn off pixels in this subset that are blocked by another
  // get a pointer to the member data in the subset that will store the list of blocked pixels
  std::set<std::pair<int_t,int_t> > & blocked_pixels =
      *obj_vec_[subset_lid]->subset()->pixels_blocked_by_other_subsets();
  blocked_pixels.clear();

  // get the list of subsets that block this one
  std::vector<int_t> * obst_ids = &obstructing_subset_ids_->find(subset_global_id)->second;
  // iterate over all the blocking subsets
  for(size_t si=0;si<obst_ids->size();++si){
    int_t global_ss = (*obst_ids)[si];
    int_t local_ss = subset_local_id(global_ss);
    assert(local_ss>=0);
    int_t cx = obj_vec_[local_ss]->subset()->centroid_x();
    int_t cy = obj_vec_[local_ss]->subset()->centroid_y();
    Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
    shape_function->initialize_parameters_from_fields(this,global_ss);
    std::set<std::pair<int_t,int_t> > subset_pixels =
        obj_vec_[local_ss]->subset()->deformed_shapes(shape_function,cx,cy,obstruction_skin_factor_);
    blocked_pixels.insert(subset_pixels.begin(),subset_pixels.end());
  } // blocking subsets loop
}

void
Schema::write_deformed_subsets_image(const bool use_gamma_as_color){
  DEBUG_MSG("Schema::write_deformed_subset_image(): called");
  if(obj_vec_.empty()) return;
  // if the subset_images folder does not exist, create it
  // TODO allow user to specify where this goes
  // If the dir is already there this step becomes a no-op
  DEBUG_MSG("Attempting to create directory : ./deformed_subsets/");
  std::string dirStr = "./deformed_subsets/";
  create_directory(dirStr);
  int_t num_zeros = 0;
  if(num_frames_>0){
    int_t num_digits_total = 0;
    int_t num_digits_image = 0;
    int_t decrement_total = first_frame_id_ + num_frames_*frame_skip_;
    int_t decrement_image = frame_id_;
    while (decrement_total){decrement_total /= 10; num_digits_total++;}
    if(decrement_image==0)
      num_digits_image = 1;
    else
      while (decrement_image){decrement_image /= 10; num_digits_image++;}
    num_zeros = num_digits_total - num_digits_image;
  }
  const int_t proc_id = comm_->get_rank();
  std::stringstream ss;
  ss << dirStr << "def_subsets_p_" << proc_id << "_";
  for(int_t i=0;i<num_zeros;++i)
    ss << "0";
  ss << frame_id_ << ".tif";

  const int_t w = ref_img_->width();
  const int_t h = ref_img_->height();

  Teuchos::ArrayRCP<storage_t> intensities(w*h,0);

  // construct a copy of the base image to use as layer 0 for the output;
  // read each sub image if motion windows are used
  for(int_t sub=0;sub<(int_t)def_imgs_.size();++sub){
    if(def_imgs_[sub]==Teuchos::null)continue;
    const int_t offset_x = def_imgs_[sub]->offset_x();
    const int_t offset_y = def_imgs_[sub]->offset_y();
    for(int_t y=0;y<def_imgs_[sub]->height();++y){
      for(int_t x=0;x<def_imgs_[sub]->width();++x){
        intensities[(y+offset_y)*w + x + offset_x] = (*def_imgs_[sub])(x,y);
      }
    }
  }
  int_t ox=0,oy=0;
  scalar_t X=0.0,Y=0.0;
  int_t px=0,py=0;
  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory(this);
  // create output for each subset
  for(size_t subset=0;subset<obj_vec_.size();++subset){
    const int_t gid = obj_vec_[subset]->correlation_point_global_id();
    shape_function->initialize_parameters_from_fields(this,gid);
    shape_function->print_parameters();
    Teuchos::RCP<DICe::Subset> ref_subset = obj_vec_[subset]->subset();
    ox = ref_subset->centroid_x();
    oy = ref_subset->centroid_y();
    scalar_t mean_sum_ref = 0.0;
    scalar_t mean_sum_def = 0.0;
    scalar_t mean_ref = 0.0;
    scalar_t mean_def = 0.0;
    if(use_gamma_as_color){
      mean_ref = obj_vec_[subset]->subset()->mean(REF_INTENSITIES,mean_sum_ref);
      mean_def = obj_vec_[subset]->subset()->mean(DEF_INTENSITIES,mean_sum_def);
      TEUCHOS_TEST_FOR_EXCEPTION(mean_sum_ref==0.0||mean_sum_def==0.0,std::runtime_error," invalid mean sum (cannot be 0.0, ZNSSD is then undefined)" <<
        mean_sum_ref << " " << mean_sum_def);
    }
    // loop over each pixel in the subset
    scalar_t pixel_gamma = 0.0;
    for(int_t i=0;i<ref_subset->num_pixels();++i){
      shape_function->map(ref_subset->x(i),ref_subset->y(i),ox,oy,X,Y);
      // get the nearest pixel location:
      px = (int_t)X;
      if(X - (int_t)X >= 0.5) px++;
      py = (int_t)Y;
      if(Y - (int_t)Y >= 0.5) py++;
      // offset the pixel locations by the sub image offsets
      if(px>=0&&px<w&&py>=0&&py<h){
        if(use_gamma_as_color){
          if(ref_subset->is_active(i)&!ref_subset->is_deactivated_this_step(i)){
            pixel_gamma =  (ref_subset->def_intensities(i)-mean_def)/mean_sum_def - (ref_subset->ref_intensities(i)-mean_ref)/mean_sum_ref;
            intensities[py*w+px] = pixel_gamma*pixel_gamma*10000.0;
          }
        }else{
          if(!ref_subset->is_active(i)){
            intensities[py*w+px] = 75;
          }
          else{
            // color shows correlation quality
            intensities[py*w+px] = 100;//ref_subset->per_pixel_gamma(i)*85000;
          }
          // trun all deactivated pixels white
          if(ref_subset->is_deactivated_this_step(i)){
            intensities[py*w+px] = 255;
          }
        } // not use_gamma_as_color
      } // range guard
    } // pixel loop

  } // subset loop

  Teuchos::RCP<Image> layer_0_image = Teuchos::rcp(new Image(w,h,intensities));
  layer_0_image->write(ss.str());
}


int_t
Schema::strain_window_size(const int_t post_processor_index)const{
  assert((int_t)post_processors_.size()>post_processor_index);
    return post_processors_[post_processor_index]->strain_window_size();
}

Output_Spec::Output_Spec(Schema * schema,
  const bool omit_row_id,
  const Teuchos::RCP<Teuchos::ParameterList> & params,
  const std::string & delimiter):
  schema_(schema),
  delimiter_(delimiter), // space delimited is default
  omit_row_id_(omit_row_id)
{

  // default output format
  if(params == Teuchos::null){
    field_names_.push_back(SUBSET_COORDINATES_X_FS.get_name_label());
    field_names_.push_back(SUBSET_COORDINATES_Y_FS.get_name_label());
    field_names_.push_back(SUBSET_DISPLACEMENT_X_FS.get_name_label());
    field_names_.push_back(SUBSET_DISPLACEMENT_Y_FS.get_name_label());
    field_names_.push_back(ROTATION_Z_FS.get_name_label());
    if(schema->shape_function_type()==DICe::AFFINE_SF){
      field_names_.push_back(NORMAL_STRETCH_XX_FS.get_name_label());
      field_names_.push_back(NORMAL_STRETCH_YY_FS.get_name_label());
      field_names_.push_back(SHEAR_STRETCH_XY_FS.get_name_label());
    }
    field_names_.push_back(SIGMA_FS.get_name_label());
    field_names_.push_back(STATUS_FLAG_FS.get_name_label());
  }
  else{
    // get the total number of field names
    int_t num_names = 0;
    // if the names are listed with bool values, count the number of true bools
    for(Teuchos::ParameterList::ConstIterator it=params->begin();it!=params->end();++it){
      std::string string_field_name = it->first;
      if(params->isType<bool>(string_field_name)){
        if(params->get<bool>(string_field_name,false))
          num_names++;
      }
      else if(params->isType<int_t>(string_field_name)){
        num_names++;
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Error, output spec has incorrect syntax.");
      }
    }
    DEBUG_MSG("Output spec has " << num_names << " active fields");
    // get the max index
    field_names_.resize(num_names);
    int_t max_index = 0;
    std::set<int_t> indices;

    // read in the names and indices by iterating the parameter list
    int_t current_position = 0;
    for(Teuchos::ParameterList::ConstIterator it=params->begin();it!=params->end();++it){
      std::string string_field_name = it->first;
      stringToUpper(string_field_name);
      bool paramValid = false;
      for(int_t j=0;j<num_fields_defined;++j){
        if(string_field_name==fs_spec_vec[j].get_name_label())
          paramValid = true;
      }
      // see if this field is in one of the post processors instead
      for(int_t j=0;j<(int_t)schema_->post_processors()->size();++j){
        for(int_t k=0;k<(int_t)(*schema_->post_processors())[j]->field_specs()->size();++k){
          if(string_field_name==(*(*schema_->post_processors())[j]->field_specs())[k].get_name_label()){
            paramValid = true;
          }
        }
      }
      if(!paramValid){
        std::cout << "Error: invalid field name requested in output spec: " << string_field_name << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      // check if the output spec was specified as integer values or bools
      int_t field_index = -1;
      if(params->isType<bool>(string_field_name)){
        if(params->get<bool>(string_field_name,false)){
          field_index = current_position;
          current_position++;
        }
        else
          continue;
      }
      else{
        field_index = params->get<int_t>(string_field_name);
      }
      DEBUG_MSG("Adding output field " << string_field_name << " in column " << field_index);
      if(field_index>num_names-1||field_index<0){
        std::cout << "Error: field index in output spec is invalid " << field_index << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      // see if this index exists already
      if(indices.find(field_index)!=indices.end()){
        std::cout << "Error: same field index assigned to multiple fields in output spec: " << field_index << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      indices.insert(field_index);
      if(field_index > max_index) max_index = field_index;
      field_names_[field_index] = string_field_name;
    } // loop over field names in the parameterlist
    if(max_index!=num_names-1){
      std::cout << "Error: The max field index in the output spec is not equal to the number of fields, num_fields " << field_names_.size() << " max_index " << max_index << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }
  }
};

void
Output_Spec::gather_fields(){
  field_vec_.clear();
  field_vec_.resize(field_names_.size());
  for(size_t i=0;i<field_names_.size();++i){
    // check for the legacy fields form old input decks:
    DICe::field_enums::Field_Spec fs = NO_SUCH_FS;
    // for fields that have been requested, but don't exist, just return a null pointer
    // so that the output will be zeros
    try{
      fs = schema_->mesh()->get_field_spec(field_names_[i]);
    }
    catch(...){
    }
    field_vec_[i] = schema_->mesh()->get_field(fs);
  }
}

void
Output_Spec::write_info(std::FILE * file,
  const bool include_time_of_day){
  assert(file);
  TEUCHOS_TEST_FOR_EXCEPTION(schema_->analysis_type()==GLOBAL_DIC,std::runtime_error,
    "Error, write_info is not intended to be used for global DIC");
  fprintf(file,"***\n");
  fprintf(file,"*** Digital Image Correlation Engine (DICe), (git sha1: %s) Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS)\n",GITSHA1);
  fprintf(file,"***\n");
  fprintf(file,"*** Reference image: %s \n",schema_->ref_img()->file_name().c_str());
  fprintf(file,"*** Deformed image: %s \n",schema_->def_img(0)->file_name().c_str());
  fprintf(file,"*** DIC method : local \n");
  fprintf(file,"*** Correlation method: ZNSSD\n");
  std::string interp_method = to_string(schema_->interpolation_method());
  fprintf(file,"*** Interpolation method: %s\n",interp_method.c_str());
  std::string grad_method = to_string(schema_->gradient_method());
  fprintf(file,"*** Image gradient method: %s\n",grad_method.c_str());
  std::string opt_method = to_string(schema_->optimization_method());
  fprintf(file,"*** Optimization method: %s\n",opt_method.c_str());
  std::string proj_method = to_string(schema_->projection_method());
  fprintf(file,"*** Projection method: %s\n",proj_method.c_str());
  std::string init_method = to_string(schema_->initialization_method());
  fprintf(file,"*** Guess initialization method: %s\n",init_method.c_str());
  fprintf(file,"*** Seed location: N/A\n");
  fprintf(file,"*** Shape functions: ");
  if(schema_->shape_function_type()==DICe::QUADRATIC_SF){
    fprintf(file,"quadratic (A,B,C,D,E,F,G,H,I,J,K,L)");
  }
//	else if(schema_->profile_shape_function_enabled()) {
//		fprintf(file, "profile (Z, PHI, THETA)");
//	}
	else{
    if(schema_->translation_enabled())
      fprintf(file,"Translation (u,v) ");
    if(schema_->rotation_enabled())
      fprintf(file,"Rotation (theta) ");
    if(schema_->normal_strain_enabled())
      fprintf(file,"Normal Strain (ex,ey) ");
    if(schema_->shear_strain_enabled())
      fprintf(file,"Shear Strain (gamma_xy) ");
  }
  fprintf(file,"\n");
  std::string inc_cor_string = schema_->use_incremental_formulation() ? "true" : "false";
  fprintf(file,"*** Incremental correlation: %s\n", inc_cor_string.c_str());
  fprintf(file,"*** Subset size: %i\n",schema_->subset_dim());
  fprintf(file,"*** Step size: x %i y %i (-1 implies not regular grid)\n",schema_->step_size_x(),schema_->step_size_y());
  if(schema_->post_processors()->size()==0)
    fprintf(file,"*** Strain window: N/A\n");
  else if(schema_->strain_window_size(0)==-1)
    fprintf(file,"*** Strain window: N/A\n");
  else
    fprintf(file,"*** Strain window size in pixels: %i (only first strain post-processor is reported)\n",schema_->strain_window_size(0));
  fprintf(file,"*** Coordinates given with (0,0) as upper left corner of image, x positive right, y positive down\n");
  fprintf(file,"***\n");
  if(include_time_of_day){
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    fprintf(file,"*** Analysis start time %i_%i_%i %i %i %i \n",
      now->tm_year + 1900,now->tm_mon + 1,now->tm_mday,now->tm_hour,now->tm_min,now->tm_sec);
  }
}

void
Output_Spec::write_stats(std::FILE * file){
  assert(file);
  TEUCHOS_TEST_FOR_EXCEPTION(schema_->analysis_type()==GLOBAL_DIC,std::runtime_error,
    "Error, write_stats is not intended to be used for global DIC");
  time_t t = time(0);   // get time now
  struct tm * now = localtime( & t );
  const int_t proc_id = schema_->mesh()->get_comm()->get_rank();
  fprintf(file,"*** Proc %i, Analysis end time %i_%i_%i %i %i %i \n",
    proc_id, now->tm_year + 1900,now->tm_mon + 1,now->tm_mday,now->tm_hour,now->tm_min,now->tm_sec);
  if(schema_->correlation_routine()!=TRACKING_ROUTINE) return;
  fprintf(file,"***\n");
  fprintf(file,"*** Analysis stats summary: \n");
  fprintf(file,"***\n");
  fprintf(file,"%-18s %-18s %-18s %-18s %-18s\n","subset","backup opt calls","search calls","init fails","jump tol fails");
  for(size_t i=0;i<schema_->obj_vec()->size();++i){
    const int_t subset_id = (*schema_->obj_vec())[i]->correlation_point_global_id();
    const int_t backup_ops = schema_->stat_container()->num_backup_opts(subset_id);
    const int_t init_fails = schema_->stat_container()->num_failed_inits(subset_id);
    const int_t searches = schema_->stat_container()->num_searches(subset_id);
    const int_t jump_fails = schema_->stat_container()->num_jump_fails(subset_id);
    fprintf(file,"%-18i %-18i %-18i %-18i %-18i\n",subset_id,backup_ops,searches,init_fails,jump_fails);
  }
  for(size_t i=0;i<schema_->obj_vec()->size();++i){
    const int_t subset_id = (*schema_->obj_vec())[i]->correlation_point_global_id();
    fprintf(file,"\n***\n");
    fprintf(file,"*** Details for subset %i: \n",subset_id);
    fprintf(file,"***\n");
    const int_t backup_ops = schema_->stat_container()->num_backup_opts(subset_id);
    const int_t init_fails = schema_->stat_container()->num_failed_inits(subset_id);
    const int_t searches = schema_->stat_container()->num_searches(subset_id);
    const int_t jump_fails = schema_->stat_container()->num_jump_fails(subset_id);
    fprintf(file,"\n Backup optimization was called for frames: \n");
    for(int_t j=0;j<backup_ops;++j){
      fprintf(file,"%i ",schema_->stat_container()->backup_optimization_call_frams()->find(subset_id)->second[j]);
    }
    fprintf(file,"\n Search initialization was done for frames: \n");
    for(int_t j=0;j<searches;++j){
      fprintf(file,"%i ",schema_->stat_container()->search_call_frames()->find(subset_id)->second[j]);
    }
    fprintf(file,"\n Jump tolerance was exceeded for frames: \n");
    for(int_t j=0;j<jump_fails;++j){
      fprintf(file,"%i ",schema_->stat_container()->jump_tol_exceeded_frames()->find(subset_id)->second[j]);
    }
    fprintf(file,"\n Initialization failed for frames: \n");
    for(int_t j=0;j<init_fails;++j){
      fprintf(file,"%i ",schema_->stat_container()->failed_init_frames()->find(subset_id)->second[j]);
    }
  }
}


void
Output_Spec::write_header(std::FILE * file,
  const std::string & row_id){
  assert(file);
  if(!omit_row_id_)
    fprintf(file,"%s%s",row_id.c_str(),delimiter_.c_str());
  for(size_t i=0;i<field_names_.size();++i){
    if(i==0)
      fprintf(file,"%s",field_names_[i].c_str());
    else
      fprintf(file,"%s%s",delimiter_.c_str(),field_names_[i].c_str());
  }
  fprintf(file,"\n");
}

void
Output_Spec::write_frame(std::FILE * file,
  const int_t row_index,
  const int_t field_value_index){
  assert(file);
  if(!omit_row_id_)
    fprintf(file,"%i%s",row_index,delimiter_.c_str());
  for(size_t i=0;i<field_names_.size();++i)
  {
    // if the field_name is from one of the schema fields, get the information from the schema
    scalar_t value = 0.0;
    if(field_vec_[i]!=Teuchos::null)
      value = field_vec_[i]->local_value(field_value_index);
    if(i==0)
      fprintf(file,"%4.4E",value);
    else{
      fprintf(file,"%s%4.4E",delimiter_.c_str(),value);
    }
  }
  fprintf(file,"\n"); // the space before end of line is important for parsing in the output diff tool
}

bool frame_should_be_skipped(const int_t trigger_based_frame_index,
  std::vector<int_t> & frame_id_vector){
  DEBUG_MSG("frame_should_be_skipped(): vector size " << frame_id_vector.size());
  if(frame_id_vector.size()==0) return true;

  int_t index = frame_id_vector.size();
  for(size_t i=0;i<frame_id_vector.size();++i){
    if(trigger_based_frame_index < frame_id_vector[i]){
      index = i;
      break;
    }
  }
  DEBUG_MSG("frame_should_be_skipped(): index " << index);
  if(index%2==0||index==0) return false;
  else return true;
}

void
Stat_Container::register_backup_opt_call(const int_t subset_id,
  const int_t frame_id){
  if(backup_optimization_call_frames_.find(subset_id) == backup_optimization_call_frames_.end()){
    std::vector<int_t> frames;
    frames.push_back(frame_id);
    backup_optimization_call_frames_.insert(std::pair<int_t,std::vector<int_t> >(subset_id,frames));
  }
  else
    backup_optimization_call_frames_.find(subset_id)->second.push_back(frame_id);
}

void
Stat_Container::register_search_call(const int_t subset_id,
  const int_t frame_id){
  if(search_call_frames_.find(subset_id) == search_call_frames_.end()){
    std::vector<int_t> frames;
    frames.push_back(frame_id);
    search_call_frames_.insert(std::pair<int_t,std::vector<int_t> >(subset_id,frames));
  }
  else
    search_call_frames_.find(subset_id)->second.push_back(frame_id);
}

void
Stat_Container::register_jump_exceeded(const int_t subset_id,
  const int_t frame_id){
  if(jump_tol_exceeded_frames_.find(subset_id) == jump_tol_exceeded_frames_.end()){
    std::vector<int_t> frames;
    frames.push_back(frame_id);
    jump_tol_exceeded_frames_.insert(std::pair<int_t,std::vector<int_t> >(subset_id,frames));
  }
  else
    jump_tol_exceeded_frames_.find(subset_id)->second.push_back(frame_id);
}

void
Stat_Container::register_failed_init(const int_t subset_id,
  const int_t frame_id){
  if(failed_init_frames_.find(subset_id) == failed_init_frames_.end()){
    std::vector<int_t> frames;
    frames.push_back(frame_id);
    failed_init_frames_.insert(std::pair<int_t,std::vector<int_t> >(subset_id,frames));
  }
  else
    failed_init_frames_.find(subset_id)->second.push_back(frame_id);
}


}// End DICe Namespace
