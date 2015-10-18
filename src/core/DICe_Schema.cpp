// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact lead developer:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#include <DICe.h>
#include <DICe_Schema.h>
#include <DICe_ObjectiveZNSSD.h>
#include <DICe_PostProcessor.h>
#include <DICe_ParameterUtilities.h>
#include <DICe_FFT.h>

#include <Teuchos_ArrayRCP.hpp>

#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <cassert>
#include <set>

#ifndef   DICE_DISABLE_BOOST_FILESYSTEM
#include    <boost/filesystem.hpp>
#endif

namespace DICe {

Schema::Schema(const std::string & refName,
  const std::string & defName,
  const Teuchos::RCP<Teuchos::ParameterList> & params)
{
  default_constructor_tasks(params);

  Teuchos::RCP<Teuchos::ParameterList> imgParams;
  if(params!=Teuchos::null) imgParams = params;
  else imgParams = Teuchos::rcp(new Teuchos::ParameterList());

  // construct the images
  // (the compute_image_gradients param is used by the image constructor)
  imgParams->set(DICe::compute_image_gradients,compute_ref_gradients_);
  imgParams->set(DICe::gauss_filter_images,gauss_filter_images_);
  ref_img_ = Teuchos::rcp( new Image(refName.c_str(),imgParams));
  prev_img_ = Teuchos::rcp( new Image(refName.c_str(),imgParams));
  // (the compute_image_gradients param is used by the image constructor)
  imgParams->set(DICe::compute_image_gradients,compute_def_gradients_);
  def_img_ = Teuchos::rcp( new Image(defName.c_str(),imgParams));

  const int_t width = ref_img_->width();
  const int_t height = ref_img_->height();
  // require that the images are the same size
  assert(width==def_img_->width() && "  DICe ERROR: Images must be the same width.");
  assert(height==def_img_->height() && "  DICe ERROR: Images must be the same height.");
}

Schema::Schema(const int_t img_width,
  const int_t img_height,
  const Teuchos::ArrayRCP<intensity_t> refRCP,
  const Teuchos::ArrayRCP<intensity_t> defRCP,
  const Teuchos::RCP<Teuchos::ParameterList> & params)
{
  default_constructor_tasks(params);

  Teuchos::RCP<Teuchos::ParameterList> imgParams;
  if(params!=Teuchos::null) imgParams = params;
  else imgParams = Teuchos::rcp(new Teuchos::ParameterList());

  // (the compute_image_gradients param is used by the image constructor)
  imgParams->set(DICe::compute_image_gradients,compute_ref_gradients_);
  imgParams->set(DICe::gauss_filter_images,gauss_filter_images_);
  ref_img_ = Teuchos::rcp( new Image(img_width,img_height,refRCP,imgParams));
  prev_img_ = Teuchos::rcp( new Image(img_width,img_height,refRCP,imgParams));
  // (the compute_image_gradients param is used by the image constructor)
  imgParams->set(DICe::compute_image_gradients,compute_def_gradients_);
  def_img_ = Teuchos::rcp( new Image(img_width,img_height,defRCP,imgParams));

  // require that the images are the same size
  assert(!(img_width<=0||img_width!=def_img_->width()) && "  DICe ERROR: Images must be the same width and nonzero.");
  assert(!(img_height<=0||img_height!=def_img_->height()) && "  DICe ERROR: Images must be the same height and nonzero.");
}

Schema::Schema(Teuchos::RCP<Image> ref_img,
  Teuchos::RCP<Image> def_img,
  const Teuchos::RCP<Teuchos::ParameterList> & params)
{
  default_constructor_tasks(params);
  if(gauss_filter_images_){
    ref_img->gauss_filter();
    def_img->gauss_filter();
  }
  if(compute_ref_gradients_&&!ref_img->has_gradients()){
    ref_img->compute_gradients();
  }
  if(compute_def_gradients_&&!def_img->has_gradients()){
    def_img->compute_gradients();
  }
  ref_img_ = ref_img;
  def_img_ = def_img;
  prev_img_ = ref_img;
}

void
Schema::set_def_image(const std::string & defName){ // TODO add option to pass params here
  DEBUG_MSG("Schema: Resetting the deformed image");
  def_img_ = Teuchos::rcp( new Image(defName.c_str()));
}

void
Schema::set_def_image(Teuchos::RCP<Image> img){ // TODO add option to pass params here
  DEBUG_MSG("Schema: Resetting the deformed image");
  def_img_ = img;
}

void
Schema::set_def_image(const int_t img_width,
  const int_t img_height,
  const Teuchos::ArrayRCP<intensity_t> defRCP){
  DEBUG_MSG("Schema:  Resetting the deformed image");
  assert(img_width>0);
  assert(img_height>0);
  def_img_ = Teuchos::rcp( new Image(img_width,img_height,defRCP)); // TODO add option to pass params to image here
}

void
Schema::set_ref_image(const std::string & refName){ // TODO add option to pass params here
  DEBUG_MSG("Schema:  Resetting the reference image");
  ref_img_ = Teuchos::rcp( new Image(refName.c_str()));
}

void
Schema::set_ref_image(const int_t img_width,
  const int_t img_height,
  const Teuchos::ArrayRCP<intensity_t> refRCP){
  DEBUG_MSG("Schema:  Resetting the reference image");
  assert(img_width>0);
  assert(img_height>0);
  ref_img_ = Teuchos::rcp( new Image(img_width,img_height,refRCP)); // TODO add option to pass params to image here
}

void
Schema::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  data_num_points_ = 0;
  subset_dim_ = -1;
  step_size_x_ = -1;
  step_size_y_ = -1;
  mesh_size_ = -1.0;
  image_frame_ = 0;
  num_image_frames_ = -1;
  has_output_spec_ = false;
  is_initialized_ = false;
  analysis_type_ = LOCAL_DIC;
  target_field_descriptor_ = ALL_OWNED;
  distributed_fields_being_modified_ = false;
  has_post_processor_ = false;
  update_obstructed_pixels_each_iteration_ = false;
  normalize_gamma_with_active_pixels_ = false;
  gauss_filter_images_ = false;
  init_params_ = params;
  phase_cor_u_x_ = 0.0;
  phase_cor_u_y_ = 0.0;
  comm_ = Teuchos::rcp(new MultiField_Comm());
  set_params(params);
}

void
Schema::set_params(const Teuchos::RCP<Teuchos::ParameterList> & params){

  const int_t proc_rank = comm_->get_rank();

  if(params!=Teuchos::null){
    if(params->get<bool>(DICe::use_global_dic,false))
      analysis_type_=GLOBAL_DIC;
    // TODO make sure only one of these is active
  }

  // start with the default params and add any that are specified by the input params
  Teuchos::RCP<Teuchos::ParameterList> diceParams = Teuchos::rcp( new Teuchos::ParameterList("Schema_Correlation_Parameters") );

  if(analysis_type_==GLOBAL_DIC){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Global DIC is not enabled");
  }
  else if(analysis_type_==LOCAL_DIC){
    bool use_sl_defaults = false;
    if(params!=Teuchos::null){
      use_sl_defaults = params->get<bool>(DICe::use_sl_default_params,false);
    }
    // First set all of the params to their defaults in case the user does not specify them:
    if(use_sl_defaults){
      sl_default_params(diceParams.getRawPtr());
      if(proc_rank == 0) DEBUG_MSG("Initializing schema params with SL default parameters");
    }
    else{
      dice_default_params(diceParams.getRawPtr());
      if(proc_rank == 0) DEBUG_MSG("Initializing schema params with DICe default parameters");
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
    assert(false && "Error, unrecognized analysis_type");
  }
#ifdef DICE_DEBUG_MSG
  if(proc_rank == 0) {
    std::cout << "Full set of correlation parameters: " << std::endl;
    diceParams->print(std::cout);
  }
#endif

  gauss_filter_images_ = diceParams->get<bool>(DICe::gauss_filter_images,false);
  compute_ref_gradients_ = diceParams->get<bool>(DICe::compute_ref_gradients,false);
  compute_def_gradients_ = diceParams->get<bool>(DICe::compute_def_gradients,false);
  if(diceParams->get<bool>(DICe::compute_image_gradients,false)) { // this flag turns them both on
    compute_ref_gradients_ = true;
    compute_def_gradients_ = true;
  }
  assert(diceParams->isParameter(DICe::projection_method));
  projection_method_ = diceParams->get<Projection_Method>(DICe::projection_method);
  assert(diceParams->isParameter(DICe::interpolation_method));
  interpolation_method_ = diceParams->get<Interpolation_Method>(DICe::interpolation_method);
  assert(diceParams->isParameter(DICe::max_evolution_iterations));
  max_evolution_iterations_ = diceParams->get<int_t>(DICe::max_evolution_iterations);
  assert(diceParams->isParameter(DICe::max_solver_iterations_fast));
  max_solver_iterations_fast_ = diceParams->get<int_t>(DICe::max_solver_iterations_fast);
  assert(diceParams->isParameter(DICe::fast_solver_tolerance));
  fast_solver_tolerance_ = diceParams->get<double>(DICe::fast_solver_tolerance);
  // make sure image gradients are on at least for the reference image for any gradient based optimization routine
  assert(diceParams->isParameter(DICe::optimization_method));
  optimization_method_ = diceParams->get<Optimization_Method>(DICe::optimization_method);
  assert(diceParams->isParameter(DICe::correlation_routine));
  correlation_routine_ = diceParams->get<Correlation_Routine>(DICe::correlation_routine);
  assert(diceParams->isParameter(DICe::initialization_method));
  initialization_method_ = diceParams->get<Initialization_Method>(DICe::initialization_method);
  assert(diceParams->isParameter(DICe::max_solver_iterations_robust));
  max_solver_iterations_robust_ = diceParams->get<int_t>(DICe::max_solver_iterations_robust);
  assert(diceParams->isParameter(DICe::robust_solver_tolerance));
  robust_solver_tolerance_ = diceParams->get<double>(DICe::robust_solver_tolerance);
  assert(diceParams->isParameter(DICe::skip_solve_gamma_threshold));
  skip_solve_gamma_threshold_ = diceParams->get<double>(DICe::skip_solve_gamma_threshold);
  assert(diceParams->isParameter(DICe::disp_jump_tol));
  disp_jump_tol_ = diceParams->get<double>(DICe::disp_jump_tol);
  assert(diceParams->isParameter(DICe::theta_jump_tol));
  theta_jump_tol_ = diceParams->get<double>(DICe::theta_jump_tol);
  assert(diceParams->isParameter(DICe::robust_delta_disp));
  robust_delta_disp_ = diceParams->get<double>(DICe::robust_delta_disp);
  assert(diceParams->isParameter(DICe::robust_delta_theta));
  robust_delta_theta_ = diceParams->get<double>(DICe::robust_delta_theta);
  assert(diceParams->isParameter(DICe::enable_translation));
  enable_translation_ = diceParams->get<bool>(DICe::enable_translation);
  assert(diceParams->isParameter(DICe::enable_rotation));
  enable_rotation_ = diceParams->get<bool>(DICe::enable_rotation);
  assert(diceParams->isParameter(DICe::enable_normal_strain));
  enable_normal_strain_ = diceParams->get<bool>(DICe::enable_normal_strain);
  assert(diceParams->isParameter(DICe::enable_shear_strain));
  enable_shear_strain_ = diceParams->get<bool>(DICe::enable_shear_strain);
  assert(diceParams->isParameter(DICe::output_deformed_subset_images));
  output_deformed_subset_images_ = diceParams->get<bool>(DICe::output_deformed_subset_images);
  assert(diceParams->isParameter(DICe::output_deformed_subset_intensity_images));
  output_deformed_subset_intensity_images_ = diceParams->get<bool>(DICe::output_deformed_subset_intensity_images);
  assert(diceParams->isParameter(DICe::output_evolved_subset_images));
  output_evolved_subset_images_ = diceParams->get<bool>(DICe::output_evolved_subset_images);
  assert(diceParams->isParameter(DICe::use_subset_evolution));
  use_subset_evolution_ = diceParams->get<bool>(DICe::use_subset_evolution);
  assert(diceParams->isParameter(DICe::obstruction_buffer_size));
  obstruction_buffer_size_ = diceParams->get<int_t>(DICe::obstruction_buffer_size);
  assert(diceParams->isParameter(DICe::pixel_integration_order));
  pixel_integration_order_ = diceParams->get<int_t>(DICe::pixel_integration_order);
  assert(diceParams->isParameter(DICe::obstruction_skin_factor));
  obstruction_skin_factor_ = diceParams->get<double>(DICe::obstruction_skin_factor);
  assert(diceParams->isParameter(DICe::use_objective_regularization));
  use_objective_regularization_ = diceParams->get<bool>(DICe::use_objective_regularization);
  assert(diceParams->isParameter(DICe::update_obstructed_pixels_each_iteration));
  update_obstructed_pixels_each_iteration_ = diceParams->get<bool>(DICe::update_obstructed_pixels_each_iteration);
  if(update_obstructed_pixels_each_iteration_)
    DEBUG_MSG("Obstructed pixel information will be updated each iteration.");
  assert(diceParams->isParameter(DICe::normalize_gamma_with_active_pixels));
  normalize_gamma_with_active_pixels_ = diceParams->get<bool>(DICe::normalize_gamma_with_active_pixels);
  if(normalize_gamma_with_active_pixels_)
    DEBUG_MSG("Gamma values will be normalized by the number of active pixels.");

  if(analysis_type_==GLOBAL_DIC){
    compute_ref_gradients_ = true;
    assert(diceParams->isParameter(DICe::use_hvm_stabilization));
    use_hvm_stabilization_ = diceParams->get<bool>(DICe::use_hvm_stabilization);
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
    Teuchos::RCP<VSG_Strain_Post_Processor> vsg_ptr = Teuchos::rcp (new VSG_Strain_Post_Processor(this,ppParams));
    post_processors_.push_back(vsg_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_nlvc_strain)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_nlvc_strain);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    Teuchos::RCP<NLVC_Strain_Post_Processor> nlvc_ptr = Teuchos::rcp (new NLVC_Strain_Post_Processor(this,ppParams));
    post_processors_.push_back(nlvc_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_keys4_strain)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_keys4_strain);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    Teuchos::RCP<Keys4_Strain_Post_Processor> keys4_ptr = Teuchos::rcp (new Keys4_Strain_Post_Processor(this,ppParams));
    post_processors_.push_back(keys4_ptr);
  }
  if(diceParams->isParameter(DICe::post_process_global_strain)){
    Teuchos::ParameterList sublist = diceParams->sublist(DICe::post_process_global_strain);
    Teuchos::RCP<Teuchos::ParameterList> ppParams = Teuchos::rcp( new Teuchos::ParameterList());
    for(Teuchos::ParameterList::ConstIterator it=sublist.begin();it!=sublist.end();++it){
      ppParams->setEntry(it->first,it->second);
    }
    Teuchos::RCP<Global_Strain_Post_Processor> global_ptr = Teuchos::rcp (new Global_Strain_Post_Processor(this,ppParams));
    post_processors_.push_back(global_ptr);
  }
  if(post_processors_.size()>0) has_post_processor_ = true;

  Teuchos::RCP<Teuchos::ParameterList> outputParams;
  if(diceParams->isParameter(DICe::output_spec)){
    if(proc_rank == 0) DEBUG_MSG("Output spec was provided by user");
    // Strip output params sublist out of params
    Teuchos::ParameterList output_sublist = diceParams->sublist(DICe::output_spec);
    outputParams = Teuchos::rcp( new Teuchos::ParameterList());
    // iterate the sublis and add the params to the output params:
    for(Teuchos::ParameterList::ConstIterator it=output_sublist.begin();it!=output_sublist.end();++it){
      outputParams->setEntry(it->first,it->second);
    }
  }
  // create the output spec:
  const std::string delimiter = diceParams->get<std::string>(DICe::output_delimiter," ");
  const bool omit_row_id = diceParams->get<bool>(DICe::omit_output_row_id,false);
  output_spec_ = Teuchos::rcp(new DICe::Output_Spec(this,omit_row_id,outputParams,delimiter));
  has_output_spec_ = true;
}

void
Schema::initialize(const int_t step_size_x,
  const int_t step_size_y,
  const int_t subset_size){

  assert(!is_initialized_ && "Error: this schema is already initialized.");
  assert(subset_size>0 && "  Error: width cannot be equal to or less than zero.");
  step_size_x_ = step_size_x;
  step_size_y_ = step_size_y;

  const int_t img_width = ref_img_->width();
  const int_t img_height = ref_img_->height();
  // create a buffer the size of one view along all edges
  const int_t trimmedWidth = img_width - 2*subset_size;
  const int_t trimmedHeight = img_height - 2*subset_size;
  // set up the control points
  assert(step_size_x > 0 && "  DICe ERROR: step size x is <= 0");
  assert(step_size_y > 0 && "  DICe ERROR: step size y is <= 0");
  const int_t numPointsX = trimmedWidth  / step_size_x + 1;
  const int_t numPointsY = trimmedHeight / step_size_y + 1;
  assert(numPointsX > 0 && "  DICe ERROR: numPointsX <= 0.");
  assert(numPointsY > 0 && "  DICe ERROR: numPointsY <= 0.");

  const int_t num_pts = numPointsX * numPointsY;

  initialize(num_pts,subset_size);
  assert(data_num_points_==num_pts);

  int_t x_it=0, y_it=0, x_coord=0, y_coord=0;
  for (int_t i=0;i<num_pts;++i)
  {
     y_it = i / numPointsX;
     x_it = i - (y_it*numPointsX);
     x_coord = subset_dim_ + x_it * step_size_x_ -1;
     y_coord = subset_dim_ + y_it * step_size_y_ -1;
     field_value(i,COORDINATE_X) = x_coord;
     field_value(i,COORDINATE_Y) = y_coord;
  }
}

void
Schema::initialize(const int_t num_pts,
  const int_t subset_size,
  Teuchos::RCP<std::map<int_t,Conformal_Area_Def> > conformal_subset_defs,
  Teuchos::RCP<std::vector<int_t> > neighbor_ids){
  assert(def_img_->width()==ref_img_->width());
  assert(def_img_->height()==ref_img_->height());
  if(is_initialized_){
    assert(data_num_points_>0);
    assert(fields_->get_num_fields()==MAX_FIELD_NAME);
    assert(fields_nm1_->get_num_fields()==MAX_FIELD_NAME);
    return;  // no need to initialize if already done
  }
  // TODO find some way to address this (for constrained optimization, the schema doesn't need any fields)
  //assert(num_pts>0);
  data_num_points_ = num_pts;
  subset_dim_ = subset_size;

  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  // evenly distributed one-to-one map
  dist_map_ = Teuchos::rcp(new MultiField_Map(data_num_points_,0,*comm_));

  // all owned map (not one-to-one)
  Teuchos::Array<int_t> all_subsets(data_num_points_);
  for(int_t i=0;i<data_num_points_;++i)
    all_subsets[i] = i;
  all_map_ = Teuchos::rcp(new MultiField_Map(-1,all_subsets.view(0,all_subsets.size()),0,*comm_));

  // if there are blocking subsets, they need to be on the same processor and put in order:
  create_obstruction_dist_map();

  create_seed_dist_map(neighbor_ids);

  importer_ = Teuchos::rcp(new MultiField_Importer(*dist_map_,*all_map_));
  exporter_ = Teuchos::rcp(new MultiField_Exporter(*all_map_,*dist_map_));
  seed_importer_ = Teuchos::rcp(new MultiField_Importer(*seed_dist_map_,*all_map_));
  seed_exporter_ = Teuchos::rcp(new MultiField_Exporter(*all_map_,*seed_dist_map_));
  fields_ = Teuchos::rcp(new MultiField(all_map_,MAX_FIELD_NAME,true));
  fields_nm1_ = Teuchos::rcp(new MultiField(all_map_,MAX_FIELD_NAME,true));
#ifdef HAVE_MPI
  dist_fields_ = Teuchos::rcp(new MultiField(dist_map_,MAX_FIELD_NAME,true));
  dist_fields_nm1_ = Teuchos::rcp(new MultiField(dist_map_,MAX_FIELD_NAME,true));
  seed_dist_fields_ = Teuchos::rcp(new MultiField(seed_dist_map_,MAX_FIELD_NAME,true));
  seed_dist_fields_nm1_ = Teuchos::rcp(new MultiField(seed_dist_map_,MAX_FIELD_NAME,true));
#endif
  // initialize the conformal subset map to avoid havng to check if its null always
  if(conformal_subset_defs==Teuchos::null)
    conformal_subset_defs_ = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def>);
  else
    conformal_subset_defs_ = conformal_subset_defs;

  assert(data_num_points_ >= conformal_subset_defs_->size() && "  DICe ERROR: data is not the right size, "
      "conformal_subset_defs_.size() is too large for the data array");
  // ensure that the ids in conformal subset defs are valid:
  typename std::map<int_t,Conformal_Area_Def>::iterator it = conformal_subset_defs_->begin();
  for( ;it!=conformal_subset_defs_->end();++it){
    assert(it->first >= 0);
    assert(it->first < data_num_points_);
  }
  // ensure that a subset size was specified if not all subsets are conformal:
  if(analysis_type_==LOCAL_DIC&&conformal_subset_defs_->size()<data_num_points_){
    assert(subset_size > 0);
  }

  // initialize the post processors
  for(int_t i=0;i<post_processors_.size();++i)
    post_processors_[i]->initialize();

  is_initialized_ = true;

  if(neighbor_ids!=Teuchos::null)
    for(int_t i=0;i<data_num_points_;++i){
      field_value(i,DICe::NEIGHBOR_ID)  = (*neighbor_ids)[i];
    }
}

void
Schema::create_obstruction_dist_map(){
  if(obstructing_subset_ids_==Teuchos::null) return;

  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  if(proc_id == 0) DEBUG_MSG("Subsets have obstruction dependencies.");
  // set up the groupings of subset ids that have to stay together:
  // Note: this assumes that the obstructions are only one relation deep
  // i.e. the blocking subset cannot itself have a subset that blocks it
  // TODO address this to make it more general
  std::set<int_t> eligible_ids;
  for(int_t i=0;i<data_num_points_;++i)
    eligible_ids.insert(i);
  std::vector<std::set<int_t> > obstruction_groups;
  std::map<int_t,int_t> earliest_id_can_appear;
  std::set<int_t> assigned_to_a_group;
  typename std::map<int_t,std::vector<int_t> >::iterator map_it = obstructing_subset_ids_->begin();
  for(;map_it!=obstructing_subset_ids_->end();++map_it){
    int_t greatest_subset_id_among_obst = 0;
    for(int_t j=0;j<map_it->second.size();++j)
      if(map_it->second[j] > greatest_subset_id_among_obst) greatest_subset_id_among_obst = map_it->second[j];
    earliest_id_can_appear.insert(std::pair<int_t,int_t>(map_it->first,greatest_subset_id_among_obst));

    if(assigned_to_a_group.find(map_it->first)!=assigned_to_a_group.end()) continue;
    std::set<int_t> dependencies;
    dependencies.insert(map_it->first);
    eligible_ids.erase(map_it->first);
    // gather for all the dependencies for this subset
    for(int_t j=0;j<map_it->second.size();++j){
      dependencies.insert(map_it->second[j]);
      eligible_ids.erase(map_it->second[j]);
    }
    // no search all the other obstruction sets for any ids currently in the dependency list
    typename std::set<int_t>::iterator dep_it = dependencies.begin();
    for(;dep_it!=dependencies.end();++dep_it){
      typename std::map<int_t,std::vector<int_t> >::iterator search_it = obstructing_subset_ids_->begin();
      for(;search_it!=obstructing_subset_ids_->end();++search_it){
        if(assigned_to_a_group.find(search_it->first)!=assigned_to_a_group.end()) continue;
        // if any of the ids are in the current dependency list, add the whole set:
        bool match_found = false;
        if(*dep_it==search_it->first) match_found = true;
        for(int_t k=0;k<search_it->second.size();++k){
          if(*dep_it==search_it->second[k]) match_found = true;
        }
        if(match_found){
          dependencies.insert(search_it->first);
          eligible_ids.erase(search_it->first);
          for(int_t k=0;k<search_it->second.size();++k){
            dependencies.insert(search_it->second[k]);
            eligible_ids.erase(search_it->second[k]);
          }
          // reset the dependency index because more items were added to the list
          dep_it = dependencies.begin();
          // remove this set of obstruction ids since they have already been added to a group
          assigned_to_a_group.insert(search_it->first);
        } // match found
      } // obstruction set
    } // dependency it
    obstruction_groups.push_back(dependencies);
  } // outer obstruction set it
  if(proc_id == 0) DEBUG_MSG("[PROC " << proc_id << "] There are " << obstruction_groups.size() << " obstruction groupings: ");
  std::stringstream ss;
  for(int_t i=0;i<obstruction_groups.size();++i){
    ss << "[PROC " << proc_id << "] Group: " << i << std::endl;
    typename std::set<int_t>::iterator j = obstruction_groups[i].begin();
    for(;j!=obstruction_groups[i].end();++j){
      ss << "[PROC " << proc_id << "]   id: " << *j << std::endl;
    }
  }
  ss << "[PROC " << proc_id << "] Eligible ids: " << std::endl;
  for(typename std::set<int_t>::iterator elig_it=eligible_ids.begin();elig_it!=eligible_ids.end();++elig_it){
    ss << "[PROC " << proc_id << "]   " << *elig_it << std::endl;
  }
  if(proc_id == 0) DEBUG_MSG(ss.str());

  // divy up the obstruction groups among the processors:
  int_t obst_group_gid = 0;
  std::vector<std::set<int_t> > local_subset_ids(num_procs);
  while(obst_group_gid < obstruction_groups.size()){
    for(int_t p_id=0;p_id<num_procs;++p_id){
      if(obst_group_gid < obstruction_groups.size()){
        //if(p_id==proc_id){
        local_subset_ids[p_id].insert(obstruction_groups[obst_group_gid].begin(),obstruction_groups[obst_group_gid].end());
        //}
        obst_group_gid++;
      }
      else break;
    }
  }
  // assign the rest based on who has the least amount of subsets
  for(typename std::set<int_t>::iterator elig_it = eligible_ids.begin();elig_it!=eligible_ids.end();++elig_it){
    int_t proc_with_fewest_subsets = 0;
    int_t lowest_num_subsets = data_num_points_;
    for(int_t i=0;i<num_procs;++i){
      if(local_subset_ids[i].size() <= lowest_num_subsets){
        lowest_num_subsets = local_subset_ids[i].size();
        proc_with_fewest_subsets = i;
      }
    }
    local_subset_ids[proc_with_fewest_subsets].insert(*elig_it);
  }
  // order the subset ids so that they respect the dependencies:
  std::vector<int_t> local_ids;
  typename std::set<int_t>::iterator set_it = local_subset_ids[proc_id].begin();
  for(;set_it!=local_subset_ids[proc_id].end();++set_it){
    // not in the list of subsets with blockers
    if(obstructing_subset_ids_->find(*set_it)==obstructing_subset_ids_->end()){
      local_ids.push_back(*set_it);
    }
    // in the list of subsets with blockers, but has no blocking ids
    else if(obstructing_subset_ids_->find(*set_it)->second.size()==0){
      local_ids.push_back(*set_it);
    }

  }
  set_it = local_subset_ids[proc_id].begin();
  for(;set_it!=local_subset_ids[proc_id].end();++set_it){
    if(obstructing_subset_ids_->find(*set_it)!=obstructing_subset_ids_->end()){
      if(obstructing_subset_ids_->find(*set_it)->second.size()>0){
        assert(earliest_id_can_appear.find(*set_it)!=earliest_id_can_appear.end());
        local_ids.push_back(*set_it);
      }
    }
  }

  ss.str(std::string());
  ss.clear();
  ss << "[PROC " << proc_id << "] Has the following subset ids: " << std::endl;
  for(int_t i=0;i<local_ids.size();++i){
    ss << "[PROC " << proc_id << "] " << local_ids[i] <<  std::endl;
  }
  DEBUG_MSG(ss.str());

  Teuchos::ArrayView<const int_t> lids_grouped_by_obstruction(&local_ids[0],local_ids.size());
  dist_map_ = Teuchos::rcp(new MultiField_Map(data_num_points_,lids_grouped_by_obstruction,0,*comm_));
  //dist_map_->describe();
  assert(dist_map_->is_one_to_one());

  // if this is a serial run, the ordering must be changed too
  if(num_procs==1)
    all_map_ = Teuchos::rcp(new MultiField_Map(data_num_points_,lids_grouped_by_obstruction,0,*comm_));
  //all_map_->describe();
}

void
Schema::create_seed_dist_map(Teuchos::RCP<std::vector<int_t> > neighbor_ids){
  // distribution according to seeds map (one-to-one, not all procs have entries)
  // If the initialization method is USE_NEIGHBOR_VALUES or USE_NEIGHBOR_VALUES_FIRST_STEP, the
  // first step has to have a special map that keeps all subsets that use a particular seed
  // on the same processor (the parallelism is limited to the number of seeds).
  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  if(neighbor_ids!=Teuchos::null){
    // catch the case that this is an SL_ROUTINE run, but seed values were specified for
    // the individual subsets. In that case, the seed map is not necessary because there are
    // no initializiation dependencies among subsets, but the seed map will still be used since it
    // will be activated when seeds are specified for a subset.
    if(obstructing_subset_ids_!=Teuchos::null){
      if(obstructing_subset_ids_->size()>0){
        bool print_warning = false;
        for(int_t i=0;i<neighbor_ids->size();++i){
          if((*neighbor_ids)[i]!=-1) print_warning = true;
        }
        if(print_warning && proc_id==0){
          std::cout << "*** Waring: Seed values were specified for an anlysis with obstructing subsets. " << std::endl;
          std::cout << "            These values will be used to initialize subsets for which a seed has been specified, but the seed map " << std::endl;
          std::cout << "            will be set to the distributed map because grouping subsets by obstruction trumps seed ordering." << std::endl;
          std::cout << "            Seed dependencies between neighbors will not be enforced." << std::endl;
        }
        seed_dist_map_ = dist_map_;
        return;
      }
    }
    std::vector<int_t> local_subset_gids_grouped_by_roi;
    assert(neighbor_ids->size()==data_num_points_);
    std::vector<int_t> this_group_gids;
    std::vector<std::vector<int_t> > seed_groupings;
    std::vector<std::vector<int_t> > local_seed_groupings;
    for(int_t i=data_num_points_-1;i>=0;--i){
      this_group_gids.push_back(i);
      // if this subset is a seed, break this grouping and insert it in the set
      if((*neighbor_ids)[i]==-1){
        seed_groupings.push_back(this_group_gids);
        this_group_gids.clear();
      }
    }
    // TODO order the sets by their sizes and load balance better:
    // divy up the seed_groupings round-robin style:
    int_t group_gid = 0;
    int_t local_total_id_list_size = 0;
    while(group_gid < seed_groupings.size()){
      // reverse the order so the subsets are computed from the seed out
      for(int_t p_id=0;p_id<num_procs;++p_id){
        if(group_gid < seed_groupings.size()){
          if(p_id==proc_id){
            std::reverse(seed_groupings[group_gid].begin(), seed_groupings[group_gid].end());
            local_seed_groupings.push_back(seed_groupings[group_gid]);
            local_total_id_list_size += seed_groupings[group_gid].size();
          }
          group_gid++;
        }
        else break;
      }
    }
    DEBUG_MSG("[PROC " << proc_id << "] Has " << local_seed_groupings.size() << " local seed grouping(s)");
    for(int_t i=0;i<local_seed_groupings.size();++i){
      DEBUG_MSG("[PROC " << proc_id << "] local group id: " << i);
      for(int_t j=0;j<local_seed_groupings[i].size();++j){
        DEBUG_MSG("[PROC " << proc_id << "] gid: " << local_seed_groupings[i][j] );
      }
    }
    // concat local subset ids:
    local_subset_gids_grouped_by_roi.reserve(local_total_id_list_size);
    for(int_t i=0;i<local_seed_groupings.size();++i){
      local_subset_gids_grouped_by_roi.insert( local_subset_gids_grouped_by_roi.end(),
        local_seed_groupings[i].begin(),
        local_seed_groupings[i].end());
    }
    Teuchos::ArrayView<const int_t> lids_grouped_by_roi(&local_subset_gids_grouped_by_roi[0],local_total_id_list_size);
    seed_dist_map_ = Teuchos::rcp(new MultiField_Map(data_num_points_,lids_grouped_by_roi,0,*comm_));
  } // end has_neighbor_ids
  else{
    seed_dist_map_ = dist_map_;
  }
}

void
Schema::execute_correlation(){

  // make sure the data is ready to go since it may have been initialized externally by an api
  assert(is_initialized_);
  assert(fields_->get_num_fields()==MAX_FIELD_NAME);
  assert(fields_nm1_->get_num_fields()==MAX_FIELD_NAME);
  assert(data_num_points_>0);

  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  DEBUG_MSG("********************");
  std::stringstream progress;
  progress << "[PROC " << proc_id << " of " << num_procs << "] IMAGE FRAME " << image_frame_;
  if(num_image_frames_>0)
    progress << " of " << num_image_frames_;
  DEBUG_MSG(progress.str());
  DEBUG_MSG("********************");

  int_t num_local_subsets = this_proc_subset_global_ids_.size();

  // PARALLEL CASE:
  if(num_procs >1){
    // first pass for a USE_FILED_VALUES run sets up the local subset list
    // for all subsequent frames, the list remains unchanged. For this case, it
    // doesn't matter if seeding is used, because neighbor values are not needed
    if(initialization_method_==USE_FIELD_VALUES){
      target_field_descriptor_ = DISTRIBUTED;
      if(this_proc_subset_global_ids_.size()==0){
        num_local_subsets = dist_map_->get_num_local_elements();
        this_proc_subset_global_ids_ = dist_map_->get_local_element_list();
      }
    }
    // if seeding is used and the init method is USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY, the first
    // frame has to be serial, the rest can be parallel
    // TODO if multiple ROIs are used, the ROIs can be split across processors
    else if(initialization_method_==USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY){
      if(image_frame_==0){
        target_field_descriptor_ = DISTRIBUTED_GROUPED_BY_SEED;
        num_local_subsets = seed_dist_map_->get_num_local_elements();
        this_proc_subset_global_ids_ = seed_dist_map_->get_local_element_list();
      }
      else if(image_frame_==1){
        target_field_descriptor_ = DISTRIBUTED;
        num_local_subsets = dist_map_->get_num_local_elements();
        this_proc_subset_global_ids_ = dist_map_->get_local_element_list();
      }
      // otherwise nothing needs to be done since the maps will not need to change after step 1
    }
    /// For use neighbor values, the run has to be serial for each grouping that has a seed
    else if(initialization_method_==USE_NEIGHBOR_VALUES){
      if(image_frame_==0){
        target_field_descriptor_ = DISTRIBUTED_GROUPED_BY_SEED;
        num_local_subsets = seed_dist_map_->get_num_local_elements();
        this_proc_subset_global_ids_ = seed_dist_map_->get_local_element_list();
      }
    }
    else{
      assert(false && "Error: unknown initialization_method in execute correlation");
    }
  }

  // SERIAL CASE:

  else{
    if(image_frame_==0){
      target_field_descriptor_ = ALL_OWNED;
      num_local_subsets = all_map_->get_num_local_elements();
      this_proc_subset_global_ids_ = all_map_->get_local_element_list();
    }
  }
#ifdef DICE_DEBUG_MSG
  std::stringstream message;
  message << std::endl;
  for(int_t i=0;i<num_local_subsets;++i){
    message << "[PROC " << proc_id << "] Owns subset global id: " << this_proc_subset_global_ids_[i] << std::endl;
  }
  DEBUG_MSG(message.str());
#endif
  DEBUG_MSG("[PROC " << proc_id << "] has target_field_descriptor " << target_field_descriptor_);

  // Complete the set up activities for the post processors
  if(image_frame_==0){
    for(int_t i=0;i<post_processors_.size();++i){
      post_processors_[i]->pre_execution_tasks();
    }
  }

  // sync the fields:
  sync_fields_all_to_dist();

  // if requested, do a phase correlation of the images to get the initial guess for u_x and u_y:
  if(initialization_method_==USE_PHASE_CORRELATION){
    DICe::phase_correlate_x_y(prev_img_,def_img_,phase_cor_u_x_,phase_cor_u_y_);
    DEBUG_MSG(" - phase correlation initial displacements ux: " << phase_cor_u_x_ << " uy: " << phase_cor_u_y_);
  }

  // The generic routine is typically used when the dataset involves numerous subsets,
  // but only a small number of images. In this case it's more efficient to re-allocate the
  // objectives at every step, since making them static would consume a lot of memory
  if(correlation_routine_==GENERIC_ROUTINE){
    for(int_t subset_index=0;subset_index<num_local_subsets;++subset_index){
      Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(this,
        this_proc_subset_global_ids_[subset_index]));
      generic_correlation_routine(obj);
    }
  }
  // In this routine there are usually only a handful of subsets, but thousands of images.
  // In this case it is a lot more efficient to make the objectives static since there won't
  // be very many of them, and we can avoid the allocation cost at every step
  else if(correlation_routine_==SL_ROUTINE || correlation_routine_==SUBSET_EVOLUTION_ROUTINE){
    // construct the static objectives if they haven't already been constructed
    if(obj_vec_.empty()){
      for(int_t subset_index=0;subset_index<num_local_subsets;++subset_index){
        DEBUG_MSG("[PROC " << proc_id << "] Adding objective to obj_vec_ " << this_proc_subset_global_ids_[subset_index]);
        obj_vec_.push_back(Teuchos::rcp(new Objective_ZNSSD(this,
          this_proc_subset_global_ids_[subset_index])));
      }
    }
    assert(obj_vec_.size()==num_local_subsets);
    // now run the correlations:
    for(int_t subset_index=0;subset_index<num_local_subsets;++subset_index){
      check_for_blocking_subsets_new(this_proc_subset_global_ids_[subset_index]);
      if(correlation_routine_==SUBSET_EVOLUTION_ROUTINE)
        subset_evolution_routine(obj_vec_[subset_index]);
      else
        generic_correlation_routine(obj_vec_[subset_index]);
    }
    if(output_deformed_subset_images_)
      write_deformed_subsets_image();
    //if(correlation_routine_==FAST_PREDICTION_ROUTINE)
    prev_img_=def_img_;
  }
  else
    assert(false && "  DICe ERROR: unknown correlation routine.");

  // sync the fields
  sync_fields_dist_to_all();

  if(proc_id==0){
    for(unsigned subset_index=0;subset_index<data_num_points_;++subset_index){
      DEBUG_MSG("[PROC " << proc_id << "] Subset " << subset_index << " synced-up solution after execute_correlation() done, u: " <<
        field_value(subset_index,DISPLACEMENT_X) << " v: " << field_value(subset_index,DISPLACEMENT_Y)
        << " theta: " << field_value(subset_index,ROTATION_Z) << " gamma: " << field_value(subset_index,GAMMA));
    }
  }

  // compute post-processed quantities
  // For now, this assumes that all the fields are synched so that everyone owns all values.
  // TODO In the future, this can be parallelized
  // Complete the set up activities for the post processors
  // TODO maybe only processor 0 does this
  for(int_t i=0;i<post_processors_.size();++i){
    post_processors_[i]->execute();
  }
  update_image_frame();

};

void
Schema::subset_evolution_routine(Teuchos::RCP<Objective> obj){
//  const int_t subset_gid = obj->correlation_point_global_id();
//  const int_t proc_id = comm_->get_rank();
//  // turn on objective regularization to deal with pixels getting turned on and off
//  DEBUG_MSG("Subset " << subset_gid << " turing on objective regularization automatically, regardless of if it's off in the user input");
//  use_objective_regularization_ = true;
//
//  assert(get_local_id(subset_gid)!=-1 && "Error: subset id is not local to this process.");
//  DEBUG_MSG("[PROC " << proc_id << "] SUBSET " << subset_gid << " (" << local_field_value(subset_gid,DICe::COORDINATE_X) <<
//    "," << local_field_value(subset_gid,DICe::COORDINATE_Y) << ")");
//
//  // determine if the subset is a blocker and if so, force it to use simplex method:
//  // also force simplex if it is a blocked subset (not enough speckles to use grad-based method)
//  bool is_blocked = false;
//  bool is_a_blocker = false;
//  scalar_t override_tol = -1.0;
//  if(obstructing_subset_ids_->find(subset_gid)!=obstructing_subset_ids_->end()){
//    if(obstructing_subset_ids_->find(subset_gid)->second.size()>0){
//      is_blocked = true;
//      DEBUG_MSG("[PROC " << proc_id << "] SUBSET " << subset_gid << " is a blocker or blocked subset, forcing simplex method for this subset.");
//    }
//  }
//  typename std::map<int_t,std::vector<int_t> >::iterator blk_it = obstructing_subset_ids_->begin();
//  typename std::map<int_t,std::vector<int_t> >::iterator blk_end = obstructing_subset_ids_->end();
//  for(;blk_it!=blk_end;++blk_it){
//    std::vector<int_t> * obst_ids = &blk_it->second;
//    for(int_t i=0;i<obst_ids->size();++i){
//      if((*obst_ids)[i]==subset_gid){
//        is_a_blocker = true;
//        DEBUG_MSG("[PROC " << proc_id << "] SUBSET " << subset_gid << " is a blocking subset, forcing simplex method for this subset.");
//        override_tol = 0.001; // TODO move this to the input file
//      }
//    }
//  }
//
//  Teuchos::RCP<std::vector<scalar_t> > deformation = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
//  const scalar_t prev_u = local_field_value(subset_gid,DICe::DISPLACEMENT_X);
//  const scalar_t prev_v = local_field_value(subset_gid,DICe::DISPLACEMENT_Y);
//  const scalar_t prev_t = local_field_value(subset_gid,DICe::ROTATION_Z);
//
//  bool search_to_initialize = false;
//  // check if the previous step failed
//  if(local_field_value(subset_gid,SIGMA) == -1.0){
//
//    local_field_value(subset_gid,SIGMA) = -1.0;
//    local_field_value(subset_gid,MATCH) = -1.0;
//    local_field_value(subset_gid,GAMMA) = -1.0;
//    local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(INITIALIZE_FAILED);
//    local_field_value(subset_gid,ITERATIONS) = 0;
//    return;
////    // search first rather than project initial guess:
////    search_to_initialize = true;
////
////    for(int_t i=0;i<DICE_DEFORMATION_SIZE;++i) (*deformation)[i] = 0.0;
////    (*deformation)[DICe::DISPLACEMENT_X] = prev_u;
////    (*deformation)[DICe::DISPLACEMENT_Y] = prev_v;
////    (*deformation)[DICe::ROTATION_Z] = prev_t;
////    Real init_gamma = 1000.0;
////    obj->search(deformation, 1, init_gamma);
////    DEBUG_MSG("Subset " << subset_gid << " initialization gamma " << init_gamma);
////    if(init_gamma > 0.6){
////      assert(false);
////      local_field_value(subset_gid,SIGMA) = -1.0;
////      local_field_value(subset_gid,MATCH) = -1.0;
////      local_field_value(subset_gid,GAMMA) = -1.0;
////      local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(INITIALIZE_FAILED);
////      local_field_value(subset_gid,ITERATIONS) = 0;
////      return;
////    }
////    else{
////      DEBUG_MSG("Subset " << subset_gid << " resetting the active flags");
////      obj->get_ref_subset()->reset_is_deactivated_this_step();
////      // initialize the def subset with the new deformation:
////      obj->get_def_subset()->initialize(deformation,interpolation_method_,def_img_);
////    }
//  }
//
//  Status_Flag corr_status = CORRELATION_FAILED;
//  int_t num_global_iterations = 0;
//  int_t num_iterations = 0;
//
//  const int_t max_global_iterations = max_evolution_iterations_;
//
//  Teuchos::ArrayRCP<bool> old_is_active(obj->get_ref_subset()->num_pixels());
//  Teuchos::ArrayRCP<bool> old_is_deactivated_this_step(obj->get_ref_subset()->num_pixels());
//
//  scalar_t ls_x = 0.0, ls_y=0.0, ls_t=0.0;
//  // least squares fit of the previous displacements (goes back up to ten steps to fit if available)
//  obj->get_ref_subset()->predict_next_step(ls_x,ls_y,ls_t);
//  DEBUG_MSG("Subset " << subset_gid << " multistep prediction u " << ls_x << " v " << ls_y << " theta " << ls_t);
//  // grab the previous solution
//  if(!search_to_initialize){
//    if(projection_method_==MULTISTEP){
//      (*deformation)[DICe::DISPLACEMENT_X] = ls_x;
//      (*deformation)[DICe::DISPLACEMENT_Y] = ls_y;
//      (*deformation)[DICe::ROTATION_Z] = ls_t;
//    }
//    else if(projection_method_==DISPLACEMENT_BASED){
//      (*deformation)[DICe::DISPLACEMENT_X] = prev_u;
//      (*deformation)[DICe::DISPLACEMENT_Y] = prev_v;
//      (*deformation)[DICe::ROTATION_Z] = prev_t;
//    }
//    else
//      assert(false && "Invalid projection method for Schema::subset_evolution_routine()");
//  }
//
//  DEBUG_MSG("Subset " << subset_gid << " init. with values: u " << (*deformation)[DICe::DISPLACEMENT_X]
//         << " v " << (*deformation)[DICe::DISPLACEMENT_Y]
//         << " theta " << (*deformation)[DICe::ROTATION_Z]
//         << " e_x " << (*deformation)[DICe::NORMAL_STRAIN_X]
//         << " e_y " << (*deformation)[DICe::NORMAL_STRAIN_Y]
//         << " g_xy " << (*deformation)[DICe::SHEAR_STRAIN_XY]);
//
//  // keep track of the last iterative solution to see if it converged
//  scalar_t global_it_u = 0.0;
//  scalar_t global_it_v = 0.0;
//  scalar_t global_it_t = 0.0;
//
//  // GLOBAL LOOP
//
//  int_t global_it = 0;
//  bool gamma_pass = true;
//  bool max_iterations_pass = true;
//  bool hard_restart = false;
//  bool inner_convergence_pass = false;
//  for(;global_it<max_global_iterations;++global_it){
//    DEBUG_MSG("Subset " << subset_gid << " ** Global iteration: " << global_it << " **");
//
//    // SET ACTIVE FLAGS
//
//    // make a copy of all the active flags to see if any change
//    DEBUG_MSG("Subset " << subset_gid << " -> making a copy of active flags");
//    for(int_t i=0;i<obj->get_ref_subset()->num_pixels();++i){
//      old_is_active[i] = obj->get_ref_subset()->is_active(i);
//      old_is_deactivated_this_step[i] = obj->get_ref_subset()->is_deactivated_this_step(i);
//    }
//    DEBUG_MSG("Subset " << subset_gid << " -> resetting the active flags");
//    obj->get_ref_subset()->reset_is_deactivated_this_step();
//    DEBUG_MSG("Subset " << subset_gid << " -> obstruction test");
//    obj->get_ref_subset()->turn_off_obstructed_pixels(deformation);
//    // TEST FOR CHANGES TO THE ACTIVE PIXELS
//
//    DEBUG_MSG("Subset " << subset_gid << " -> testing for active flag changes since last step");
//    bool test_failed = false;
//    for(int_t i=0;i<obj->get_ref_subset()->num_pixels();++i){
//      if(old_is_active[i] != obj->get_ref_subset()->is_active(i)){
//        test_failed = true;
//      }
//      if(old_is_deactivated_this_step[i] != obj->get_ref_subset()->is_deactivated_this_step(i)){
//        test_failed = true;
//      }
//    }
//    //////////////////////////////////
//    if((!test_failed||inner_convergence_pass)&&global_it>=1&&gamma_pass&&max_iterations_pass){
//      DEBUG_MSG("Subset " << subset_gid << " ***** EVOLUTION LOOP CONVERGED *****");
//      break;
//    }
//    //////////////////////////////////
//    if(global_it>=1) DEBUG_MSG("Subset " << subset_gid << " -> active flags for some pixels have changed since last iteration");
//
//    // SOLVE
//
//    DEBUG_MSG("Subset " << subset_gid << " -> attempting to solve");
//    num_iterations = -1;
//
//    try{
//      if(optimization_method_==SIMPLEX || is_a_blocker || is_blocked){
//       corr_status = obj->computeUpdateRobust(deformation,num_iterations,override_tol);
//      }
//      else if(optimization_method_==GRADIENT_BASED || optimization_method_==GRADIENT_BASED_THEN_SIMPLEX){
//        corr_status = obj->computeUpdateFast(deformation,num_iterations);
//      }
//      else
//        assert(false && "Invalid optimization method for Schema::subset_evolution_routine()"
//            " (valid methods are: GRADIENT_BASED, SIMPLEX or GRADIENT_BASED_THEN_SIMPLEX"
//            "the simplex methods and mixed methods haven't been implemented for this routine");
//    }
//    catch (std::logic_error &err) { //a non-graceful exception occurred
//      corr_status = CORRELATION_FAILED_BY_EXCEPTION;
//    };
//    max_iterations_pass = corr_status!=MAX_ITERATIONS_REACHED;
//    DEBUG_MSG("Subset " << subset_gid << " corr_status " << corr_status);
//
//    // TEST FOR CONVERGENCE OF U V THETA (this prevents flip-flopping between two close values, important for image blur case):
//
//    if(std::abs(global_it_u - (*deformation)[DISPLACEMENT_X])<0.1 &&
//        std::abs(global_it_v - (*deformation)[DISPLACEMENT_Y])<0.1 &&
//        std::abs(global_it_t - (*deformation)[ROTATION_Z])<0.05){
//      DEBUG_MSG("Subset " << subset_gid << " ** convergence of evolution loop detected.");
//      inner_convergence_pass = true;
//    }
//    global_it_u = (*deformation)[DISPLACEMENT_X];
//    global_it_v = (*deformation)[DISPLACEMENT_Y];
//    global_it_t = (*deformation)[ROTATION_Z];
//
//    // TEST ON GAMMA
//
//    const scalar_t gamma = obj->gamma(deformation);
//    DEBUG_MSG("Subset " << subset_gid << " evolution iteration GAMMA " << gamma);
//    if(gamma >=0.8){ // detecting decorrelation
//      DEBUG_MSG("Subset " << subset_gid << " ** gamma failed.");
//      gamma_pass = false;
//    }
//    else{
//      gamma_pass = true;
//    }
//
//    // test for jump failure (too high displacement or rotation from last step due to subset getting lost)
//    bool jump_pass = true;
//    const scalar_t diffX = ((*deformation)[DISPLACEMENT_X] - prev_u);
//    const scalar_t diffY = ((*deformation)[DISPLACEMENT_Y] - prev_v);
//    const scalar_t diffT = ((*deformation)[ROTATION_Z] - prev_t);
//    DEBUG_MSG("Subset " << subset_gid << " DIFF X " << diffX << " DIFF Y " << diffY << " DIFF T " << diffT);
//    if(std::abs(diffX) > disp_jump_tol_ || std::abs(diffY) > disp_jump_tol_ || std::abs(diffT) > theta_jump_tol_)
//      jump_pass = false;
//    DEBUG_MSG("Subset " << subset_gid << " jump pass: " << jump_pass);
//
//    DEBUG_MSG("Subset " << subset_gid << " ** End of global iteration: " << global_it << " **");
//    DEBUG_MSG("Subset " << subset_gid << " gamma pass: " << gamma_pass << " max_iterations_pass " <<
//      max_iterations_pass << " jump pass " << jump_pass <<
//      " hard restart " << hard_restart << " inner_convrgence_pass " << inner_convergence_pass);
//
//    // catch complete failures due to max iterations reached or if gamma is still failing at the end of the evolution iteration loop
//    if(!hard_restart && global_it==max_global_iterations-1 && (!gamma_pass || !max_iterations_pass || !jump_pass)){
//      hard_restart = true;
//      DEBUG_MSG("Subset " << subset_gid << " Gamma still failing after max iterations, "
//          "conducting localized search around the last converged solution and re-initializing:");
//      for(int_t i=0;i<DICE_DEFORMATION_SIZE;++i) (*deformation)[i] = 0.0;
//      (*deformation)[DICe::DISPLACEMENT_X] = prev_u;
//      (*deformation)[DICe::DISPLACEMENT_Y] = prev_v;
//      (*deformation)[DICe::ROTATION_Z] = prev_t;
//      scalar_t init_gamma = 1000.0;
//      obj->search(deformation, 1, init_gamma);
//      DEBUG_MSG("Subset " << subset_gid << " resetting the active flags");
//      obj->get_ref_subset()->reset_is_deactivated_this_step();
//      // initialize the def subset with the new deformation:
//      obj->get_def_subset()->initialize(deformation,interpolation_method_,def_img_);
//      global_it = -1;
//    }
//  }  // END GLOBAL LOOP
//
//  if(global_it>=max_global_iterations){
//    DEBUG_MSG("Subset " << subset_gid << " **** WARNING: max global iterations (" << max_global_iterations <<
//      ") reached for evolution loop for subset " << subset_gid << " global iteration " << global_it);
//
//    if(optimization_method_==GRADIENT_BASED_THEN_SIMPLEX){
//      DEBUG_MSG("Subset " << subset_gid << " Searching then switching to simplex method and moving on ");
//      for(int_t i=0;i<DICE_DEFORMATION_SIZE;++i) (*deformation)[i] = 0.0;
//      (*deformation)[DICe::DISPLACEMENT_X] = prev_u;
//      (*deformation)[DICe::DISPLACEMENT_Y] = prev_v;
//      (*deformation)[DICe::ROTATION_Z] = prev_t;
//      scalar_t init_gamma = 1000.0;
//      const Status_Flag search_flag = obj->search(deformation, 1, init_gamma);
//      const bool init_gamma_fail = init_gamma > 0.8;
//      bool simplex_failed = true;
//      if(search_flag==SEARCH_SUCCESSFUL){
//        corr_status = obj->computeUpdateRobust(deformation,num_iterations);
//        simplex_failed = corr_status==CORRELATION_FAILED;
//      }
//      if(init_gamma_fail||simplex_failed){
//        local_field_value(subset_gid,SIGMA) = -1.0;
//        local_field_value(subset_gid,MATCH) = -1.0;
//        local_field_value(subset_gid,GAMMA) = -1.0;
//        local_field_value(subset_gid,DISPLACEMENT_X) = (*deformation)[DICe::DISPLACEMENT_X];
//        local_field_value(subset_gid,DISPLACEMENT_Y) = (*deformation)[DICe::DISPLACEMENT_Y];
//        local_field_value(subset_gid,ROTATION_Z) = (*deformation)[DICe::ROTATION_Z];
//        local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(MAX_GLOBAL_ITERATIONS_REACHED_IN_EVOLUTION_LOOP);
//        local_field_value(subset_gid,ITERATIONS) = num_iterations;
//        return;
//      }
//    }
//    else{
//      local_field_value(subset_gid,SIGMA) = -1.0;
//      local_field_value(subset_gid,MATCH) = -1.0;
//      local_field_value(subset_gid,GAMMA) = -1.0;
//      local_field_value(subset_gid,DISPLACEMENT_X) = (*deformation)[DICe::DISPLACEMENT_X];
//      local_field_value(subset_gid,DISPLACEMENT_Y) = (*deformation)[DICe::DISPLACEMENT_Y];
//      local_field_value(subset_gid,ROTATION_Z) = (*deformation)[DICe::ROTATION_Z];
//      local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(MAX_GLOBAL_ITERATIONS_REACHED_IN_EVOLUTION_LOOP);
//      local_field_value(subset_gid,ITERATIONS) = num_iterations;
//      return;
//    }
//  }
//  // SUCCESS
//
//  const scalar_t gamma = obj->gamma(deformation);
//  DEBUG_MSG("Subset " << subset_gid << " solution gamma " << gamma);
//  const scalar_t sigma = obj->sigma(deformation);
//
//  DEBUG_MSG("Subset " << subset_gid << " saving previous fields for u v and theta");
//  obj->get_ref_subset()->save_previous_fields(image_frame_,(*deformation)[DISPLACEMENT_X],
//    (*deformation)[DISPLACEMENT_Y],(*deformation)[DICe::ROTATION_Z]);
//
//  // test if the prediction has started to diverge:
//  DEBUG_MSG("Subset " << subset_gid << " detecting prediction divergence");
//  obj->get_ref_subset()->detect_prediction_divergence((*deformation)[DISPLACEMENT_X],
//    (*deformation)[DISPLACEMENT_Y],(*deformation)[DICe::ROTATION_Z],ls_x,ls_y,ls_t);
//
//  // TODO this may not be necessary
//  save_off_fields(subset_gid);
//
//  //obj->get_ref_subset()->save_converged_def_subset(obj->get_def_subset());
//
//  local_field_value(subset_gid,DISPLACEMENT_X) = (*deformation)[DISPLACEMENT_X];
//  local_field_value(subset_gid,DISPLACEMENT_Y) = (*deformation)[DISPLACEMENT_Y];
//  local_field_value(subset_gid,NORMAL_STRAIN_X) = (*deformation)[NORMAL_STRAIN_X];
//  local_field_value(subset_gid,NORMAL_STRAIN_Y) = (*deformation)[NORMAL_STRAIN_Y];
//  local_field_value(subset_gid,SHEAR_STRAIN_XY) = (*deformation)[SHEAR_STRAIN_XY];
//  local_field_value(subset_gid,ROTATION_Z) = (*deformation)[DICe::ROTATION_Z];
//  local_field_value(subset_gid,SIGMA) = sigma;
//  local_field_value(subset_gid,MATCH) = 0.0; // 0 means data is successful
//  local_field_value(subset_gid,GAMMA) = gamma;
//  local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(corr_status);
//  local_field_value(subset_gid,ITERATIONS) = num_iterations;
//
//  if(output_deformed_subset_intensity_images_){
//#ifndef DICE_DISABLE_BOOST_FILESYSTEM
//    DEBUG_MSG("[PROC " << proc_id << "] Attempting to create directory : ./deformed_subset_intensities/");
//    std::string dirStr = "./deformed_subset_intensities/";
//    boost::filesystem::path dir(dirStr);
//    if(boost::filesystem::create_directory(dir)) {
//      DEBUG_MSG("[PROC " << proc_id << "] Directory successfully created");
//    }
//    int_t num_zeros = 0;
//    if(num_image_frames_>0){
//      int_t num_digits_total = 0;
//      int_t num_digits_image = 0;
//      int_t decrement_total = num_image_frames_;
//      int_t decrement_image = image_frame_;
//      while (decrement_total){decrement_total /= 10; num_digits_total++;}
//      if(image_frame_==0) num_digits_image = 1;
//      else
//        while (decrement_image){decrement_image /= 10; num_digits_image++;}
//      num_zeros = num_digits_total - num_digits_image;
//    }
//    std::stringstream ss;
//    ss << dirStr << "deformedSubset_" << subset_gid << "_";
//    for(int_t i=0;i<num_zeros;++i)
//      ss << "0";
//    ss << image_frame_;
//    obj->get_def_subset()->write_tiff(ss.str());
//#endif
//  }
//  if(output_evolved_subset_images_){
//#ifndef DICE_DISABLE_BOOST_FILESYSTEM
//    DEBUG_MSG("[PROC " << proc_id << "] Attempting to create directory : ./evolved_subsets/");
//    std::string dirStr = "./evolved_subsets/";
//    boost::filesystem::path dir(dirStr);
//    if(boost::filesystem::create_directory(dir)) {
//      DEBUG_MSG("[PROC " << proc_id << "[ Directory successfully created");
//    }
//    int_t num_zeros = 0;
//    if(num_image_frames_>0){
//      int_t num_digits_total = 0;
//      int_t num_digits_image = 0;
//      int_t decrement_total = num_image_frames_;
//      int_t decrement_image = image_frame_;
//      while (decrement_total){decrement_total /= 10; num_digits_total++;}
//      if(image_frame_==0) num_digits_image = 1;
//      else
//        while (decrement_image){decrement_image /= 10; num_digits_image++;}
//      num_zeros = num_digits_total - num_digits_image;
//    }
//    std::stringstream ss;
//    ss << dirStr << "evolvedSubset_" << subset_gid << "_";
//    for(int_t i=0;i<num_zeros;++i)
//      ss << "0";
//    ss << image_frame_;
//    obj->get_ref_subset()->write_tiff(ss.str());
//#endif
//  }
}

void
Schema::generic_correlation_routine(Teuchos::RCP<Objective> obj){

  TEUCHOS_TEST_FOR_EXCEPTION(use_subset_evolution_,std::runtime_error,
    "use_subset_evolution is not allowed for the generic correlation routine");

  const int_t subset_gid = obj->correlation_point_global_id();
  const int_t proc_id = comm_->get_rank();
  assert(get_local_id(subset_gid)!=-1 && "Error: subset id is not local to this process.");
  DEBUG_MSG("[PROC " << proc_id << "] SUBSET " << subset_gid << " (" << local_field_value(subset_gid,DICe::COORDINATE_X) <<
    "," << local_field_value(subset_gid,DICe::COORDINATE_Y) << ")");

  Status_Flag init_status = INITIALIZE_FAILED;
  Status_Flag corr_status = CORRELATION_FAILED;
  int_t num_iterations = -1;

  Teuchos::RCP<std::vector<scalar_t> > deformation = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));

  // INITIALIZATION
  // The first subset uses the solution in the field values as the initial guess
  if(initialization_method_==DICe::USE_FIELD_VALUES ||
      (initialization_method_==DICe::USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY && image_frame_>0)){
    try{
      init_status = obj->initialize_from_previous_frame(deformation);
    }
    catch (std::logic_error &err) { // a non-graceful exception occurred
      local_field_value(subset_gid,SIGMA) = -1.0;
      local_field_value(subset_gid,MATCH) = -1.0;
      local_field_value(subset_gid,GAMMA) = -1.0; // TODO should -1 be used here? could this affect the minimization since it's < 0?
      local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(INITIALIZE_FAILED_BY_EXCEPTION);
      local_field_value(subset_gid,ITERATIONS) = num_iterations;
      return;
    };
  }
  else if(initialization_method_==DICe::USE_PHASE_CORRELATION){
    (*deformation)[0] = phase_cor_u_x_ + local_field_value(subset_gid,DISPLACEMENT_X);
    (*deformation)[1] = phase_cor_u_y_ + local_field_value(subset_gid,DISPLACEMENT_Y);
//    for(int_t i=2;i<DICE_DEFORMATION_SIZE;++i)
//      (*deformation)[i] = 0.0;
//    const int_t window_size = 6;
//    const scalar_t step_size = 0.5;
//    scalar_t return_gamma = 0.0;
//    init_status = obj->search_step(deformation,window_size,step_size,return_gamma);
    init_status = DICe::INITIALIZE_SUCCESSFUL;
  }
  // The rest of the subsets use the soluion from the previous subset as the initial guess
  else{
    try{
      init_status = obj->initialize_from_neighbor(deformation);
    }
    catch(std::logic_error &err){ // a non-graceful exception occurred
      local_field_value(subset_gid,SIGMA) = -1.0;
      local_field_value(subset_gid,MATCH) = -1.0;
      local_field_value(subset_gid,GAMMA) = -1.0;
      local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(INITIALIZE_FAILED_BY_EXCEPTION);
      local_field_value(subset_gid,ITERATIONS) = num_iterations;
      return;
    }
  }

  // TODO add seraching methods

  // CHECK THAT THE INITIALIZATION WAS SUCCESSFUL

  if(init_status == SEARCH_FAILED || init_status==INITIALIZE_FAILED){
    local_field_value(subset_gid,SIGMA) = -1.0;
    local_field_value(subset_gid,MATCH) = -1.0;
    local_field_value(subset_gid,GAMMA) = -1.0;
    local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(init_status);
    local_field_value(subset_gid,ITERATIONS) = num_iterations;
    return;
  }

  // TODO EVOLVE THE SUBSETS BASED ON OBSTRUCTIONS, ETC

  // CORRELATE

  if(optimization_method_==DICe::GRADIENT_BASED||optimization_method_==DICe::GRADIENT_BASED_THEN_SIMPLEX){
    try{
      corr_status = obj->computeUpdateFast(deformation,num_iterations);
    }
    catch (std::logic_error &err) { //a non-graceful exception occurred
      corr_status = CORRELATION_FAILED_BY_EXCEPTION;
    };
  }
  else if(optimization_method_==DICe::SIMPLEX||optimization_method_==DICe::SIMPLEX_THEN_GRADIENT_BASED){
    try{
      corr_status = obj->computeUpdateRobust(deformation,num_iterations);
    }
    catch (std::logic_error &err) { //a non-graceful exception occurred
      corr_status = CORRELATION_FAILED_BY_EXCEPTION;
    };
  }
  if(corr_status!=CORRELATION_SUCCESSFUL){
    if(optimization_method_==DICe::SIMPLEX||optimization_method_==DICe::GRADIENT_BASED){
      // no more tries just output the values and a failed sigma = -1 and gamma = -1;
      local_field_value(subset_gid,SIGMA) = -1.0;
      local_field_value(subset_gid,MATCH) = -1.0;
      local_field_value(subset_gid,GAMMA) = -1.0;
      local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(corr_status);
      local_field_value(subset_gid,ITERATIONS) = num_iterations;
      return;
    }
    else if(optimization_method_==DICe::GRADIENT_BASED_THEN_SIMPLEX){
      // try again using simplex
      if(initialization_method_==DICe::USE_FIELD_VALUES || (initialization_method_==DICe::USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY && image_frame_>0))
        init_status = obj->initialize_from_previous_frame(deformation);
      else if(initialization_method_==DICe::USE_PHASE_CORRELATION){
        (*deformation)[0] = phase_cor_u_x_ + local_field_value(subset_gid,DISPLACEMENT_X);
        (*deformation)[1] = phase_cor_u_y_ + local_field_value(subset_gid,DISPLACEMENT_Y);
        for(int_t i=2;i<DICE_DEFORMATION_SIZE;++i)
          (*deformation)[i] = 0.0;
        init_status = DICe::INITIALIZE_SUCCESSFUL;
      }
      else init_status = obj->initialize_from_neighbor(deformation);
      try{
        corr_status = obj->computeUpdateRobust(deformation,num_iterations);
      }
      catch (std::logic_error &err) { //a non-graceful exception occurred
        corr_status = CORRELATION_FAILED_BY_EXCEPTION;
      };
      if(corr_status!=CORRELATION_SUCCESSFUL){
        // no more tries just output the values and a failed sigma = -1 and gamma = -1;
        local_field_value(subset_gid,SIGMA) = -1.0;
        local_field_value(subset_gid,MATCH) = -1.0;
        local_field_value(subset_gid,GAMMA) = -1.0;
        local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(corr_status);
        local_field_value(subset_gid,ITERATIONS) = num_iterations;
        return;
      }
    }
    else if(optimization_method_==DICe::SIMPLEX_THEN_GRADIENT_BASED){
      // try again using gradient based
      if(initialization_method_==DICe::USE_FIELD_VALUES || (initialization_method_==DICe::USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY && image_frame_>0))
        init_status = obj->initialize_from_previous_frame(deformation);
      else if(initialization_method_==DICe::USE_PHASE_CORRELATION){
        (*deformation)[0] = phase_cor_u_x_ + local_field_value(subset_gid,DISPLACEMENT_X);
        (*deformation)[1] = phase_cor_u_y_ + local_field_value(subset_gid,DISPLACEMENT_Y);
        for(int_t i=2;i<DICE_DEFORMATION_SIZE;++i)
          (*deformation)[i] = 0.0;
        init_status = DICe::INITIALIZE_SUCCESSFUL;
      }
      else init_status = obj->initialize_from_neighbor(deformation);
      try{
        corr_status = obj->computeUpdateFast(deformation,num_iterations);
      }
      catch (std::logic_error &err) { //a non-graceful exception occurred
        corr_status = CORRELATION_FAILED_BY_EXCEPTION;
      };
      if(corr_status!=CORRELATION_SUCCESSFUL){
        // no more tries just output the values and a failed sigma = -1 and gamma = -1;
        local_field_value(subset_gid,SIGMA) = -1.0;
        local_field_value(subset_gid,MATCH) = -1.0;
        local_field_value(subset_gid,GAMMA) = -1.0;
        local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(corr_status);
        local_field_value(subset_gid,ITERATIONS) = num_iterations;
        return;
      }
    }
  }

  // SUCCESS

  const scalar_t gamma = obj->gamma(deformation);
  const scalar_t sigma = obj->sigma(deformation);

  if(projection_method_==VELOCITY_BASED) save_off_fields(subset_gid);

  local_field_value(subset_gid,DISPLACEMENT_X) = (*deformation)[DISPLACEMENT_X];
  local_field_value(subset_gid,DISPLACEMENT_Y) = (*deformation)[DISPLACEMENT_Y];
  local_field_value(subset_gid,NORMAL_STRAIN_X) = (*deformation)[NORMAL_STRAIN_X];
  local_field_value(subset_gid,NORMAL_STRAIN_Y) = (*deformation)[NORMAL_STRAIN_Y];
  local_field_value(subset_gid,SHEAR_STRAIN_XY) = (*deformation)[SHEAR_STRAIN_XY];
  local_field_value(subset_gid,ROTATION_Z) = (*deformation)[DICe::ROTATION_Z];
  local_field_value(subset_gid,SIGMA) = sigma;
  local_field_value(subset_gid,MATCH) = 0.0; // 0 means data is successful
  local_field_value(subset_gid,GAMMA) = gamma;
  local_field_value(subset_gid,STATUS_FLAG) = static_cast<int_t>(init_status);
  local_field_value(subset_gid,ITERATIONS) = num_iterations;

  if(output_deformed_subset_intensity_images_){
#ifndef DICE_DISABLE_BOOST_FILESYSTEM
    DEBUG_MSG("[PROC " << proc_id << "] Attempting to create directory : ./deformed_subset_intensities/");
    std::string dirStr = "./deformed_subset_intensities/";
    boost::filesystem::path dir(dirStr);
    if(boost::filesystem::create_directory(dir)) {
      DEBUG_MSG("[PROC " << proc_id << "] Directory successfully created");
    }
    int_t num_zeros = 0;
    if(num_image_frames_>0){
      int_t num_digits_total = 0;
      int_t num_digits_image = 0;
      int_t decrement_total = num_image_frames_;
      int_t decrement_image = image_frame_;
      while (decrement_total){decrement_total /= 10; num_digits_total++;}
      if(image_frame_==0) num_digits_image = 1;
      else
        while (decrement_image){decrement_image /= 10; num_digits_image++;}
      num_zeros = num_digits_total - num_digits_image;
    }
    std::stringstream ss;
    ss << dirStr << "deformedSubset_" << subset_gid << "_";
    for(int_t i=0;i<num_zeros;++i)
      ss << "0";
    ss << image_frame_;
    obj->subset()->write_tiff(ss.str(),true);
#endif
  }
  if(output_evolved_subset_images_){
#ifndef DICE_DISABLE_BOOST_FILESYSTEM
    DEBUG_MSG("[PROC " << proc_id << "] Attempting to create directory : ./evolved_subsets/");
    std::string dirStr = "./evolved_subsets/";
    boost::filesystem::path dir(dirStr);
    if(boost::filesystem::create_directory(dir)) {
      DEBUG_MSG("[PROC " << proc_id << "[ Directory successfully created");
    }
    int_t num_zeros = 0;
    if(num_image_frames_>0){
      int_t num_digits_total = 0;
      int_t num_digits_image = 0;
      int_t decrement_total = num_image_frames_;
      int_t decrement_image = image_frame_;
      while (decrement_total){decrement_total /= 10; num_digits_total++;}
      if(image_frame_==0) num_digits_image = 1;
      else
        while (decrement_image){decrement_image /= 10; num_digits_image++;}
      num_zeros = num_digits_total - num_digits_image;
    }
    std::stringstream ss;
    ss << dirStr << "evolvedSubset_" << subset_gid << "_";
    for(int_t i=0;i<num_zeros;++i)
      ss << "0";
    ss << image_frame_;
    obj->subset()->write_tiff(ss.str());
#endif
  }
}

// TODO fix this up so that it works with conformal subsets:
void
Schema::write_control_points_image(const std::string & fileName,
  const bool use_def_image,
  const bool use_one_point){

  assert(subset_dim_>0);
  Teuchos::RCP<Image> img = (use_def_image) ? def_img_ : ref_img_;

  const int_t width = img->width();
  const int_t height = img->height();

  // first, create new intensities based on the old
  Teuchos::ArrayRCP<intensity_t> intensities(width*height,0.0);
  Teuchos::ArrayRCP<intensity_t> img_intensity_values = img->intensity_array();
  for (int_t i=0;i<width*height;++i)
    intensities[i] = img_intensity_values[i];

  int_t x=0,y=0,xAlt=0,yAlt=0;
  const int_t numLocalControlPts = data_num_points_; //cp_map_->getNodeNumElements();
  // put a black box around the subset
  int_t i_start = 0;
  if(use_one_point) i_start = numLocalControlPts/2;
  const int_t i_end = use_one_point ? i_start + 1 : numLocalControlPts;
  const int_t color = use_one_point ? 255 : 0;
  for (int_t i=i_start;i<i_end;++i){
    x = field_value(i,COORDINATE_X);
    y = field_value(i,COORDINATE_Y);
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
    x = field_value(i,COORDINATE_X);
    y = field_value(i,COORDINATE_Y);
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
    if(field_value(i,SIGMA)<=0) return;
    x = field_value(i,COORDINATE_X);
    y = field_value(i,COORDINATE_Y);
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
  new_img->write_tiff(fileName);
}

void
Schema::write_output(const std::string & output_folder,
  const std::string & prefix,
  const bool separate_files_per_subset, const Output_File_Type type){
//  assert(analysis_type_!=CONSTRAINED_OPT && "Error, writing output from a schema using constrained optimization is not enabled.");
//  assert(analysis_type_!=INTEGRATED_DIC && "Error, writing output from a schema using integrated DIC is not enabled.");
  int_t my_proc = comm_->get_rank();
  if(my_proc!=0) return;
  int_t proc_size = comm_->get_size();

  assert(type==TEXT_FILE && "Currently only TEXT_FILE output is implemented");
  assert(output_spec_!=Teuchos::null);

  if(separate_files_per_subset){
    for(int_t subset=0;subset<data_num_points_;++subset){
      // determine the number of digits to append:
      int_t num_digits_total = 0;
      int_t num_digits_subset = 0;
      int_t decrement_total = data_num_points_;
      int_t decrement_subset = subset;
      while (decrement_total){decrement_total /= 10; num_digits_total++;}
      if(subset==0) num_digits_subset = 1;
      else
        while (decrement_subset){decrement_subset /= 10; num_digits_subset++;}
      int_t num_zeros = num_digits_total - num_digits_subset;

      // determine the file name for this subset
      std::stringstream fName;
      fName << output_folder << prefix << "_";
      for(int_t i=0;i<num_zeros;++i)
        fName << "0";
      fName << subset;
      if(proc_size>1)
        fName << "." << proc_size;
      fName << ".txt";
      if(image_frame_==1){
        std::FILE * filePtr = fopen(fName.str().c_str(),"w"); // overwrite the file if it exists
        output_spec_->write_header(filePtr,"FRAME");
        fclose (filePtr);
      }
      // append the latest result to the file
      std::FILE * filePtr = fopen(fName.str().c_str(),"a");
      output_spec_->write_frame(filePtr,image_frame_,subset);
      fclose (filePtr);
    } // subset loop
  }
  else{
    std::stringstream fName;
    // determine the number of digits to append:
    int_t num_zeros = 0;
    if(num_image_frames_>0){
      int_t num_digits_total = 0;
      int_t num_digits_image = 0;
      int_t decrement_total = num_image_frames_;
      int_t decrement_image = image_frame_ -1;
      while (decrement_total){decrement_total /= 10; num_digits_total++;}
      if(image_frame_-1==0) num_digits_image = 1;
      else
        while (decrement_image){decrement_image /= 10; num_digits_image++;}
      num_zeros = num_digits_total - num_digits_image;
    }
    fName << output_folder << prefix << "_";
    for(int_t i=0;i<num_zeros;++i)
      fName << "0";
    fName << image_frame_ - 1;
    if(proc_size >1)
      fName << "." << proc_size;
    fName << ".txt";
    std::FILE * filePtr = fopen(fName.str().c_str(),"w");
    output_spec_->write_header(filePtr,"SUBSET_ID");
    for(int_t i=0;i<data_num_points_;++i){
      output_spec_->write_frame(filePtr,i,i);
    }
    fclose (filePtr);
  }
}

void
Schema::print_fields(const std::string & fileName){

  if(data_num_points_==0){
    std::cout << " Schema has 0 control points." << std::endl;
    return;
  }
  if(fields_->get_num_fields()==0){
    std::cout << " Schema fields are emplty." << std::endl;
    return;
  }
  const int_t proc_id = comm_->get_rank();

  if(fileName==""){
    std::cout << "[PROC " << proc_id << "] DICE::Schema Fields and Values: " << std::endl;
    for(int_t i=0;i<data_num_points_;++i){
      std::cout << "[PROC " << proc_id << "] Control Point ID: " << i << std::endl;
      for(int_t j=0;j<DICe::MAX_FIELD_NAME;++j){
        std::cout << "[PROC " << proc_id << "]   " << fieldNameStrings[j] <<  " " <<
            field_value(i,static_cast<Field_Name>(j)) << std::endl;
        if(dist_map_->get_local_element(i)!=-1){
          std::cout << "[PROC " << proc_id << "]   " << fieldNameStrings[j] <<  " (has distributed value)  " <<
              local_field_value(i,static_cast<Field_Name>(j)) << std::endl;
        }
      }
    }

  }
  else{
    std::FILE * outFile;
    outFile = fopen(fileName.c_str(),"a");
    for(int_t i=0;i<data_num_points_;++i){
      fprintf(outFile,"%i ",i);
      for(int_t j=0;j<DICe::MAX_FIELD_NAME;++j){
        fprintf(outFile," %4.4E ",field_value(i,static_cast<Field_Name>(j)));
      }
      fprintf(outFile,"\n");
    }
    fclose(outFile);
  }
}

void
Schema::check_for_blocking_subsets(const int_t subset_global_id){
//  if(obstructing_subset_ids_==Teuchos::null) return;
//  if(obstructing_subset_ids_->find(subset_global_id)==obstructing_subset_ids_->end()) return;
//
//  const int_t subset_local_id = get_local_id(subset_global_id);
//
//  // get a pointer to the member data in the subset that will store the list of blocked pixels
//  std::set<std::pair<int_t,int_t> > & blocked_pixels = *obj_vec_[subset_local_id]->get_ref_subset()->pixels_blocked_by_other_subsets();
//  blocked_pixels.clear();
//  // get the list of subsets that block this one
//  std::vector<int_t> * obst_ids = &obstructing_subset_ids_->find(subset_global_id)->second;
//  // iterate over all the blocking subsets
//  for(int_t si=0;si<obst_ids->size();++si){
//    int_t global_ss = (*obst_ids)[si];
//    int_t local_ss = get_local_id(global_ss);
//    assert(local_ss>=0);
//    scalar_t u     = local_field_value(global_ss,DICe::DISPLACEMENT_X);
//    scalar_t v     = local_field_value(global_ss,DICe::DISPLACEMENT_Y);
//    scalar_t theta = local_field_value(global_ss,DICe::ROTATION_Z);
//    scalar_t dudx  = local_field_value(global_ss,DICe::NORMAL_STRAIN_X);
//    scalar_t dvdy  = local_field_value(global_ss,DICe::NORMAL_STRAIN_Y);
//    scalar_t gxy   = local_field_value(global_ss,DICe::SHEAR_STRAIN_XY);
//    scalar_t dx=0.0,dy=0.0;
//    scalar_t X=0.0,Y=0.0;
//    int_t cx=0,cy=0;
//    // iterate over all the pixels in the reference blocking subset and compute the current position
//    for(int_t px=0;px<obj_vec_[local_ss]->get_ref_subset()->num_pixels();++px){
//      dx = (1.0+dudx)*obj_vec_[local_ss]->get_ref_subset()->x(px) + gxy*obj_vec_[local_ss]->get_ref_subset()->y(px);
//      dy = (1.0+dvdy)*obj_vec_[local_ss]->get_ref_subset()->y(px) + gxy*obj_vec_[local_ss]->get_ref_subset()->x(px);
//      X = std::cos(theta)*dx - std::sin(theta)*dy + u            + obj_vec_[local_ss]->get_ref_subset()->centroid_x();
//      Y = std::sin(theta)*dx + std::cos(theta)*dy + v            + obj_vec_[local_ss]->get_ref_subset()->centroid_y();
//      cx = (int_t)X;
//      if(X - (int_t)X >= 0.5) cx++;
//      cy = (int_t)Y;
//      if(Y - (int_t)Y >= 0.5) cy++;
//      // insert a few pixels on each side of the centroid pixel:
//      for(int_t i=-obstruction_buffer_size_;i<=obstruction_buffer_size_;++i){
//        for(int_t j=-obstruction_buffer_size_;j<=obstruction_buffer_size_;++j){
//          // add these pixels to the list
//          blocked_pixels.insert(std::pair<int_t,int_t>(cy+j,cx+i));
//        }
//      } // loop over region surrounding the pixels in question
//    } // loop over blocking subset pixels
//  } // blocking subsets loop
}

void
Schema::check_for_blocking_subsets_new(const int_t subset_global_id){
//  if(obstructing_subset_ids_==Teuchos::null) return;
//  if(obstructing_subset_ids_->find(subset_global_id)==obstructing_subset_ids_->end()) return;
//  if(obstructing_subset_ids_->find(subset_global_id)->second.size()==0) return;
//
//  const int_t subset_local_id = get_local_id(subset_global_id);
//
//  // turn off pixels in subset 0 that are blocked by 1 and 2
//  // get a pointer to the member data in the subset that will store the list of blocked pixels
//  std::set<std::pair<int_t,int_t> > & blocked_pixels = *obj_vec_[subset_local_id]->get_ref_subset()->pixels_blocked_by_other_subsets();
//  blocked_pixels.clear();
//
//  // get the list of subsets that block this one
//  std::vector<int_t> * obst_ids = &obstructing_subset_ids_->find(subset_global_id)->second;
//  // iterate over all the blocking subsets
//  for(int_t si=0;si<obst_ids->size();++si){
//    int_t global_ss = (*obst_ids)[si];
//    int_t local_ss = get_local_id(global_ss);
//    assert(local_ss>=0);
//    int_t cx = obj_vec_[local_ss]->get_ref_subset()->centroid_x();
//    int_t cy = obj_vec_[local_ss]->get_ref_subset()->centroid_y();
//    Teuchos::RCP<std::vector<scalar_t> > def = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
//    (*def)[DICe::DISPLACEMENT_X]  = local_field_value(global_ss,DICe::DISPLACEMENT_X);
//    (*def)[DICe::DISPLACEMENT_Y]  = local_field_value(global_ss,DICe::DISPLACEMENT_Y);
//    (*def)[DICe::ROTATION_Z]      = local_field_value(global_ss,DICe::ROTATION_Z);
//    (*def)[DICe::NORMAL_STRAIN_X] = local_field_value(global_ss,DICe::NORMAL_STRAIN_X);
//    (*def)[DICe::NORMAL_STRAIN_Y] = local_field_value(global_ss,DICe::NORMAL_STRAIN_Y);
//    (*def)[DICe::SHEAR_STRAIN_XY] = local_field_value(global_ss,DICe::SHEAR_STRAIN_XY);
//    std::set<std::pair<int_t,int_t> > subset_pixels = obj_vec_[local_ss]->get_ref_subset()->get_deformed_shapes(def,cx,cy,obstruction_skin_factor_);
//    blocked_pixels.insert(subset_pixels.begin(),subset_pixels.end());
//  } // blocking subsets loop
}

void
Schema::write_deformed_subsets_image(){
#ifndef DICE_DISABLE_BOOST_FILESYSTEM
  if(obj_vec_.empty()) return;
  // if the subset_images folder does not exist, create it
  // TODO allow user to specify where this goes
  // If the dir is already there this step becomes a no-op
  DEBUG_MSG("Attempting to create directory : ./deformed_subsets/");
  std::string dirStr = "./deformed_subsets/";
  boost::filesystem::path dir(dirStr);
  if(boost::filesystem::create_directory(dir)) {
    DEBUG_MSG("Directory successfully created");
  }

  int_t num_zeros = 0;
  if(num_image_frames_>0){
    int_t num_digits_total = 0;
    int_t num_digits_image = 0;
    int_t decrement_total = num_image_frames_;
    int_t decrement_image = image_frame_;
    while (decrement_total){decrement_total /= 10; num_digits_total++;}
    if(image_frame_==0) num_digits_image = 1;
    else
      while (decrement_image){decrement_image /= 10; num_digits_image++;}
    num_zeros = num_digits_total - num_digits_image;
  }
  const int_t proc_id = comm_->get_rank();
  std::stringstream ss;
  ss << dirStr << "def_subsets_p_" << proc_id << "_";
  for(int_t i=0;i<num_zeros;++i)
    ss << "0";
  ss << image_frame_ << ".tif";

  // construct a copy of the base image to use as layer 0 for the output;

  const int_t w = def_img_->width();
  const int_t h = def_img_->height();

  Teuchos::ArrayRCP<intensity_t> intensities = def_img_->intensity_array();

  int_t x=0,y=0;
  int_t ox=0,oy=0;
  int_t dx=0,dy=0;
  scalar_t X=0.0,Y=0.0;

  // create output for each subset
  //for(int_t subset=0;subset<1;++subset){
  for(int_t subset=0;subset<obj_vec_.size();++subset){
    const int_t gid = obj_vec_[subset]->correlation_point_global_id();
    //if(gid==1) continue;
    // get the deformation vector for each subset
    const scalar_t u     = local_field_value(gid,DICe::DISPLACEMENT_X);
    const scalar_t v     = local_field_value(gid,DICe::DISPLACEMENT_Y);
    const scalar_t theta = local_field_value(gid,DICe::ROTATION_Z);
    const scalar_t dudx  = local_field_value(gid,DICe::NORMAL_STRAIN_X);
    const scalar_t dvdy  = local_field_value(gid,DICe::NORMAL_STRAIN_Y);
    const scalar_t gxy   = local_field_value(gid,DICe::SHEAR_STRAIN_XY);

    DEBUG_MSG("Write deformed subset " << gid << " u " << u << " v " << v << " theta " << theta << " dudx " << dudx << " dvdy " << dvdy << " gxy " << gxy);

    Teuchos::RCP<DICe::Subset> ref_subset = obj_vec_[subset]->subset();

    ox = ref_subset->centroid_x();
    oy = ref_subset->centroid_y();

    // loop over each pixel in the subset
    for(int_t px=0;px<ref_subset->num_pixels();++px){
      x = ref_subset->x(px) - ox;
      y = ref_subset->y(px) - oy;
      // stretch and shear the coordinate
      dx = (1.0+dudx)*x + gxy*y;
      dy = (1.0+dvdy)*y + gxy*x;
      // Rotation                             // translation // convert to global coordinates
      X = std::cos(theta)*dx - std::sin(theta)*dy + u            + ox;
      Y = std::sin(theta)*dx + std::cos(theta)*dy + v            + oy;
      X = static_cast<int_t>(X);
      Y = static_cast<int_t>(Y);
      // TODO add checks for is deactivated
      if(X>=0&&X<w&&Y>=0&&Y<h){
//        if(!ref_subset->is_active(px)){
//          intensities[Y*w+X] = 75;
//        }
//        else{
          // color shows correlation quality
          intensities[Y*w+X] = 100;//ref_subset->per_pixel_gamma(px)*85000;
//        }
        // trun all deactivated pixels white
//        if(ref_subset->is_deactivated_this_step(px)){
//          intensities[Y*w+X] = 255;
//        }
      }
    } // pixel loop

  } // subset loop

  Teuchos::RCP<Image> layer_0_image = Teuchos::rcp(new Image(w,h,intensities));
  layer_0_image->write_tiff(ss.str());
#else
  DEBUG_MSG("Warning, write_deformed_image() was called, but Boost::filesystem is not enabled making this a no-op.");
#endif
}


int_t
Schema::strain_window_size(const int_t post_processor_index)const{
  assert(post_processors_.size()>post_processor_index);
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
    field_names_.push_back(to_string(DICe::COORDINATE_X));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::COORDINATE_Y));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::DISPLACEMENT_X));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::DISPLACEMENT_Y));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::ROTATION_Z));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::NORMAL_STRAIN_X));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::NORMAL_STRAIN_Y));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::SHEAR_STRAIN_XY));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::SIGMA));
    post_processor_ids_.push_back(-1);
    field_names_.push_back(to_string(DICe::STATUS_FLAG));
    post_processor_ids_.push_back(-1);
  }
  else{
    // get the total number of field names
    const int_t num_names = params->numParams();
    // get the max index
    field_names_.resize(num_names);
    post_processor_ids_.resize(num_names);
    int_t max_index = 0;
    std::set<int_t> indices;

    // read in the names and indices by iterating the parameter list
    for(Teuchos::ParameterList::ConstIterator it=params->begin();it!=params->end();++it){
      std::string string_field_name = it->first;
      stringToUpper(string_field_name);
      int_t post_processor_id = -1;
      bool paramValid = false;
      for(int_t j=0;j<MAX_FIELD_NAME;++j){
        if(string_field_name==fieldNameStrings[j])
          paramValid = true;
      }
      // see if this field is in one of the post processors instead
      for(int_t j=0;j<schema_->post_processors()->size();++j){
        for(int_t k=0;k<(*schema_->post_processors())[j]->field_names()->size();++k){
          if(string_field_name==(*(*schema_->post_processors())[j]->field_names())[k]){
            paramValid = true;
            post_processor_id = j;
          }
        }
      }
      if(!paramValid){
        std::cout << "Error: invalid field name requested in output spec: " << string_field_name << std::endl;
        assert(false);
      }
      const int_t field_index = params->get<int_t>(string_field_name);
      if(field_index>num_names-1||field_index<0){
        std::cout << "Error: field index in output spec is invalid " << field_index << std::endl;
        assert(false);
      }
      // see if this index exists already
      if(indices.find(field_index)!=indices.end()){
        std::cout << "Error: same field index assigned to multiple fields in output spec" << field_index << std::endl;
        assert(false);
      }
      indices.insert(field_index);
      if(field_index > max_index) max_index = field_index;
      field_names_[field_index] = string_field_name;
      post_processor_ids_[field_index] = post_processor_id;
    } // loop over field names in the parameterlist
    if(max_index!=num_names-1){
      std::cout << "Error: The max field index in the output spec is not equal to the number of fields, num_fields " << field_names_.size() << " max_index " << max_index << std::endl;
      assert(false);
    }
  }
};

void
Output_Spec::write_header(std::FILE * file,
  const std::string & row_id){
  assert(file);
  fprintf(file,"***\n");
  fprintf(file,"*** Digital Image Correlation Engine (DICe), Copyright 2014 Sandia Corporation\n");
  fprintf(file,"***\n");
  fprintf(file,"*** Reference image: %s \n",schema_->ref_img()->file_name().c_str());
  fprintf(file,"*** Deformed image: %s \n",schema_->def_img()->file_name().c_str());
  if(schema_->analysis_type()==GLOBAL_DIC){
    fprintf(file,"*** DIC method : global \n");
  }
  else{
    fprintf(file,"*** DIC method : local \n");
  }
  fprintf(file,"*** Correlation method: ZNSSD\n");
  std::string interp_method = to_string(schema_->interpolation_method());
  fprintf(file,"*** Interpolation method: %s\n",interp_method.c_str());
  fprintf(file,"*** Image gradient method: FINITE_DIFFERENCE\n");
  std::string opt_method = to_string(schema_->optimization_method());
  fprintf(file,"*** Optimization method: %s\n",opt_method.c_str());
  std::string proj_method = to_string(schema_->projection_method());
  fprintf(file,"*** Projection method: %s\n",proj_method.c_str());
  std::string init_method = to_string(schema_->initialization_method());
  fprintf(file,"*** Guess initialization method: %s\n",init_method.c_str());
  fprintf(file,"*** Seed location: N/A\n");
  fprintf(file,"*** Shape functions: ");
  if(schema_->translation_enabled())
    fprintf(file,"Translation (u,v) ");
  if(schema_->rotation_enabled())
    fprintf(file,"Rotation (theta) ");
  if(schema_->normal_strain_enabled())
    fprintf(file,"Normal Strain (ex,ey) ");
  if(schema_->shear_strain_enabled())
    fprintf(file,"Shear Strain (gamma_xy) ");
  fprintf(file,"\n");
  fprintf(file,"*** Incremental correlation: false\n");
  if(schema_->analysis_type()==GLOBAL_DIC){
    fprintf(file,"*** Mesh size: %i\n",schema_->mesh_size());
    fprintf(file,"*** Step size: N/A\n");
  }
  else{
    fprintf(file,"*** Subset size: %i\n",schema_->subset_dim());
    fprintf(file,"*** Step size: x %i y %i (-1 implies not regular grid)\n",schema_->step_size_x(),schema_->step_size_y());
  }
  if(schema_->post_processors()->size()==0)
    fprintf(file,"*** Strain window: N/A\n");
  else
    fprintf(file,"*** Strain window size in pixels: %i (only first strain post-processor is reported)\n",schema_->strain_window_size(0));
  fprintf(file,"*** Coordinates given with (0,0) as upper left corner of image, x positive right, y positive down\n");
  fprintf(file,"***\n");
  if(!omit_row_id_)
    fprintf(file,"%s%s",row_id.c_str(),delimiter_.c_str());
  for(int_t i=0;i<field_names_.size();++i){
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
  assert(field_names_.size()==post_processor_ids_.size());
  for(int_t i=0;i<field_names_.size();++i)
  {
    // if the field_name is from one of the schema fields, get the information from the schema
    scalar_t value = 0.0;
    if(post_processor_ids_[i]==-1)
      value = schema_->field_value(field_value_index,string_to_field_name(field_names_[i]));
    // otherwise the field must belong to a post processor
    else{
      assert(post_processor_ids_[i]>=0 && post_processor_ids_[i] < post_processor_ids_[i]<schema_->post_processors()->size());
      value = (*schema_->post_processors())[post_processor_ids_[i]]->field_value(field_value_index,field_names_[i]);
    }
    if(i==0)
      fprintf(file,"%4.4E",value);
    else
      fprintf(file,"%s%4.4E",delimiter_.c_str(),value);
  }
  fprintf(file,"\n"); // the space before end of line is important for parsing in the output diff tool
}

}// End DICe Namespace
