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

#ifndef DICE_SCHEMA_H
#define DICE_SCHEMA_H

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Shape.h>
#include <DICe_MultiField.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <map>

namespace DICe {

// forward declaration of Objective
class Objective;

// forward declaration of Output_Spec
class Output_Spec;

// forward declaration of Post_Processor
class Post_Processor;

/// \class DICe::Schema
/// \brief The centralized container for the correlation plan and parameters
///
/// This class holds all of the correlation parameters and provides the basic structure for
/// how the correlation steps are executed. Each schema holds a pointer to a reference and
/// deformed DICe::Image. Many of the pre-correlation steps are also done by a DICe::Schema
/// (for example filtering the images, or computing the gradients).
///
/// All of the field values are stored by a DICe::Schema which provides an interface to external
/// data structures as well.
///
/// Several methods are provided that expose the parameters.
class DICE_LIB_DLL_EXPORT
Schema {

public:

  /// Multifield RCP
  typedef typename Teuchos::RCP<MultiField> mf_rcp;
  /// Map RCP
  typedef typename Teuchos::RCP<MultiField_Map> map_rcp;
  /// Exporter RCP
  typedef typename Teuchos::RCP<MultiField_Exporter> exp_rcp;
  /// Importer RCP
  typedef typename Teuchos::RCP<MultiField_Importer> imp_rcp;
  /// Communicator RCP
  typedef typename Teuchos::RCP<MultiField_Comm> comm_rcp;

  /// \brief Constructor that takes string names of images as inputs
  /// \param refName String name of reference image
  /// \param defName String name of deformed image
  /// \param params Correlation parameters
  ///
  /// NOTE: Only enabled if boost is enabled
  Schema(const std::string & refName,
    const std::string & defName,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// \brief Constructor that takes arrays of intensity values as inputs
  /// \param img_width Image width (must be the same for the reference and deformed images)
  /// \param img_height Image height (must be the same for the reference and deformed images)
  /// \param refRCP Array of intensity values for the reference image
  /// \param defRCP Array of intensity values for deformed image
  /// \param params Correlation parameters
  Schema(const int_t img_width,
    const int_t img_height,
    const Teuchos::ArrayRCP<intensity_t> refRCP,
    const Teuchos::ArrayRCP<intensity_t> defRCP,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// \brief Constructor that takes already instantiated images as inputs
  /// \param ref_img Reference DICe::Image
  /// \param def_img Deformed DICe::Image
  /// \param params Correlation Parameters
  Schema(Teuchos::RCP<Image> ref_img,
    Teuchos::RCP<Image> def_img,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  virtual ~Schema(){};

  /// \brief Sets the default values for the schema's member data and other initialization tasks
  /// \param params Optional correlation parameters
  void default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// If a schema's parameters are changed, set_params() must be called again
  /// any params that aren't set are reset to the default value (so this method
  /// can be used with a null params to reset the schema's parameters).
  void set_params(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Replace the deformed image for this Schema (only enabled with boost)
  void set_def_image(const std::string & defName);

  /// Replace the deformed image using an intensity array
  void set_def_image(const int_t img_width,
    const int_t img_height,
    const Teuchos::ArrayRCP<intensity_t> defRCP);

  /// Replace the deformed image using an image
  void set_def_image(Teuchos::RCP<Image> img);

  /// Replace the deformed image for this Schema (only enabled with boost)
  void set_ref_image(const std::string & refName);

  /// Replace the deformed image using an intensity array
  void set_ref_image(const int_t img_width,
    const int_t img_height,
    const Teuchos::ArrayRCP<intensity_t> refRCP);

  /// Set the element size of the mesh (only for global DIC)
  void set_mesh_size(const int_t mesh_size){
    assert(mesh_size>=1);
    mesh_size_ = mesh_size;
  }

  /// Returns the element size for global DIC (-1 if local DIC)
  int_t mesh_size()const{
    return mesh_size_;
  }

  /// \brief Initializes the data structures for the schema
  /// \param num_pts Number of correlation points
  /// \param subset_size int_t of the subsets, (use -1 if conformals are defined for all subsets, otherwise all subsets without a
  /// conformal subset def will be square and assigned this subset size.
  /// \param conformal_subset_defs Optional definition of conformal subsets
  /// \param neighbor_ids A vector (of length num_pts) that contains the neighbor id to use when initializing the solution by neighbor value
  void initialize(const int_t num_pts,
    const int_t subset_size,
    Teuchos::RCP<std::map<int_t,Conformal_Area_Def> > conformal_subset_defs=Teuchos::null,
    Teuchos::RCP<std::vector<int_t> > neighbor_ids=Teuchos::null);

  /// \brief Initializes the guess for a subset based on a phase correlation of the whole image
  /// \param deformation the deformation guess
  void initialize_by_phase_correlation(Teuchos::RCP<std::vector<scalar_t> > & deformation);

  /// \brief Simple initialization routine that sets up a regular grid of correlation points throughout the image spaced according to the step sizes
  /// \param step_size_x Spacing of the correlation points in x
  /// \param step_size_y Spacing of the correlation points in y
  /// \param subset_size int_t of the square subsets in pixels
  void initialize(const int_t step_size_x,
    const int_t step_size_y,
    const int_t subset_size);

  /// Conduct the correlation
  void execute_correlation();

  /// Returns if the field storage is initilaized
  int_t is_initialized()const{
    return is_initialized_;
  }

  /// Returns a pointer to the reference DICe::Image
  Teuchos::RCP<Image> ref_img()const{
    return ref_img_;
  }

  /// Returns a pointer to the deformed DICe::Image
  Teuchos::RCP<Image> def_img()const{
    return def_img_;
  }

  /// Returns the max solver iterations allowed for the fast (gradient based) algorithm
  int_t max_solver_iterations_fast()const{
    return max_solver_iterations_fast_;
  }

  /// Returns the max solver iterations allowed for the robust (simplex) algorithm
  int_t max_solver_iterations_robust()const{
    return max_solver_iterations_robust_;
  }

  /// Returns the robust solver convergence tolerance
  double robust_solver_tolerance()const{
    return robust_solver_tolerance_;
  }

  /// Returns the threshold for gamma where the solve will be skipped if gamma < threshold
  double skip_solve_gamma_threshold()const{
    return skip_solve_gamma_threshold_;
  }

  /// Returns the fast solver convergence tolerance
  double fast_solver_tolerance()const{
    return fast_solver_tolerance_;
  }

  /// Returns the variation applied to the displacement initial guess in the simplex method
  double robust_delta_disp()const{
    return robust_delta_disp_;
  }

  /// Returns the variation applied to the rotation initial guess in the simplex method
  double robust_delta_theta()const{
    return robust_delta_theta_;
  }

  /// Returns the reference and deformed image width
  int_t img_width()const{
    return ref_img_->width();
  }

  /// Returns the reference and deformed image height
  int_t img_height()const{
    return ref_img_->height();
  }

  /// Returns the size of the strain window in pixels for the selected post_processor
  /// \param post_processor_index The index of the post processor to get the window size for
  int_t strain_window_size(const int_t post_processor_index)const;

  /// Returns the size of a square subset
  ///
  /// This is called subset_dim to discourage use as a call to subset_size
  /// subset_size does not exist for a conformal subset (there is no notion of width and height for conformal)
  int_t subset_dim()const{
    return subset_dim_;
  }

  /// \brief Sets the size of square subsets (not used for conformal)
  /// \param subset_dim Square size of the subsets
  void set_subset_dim(const int_t subset_dim){
    subset_dim_=subset_dim;
  }

  /// Returns the step size in x direction for a square subset
  ///
  /// If this is a conformal analysis or if the subsets are not in a regular grid this returns -1
  int_t step_size_x()const{
    return step_size_x_;
  }

  /// \brief Sets the step sizes for this schema
  ///
  /// \param step_size_x Step size in the x-direction
  /// \param step_size_y Step size in the y-direction
  void set_step_size(const int_t step_size_x,
    const int_t step_size_y){
    step_size_x_=step_size_x;
    step_size_y_=step_size_y;
  }

  /// \brief Sets the step sizes for this schema
  ///
  /// \param step_size Step size in the x and y-direction
  void set_step_size(const int_t step_size){
    step_size_x_=step_size;
    step_size_y_=step_size;
  }

  /// Returns the step size in y direction for a square subset
  ///
  /// If this is a conformal analysis or if the subsets are not in a regular grid this returns -1
  int_t step_size_y()const{
    return step_size_y_;
  }

  /// Returns the number of correlation points
  int_t data_num_points()const{
    return data_num_points_;
  }

  /// Returns true if the analysis is global DIC
  Analysis_Type analysis_type()const{
    return analysis_type_;
  }

  /// \brief Return either the distributed field vector value or the all-owned
  /// vector value, depending on the flag argument
  /// \param global_id Global ID of the element
  /// \param name Field name (see DICe_Types.h for valid field names)
  scalar_t & field_value(const int_t global_id,
    const Field_Name name){
    assert(!distributed_fields_being_modified_ && "Error: Attempting to modify or access an all-owned field, but the distributed"
      " fields have the lock, sync_dist_to_all() must be called first to re-enable access to the all-owned fields.");
    assert(global_id<data_num_points_);
    assert(name<MAX_FIELD_NAME);
    return fields_->global_value(global_id,name);
  }

  /// \brief Return either the distributed field vector value or the all-owned
  /// vector value, depending on the flag argument for frame_(n-1)
  /// \param global_id Global ID of the element
  /// \param name Field name (see DICe_Types.h for valid field names)
  scalar_t & field_value_nm1(const int_t global_id,
    const Field_Name name){
    assert(!distributed_fields_being_modified_ && "Error: Attempting to modify or access an all-owned field, but the distributed"
      " fields have the lock, sync_dist_to_all() must be called first to re-enable access to the all-owned fields.");
    assert(global_id<data_num_points_);
    assert(name<MAX_FIELD_NAME);
    return fields_nm1_->global_value(global_id,name);
  }

  /// \brief Return either the distributed field vector value or the all-owned
  /// vector value, depending on the flag argument
  /// \param global_id Global ID of the element
  /// \param name Field name (see DICe_Types.h for valid field names)
  scalar_t & local_field_value(const int_t global_id,
    const Field_Name name){
    assert(global_id<data_num_points_);
    assert(name<MAX_FIELD_NAME);
#ifdef HAVE_MPI
    if(target_field_descriptor_==DISTRIBUTED){
      assert(distributed_fields_being_modified_ && "Error: Attempting to modify or access a distributed field, but the all-owned"
          " fields have the lock, sync_all_to_dist() must be called first to enable access to the distributed fields.");
      assert(dist_map_->get_local_element(global_id)>=0);
      return dist_fields_->global_value(global_id,name);
    }
    else if(target_field_descriptor_==DISTRIBUTED_GROUPED_BY_SEED){
      assert(distributed_fields_being_modified_ && "Error: Attempting to modify or access a distributed field, but the all-owned"
          " fields have the lock, sync_all_to_dist() must be called first to enable access to the distributed fields.");
      assert(seed_dist_map_->get_local_element(global_id)>=0);
      return seed_dist_fields_->global_value(global_id,name);
    }
#endif
    assert(target_field_descriptor_==ALL_OWNED);
    return fields_->global_value(global_id,name);
  }

  /// \brief Return either the distributed field vector value or the all-owned
  /// vector value, depending on the flag argument for frame_(n-1)
  /// \param global_id Global ID of the element
  /// \param name Field name (see DICe_Types.h for valid field names)
  scalar_t & local_field_value_nm1(const int_t global_id,
    const Field_Name name){
    assert(global_id<data_num_points_);
    assert(name<MAX_FIELD_NAME);
#ifdef HAVE_MPI
    if(target_field_descriptor_==DISTRIBUTED){
      assert(distributed_fields_being_modified_ && "Error: Attempting to modify or access a distributed field, but the all-owned"
        " fields have the lock, sync_all_to_dist() must be called first to enable access to the distributed fields.");
      assert(dist_map_->get_local_element(global_id)>=0);
      return dist_fields_nm1_->global_value(global_id,name);
    }
    else if(target_field_descriptor_==DISTRIBUTED_GROUPED_BY_SEED){
      assert(distributed_fields_being_modified_ && "Error: Attempting to modify or access a distributed field, but the all-owned"
          " fields have the lock, sync_all_to_dist() must be called first to enable access to the distributed fields.");
      assert(seed_dist_map_->get_local_element(global_id)>=0);
      return seed_dist_fields_nm1_->global_value(global_id,name);
    }
#endif
    assert(target_field_descriptor_==ALL_OWNED);
    return fields_nm1_->global_value(global_id,name);
  }

  /// \brief Save off the current solution into the storage for frame n - 1 (only used if projection_method is VELOCITY_BASED)
  /// \param global_id global ID of correlation point
  void save_off_fields(const int_t global_id){
    DEBUG_MSG("Saving off solution nm1 for subset (global id) " << global_id);
    for(int_t i=0;i<MAX_FIELD_NAME;++i){
#ifdef HAVE_MPI
      if(target_field_descriptor_==DISTRIBUTED)
        dist_fields_nm1_->global_value(global_id,i) = dist_fields_->global_value(global_id,i);
      else if(target_field_descriptor_==DISTRIBUTED_GROUPED_BY_SEED)
        seed_dist_fields_nm1_->global_value(global_id,i) = seed_dist_fields_->global_value(global_id,i);
      else
        fields_nm1_->global_value(global_id,i) = fields_->global_value(global_id,i);
#else
      fields_nm1_->global_value(global_id,i) = fields_->global_value(global_id,i);
#endif
    }
  };

  /// \brief Print the field values to screen or to a file
  /// \param fileName Optional file name
  ///
  /// If fileName is empty, write the fields to the screen.
  /// If fielName is not empty, write the fields to a file (appending)
  void print_fields(const std::string & fileName = "");

  /// Returns a pointer to the Conformal_Subset_Def vector
  Teuchos::RCP<std::map<int_t,Conformal_Area_Def> > conformal_subset_defs(){
    return conformal_subset_defs_;
  }

  /// Returns the correlation routine (see DICe_Types.h for valid values)
  Correlation_Routine correlation_routine()const{
    return correlation_routine_;
  }

  /// Returns the interpolation method (see DICe_Types.h for valid values)
  Interpolation_Method interpolation_method()const{
    return interpolation_method_;
  }

  /// Returns the optimization method (see DICe_Types.h for valid values)
  Optimization_Method optimization_method()const{
    return optimization_method_;
  }

  /// Returns the initilaization method (see DICe_Types.h for valid values)
  Initialization_Method initialization_method()const{
    return initialization_method_;
  }

  /// Returns the projection method (see DICe_Types.h for valid values)
  Projection_Method projection_method()const{
    return projection_method_;
  }

  /// set up the distributed map so that it respects dependencies among obstructions
  void create_obstruction_dist_map();

  /// set up the distributed map so that it respects dependencies among seeds
  void create_seed_dist_map(Teuchos::RCP<std::vector<int_t> > neighbor_ids);

  /// \brief Create an image that shows the correlation points
  /// \param fileName String name of file to for output
  /// \param use_def_image True if the deformed image should be used, otherwise the reference image is used
  /// \param use_one_point True if only one correlation point should be used (helpful if the point density is high)
  ///
  /// Only works if boost is enabled.
  /// WARNING: This only works for square subsets.
  // TODO this only works for square subsets, make it work for the conformal
  void write_control_points_image(const std::string & fileName,
    const bool use_def_image = false,
    const bool use_one_point = false);

  /// \brief Write the solution vector to a text file
  /// \param output_folder Name of the folder for output (the file name is fixed)
  /// \param prefix Optional string to use as the file prefix
  /// \param separate_files_per_subset Forces the output to be one file per subset listing each frame on a new row.
  /// The default is one file per frame with all subsets listed one per row.
  /// \param type Type of file to write (currently only TEXT_FILE is implemented)
  // TODO export in exodus format
  void write_output(const std::string & output_folder,
    const std::string & prefix="DICe_solution",
    const bool separate_files_per_subset=false,
    const Output_File_Type type = TEXT_FILE);

  /// \brief Write an image that shows all the subsets current position and shape
  /// field values from.
  ///
  /// WARNING: This is meant only for the SL_ROUTINE where there are only a few subsets to track
  void write_deformed_subsets_image();

  /// \brief Write an image that shows all the subsets current position and shape
  /// field values from.
  ///
  /// WARNING: This is meant only for the SL_ROUTINE where there are only a few subsets to track
  void write_deformed_subsets_image_new();

  /// \brief See if any of the subsets have crossed each other's paths blocking each other
  ///
  /// The potential for two subsets to block each other should have been set by calling
  /// set_obstructing_subset_ids() on the schema by this point.
  /// WARNING: This is meant only for the SL_ROUTINE where there are only a few subsets to track
  void check_for_blocking_subsets(const int_t subset_global_id);

  /// \brief See if any of the subsets have crossed each other's paths blocking each other
  ///
  /// The potential for two subsets to block each other should have been set by calling
  /// set_obstructing_subset_ids() on the schema by this point.
  /// WARNING: This is meant only for the SL_ROUTINE where there are only a few subsets to track
  void check_for_blocking_subsets_new(const int_t subset_global_id);

  /// \brief Orchestration of how the correlation is conducted
  /// \param obj A single DICe::Objective
  ///
  /// The process is as followa:
  /// 1. The first subset is correlated based on whatever initial guess
  /// happens to be in the data array (could have been specified by user or zeros).
  /// 2. The rest of the subsets use the neighboring subset's solution as an initial guess if the
  /// initialization method is DICe::USE_NEIGHBOR_VALUES. Otherwise, the initial guess is taken
  /// from whatever solution is in the field_values (DICe::USE_FIELD_VALUES).
  ///
  /// 3. For the optimization, if the method is DICe::SIMPLEX, only the simpex method (computeUpdateRobust()) will be
  /// used. For the DICe::GRADIENT_BASED method, only the computeUpdateFast() method will be used. The other
  /// options are first try simplex, then gradient based if it fails (DICe::SIMPLEX_THEN_GRADIENT_BASED) or start
  /// with gradient based then simplex (DICe::GRADIENT_BASED_THEN_SIMPLEX).
  void generic_correlation_routine(Teuchos::RCP<Objective> obj);

  /// \brief As an alternative to the generic routine, this routine spends a lot
  /// more time handling various failure cases. It tests for the solution gamma value after each
  /// step to ensure the correlation is still good. It also tests for jumps in the solution values.
  /// It uses a 10 step projection method to initialize the solution and checks for errors in the
  /// predicted vs. computed solution to restart the multistep projection process. In this routine
  /// obstructions are detected using statistics on the pixel intensity values rather than based on
  /// obstruction subsets that need to be tracked in the generic correlation routine.
  /// \param obj The objective function that holds the ref and def subset along with several methods
  void subset_evolution_routine(Teuchos::RCP<Objective> obj);

  /// Returne true if regularization should be used in the objective evaluation
  bool use_objective_regularization()const{
    return use_objective_regularization_;
  }

  // shape function controls:
  /// Returns true if translation shape functions are enabled
  bool translation_enabled() const {
    return enable_translation_;
  }

  /// Returns true if rotation shape functions are enabled
  bool rotation_enabled() const {
    return enable_rotation_;
  }

  /// Returns true if normal strain shape functions are enabled
  bool normal_strain_enabled() const {
    return enable_normal_strain_;
  }

  /// Returns true if shear strain shape functions are enabled
  bool shear_strain_enabled() const {
    return enable_shear_strain_;
  }

  /// Print images with the deformed shape of the subset:
  bool output_deformed_subset_images()const{
    return output_deformed_subset_images_;
  }

  /// Print images with the evolved intensity profile
  bool output_evolved_subset_images()const{
    return output_evolved_subset_images_;
  }

  /// Evolve subsets as more pixels become visible that were previously obstructed
  bool use_subset_evolution()const{
    return use_subset_evolution_;
  }

  /// True if the check for obstructed pixels should ocurr at every iteration
  bool update_obstructed_pixels_each_iteration()const{
    return update_obstructed_pixels_each_iteration_;
  }

  /// True if the gamma values should be normalized by the number of active pixels
  bool normalize_gamma_with_active_pixels()const{
    return normalize_gamma_with_active_pixels_;
  }

  /// Enable translation shape functions
  void enable_translation(const bool flag){
    enable_translation_ = flag;
  }

  /// Enable rotation shape functions
  void enable_rotation(const bool flag){
    enable_rotation_ = flag;
  }

  /// Enable normal strain shape functions
  void enable_normal_strain(const bool flag){
    enable_normal_strain_ = flag;
  }

  /// Enable shear strain shape functions
  void enable_shear_strain(const bool flag){
    enable_shear_strain_ = flag;
  }

  /// Updates the current image frame number
  void update_image_frame(){
    image_frame_++;
  }

  /// Returns the current image frame (Nonzero only if multiple images are included in the sequence)
  int_t image_frame()const{
    return image_frame_;
  }

  /// Returns the number of images in the set (-1 if it has not been set)
  int_t num_image_frames() const{
    return num_image_frames_;
  }

  /// Set the number of images in the sequence (defualts to -1 if not set)
  void set_num_image_frames(const int_t num_frames){
    num_image_frames_ = num_frames;
  }

  /// Returns true if the output has a specific order to the fields
  bool has_output_spec()const{
    return has_output_spec_;
  }

  /// Returns the size of the obstruction buffer
  int_t obstruction_buffer_size()const{
    return obstruction_buffer_size_;
  }

  /// Returns the size of the obstruction skin
  double obstruction_skin_factor()const{
    return obstruction_skin_factor_;
  }

  /// Returns the integration order for each pixel
  int_t pixel_integration_order()const{
    return pixel_integration_order_;
  }

  /// Return access to the post processors vector
  const std::vector<Teuchos::RCP<Post_Processor> > * post_processors(){
    return &post_processors_;
  }

  /// Provide access to the list of owning elements for each pixel
  std::vector<int_t> * pixels_owning_element_global_id(){
    return &pixels_owning_element_global_id_;
  }

  /// Return a pointer to the connectivity matrix
  Teuchos::SerialDenseMatrix<int_t,int_t> * connectivity(){
    return &connectivity_;
  }

  /// Return the jump tolerance for rotations
  double theta_jump_tol()const{
    return theta_jump_tol_;
  }

  /// Return the jump tolerance for displacements
  double disp_jump_tol()const{
    return disp_jump_tol_;
  }

  /// \brief EXPERIMENTAL sets the layering order of subets
  /// \param id_vec Pointer to map of vectors, each vector contains the ids for that particular subset in the order they will be layered if they cross paths
  ///
  /// If two subsets have the potential to cross each other's path, the Schema needs to know which one will
  /// be on top so that the pixels obscured on the lower subset can be turned off
  void set_obstructing_subset_ids(Teuchos::RCP<std::map<int_t,std::vector<int_t> > > id_vec){
    if(id_vec==Teuchos::null)return;
    obstructing_subset_ids_ = id_vec;
  }

  /// Return a pointer to the distribution map
  const map_rcp dist_map()const{
    return dist_map_;
  }

  /// Copy distributed fields to serial fields
  void sync_fields_dist_to_all(){
    if(target_field_descriptor_==ALL_OWNED) return; // NO-OP
#ifdef HAVE_MPI
    if(target_field_descriptor_==DISTRIBUTED){
      fields_->do_import(*dist_fields_,*importer_);
      fields_nm1_->do_import(*dist_fields_nm1_,*importer_);
    }
    else if(target_field_descriptor_==DISTRIBUTED_GROUPED_BY_SEED){
      fields_->do_import(*seed_dist_fields_,*seed_importer_);
      fields_nm1_->do_import(*seed_dist_fields_,*seed_importer_);
    }
    else{
      assert(false && "Error: unknown field descriptor.");
    }
    distributed_fields_being_modified_ = false;
#else
    return; // NO-OP without MPI
#endif
  }

  /// Copy serial fields to distributedsinsert
  void sync_fields_all_to_dist(){
    if(target_field_descriptor_==ALL_OWNED) return; // NO-OP
#ifdef HAVE_MPI
    distributed_fields_being_modified_ = true;
    if(target_field_descriptor_==DISTRIBUTED){
      dist_fields_->do_export(*fields_,*exporter_);
      dist_fields_nm1_->do_export(*fields_nm1_,*exporter_);
    }
    else if(target_field_descriptor_==DISTRIBUTED_GROUPED_BY_SEED){
      seed_dist_fields_->do_export(*fields_,*seed_exporter_);
      seed_dist_fields_nm1_->do_export(*fields_nm1_,*seed_exporter_);
    }
    else{
      assert(false && "Error: unknown field descriptor.");
    }
    //Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    //dist_fields_->describe(*fos,Teuchos::VERB_EXTREME);
#else
    return; // NO-OP without MPI
#endif
  }

  /// Provide access to the distributed map:
  int_t get_local_id(const int_t gid)const{
    if(target_field_descriptor_==DISTRIBUTED)
      return dist_map_->get_local_element(gid);
    else if(target_field_descriptor_==DISTRIBUTED_GROUPED_BY_SEED)
      return seed_dist_map_->get_local_element(gid);
    else
      return all_map_->get_local_element(gid);
  }

  /// Returns a pointer to the params that were used to construct this schema
  Teuchos::RCP<Teuchos::ParameterList> get_params(){
    return init_params_;
  }

private:
  /// Pointer to communicator (can be serial)
  comm_rcp comm_;
  /// Pointer to map that defines the parallel decomposition
  map_rcp dist_map_;
  /// Pointer to map that defines parallel decomposition according to
  /// the number of seeds or regions of interest in the analysis
  map_rcp seed_dist_map_;
  /// Pointer to map that holds all elements on all processors
  map_rcp all_map_;
  /// Pointer to importer for managing distributed fields
  imp_rcp importer_;
  /// Pointer to exporter for managing distributed fields
  exp_rcp exporter_;
  /// Pointer to importer for managing distributed fields grouped by seed
  imp_rcp seed_importer_;
  /// Pointer to exporter for managing distributed fields grouped by seed
  exp_rcp seed_exporter_;
#ifdef HAVE_MPI
  /// Pointer to a map of vectors that hold the distributed fields.
  /// These are meant to be internal and not exposed to the user via field_values();
  mf_rcp dist_fields_;
  /// Pointer to a map of vectors that hold the distributed fields at step n_minus_1
  /// These are meant to be internal and not exposed to the user via field_values();
  mf_rcp dist_fields_nm1_;
  /// Pointer to a map of vectors that hold the distributed fields grouped by seed.
  /// These are meant to be internal and not exposed to the user via field_values();
  mf_rcp seed_dist_fields_;
  /// Pointer to a map of vectors that hold the distributed fields grouped by seed at step n_minus_1
  /// These are meant to be internal and not exposed to the user via field_values();
  mf_rcp seed_dist_fields_nm1_;
#endif
  /// Determines where the field_value calls should look for the field vectors
  Target_Field_Descriptor target_field_descriptor_;
  /// Lock the fields if the distributed vectors are being modified
  bool distributed_fields_being_modified_;
  /// Pointer to a map of vectors that hold the fields
  mf_rcp fields_;
  /// Pointer to a map of vectors that hold the fields at step n_minus_1
  ///
  /// Managing the fields is a little tricky. If MPI is not enabled
  /// There is only one set of vectors. All calls to local_field_value and field_value
  /// do the same thing (these calls point to the same vector, which is not distributed).
  /// However, if MPI is enabled, there are two sets of field vectors, for the fields_ and
  /// fields_nm1_ vectors, all processor own all elements. For the dist_fields_ and
  /// dist_fields_nm1_ vectors, these are distributed across processors with a one-to-one map.
  /// In this case, calls to field_value point to the all shared vectors, and local_field_value points
  /// to the distributed vectors and can only be given an index that is a valid local index for this
  /// processor. All of the work on the distributed vector occurrs inside the execute_correlation
  /// function. When this function is entered the data has to be copied over from the all-owned
  /// vectors to the distributed. When the calculations are finished from execute_correlation(), the data
  /// has to be copied back to the all-owned vectors.
  mf_rcp fields_nm1_;
  /// Keeps track of the gids of subsets that are local to this process.
  Teuchos::ArrayView<const int_t> this_proc_subset_global_ids_;
  /// Vector of objective classes
  std::vector<Teuchos::RCP<Objective> > obj_vec_;
  /// Pointer to reference image
  Teuchos::RCP<Image> ref_img_;
  /// Pointer to deformed image
  Teuchos::RCP<Image> def_img_;
  /// Pointer to previous image
  Teuchos::RCP<Image> prev_img_;
  /// Vector of pointers to the post processing utilities
  std::vector<Teuchos::RCP<Post_Processor> > post_processors_;
  /// True if any post_processors have been activated
  bool has_post_processor_;
  /// For constrained optimiation, this lists the owning element global id for each pixel:
  std::vector<int_t> pixels_owning_element_global_id_;
  /// Connectivity matrix for the global DIC method
  Teuchos::SerialDenseMatrix<int_t,int_t> connectivity_;
  /// Square subset size (used only if subsets are not conformal)
  int_t subset_dim_;
  /// Regular grid subset spacing in x direction (used only if subsets are not conformal)
  int_t step_size_x_;
  /// Regular grid subset spacing in y direction (used only if subsets are not conformal)
  int_t step_size_y_;
  /// Element size for the global method
  int_t mesh_size_;
  /// Generic strain window size (horizon for nlvc, convolution support for keys, strain window size for vsg)
  int_t strain_window_size_;
  /// Map of subset id and geometry definition
  Teuchos::RCP<std::map<int_t,Conformal_Area_Def> > conformal_subset_defs_;
  /// Maximum number of iterations in the subset evolution routine
  int_t max_evolution_iterations_;
  /// Maximum solver iterations for computeUpdateFast() for an objective
  int_t max_solver_iterations_fast_;
  /// Maximum solver iterations for computeUpdateRobust() for an objective
  int_t max_solver_iterations_robust_;
  /// Fase solver convergence tolerance
  double fast_solver_tolerance_;
  /// Robust solver convergence tolerance
  double robust_solver_tolerance_;
  /// If gamma is less than this for the initial guess, the solve is skipped
  double skip_solve_gamma_threshold_;
  /// The number of correlation points
  int_t data_num_points_;
  /// Are the output fields and columns specified by the user?
  bool has_output_spec_;
  /// Determines how the output is formatted
  Teuchos::RCP<DICe::Output_Spec> output_spec_;
  /// Stores current fame number for a sequence of images
  int_t image_frame_;
  /// Stores the number of images in the sequence
  int_t num_image_frames_;
  /// Displacement jump tolerance. If the displacement solution is larger than this from the previous frame
  /// it is rejected
  double disp_jump_tol_;
  /// Theta jump tolerance. If the theta solution is larger than this from the previous frame
  /// it is rejected
  double theta_jump_tol_;
  /// int_t of the variation to apply to the initial displacement guess used to construct the simplex
  double robust_delta_disp_;
  /// int_t of the variation to apply to the initial rotation guess used to construct the simplex
  double robust_delta_theta_;
  /// int_t of the buffer to add around subset obstructions
  int_t obstruction_buffer_size_;
  /// Determines how many subdivisions each pixel is cut into for integration purposes
  int_t pixel_integration_order_;
  /// Factor that increases or decreases the size of obstructions
  double obstruction_skin_factor_;
  /// DICe::Correlation_Routine
  Correlation_Routine correlation_routine_;
  /// DICe::Interpolation_Method
  Interpolation_Method interpolation_method_;
  /// DICe::Optimization_Method
  Optimization_Method optimization_method_;
  /// DICe::Initialization_Method
  Initialization_Method initialization_method_;
  /// DICe::Projection_Method
  Projection_Method projection_method_;
  /// Analysis type
  Analysis_Type analysis_type_;
  /// Enable translation
  bool enable_translation_;
  /// Enable rotation
  bool enable_rotation_;
  /// Enable normal strain
  bool enable_normal_strain_;
  /// Enable shear strain
  bool enable_shear_strain_;
  /// \brief If subsets cross paths and obstruct each other,
  /// the obstructed pixels are removed for the blocked subset
  /// this vector stores a vector of obstructing subset ids that have
  /// the potential to block the subset associated with the outer vector index
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids_;
  /// filter the images using a 7 point gauss filter
  bool gauss_filter_images_;
  /// Compute the reference image gradients
  bool compute_ref_gradients_;
  /// Compute the deformed image gradients
  bool compute_def_gradients_;
  /// Output images of the deformed subsets at each frame (shapes not intensity profiles)
  bool output_deformed_subset_images_;
  /// Output images of the deformed subsets at each frame (intensity profiles)
  bool output_deformed_subset_intensity_images_;
  /// Output images of the evolved subset at each frame
  bool output_evolved_subset_images_;
  /// Use subset evolution (Fill in the reference subset pixels as they become visible in subsequent frames)
  bool use_subset_evolution_;
  /// True if the check for obstructed pixels should ocurr at every iteration, otherwise it happens once a frame
  bool update_obstructed_pixels_each_iteration_;
  /// True if the gamma values (match quality) should be normalized with the number of active pixels
  bool normalize_gamma_with_active_pixels_;
  /// True if this anlysis is global DIC and the hvm terms should be used
  bool use_hvm_stabilization_;
  /// True if regularization is used in the objective function
  bool use_objective_regularization_;
  /// Solution vector and subsets are initialized
  bool is_initialized_;
  /// Pointer to the parameters which which this schema was initialized
  Teuchos::RCP<Teuchos::ParameterList> init_params_;
  /// Phase correlation initial guess values (only computed if USE_PHASE_CORRELATION is set as the initialization method)
  scalar_t phase_cor_u_x_;
  scalar_t phase_cor_u_y_;
};

/// \class DICe::Output_Spec
/// \brief A simple class to hold the fields to write to the output files and the order to write them
///
/// The user may want the output in a specific format (in terms of the order of the fields). A DICe::Output_Spec
/// holds this order and also provides some output helper functions
class DICE_LIB_DLL_EXPORT
Output_Spec {
public:
  /// \brief Output_Spec constructor with several optional input params
  /// \param schema Pointer to the parent DICe::Schema (used to access field values, etc.)
  /// \param omit_row_id True if each row should not be labelled with a subset_id or frame_number (skips the first column of output)
  /// \param params ParameterList of the field names and their order
  /// \param delimiter String value of the delimiter (space " " or comma ",", could also use a combo ", ")
  Output_Spec(Schema * schema,
    const bool omit_row_id=false,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null,
    const std::string & delimiter=" ");
  virtual ~Output_Spec(){};

  /// \brief Writes the file header information
  /// \param file Pointer to the output file (must already be open)
  /// \param row_id Optional name for the row_id column (typically "FRAME" or "SUBSET_ID")
  void write_header(std::FILE * file,
    const std::string & row_id="");

  /// \brief Writes the fields for the current frame
  /// \param file Pointer to the output file (must already be open)
  /// \param row_index The label for the current row (typically frame_number or subset_id)
  /// \param field_value_index Index of the field values. If the output_separate_file_for_each_subset
  /// option is enabled, the row_index and field_value_index will not be the same. The row_index will
  /// be the frame_number and the field_value_index will be the subset_id. In the default output (one
  /// file per frame that lists all subsets, one per row) row_index and field_value_index will be the same.
  void write_frame(std::FILE * file,
    const int_t row_index,
    const int_t field_value_index);

private:
  /// Vector of Field_Names that will be output to file
  std::vector<std::string> field_names_;
  /// Vector of corresponding ids of the post processor that the field belongs to (-1 id means the field is on the schema)
  std::vector<int_t> post_processor_ids_;
  /// Pointer to the parent schema (used to obtain field values)
  Schema * schema_;
  /// Delimeter to use in the output file
  std::string delimiter_;
  /// True if the row_id should be omited (first column of output)
  bool omit_row_id_;
};

}// End DICe Namespace

#endif
