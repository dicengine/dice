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

#ifndef DICE_H
#define DICE_H

#define DICE_PI 3.14159265358979323846
#define DICE_TWOPI 6.28318530717958647692

#if defined(WIN32)
#  if defined(DICE_LIB_EXPORTS_MODE)
#    define DICE_LIB_DLL_EXPORT __declspec(dllexport)
#  else
#    define DICE_LIB_DLL_EXPORT __declspec(dllimport)
#  endif
#else
#  define DICE_LIB_DLL_EXPORT
#endif

// debugging macros:
#ifdef DICE_DEBUG_MSG
#  define DEBUG_MSG(x) do { std::cout << "[DICe_DEBUG]: " << x << std::endl; } while (0)
#else
#  define DEBUG_MSG(x) do {} while (0)
#endif

#define VERSION "v1.0"
#ifndef GITSHA1
  #define GITSHA1 "not available"
#endif
// size of the deformation vectors used in the correlation
// includes fields below from 0 to (DICE_DEFORMATION_SIZE-1)
#define DICE_DEFORMATION_SIZE 6


#include <string>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe{

/// basic types

#if DICE_USE_DOUBLE
  /// image intensity type
  typedef double intensity_t;
  /// generic scalar type
  typedef double scalar_t;
#else
  /// image intensity type
  typedef float intensity_t;
  /// generic scalar type
  typedef float scalar_t;
#endif

/// integer type
typedef int int_t;

/// Print the executable information banner
DICE_LIB_DLL_EXPORT
void print_banner();

/// Initialization function (mpi and kokkos if enabled):
/// \param argc argument count
/// \param argv array of argument chars
DICE_LIB_DLL_EXPORT
void initialize(int argc,
  char *argv[]);

/// Finalize function (mpi and kokkos if enabled):
DICE_LIB_DLL_EXPORT
void finalize();

/// returns true if the data layout is LayoutRight
DICE_LIB_DLL_EXPORT
bool default_is_layout_right();

/// parameters (all lower case)

/// String parameter names using globals to prevent misspelling in the parameter lists:
const char* const image_grad_use_hierarchical_parallelism = "image_grad_use_hierarchical_parallelism";
/// String parameter name
const char* const image_grad_team_size = "image_grad_team_size";
/// String parameter name
const char* const gauss_filter_images = "gauss_filter_images";
/// String parameter name
const char* const time_average_cine_ref_frame = "time_average_cine_ref_frame";
/// String parameter name
const char* const gauss_filter_use_hierarchical_parallelism = "gauss_filter_use_hierarchical_parallelism";
/// String parameter name
const char* const gauss_filter_team_size = "gauss_filter_team_size";
/// String parameter name
const char* const gauss_filter_mask_size = "gauss_filter_mask_size";
/// String parameter name
const char* const correlation_routine = "correlation_routine";
/// String parameter name
const char* const use_global_dic = "use_global_dic";
/// String parameter name
const char* const use_constrained_opt_dic = "use_constrained_opt_dic";
/// String parameter name
const char* const use_integrated_dic = "use_integrated_dic";
/// String parameter name
const char* const interpolation_method = "interpolation_method";
/// String parameter name
const char* const gradient_method = "gradient_method";
/// String parameter name
const char* const initialization_method = "initialization_method";
/// String parameter name
const char* const optimization_method = "optimization_method";
/// String parameter name
const char* const projection_method = "projection_method";
/// String parameter name
const char* const compute_ref_gradients = "compute_ref_gradients";
/// String parameter name
const char* const compute_def_gradients = "compute_def_gradients";
/// String parameter name
const char* const compute_image_gradients = "compute_image_gradients";
/// String parameter name
const char* const filter_failed_cine_pixels = "filter_failed_cine_pixels";
/// String parameter name
const char* const initial_condition_file = "initial_condition_file";
/// String parameter name
const char* const enable_translation = "enable_translation";
/// String parameter name
const char* const enable_rotation = "enable_rotation";
/// String parameter name
const char* const enable_normal_strain = "enable_normal_strain";
/// String parameter name
const char* const enable_shear_strain = "enable_shear_strain";
/// String parameter name
const char* const max_evolution_iterations = "max_evolution_iterations";
/// String parameter name
const char* const max_solver_iterations_fast = "max_solver_iterations_fast";
/// String parameter name
const char* const max_solver_iterations_robust = "max_solver_iterations_robust";
/// String parameter name
const char* const robust_solver_tolerance = "robust_solver_tolerance";
/// String parameter name
const char* const initial_gamma_threshold = "initial_gamma_threshold";
/// String parameter name
const char* const final_gamma_threshold = "final_gamma_threshold";
/// String parameter name
const char* const path_distance_threshold = "path_distance_threshold";
/// String parameter name
const char* const skip_all_solves = "skip_all_solves";
/// String parameter name
const char* const skip_solve_gamma_threshold = "skip_solve_gamma_threshold";
/// String parameter name
const char* const fast_solver_tolerance = "fast_solver_tolerance";
/// String parameter name
const char* const pixel_size_in_mm = "pixel_size_in_mm";
/// String parameter name
const char* const disp_jump_tol = "disp_jump_tol";
/// String parameter name
const char* const theta_jump_tol = "theta_jump_tol";
/// String parameter name
const char* const robust_delta_disp = "robust_delta_disp";
/// String parameter name
const char* const robust_delta_theta = "robust_delta_theta";
/// String parameter name
const char* const output_deformed_subset_images = "output_deformed_subset_images";
/// String parameter name
const char* const output_deformed_subset_intensity_images = "output_deformed_subset_intensity_images";
/// String parameter name
const char* const output_evolved_subset_images = "output_evolved_subset_images";
/// String parameter name
const char* const use_subset_evolution = "use_subset_evolution";
/// String parameter name
const char* const output_beta = "output_beta";
/// String parameter name
const char* const global_regularization_alpha = "global_regularization_alpha";
/// String parameter name
const char* const global_stabilization_tau = "global_stabilization_tau";
/// String parameter name
const char* const max_iterations = "max_iterations";
/// String parameter name
const char* const tolerance = "tolerance";
/// String parameter name
const char* const output_spec = "output_spec";
/// String parameter name
const char* const mms_spec = "mms_spec";
/// String parameter name
const char* const output_delimiter = "output_delimiter";
/// String parameter name
const char* const omit_output_row_id = "omit_output_row_id";
/// String parameter name
const char* const obstruction_skin_factor = "obstruction_skin_factor";
/// String parameter name
const char* const use_tracking_default_params = "use_tracking_default_params";
/// String parameter name
const char* const override_force_simplex = "override_force_simplex";
/// String parameter name
const char* const use_search_initialization_for_failed_steps = "use_search_initialization_for_failed_steps";
/// String parameter name
const char* const normalize_gamma_with_active_pixels = "normalize_gamma_with_active_pixels";
/// String parameter name
const char* const levenberg_marquardt_regularization_factor = "levenberg_marquardt_regularization_factor";
/// String parameter name
const char* const pixel_integration_order = "pixel_integration_order";
/// String parameter name
const char* const rotate_ref_image_90 = "rotate_ref_image_90";
/// String parameter name
const char* const rotate_def_image_90 = "rotate_def_image_90";
/// String parameter name
const char* const rotate_ref_image_180 = "rotate_ref_image_180";
/// String parameter name
const char* const rotate_def_image_180 = "rotate_def_image_180";
/// String parameter name
const char* const rotate_ref_image_270 = "rotate_ref_image_270";
/// String parameter name
const char* const rotate_def_image_270 = "rotate_def_image_270";
/// String parameter name
const char* const estimate_resolution_error = "estimate_resolution_error";
/// String parameter name
const char* const estimate_resolution_error_min_period = "estimate_resolution_error_min_period";
/// String parameter name
const char* const estimate_resolution_error_max_period = "estimate_resolution_error_max_period";
/// String parameter name
const char* const estimate_resolution_error_period_factor = "estimate_resolution_error_period_factor";
/// String parameter name
const char* const estimate_resolution_error_min_amplitude = "estimate_resolution_error_min_amplitude";
/// String parameter name
const char* const estimate_resolution_error_max_amplitude = "estimate_resolution_error_max_amplitude";
/// String parameter name
const char* const estimate_resolution_error_amplitude_step = "estimate_resolution_error_amplitude_step";
/// String parameter name
const char* const estimate_resolution_error_speckle_size = "estimate_resolution_error_speckle_size";
/// String parameter name
const char* const estimate_resolution_error_noise_percent = "estimate_resolution_error_noise_percent";
/// String parameter name
const char* const use_incremental_formulation = "use_incremental_formulation";
/// String parameter name
const char* const use_nonlinear_projection = "use_nonlinear_projection";
/// String parameter name
const char* const sort_txt_output = "sort_txt_output";
/// String parameter name, only for global DIC
const char* const global_solver = "global_solver";
/// String parameter name, only for global DIC
const char* const global_formulation = "global_formulation";
/// String parameter name, only for global DIC
const char* const problem_name = "problem_name";
/// String parameter name, only for global DIC
const char* const phi_coeff = "phi_coeff";
/// String parameter name, only for global DIC
const char* const b_coeff = "b_coeff";
/// String parameter name, only for global DIC
const char* const curl_coeff = "curl_coeff";
/// String parameter name, only for global DIC
const char* const num_image_integration_points = "num_image_integration_points";
/// String parameter name, only for global DIC
const char* const global_element_type = "global_element_type";
/// String parameter name, only for global DIC
const char* const use_fixed_point_iterations = "use_fixed_point_iterations";


/// enums:
enum Subset_View_Target{
  REF_INTENSITIES=0,
  DEF_INTENSITIES,
  // *** DO NOT PUT NEW ENUMS BELOW THIS ONE ***
  // (this is used for striding and converting enums to strings)
  MAX_SUBSET_VIEW_TARGET,
  NO_SUCH_SUBSET_VIEW_TARGET,
};

/// Valid names for fields
enum Field_Name {
  // *** DON'T FORGET TO ADD A STRING TO THE to_string(Field_name) function ***
  //
  // *** ALSO UPDATE THE SIZE OF THE DEFORMATION GIVEN IN #define DICE_DEFORMATION_SIZE
  //     IF MORE DEFORMATION FIELDS ARE ADDED
  // 0
  DISPLACEMENT_X=0,  // u
  // 1
  DISPLACEMENT_Y,    // v
  // 2
  ROTATION_Z,        // (theta) rotation about the z-axis, z is out of the plane
  // 3
  NORMAL_STRAIN_X,   // stretch in the x direction
  // 4
  NORMAL_STRAIN_Y,   // stretch in the y direction
  // 5
  SHEAR_STRAIN_XY,   // shear strain in the x-y plane
  // *** Fields >= DICE_DEFORMATION_SIZE are not part of the deformation state of a subset
  //     but used elsewhere in the schema
  // 6
  COORDINATE_X,      // x position of the subset in sensor (image) coordinates
  // 7
  COORDINATE_Y,      // y position of the subset in sensor (image) coordinates
  // 8
  STEREO_COORDINATE_X,// x position of the stereo subset in sensor (right image) coordinates (used for stereo only)
  // 9
  STEREO_COORDINATE_Y,// y position of the stereo subset in sensor (right image) coordinates (used for stereo only)
  // 10
  MODEL_COORDINATE_X, // x position in world or physical coordinates (used for stereo only)
  // 11
  MODEL_COORDINATE_Y,// y position in world or physical coordinates (used for stereo only)
  // 12
  MODEL_COORDINATE_Z,// z position in world or physical coordinates (used for stereo only)
  // 13
  STEREO_DISPLACEMENT_X,// right image u (used for stereo only)
  // 14
  STEREO_DISPLACEMENT_Y,// right image v (used for stereo only)
  // 15
  MODEL_DISPLACEMENT_X,// x displacement in world or physical coordinates
  // 16
  MODEL_DISPLACEMENT_Y,// y displacement in world or physical coordinates
  // 17
  MODEL_DISPLACEMENT_Z,// z displacement in world or physical coordinates
  // 18
  VAR_X,             // auxiliary variable
  // 19
  VAR_Y,             // auxiliary variable
  // 20
  VAR_Z,             // auxiliary varaible
  // 21
  SIGMA,             // predicted std. dev. of the displacement solution given std. dev. of image
                     // noise and interpolation bias, smaller sigma is better
  // 22
  GAMMA,             // template match quality (value of the cost function),
                     // smaller gamma is better, 0.0 is perfect match
  // 23
  BETA,              // sensitivity of the cost function to small perturbations in the displacement solution
  // 24
  NOISE_LEVEL,       // estimated std. dev. of the image noise
  // 25
  CONTRAST_LEVEL,    // estimated std. dev. of the image intensity values
  // 26
  ACTIVE_PIXELS,     // number of active pixels for the subset
  // 27
  MATCH,             // 0 means match was found -1 means match failed
  // 28
  ITERATIONS,        // number of iterations taken by the solution algorithm
  // 29
  STATUS_FLAG,       // information about the initialization method or error flags on failed steps
  // 30
  NEIGHBOR_ID,       // the global id of the neighboring subset to use for initialization by neighbor value
  // 31
  CONDITION_NUMBER,  // quality metric for the pseudoinverse matrix in the gradient-based method
  // 32
  CROSS_CORR_Q,      // cross correlation Q coordinate
  // 33
  CROSS_CORR_R,      // cross correlation R coordinate
  // *** DO NOT PUT ANY FIELDS UNDER THIS ONE ***
  // (this is how the field stride is automatically set if another field is added)
  MAX_FIELD_NAME,
  NO_SUCH_FIELD_NAME
};

/// Subset_File_Info types
enum Subset_File_Info_Type {
  SUBSET_INFO=0,
  REGION_OF_INTEREST_INFO
};

/// Enum that determines which distributed vector to send
/// the field values to
enum Target_Field_Descriptor {
  ALL_OWNED=0,
  DISTRIBUTED,
  DISTRIBUTED_GROUPED_BY_SEED,
  MAX_TARGET_FIELD_DESCRIPTOR
};

/// Analysis Type
enum Analysis_Type {
  LOCAL_DIC=0,
  GLOBAL_DIC,
  // DON'T ADD ANY BELOW THIS
  MAX_ANALYSIS_TYPE,
  NO_SUCH_ANALYSIS_TYPE
};

/// Projection method
enum Global_Formulation {
  HORN_SCHUNCK=0,
  MIXED_HORN_SCHUNCK,
  LEVENBERG_MARQUARDT,
  LEHOUCQ_TURNER,
  UNREGULARIZED,
  METHOD_OF_MANUFACTURED_SOLUTIONS,
  // DON'T ADD ANY BELOW MAX
  MAX_GLOBAL_FORMULATION,
  NO_SUCH_GLOBAL_FORMULATION
};

const static char * globalFormulationStrings[] = {
  "HORN_SCHUNCK",
  "MIXED_HORN_SCHUNCK",
  "LEVENBERG_MARQUARDT",
  "LEHOUCQ_TURNER",
  "UNREGULARIZED",
  "METHOD_OF_MANUFACTURED_SOLUTIONS"
};

/// Projection method
enum Projection_Method {
  DISPLACEMENT_BASED=0,
  VELOCITY_BASED,
  MULTISTEP,
  // DON'T ADD ANY BELOW MAX
  MAX_PROJECTION_METHOD,
  NO_SUCH_PROJECTION_METHOD
};

const static char * projectionMethodStrings[] = {
  "DISPLACEMENT_BASED",
  "VELOCITY_BASED",
  "MULTISTEP"
};

/// Initialization method
enum Initialization_Method {
  USE_FIELD_VALUES=0,
  USE_NEIGHBOR_VALUES,
  USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY,
  USE_PHASE_CORRELATION,
  USE_OPTICAL_FLOW,
  USE_ZEROS,
  INITIALIZATION_METHOD_NOT_APPLICABLE,
  // DON'T ADD ANY BELOW MAX
  MAX_INITIALIZATION_METHOD,
  NO_SUCH_INITIALIZATION_METHOD
};

const static char * initializationMethodStrings[] = {
  "USE_FIELD_VALUES",
  "USE_NEIGHBOR_VALUES",
  "USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY",
  "USE_PHASE_CORRELATION",
  "USE_OPTICAL_FLOW",
  "USE_ZEROS",
  "INITIALIZATION_METHOD_NOT_APPLICABLE"
};

/// Optimization method
enum Optimization_Method {
  SIMPLEX=0,
  GRADIENT_BASED,
  SIMPLEX_THEN_GRADIENT_BASED,
  GRADIENT_BASED_THEN_SIMPLEX,
  OPTIMIZATION_METHOD_NOT_APPLICABLE,
  // DON'T ADD ANY BELOW MAX
  MAX_OPTIMIZATION_METHOD,
  NO_SUCH_OPTIMIZATION_METHOD
};

const static char * optimizationMethodStrings[] = {
  "SIMPLEX",
  "GRADIENT_BASED",
  "SIMPLEX_THEN_GRADIENT_BASED",
  "GRADIENT_BASED_THEN_SIMPLEX",
  "OPTIMIZATION_METHOD_NOT_APPLICABLE"
};

/// Interpolation method
enum Interpolation_Method {
  BILINEAR=0,
  BICUBIC,
  KEYS_FOURTH,
  // DON'T ADD ANY BELOW MAX
  MAX_INTERPOLATION_METHOD,
  NO_SUCH_INTERPOLATION_METHOD
};

const static char * interpolationMethodStrings[] = {
  "BILINEAR",
  "BICUBIC",
  "KEYS_FOURTH"
};

/// Gradient method
enum Gradient_Method {
  FINITE_DIFFERENCE=0,
  CONVOLUTION_5_POINT,
  // DON'T ADD ANY BELOW MAX
  MAX_GRADIENT_METHOD,
  NO_SUCH_GRADIENT_METHOD
};

const static char * gradientMethodStrings[] = {
  "FINITE_DIFFERENCE",
  "CONVOLUTION_5_POINT"
};


/// Correlation routine (determines how the correlation steps are executed).
/// Can be customized for a particular application
enum Correlation_Routine {
  GENERIC_ROUTINE=0,
  TRACKING_ROUTINE,
  CORRELATION_ROUTINE_NOT_APPLICABLE,
  // DON'T ADD ANY BELOW MAX
  MAX_CORRELATION_ROUTINE,
  NO_SUCH_CORRELATION_ROUTINE
};

const static char * correlationRoutineStrings[] = {
  "GENERIC_ROUTINE",
  "TRACKING_ROUTINE",
  "CORRELATION_ROUTINE_NOT_APPLICABLE"
};

/// Status flags
enum Status_Flag{
  // 0
  CORRELATION_SUCCESSFUL=0,
  // 1
  INITIALIZE_USING_PREVIOUS_FRAME_SUCCESSFUL,
  // 2
  INITIALIZE_USING_CONNECTED_SUBSET_VALUE_SUCCESSFUL,
  // 3
  INITIALIZE_USING_NEIGHBOR_VALUE_SUCCESSFUL,
  // 4
  INITIALIZE_SUCCESSFUL,
  // 5
  INITIALIZE_FAILED,
  // 6
  SEARCH_SUCCESSFUL,
  // 7
  SEARCH_FAILED,
  // 8
  CORRELATION_FAILED,
  // 9
  SUBSET_CONSTRUCTION_FAILED,
  // 10
  LINEAR_SOLVE_FAILED,
  // 11
  MAX_ITERATIONS_REACHED,
  // 12
  INITIALIZE_FAILED_BY_EXCEPTION,
  // 13
  SEARCH_FAILED_BY_EXCEPTION,
  // 14
  CORRELATION_FAILED_BY_EXCEPTION,
  // 15
  CORRELATION_BY_AVERAGING_CONNECTED_VALUES,
  // 16
  JUMP_TOLERANCE_EXCEEDED,
  // 17
  ZERO_HESSIAN_DETERMINANT,
  // 18
  SEARCH_USING_PREVIOUS_STEP_SUCCESSFUL,
  // 19
  LINEARIZED_GAMMA_OUT_OF_BOUNDS,
  // 20
  NAN_IN_HESSIAN_OR_RESIDUAL,
  // 21
  HESSIAN_SINGULAR,
  // 22
  SKIPPED_FRAME_DUE_TO_HIGH_GAMMA,
  // 23
  FRAME_FAILED_DUE_TO_HIGH_GAMMA,
  // 24
  FRAME_FAILED_DUE_TO_NEGATIVE_SIGMA,
  // 25
  FRAME_FAILED_DUE_TO_HIGH_PATH_DISTANCE,
  // 26
  RESET_REF_SUBSET_DUE_TO_HIGH_GAMMA,
  // 27
  MAX_GLOBAL_ITERATIONS_REACHED_IN_EVOLUTION_LOOP,
  // 28
  FAILURE_DUE_TO_TOO_MANY_RESTARTS,
  // 29
  FAILURE_DUE_TO_DEVIATION_FROM_PATH,
  // 30
  FRAME_SKIPPED,
  // 31
  FRAME_SKIPPED_DUE_TO_NO_MOTION,
  // DON'T ADD ANY BELOW MAX
  MAX_STATUS_FLAG,
  NO_SUCH_STATUS_FLAG
};

/// Specific values of rotation used for transformation
enum Rotation_Value{
  ZERO_DEGREES=0,
  NINTY_DEGREES,
  ONE_HUNDRED_EIGHTY_DEGREES,
  TWO_HUNDRED_SEVENTY_DEGREES
};

/// Specifies whether motion is occurring in the frame or not
enum Motion_State{
  MOTION_NOT_SET=0,
  MOTION_TRUE,
  MOTION_FALSE
};

/// The type of image file
enum Image_File_Type{
  RAWI=0,
  TIFF,
  JPEG,
  PNG,
  NETCDF,
  MAX_IMAGE_FILE_TYPE,
  NO_SUCH_IMAGE_FILE_TYPE
};

/// The type of correlation parameter, used for creating template input files
enum Correlation_Parameter_Type{
  STRING_PARAM=0,
  PARAM_PARAM, // parameter that is another parameter list
  SCALAR_PARAM,
  SIZE_PARAM,
  BOOL_PARAM
};

/// Combine mode for fields
enum Combine_Mode{
  INSERT=0,
  ADD
};

/// Global method terms to include in the residual
enum Global_EQ_Term{
  IMAGE_TIME_FORCE=0,
  IMAGE_GRAD_TENSOR,
  DIV_SYMMETRIC_STRAIN_REGULARIZATION,
  TIKHONOV_REGULARIZATION,
  GRAD_LAGRANGE_MULTIPLIER,
  DIV_VELOCITY,
  MMS_IMAGE_GRAD_TENSOR,
  MMS_FORCE,
  MMS_IMAGE_TIME_FORCE,
  MMS_GRAD_LAGRANGE_MULTIPLIER,
  DIRICHLET_DISPLACEMENT_BC,
  MMS_DIRICHLET_DISPLACEMENT_BC,
  MMS_LAGRANGE_BC,
  CORNER_BC,
  OPTICAL_FLOW_DISPLACEMENT_BC,
  SUBSET_DISPLACEMENT_BC,
  SUBSET_DISPLACEMENT_IC,
  LAGRANGE_BC,
  CONSTANT_IC,
  STAB_LAGRANGE,
  NO_SUCH_GLOBAL_EQ_TERM
};


/// Global solver type
enum Global_Solver{
  CG_SOLVER=0,
  GMRES_SOLVER,
  LSQR_SOLVER,
  NO_SUCH_GLOBAL_SOLVER
};

/// \class DICe::Extents
/// \brief collection of origin x, y and width and height
struct Extents {
  /// constructor
  /// \param origin_x upper left corner x loc
  /// \param origin_y upper left corner y loc
  /// \param width box width
  /// \param height box height
  Extents(const int_t origin_x,
    const int_t origin_y,
    const int_t width,
    const int_t height):
    origin_x_(origin_x),
    origin_y_(origin_y),
    width_(width),
    height_(height){}
  /// upper left corner x loc
  int_t origin_x_;
  /// upper_left corner y loc
  int_t origin_y_;
  /// widht
  int_t width_;
  /// height
  int_t height_;
};

/// \class DICe::Correlation_Parameter
/// \brief Simple struct to hold information about correlation parameters
struct Correlation_Parameter {
  /// \brief Only constructor with several optional arguments
  /// \param name The string name of the correlation parameter
  /// \param type Defines if this is a bool, string, integer value, etc.
  /// \param expose_to_user Signifies that this should be included when template input files are made
  /// \param desc Correlation parameter description
  /// \param stringNames Pointer to the array of string names for this parameter if this is a string parameter, otherwise null pointer
  /// \param size The number of available options for this correlation parameter
  Correlation_Parameter(const std::string & name, const Correlation_Parameter_Type & type,
    const bool expose_to_user=true,
    const std::string & desc="No description",
    const char ** stringNames=0,
    const int_t size=0){
    name_ = name;
    type_ = type;
    desc_ = desc;
    stringNamePtr_ = stringNames;
    size_ = size;
    expose_to_user_ = expose_to_user;
  }
  /// Name of the parameter (what will be used from the input file if specified)
  std::string name_;
  /// Type of parameter (bool, size, real, string)
  Correlation_Parameter_Type type_;
  /// Pointer to the string names of all the options
  const char ** stringNamePtr_;
  /// The number of potential options
  int_t size_;
  /// Short description of the correlation parameter
  std::string desc_;
  /// Determines if this param shows up in the template input files exposed to the user
  bool expose_to_user_;
};

/// Correlation parameter and properties
const Correlation_Parameter rotate_ref_image_90_param(rotate_ref_image_90,
  BOOL_PARAM,
  true,
  "True if the reference image should be rotated 90 degrees.");
/// Correlation parameter and properties
const Correlation_Parameter rotate_def_image_90_param(rotate_def_image_90,
  BOOL_PARAM,
  true,
  "True if deformed image(s) should be rotated 90 degrees.");
/// Correlation parameter and properties
const Correlation_Parameter rotate_ref_image_180_param(rotate_ref_image_180,
  BOOL_PARAM,
  true,
  "True if the reference image should be rotated 180 degrees.");
/// Correlation parameter and properties
const Correlation_Parameter rotate_def_image_180_param(rotate_def_image_180,
  BOOL_PARAM,
  true,
  "True if deformed image(s) should be rotated 180 degrees.");
/// Correlation parameter and properties
const Correlation_Parameter rotate_ref_image_270_param(rotate_ref_image_270,
  BOOL_PARAM,
  true,
  "True if the reference image should be rotated 270 degrees.");
/// Correlation parameter and properties
const Correlation_Parameter rotate_def_image_270_param(rotate_def_image_270,
  BOOL_PARAM,
  true,
  "True if deformed image(s) should be rotated 270 degrees.");
/// Correlation parameter and properties
const Correlation_Parameter image_grad_use_hierarchical_parallelism_param(image_grad_use_hierarchical_parallelism,
  BOOL_PARAM,
  true,
  "True if higherarchical parallelism should be used when computing image gradients (parallel in x and y)");
/// Correlation parameter and properties
const Correlation_Parameter image_grad_team_size_param(image_grad_team_size,
  SIZE_PARAM,
  true,
  "The team size to use for thread teams when computing image gradients.");
/// Correlation parameter and properties
const Correlation_Parameter gauss_filter_use_hierarchical_parallelism_param(gauss_filter_use_hierarchical_parallelism,
  BOOL_PARAM,
  true,
  "True if higherarchical parallelism should be used when computing image Gaussian filter (parallel in x and y)");
/// Correlation parameter and properties
const Correlation_Parameter gauss_filter_team_size_param(gauss_filter_team_size,
  SIZE_PARAM,
  true,
  "The team size to use for thread teams when computing Gaussian filter.");
/// Correlation parameter and properties
const Correlation_Parameter gauss_filter_mask_size_param(gauss_filter_mask_size,
  SIZE_PARAM,
  true,
  "The size in pixels of the Gaussian filter (3, 5, 7, 9, 11, or 13).");
/// Correlation parameter and properties
const Correlation_Parameter pixel_integration_order_param(pixel_integration_order,
  SIZE_PARAM,
  true,
  "Specifies the integration order to use (number of subdivisions for each pixel). Used only in the constrained optimization formulation.");

/// Correlation parameter and properties
const Correlation_Parameter obstruction_skin_factor_param(obstruction_skin_factor,
  SCALAR_PARAM,
  true,
  "Stretches the obstruction subsets to make them larger (factor > 1.0) or smaller (factor < 1.0) than they actually are.");

/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_param(estimate_resolution_error,
  BOOL_PARAM,
  true,
  "Only evaluates the error from a known solution synthetically applied to the reference image,"
  " determines the spatial resolution. Integer value affects the wave number of the sin() function");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_min_period_param(estimate_resolution_error_min_period,
  SCALAR_PARAM,
  true,
  "minimum motion period to use in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_max_period_param(estimate_resolution_error_max_period,
  SCALAR_PARAM,
  true,
  "maximum motion period to use in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_period_factor_param(estimate_resolution_error_period_factor,
  SCALAR_PARAM,
  true,
  "reduction factor to apply to the period for each step in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_min_amplitude_param(estimate_resolution_error_min_amplitude,
  SCALAR_PARAM,
  true,
  "minimum amplitude of the motion to use in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_max_amplitude_param(estimate_resolution_error_max_amplitude,
  SCALAR_PARAM,
  true,
  "maximum amplitude of the motion to use in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_amplitude_step_param(estimate_resolution_error_amplitude_step,
  SCALAR_PARAM,
  true,
  "amount of amplitude added to the motion amplitude for each step in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_speckle_size_param(estimate_resolution_error_speckle_size,
  SCALAR_PARAM,
  true,
  "create synthetic images with speckles of a regular size in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter estimate_resolution_error_noise_percent_param(estimate_resolution_error_noise_percent,
  SCALAR_PARAM,
  true,
  "amount of noise to add in percent of counts to the deformed image in the estimation of resolution error");
/// Correlation parameter and properties
const Correlation_Parameter use_incremental_formulation_param(use_incremental_formulation,
  BOOL_PARAM,
  true,
  "Use the previous image as the reference rather than the original ref image. Displacements become cumulative");
/// Correlation parameter and properties
const Correlation_Parameter use_nonlinear_projection_param(use_nonlinear_projection,
  BOOL_PARAM,
  true,
  "Project the right image onto the left frame of reference before correlation using a nonlinear projection operator (used for stereo only)");
/// Correlation parameter and properties
const Correlation_Parameter sort_txt_output_param(sort_txt_output,
  BOOL_PARAM,
  true,
  "Sort the text output file according to the subset location in x then y for the full field results");
/// Correlation parameter and properties
const Correlation_Parameter output_delimiter_param(output_delimiter,
  STRING_PARAM,
  true,
  "Delimeter to separate column values in output files, (comma or space, etc.)");
/// Correlation parameter and properties
const Correlation_Parameter omit_output_row_id_param(omit_output_row_id,
  BOOL_PARAM,
  true,
  "True if the row id should be omitted from the output (column zero is skipped)");
/// Correlation parameter and properties
const Correlation_Parameter mms_spec_param(mms_spec,
  STRING_PARAM,
  false, // turned off because this one is manually added to the template output files
  "Set of parameters for the method of manufactured solutions problems");

/// Correlation parameter and properties
const Correlation_Parameter output_spec_param(output_spec,
  STRING_PARAM,
  false, // turned off because this one is manually added to the template output files
  "Determines what output to write and in what order");
/// Correlation parameter and properties
const Correlation_Parameter correlation_routine_param(correlation_routine,
  STRING_PARAM,
  true,
  "Determines the correlation order of execution (see DICe::Schema)",
  correlationRoutineStrings,
  MAX_CORRELATION_ROUTINE);
/// Correlation parameter and properties
const Correlation_Parameter interpolation_method_param(interpolation_method,
  STRING_PARAM,
  true,
  "Determines which interpolation method to use (can also affect the image gradients)",
  interpolationMethodStrings,
  MAX_INTERPOLATION_METHOD);
/// Correlation parameter and properties
const Correlation_Parameter gradient_method_param(gradient_method,
  STRING_PARAM,
  true,
  "Determines which image gradient method to use",
  gradientMethodStrings,
  MAX_GRADIENT_METHOD);
/// Correlation parameter and properties
const Correlation_Parameter initialization_method_param(initialization_method,
  STRING_PARAM,
  true,
  "Determines how solution values are initialized for each frame",
  initializationMethodStrings,
  MAX_INITIALIZATION_METHOD);
/// Correlation parameter and properties
const Correlation_Parameter optimization_method_param(optimization_method,
  STRING_PARAM,
  true,
  "Determines if gradient based (fast, but not as robust) or simplex based (no gradients needed, but requires more iterations) optimization algorithm will be used",
  optimizationMethodStrings,
  MAX_OPTIMIZATION_METHOD);
/// Correlation parameter and properties
const Correlation_Parameter projection_method_param(projection_method,
  STRING_PARAM,
  true,
  "Determines how solution values from previous frames are used to predict the current solution",
  projectionMethodStrings,
  MAX_PROJECTION_METHOD);
/// Correlation parameter and properties
const Correlation_Parameter initial_condition_file_param(initial_condition_file,
  BOOL_PARAM,
  true,
  "Denotes a file to read the solution from as the initial condition");
/// Correlation parameter and properties
const Correlation_Parameter enable_translation_param(enable_translation,
  BOOL_PARAM,
  true,
  "Enables the translation shape function degrees of freedom (u and v)");
/// Correlation parameter and properties
const Correlation_Parameter enable_rotation_param(enable_rotation,
  BOOL_PARAM,
  true,
  "Enables the rotation shape function degree of freedom (theta)");
/// Correlation parameter and properties
const Correlation_Parameter enable_normal_strain_param(enable_normal_strain,
  BOOL_PARAM,
  true,
  "Enables the normal strain shape function degrees of freedom (epsilon_x and epsilon_y)");
/// Correlation parameter and properties
const Correlation_Parameter enable_shear_strain_param(enable_shear_strain,
  BOOL_PARAM,
  true,
  "Enables the shear strain shape function defree of freedom (gamma_xy = gamma_yx)");
/// Correlation parameter and properties
const Correlation_Parameter levenberg_marquardt_regularization_factor_param(levenberg_marquardt_regularization_factor,
  SCALAR_PARAM,
  true,
  "The coefficient applied to the regularization term if active");
/// Correlation parameter and properties
const Correlation_Parameter max_evolution_iterations_param(max_evolution_iterations,SIZE_PARAM,true,
  "Maximum evolution iterations to use (only valid for subset_evolution_routine)");
/// Correlation parameter and properties
const Correlation_Parameter max_solver_iterations_fast_param(max_solver_iterations_fast,SIZE_PARAM);
/// Correlation parameter and properties
const Correlation_Parameter max_solver_iterations_robust_param(max_solver_iterations_robust,SIZE_PARAM);
/// Correlation parameter and properties
const Correlation_Parameter fast_solver_tolerance_param(fast_solver_tolerance,SCALAR_PARAM);
/// Correlation parameter and properties
const Correlation_Parameter robust_solver_tolerance_param(robust_solver_tolerance,SCALAR_PARAM);
/// Correlation parameter and properties
const Correlation_Parameter skip_all_solves_param(skip_all_solves,BOOL_PARAM,true,
  "This option will use the initial guess for the displacement solution as the solution and skip the solves. "
  "It can be helpful for testing if the initialization routine is working properly");
/// Correlation parameter and properties
const Correlation_Parameter skip_solve_gamma_threshold_param(skip_solve_gamma_threshold,SCALAR_PARAM,true,
  "If the gamma evaluation for the initial deformation guess is below this value, the solve is skipped because"
  " the match is already good enough");
/// Correlation parameter and properties
const Correlation_Parameter initial_gamma_threshold_param(initial_gamma_threshold,SCALAR_PARAM,true,
  "If the gamma evaluation for the initial deformation guess is not below this value, initialization will fail");
/// Correlation parameter and properties
const Correlation_Parameter final_gamma_threshold_param(final_gamma_threshold,SCALAR_PARAM,true,
  "If the gamma evaluation for the final deformation guess is not below this value, the step will fail for this subset");
/// Correlation parameter and properties
const Correlation_Parameter path_distance_threshold_param(path_distance_threshold,SCALAR_PARAM,true,
  "If the final deformation solution is farther than this threshold from a segment in the path file"
  " (which must be specified in the subset file) the step will fail for this subset");
/// Correlation parameter and properties
const Correlation_Parameter pixel_size_in_mm_param(pixel_size_in_mm,
  SCALAR_PARAM,
  true,
  "The spatial size of one pixel (1 pixel is equivalent to ? mm");
/// Correlation parameter and properties
const Correlation_Parameter disp_jump_tol_param(disp_jump_tol,
  SCALAR_PARAM,
  true,
  "Displacement solutions greater than this from the previous frame will be rejected as unsuccessful");
/// Correlation parameter and properties
const Correlation_Parameter theta_jump_tol_param(theta_jump_tol,
  SCALAR_PARAM,
  true,
  "Rotation solutions greater than this from the previous frame will be rejected as unsuccessful");
/// Correlation parameter and properties
const Correlation_Parameter robust_delta_disp_param(robust_delta_disp,
  SCALAR_PARAM,
  true,
  "Variation on initial displacement guess used to construct simplex");
/// Correlation parameter and properties
const Correlation_Parameter robust_delta_theta_param(robust_delta_theta,
  SCALAR_PARAM,
  true,
  "Variation on initial rotation guess used to construct simplex");
/// Correlation parameter and properties
const Correlation_Parameter output_deformed_subset_images_param(output_deformed_subset_images,
  BOOL_PARAM,
  true,
  "Write images that show the deformed position of the subsets (Currently only available for TRACKING_ROUTINE correlation_routine, not GENERIC)");
/// Correlation parameter and properties
const Correlation_Parameter output_deformed_subset_intensity_images_param(output_deformed_subset_intensity_images,
  BOOL_PARAM,
  true,
  "Write images that show the intensity profile of the deformed subsets");
/// Correlation parameter and properties
const Correlation_Parameter output_evolved_subset_images_param(output_evolved_subset_images,
  BOOL_PARAM,
  true,
  "Write images that show the reference subset as its intensity profile evolves");
/// Correlation parameter and properties
const Correlation_Parameter use_subset_evolution_param(use_subset_evolution,
  BOOL_PARAM,
  true,
  "Used to evolve subsets that are initially obscured (Currently only available for TRACKING_ROUTINE correlation routine, not GENERIC)");
/// Correlation parameter and properties
const Correlation_Parameter output_beta_param(output_beta,
  BOOL_PARAM,
  true,
  "True if the beta parameter should be computed (still needs to be added to the output spec if it should be included in the output file)"
  " This parameter measures the distinguishability of a pattern for template matching");
/// Correlation parameter and properties
const Correlation_Parameter global_regularization_alpha_param(global_regularization_alpha,
  SCALAR_PARAM,
  true,
  "Used only for global, this is the coefficient for the alpha regularization term");
/// Correlation parameter and properties
const Correlation_Parameter global_stabilization_tau_param(global_stabilization_tau,
  SCALAR_PARAM,
  true,
  "Used only for global, this is the stabilization coefficient (overrides automatically calculated one)");
/// Correlation parameter and properties
const Correlation_Parameter global_formulation_param(global_formulation,
  STRING_PARAM,
  true,
  "Used only for global, this is the formulation to use (which terms are included, etc.)",
  globalFormulationStrings,
  MAX_GLOBAL_FORMULATION
);
/// Correlation parameter and properties
const Correlation_Parameter global_solver_param(global_solver,
  STRING_PARAM,
  true,
  "Used only for global, this is the solver to use for the global method."
);
/// Correlation parameter and properties
const Correlation_Parameter use_fixed_point_iterations_param(use_fixed_point_iterations,
  BOOL_PARAM,
  true,
  "Used only for global, uses the fixed point iteration scheme for the global method."
);
/// Correlation parameter and properties
const Correlation_Parameter global_element_type_param(global_element_type,
  STRING_PARAM,
  true,
  "Used only for global, this is the element type to use for the global method."
);
/// Correlation parameter and properties
const Correlation_Parameter num_image_integration_points_param(num_image_integration_points,
  SIZE_PARAM,
  true,
  "Used only for global, this is the number of integration points (in each dim per element) to use for the global method."
);
/// Correlation parameter and properties
const Correlation_Parameter use_tracking_default_params_param(use_tracking_default_params,
  BOOL_PARAM,
  true,
  "Use the TRACKING default parameters instead of the GENERIC defaults (Not commonly used).");
/// Correlation parameter and properties
const Correlation_Parameter override_force_simplex_param(override_force_simplex,
  BOOL_PARAM,
  true,
  "Override the forcing of the simplex method for blocking subsets or those specified using the force_simplex keyword in the subset definition file.");
/// Correlation parameter and properties
const Correlation_Parameter use_search_initialization_for_failed_steps_param(use_search_initialization_for_failed_steps,
  BOOL_PARAM,
  true,
  "Use a searching routine whenever a step fails in the TRACKING_ROUTINE.");
/// Correlation parameter and properties
const Correlation_Parameter normalize_gamma_with_active_pixels_param(normalize_gamma_with_active_pixels,
  BOOL_PARAM,
  true,
  "True if the computed gamma value (or matching quality) will be normalized by the number of active pixels.");
/// Correlation parameter and properties
const Correlation_Parameter use_global_dic_param(use_global_dic,
  BOOL_PARAM,
  false,
  "True if the global method should be used rather than subset or local DIC.");
/// Correlation parameter and properties
const Correlation_Parameter use_constrained_opt_dic_param(use_constrained_opt_dic,
  BOOL_PARAM,
  false,
  "True if the constrained optimization method should be used rather than subset or local DIC.");
/// Correlation parameter and properties
const Correlation_Parameter use_integrated_dic_param(use_integrated_dic,
  BOOL_PARAM,
  false,
  "True if the integrated DIC algorithms should be used rather than subset or local DIC.");

// These are not exposed automatically to the user (they are internal params used by the schema and image classes)

/// Correlation parameter and properties
const Correlation_Parameter compute_ref_gradients_param(compute_ref_gradients,
  BOOL_PARAM,
  false,
  "Compute image gradients for the reference frame");
/// Correlation parameter and properties
const Correlation_Parameter gauss_filter_images_param(gauss_filter_images,
  BOOL_PARAM,
  false,
  "Filter the images using a 7 point gauss filter (eliminates high frequnecy content)");
/// Correlation parameter and properties
const Correlation_Parameter time_average_cine_ref_frame_param(time_average_cine_ref_frame,
  SIZE_PARAM,
  false,
  "Select the number of frames over which to time average the reference frame of a cine file");
/// Correlation parameter and properties
const Correlation_Parameter compute_def_gradients_param(compute_def_gradients,
  BOOL_PARAM,
  false,
  "Compute image gradients for the deformed image of the current frame");
/// Correlation parameter and properties
const Correlation_Parameter compute_image_gradients_param(compute_image_gradients,
  BOOL_PARAM,
  false,
  "Compute image gradients");
/// Correlation parameter and properties
const Correlation_Parameter filter_failed_cine_pixels_param(filter_failed_cine_pixels,
  BOOL_PARAM,
  false,
  "Filter out any pixels that failed during cine acquisition");

// TODO don't forget to update this when adding a new one
/// The total number of valid correlation parameters
/// Vector of valid parameter names
const int_t num_valid_correlation_params = 79;
/// Vector oIf valid parameter names
const Correlation_Parameter valid_correlation_params[num_valid_correlation_params] = {
  correlation_routine_param,
  interpolation_method_param,
  gradient_method_param,
  initialization_method_param,
  optimization_method_param,
  projection_method_param,
  compute_ref_gradients_param,
  compute_def_gradients_param,
  compute_image_gradients_param,
  gauss_filter_images_param,
  initial_condition_file_param,
  enable_translation_param,
  enable_rotation_param,
  enable_normal_strain_param,
  enable_shear_strain_param,
  max_evolution_iterations_param,
  max_solver_iterations_fast_param,
  max_solver_iterations_robust_param,
  fast_solver_tolerance_param,
  robust_solver_tolerance_param,
  skip_all_solves_param,
  skip_solve_gamma_threshold_param,
  initial_gamma_threshold_param,
  final_gamma_threshold_param,
  path_distance_threshold_param,
  disp_jump_tol_param,
  theta_jump_tol_param,
  robust_delta_disp_param,
  robust_delta_theta_param,
  output_deformed_subset_images_param,
  output_deformed_subset_intensity_images_param,
  output_evolved_subset_images_param,
  use_subset_evolution_param,
  output_beta_param,
  output_spec_param,
  output_delimiter_param,
  omit_output_row_id_param,
  obstruction_skin_factor_param,
  estimate_resolution_error_param,
  estimate_resolution_error_min_period_param,
  estimate_resolution_error_max_period_param,
  estimate_resolution_error_period_factor_param,
  estimate_resolution_error_min_amplitude_param,
  estimate_resolution_error_max_amplitude_param,
  estimate_resolution_error_amplitude_step_param,
  estimate_resolution_error_speckle_size_param,
  estimate_resolution_error_noise_percent_param,
  use_incremental_formulation_param,
  use_nonlinear_projection_param,
  sort_txt_output_param,
  use_search_initialization_for_failed_steps_param,
  use_tracking_default_params_param,
  override_force_simplex_param,
  normalize_gamma_with_active_pixels_param,
  use_global_dic_param,
  use_constrained_opt_dic_param,
  use_integrated_dic_param,
  pixel_integration_order_param,
  image_grad_use_hierarchical_parallelism_param,
  image_grad_team_size_param,
  gauss_filter_use_hierarchical_parallelism_param,
  gauss_filter_team_size_param,
  gauss_filter_mask_size_param,
  rotate_ref_image_90_param,
  rotate_def_image_90_param,
  rotate_ref_image_180_param,
  rotate_def_image_180_param,
  rotate_ref_image_270_param,
  rotate_def_image_270_param,
  levenberg_marquardt_regularization_factor_param,
  filter_failed_cine_pixels_param,
  time_average_cine_ref_frame_param,
  global_regularization_alpha_param,
  global_stabilization_tau_param,
  global_formulation_param,
  global_solver_param,
  global_element_type_param,
  num_image_integration_points_param,
  use_fixed_point_iterations_param
};

// TODO don't forget to update this when adding a new one
/// The total number of valid correlation parameters
const int_t num_valid_global_correlation_params = 31;
/// Vector of valid parameter names
const Correlation_Parameter valid_global_correlation_params[num_valid_global_correlation_params] = {
  use_global_dic_param,
  interpolation_method_param,
  fast_solver_tolerance_param,
  max_solver_iterations_fast_param,
  gradient_method_param,
  gauss_filter_images_param,
  gauss_filter_mask_size_param,
  output_spec_param,
  output_delimiter_param,
  omit_output_row_id_param,
  estimate_resolution_error_param,
  estimate_resolution_error_min_period_param,
  estimate_resolution_error_max_period_param,
  estimate_resolution_error_period_factor_param,
  estimate_resolution_error_min_amplitude_param,
  estimate_resolution_error_max_amplitude_param,
  estimate_resolution_error_amplitude_step_param,
  estimate_resolution_error_speckle_size_param,
  estimate_resolution_error_noise_percent_param,
  use_incremental_formulation_param,
  use_nonlinear_projection_param,
  sort_txt_output_param,
  global_regularization_alpha_param,
  global_stabilization_tau_param,
  global_formulation_param,
  global_solver_param,
  mms_spec_param,
  num_image_integration_points_param,
  global_element_type_param,
  use_fixed_point_iterations_param,
  initial_condition_file_param
};

} // end DICe namespace

#endif
