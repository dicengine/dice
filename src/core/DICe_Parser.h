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

#ifndef DICE_PARSER_H
#define DICE_PARSER_H

#include <DICe.h>
#include <DICe_Shape.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

namespace DICe {

// input parameter names
// using globals here to avoid misspellings
/// Input parameter, location to place the output files
const char* const output_folder = "output_folder";
/// Input parameter to specify output prefix
const char* const output_prefix = "output_prefix";
/// Input parameter, location of the input images
const char* const image_folder = "image_folder";
/// Input parameter, only for local DIC
const char* const subset_size = "subset_size";
/// Input parameter, only for local DIC
const char* const step_size = "step_size";
/// Optional input parameter to specify the x and y coordinates of the subset centroids
const char* const subset_file = "subset_file";
/// Input parameter, only for constrained optimization DIC
const char* const mesh_file = "mesh_file";
/// Input parameter, only for constrained optimization DIC
const char* const time_force_file = "time_force_file";
/// Input parameter, only for constrained optimization DIC
const char* const mesh_output_file = "mesh_output_file";
/// Input parameter, only for global DIC
const char* const mesh_size = "mesh_size";
/// Input parameter, only for global DIC
const char* const image_edge_buffer_size = "image_edge_buffer_size";
/// Input parameter
const char* const print_timing = "print_timing";
/// Input parameter
const char* const correlation_parameters_file = "correlation_parameters_file";
/// Input parameter
const char* const physics_parameters_file = "physics_parameters_file";
// For simple two image correlation
/// Input parameter
const char* const reference_image = "reference_image";
/// Input parameter
const char* const cine_file = "cine_file";
/// Input parameter
const char* const cine_ref_index = "cine_ref_index";
/// Input parameter
const char* const cine_start_index = "cine_start_index";
/// Input parameter
const char* const cine_end_index = "cine_end_index";
/// Input parameter (multiple deformed images not allowed)
const char* const deformed_images = "deformed_images";
// For image sequences
/// Input parameter
const char* const reference_image_index = "reference_image_index";
/// Input parameter
const char* const num_images = "num_images";
/// Input parameter
const char* const num_file_suffix_digits = "num_file_suffix_digits";
/// Input parameter
const char* const image_file_extension = "image_file_extension";
/// Input parameter
const char* const image_file_prefix = "image_file_prefix";
/// Input parameter
const char* const separate_output_file_for_each_subset = "separate_output_file_for_each_subset";
/// Input parameter
const char* const create_separate_run_info_file = "create_separate_run_info_file";

/// \brief Parses the options set in the command line when dice is invoked
/// \param argc typical argument from main
/// \param argv typical argument from main
/// \param outStream [out] The output stream, gets set to cout or blackholeStream depending on command line options
/// \param analysis_type Global, Local, Constrained Optimziation, ...
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> parse_command_line(int argc,
  char *argv[],
  Teuchos::RCP<std::ostream> & outStream,
  const Analysis_Type analysis_type=LOCAL_DIC);

/// \brief Read the correlation parameters from a file
/// \param paramFileName File name of the correlation parameters file
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> read_correlation_params(const std::string & paramFileName);

/// \brief Read the physics parameters from a file
/// \param paramFileName File name of the physics parameters file
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> read_physics_params(const std::string & paramFileName);

/// struct to hold motion window around a subset to test for movement
struct
DICE_LIB_DLL_EXPORT
Motion_Window_Params {
  /// constructor
  Motion_Window_Params():
  start_x_(0),
  start_y_(0),
  end_x_(0),
  end_y_(0),
  tol_(-1.0),
  use_subset_id_(-1),
  use_motion_detection_(false){};
  /// upper left corner x coord
  int_t start_x_;
  /// upper left corner y coord
  int_t start_y_;
  /// lower right corner x coord
  int_t end_x_;
  /// lower right corner y coord
  int_t end_y_;
  /// tolerance for motion detection
  scalar_t tol_;
  /// point to another subsets' motion window if multiple subsets share one
  int_t use_subset_id_;
  /// use motion detection
  bool use_motion_detection_;
};

/// Simple struct for passing info back and forth from read_subset_file:
struct
DICE_LIB_DLL_EXPORT
Subset_File_Info {
  /// Generic constructor
  /// \param info_type optional type argument (assumes SUBSET_INFO)
  Subset_File_Info(const Subset_File_Info_Type info_type=SUBSET_INFO){
    conformal_area_defs = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def>());
    coordinates_vector = Teuchos::rcp(new std::vector<int_t>());
    neighbor_vector = Teuchos::rcp(new std::vector<int_t>());
    id_sets_map = Teuchos::rcp(new std::map<int_t,std::vector<int_t> >());
    size_map = Teuchos::rcp(new std::map<int_t,std::pair<int_t,int_t> >());
    displacement_map = Teuchos::rcp(new std::map<int_t,std::pair<scalar_t,scalar_t> >());
    normal_strain_map = Teuchos::rcp(new std::map<int_t,std::pair<scalar_t,scalar_t> >());
    shear_strain_map = Teuchos::rcp(new std::map<int_t,scalar_t>());
    rotation_map = Teuchos::rcp(new std::map<int_t,scalar_t>());
    seed_subset_ids = Teuchos::rcp(new std::map<int_t,int_t>());
    path_file_names = Teuchos::rcp(new std::map<int_t,std::string>());
    optical_flow_flags = Teuchos::rcp(new std::map<int_t,bool>());
    skip_solve_flags = Teuchos::rcp(new std::map<int_t,std::vector<int_t> >());
    motion_window_params = Teuchos::rcp(new std::map<int_t,Motion_Window_Params>());
    type = info_type;
  }
  /// Pointer to map of conformal subset defs (these are used to define conformal subsets)
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs;
  /// Pointer to the vector of subset centroid coordinates
  Teuchos::RCP<std::vector<int_t> > coordinates_vector;
  /// Pointer to the vector of neighbor ids
  Teuchos::RCP<std::vector<int_t> > neighbor_vector;
  /// Pointer to a map that has vectos of subset ids (used to denote blocking subsets)
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > id_sets_map;
  /// Type of information (subset or region of interest)
  Subset_File_Info_Type type;
  /// Pointer to a map of std::pairs of size values
  Teuchos::RCP<std::map<int_t,std::pair<int_t,int_t> > > size_map;
  /// Pointer to a map of initial guesses for displacement, the map index is the subset id
  Teuchos::RCP<std::map<int_t,std::pair<scalar_t,scalar_t> > > displacement_map;
  /// Pointer to a map of initial guesses for normal strain, the map index is the subset id
  Teuchos::RCP<std::map<int_t,std::pair<scalar_t,scalar_t> > > normal_strain_map;
  /// Pointer to a map of initial guesses for shear strain, the map index is the subset id
  Teuchos::RCP<std::map<int_t,scalar_t> > shear_strain_map;
  /// Pointer to a map of initial guesses for rotation, the map index is the subset id
  Teuchos::RCP<std::map<int_t,scalar_t> > rotation_map;
  /// Map that lists the subset ids for each of the seeds (first value is subset_id, second is roi_id)
  Teuchos::RCP<std::map<int_t,int_t> > seed_subset_ids;
  /// Map that lists the names of the path files for each subset
  Teuchos::RCP<std::map<int_t,std::string> > path_file_names;
  /// Map that turns on optical flow initializer for certain subsets
  Teuchos::RCP<std::map<int_t,bool> > optical_flow_flags;
  /// Map that turns off the solve (initialize only) for certain subsets
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > skip_solve_flags;
  /// Map that tests each frame for motion before performing DIC optimization
  Teuchos::RCP<std::map<int_t,Motion_Window_Params> > motion_window_params;
};

/// \brief Read a list of coordinates for correlation points from file
/// \param fileName String name of the file that defines the subsets
/// \param width The image width (used to check for valid coords)
/// \param height The image height (used to check for valid coords)
/// TODO add conformal subset notes here.
DICE_LIB_DLL_EXPORT
const Teuchos::RCP<Subset_File_Info> read_subset_file(const std::string & fileName,
  const int_t width,
  const int_t height);

/// \brief Turns a string read from getline() into tokens
/// \param dataFile fstream file to read line from (assumed to be open)
/// \param delim Delimiter character
/// \param capitalize true if the tokens should be automatically capitalized
DICE_LIB_DLL_EXPORT
Teuchos::ArrayRCP<std::string> tokenize_line(std::fstream &dataFile,
  const std::string & delim=" \t",
  const bool capitalize = true);

/// \brief Read a circle from the input file
/// \param dataFile fstream file to read line from (assumed to be open)
DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::Circle> read_circle(std::fstream &dataFile);

/// \brief Read a circle from the input file
/// \param dataFile fstream file to read line from (assumed to be open)
DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::Rectangle> read_rectangle(std::fstream &dataFile);

/// \brief Read a polygon from the input file
/// \param dataFile fstream file to read line from (assumed to be open)
DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::Polygon> read_polygon(std::fstream &dataFile);

/// \brief Read Several shapes from the subset file
/// \param dataFile fstream file to read line from (assumed to be open)
DICE_LIB_DLL_EXPORT
multi_shape read_shapes(std::fstream & dataFile);

/// \brief Converts params into string names of all the images in a sequence
/// \param params Defines the prefix for the image names, number of images, etc.
DICE_LIB_DLL_EXPORT
const std::vector<std::string> decipher_image_file_names(Teuchos::RCP<Teuchos::ParameterList> params);

/// \brief Creates a regular square grid of correlation points
/// \param correlation_points Vector of global point coordinates
/// \param neighbor_ids Vector of neighbor ids (established if there is a seed)
/// \param params Used to determine the step size (spacing of points)
/// \param width The image width
/// \param height The image height
/// \param subset_file_info Optional information from the subset file (ROIs, etc.)
DICE_LIB_DLL_EXPORT
void create_regular_grid_of_correlation_points(std::vector<int_t> & correlation_points,
  std::vector<int_t> & neighbor_ids,
  Teuchos::RCP<Teuchos::ParameterList> params,
  const int_t width, const int_t height,
  Teuchos::RCP<DICe::Subset_File_Info> subset_file_info=Teuchos::null);

/// \brief Test to see that the point falls with the boundary of a conformal def and not in the excluded area
/// \param x_coord X coordinate of the point in question
/// \param y_coord Y coordinate of the point in question
/// \param width Image width
/// \param height Image height
/// \param subset_size Size of the subset
/// \param coords Set of valid coordinates in the area
/// \param excluded_coords Set of coordinates that should be excluded
DICE_LIB_DLL_EXPORT
bool valid_correlation_point(const int_t x_coord,
  const int_t y_coord,
  const int_t width,
  const int_t height,
  const int_t subset_size,
  std::set<std::pair<int_t,int_t> > & coords,
  std::set<std::pair<int_t,int_t> > & excluded_coords);

/// \brief Create template input files with lots of comments
/// \param file_prefix The prefix used to name the template files
DICE_LIB_DLL_EXPORT
void generate_template_input_files(const std::string & file_prefix);

/// \brief Create the opening parameter syntax
/// \param file_name The name of the xml file
DICE_LIB_DLL_EXPORT
void initialize_xml_file(const std::string & file_name);

/// \brief Create the closing parameter syntax
/// \param file_name The name of the xml file
DICE_LIB_DLL_EXPORT
void finalize_xml_file(const std::string & file_name);

/// \brief Write a comment to the xml file specified
/// \param file_name The name of the xml file
/// \param comment String comment to print to file
DICE_LIB_DLL_EXPORT
void write_xml_comment(const std::string & file_name,
  const std::string & comment);

/// \brief Write the opening part of a ParameterList
/// \param file_name The xml file
/// \param name Name of the ParameterList
/// \param commented String comment to print to file
DICE_LIB_DLL_EXPORT
void write_xml_param_list_open(const std::string & file_name,
  const std::string & name,
  const bool commented=true);

/// \brief Write the closing part of a ParameterList
/// \param file_name The xml file
/// \param commented String comment to print to file
DICE_LIB_DLL_EXPORT
void write_xml_param_list_close(const std::string & file_name,
  const bool commented=true);

/// \brief Write a general xml parameter to file
/// \param file_name The name of the xml file to write to
/// \param name The name of the parameter
/// \param type The type of the parameter (bool, int, double, string)
/// \param value The string value of the parameter
/// \param commented Determines if this parameter should be commented out
DICE_LIB_DLL_EXPORT
void write_xml_param(const std::string & file_name,
  const std::string & name,
  const std::string & type,
  const std::string & value,
  const bool commented=true);

/// \brief Write a string parameter to file
/// \param file_name The name of the xml file to write to
/// \param name The name of the parameter
/// \param value The string value of the parameter
/// \param commented Determines if this parameter should be commented out
DICE_LIB_DLL_EXPORT
void write_xml_string_param(const std::string & file_name,
  const std::string & name,
  const std::string & value="<value>",
  const bool commented=true);

/// \brief Write an integer valued parameter to file
/// \param file_name The name of the xml file to write to
/// \param name The name of the parameter
/// \param value The string value of the parameter
/// \param commented Determines if this parameter should be commented out
DICE_LIB_DLL_EXPORT
void write_xml_size_param(const std::string & file_name,
  const std::string & name,
  const std::string & value="<value>",
  const bool commented=true);

/// \brief Write a real valued parameter to file
/// \param file_name The name of the xml file to write to
/// \param name The name of the parameter
/// \param value The string value of the parameter
/// \param commented Determines if this parameter should be commented out
DICE_LIB_DLL_EXPORT
void write_xml_real_param(const std::string & file_name,
  const std::string & name,
  const std::string & value="<value>",
  const bool commented=true);

/// \brief Write a boolean valued parameter to file
/// \param file_name The name of the xml file to write to
/// \param name The name of the parameter
/// \param value The string value of the parameter
/// \param commented Determines if this parameter should be commented out
DICE_LIB_DLL_EXPORT
void write_xml_bool_param(const std::string & file_name,
  const std::string & name,
  const std::string & value="<value>",
  const bool commented=true);

/// \brief Determines if a string is a number
/// \param s Input string
DICE_LIB_DLL_EXPORT
bool is_number(const std::string& s);

/// Parser string
const char* const parser_comment_char = "#";
/// Parser string
const char* const parser_begin = "BEGIN";
/// Parser string
const char* const parser_end = "END";
/// Parser string
const char* const parser_subset_coordinates = "SUBSET_COORDINATES";
/// Parser string
const char* const parser_region_of_interest = "REGION_OF_INTEREST";
/// Parser string
const char* const parser_conformal_subset = "CONFORMAL_SUBSET";
/// Parser string
const char* const parser_subset_id = "SUBSET_ID";
/// Parser string
const char* const parser_boundary = "BOUNDARY";
/// Parser string
const char* const parser_excluded = "EXCLUDED";
/// Parser string
const char* const parser_obstructed = "OBSTRUCTED";
/// Parser string
const char* const parser_blocking_subsets = "BLOCKING_SUBSETS";
/// Parser string
const char* const parser_polygon = "POLYGON";
/// Parser string
const char* const parser_circle = "CIRCLE";
/// Parser string
const char* const parser_rectangle = "RECTANGLE";
/// Parser string
const char* const parser_center = "CENTER";
/// Parser string
const char* const parser_radius = "RADIUS";
/// Parser string
const char* const parser_vertices = "VERTICES";
/// Parser string
const char* const parser_width = "WIDTH";
/// Parser string
const char* const parser_height = "HEIGHT";
/// Parser string
const char* const parser_upper_left = "UPPER_LEFT";
/// Parser string
const char* const parser_lower_right = "LOWER_RIGHT";
/// Parser string
const char* const parser_seed = "SEED";
/// Parser string
const char* const parser_use_optical_flow = "USE_OPTICAL_FLOW";
/// Parser string
const char* const parser_use_path_file = "USE_PATH_FILE";
/// Parser string
const char* const parser_skip_solve = "SKIP_SOLVE";
/// Parser string
const char* const parser_test_for_motion = "TEST_FOR_MOTION";
/// Parser string
const char* const parser_motion_window = "MOTION_WINDOW";
/// Parser string
const char* const parser_location = "LOCATION";
/// Parser string
const char* const parser_displacement = "DISPLACEMENT";
/// Parser string
const char* const parser_normal_strain = "NORMAL_STRAIN";
/// Parser string
const char* const parser_shear_strain = "SHEAR_STRAIN";
/// Parser string
const char* const parser_rotation = "ROTATION";


}// End DICe Namespace

#endif
