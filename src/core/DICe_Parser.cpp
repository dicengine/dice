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

#include <DICe_Parser.h>
#include <DICe_XMLUtils.h>

#include <DICe_ParameterUtilities.h>
#include <DICe.h>
#include <DICe_ImageIO.h>
#include <DICe_NetCDF.h>
#include <DICe_FieldEnums.h>

#include <hypercine.h>

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <iostream>
#include <fstream>
#include <cctype>
#include <string>
#include <algorithm>
#include <vector>
#if DICE_MPI
#  include <mpi.h>
#endif

namespace DICe {

DICE_LIB_DLL_EXPORT
void create_directory(const std::string & folder){
#if defined(WIN32)
  CreateDirectory(folder.c_str(), NULL);
#else
  mkdir(folder.c_str(), 0777);
#endif
}

Command_Line_Parser::Command_Line_Parser (int &argc, char **argv){
  for (int i=1; i < argc; ++i)
    this->tokens.push_back(std::string(argv[i]));
}

const std::string&
Command_Line_Parser::get_option(const std::string &option) const
{
  std::vector<std::string>::const_iterator itr;
  itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
  if (itr != this->tokens.end() && ++itr != this->tokens.end()){
    return *itr;
  }
  static const std::string empty_string("");
  return empty_string;
}

bool
Command_Line_Parser::option_exists(const std::string &option) const
{
  return std::find(this->tokens.begin(), this->tokens.end(), option)
  != this->tokens.end();
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> parse_command_line(int argc,
  char *argv[],
  bool & force_exit,
  Teuchos::RCP<std::ostream> & outStream){

  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif
  Teuchos::RCP<Teuchos::ParameterList> inputParams = Teuchos::rcp( new Teuchos::ParameterList() );
  force_exit = false;

  Command_Line_Parser cmp(argc, argv);

  if(cmp.option_exists("-h")){
    print_banner();
    std::cout << "Valid options: " << std::endl;
    std::cout << "h: produce help message" << std::endl;
    std::cout << "v: output verbose log to screen" << std::endl;
    std::cout << "t: print timing statistics to screen" << std::endl;
    std::cout << "version: output version information to screen" << std::endl;
    std::cout << "i <string>: XML input file name" << std::endl;
    std::cout << "g <string>: create XML input file template" << std::endl;
    std::cout << "s: print field statistics to the screen" << std::endl;
    std::cout << "ss_locs: create a file (ss_locs.txt) with the subset locations and exit prior to executing the analysis" << std::endl;
    std::cout << "d: returns 1 if the debugging messages are on, 0 if they are off" << std::endl;
    std::cout << "tracklib: returns 1 if DICe was compiled with tracklib on, 0 if not" << std::endl;
    force_exit = true;
    return inputParams;
  }

  Teuchos::RCP<std::ostream> bhs = Teuchos::rcp(new Teuchos::oblackholestream); // outputs nothing
  outStream = Teuchos::rcp(&std::cout, false);
  // Print info to screen
  if(!cmp.option_exists("-v") || proc_rank!=0){
    outStream = bhs;
  }

  // Handle version requests
  if(cmp.option_exists("--version")){
    if(proc_rank==0)
      print_banner();
    force_exit = true;
    return inputParams;
  }

  // Handle debug message on or off requests
  if(cmp.option_exists("-d")){
    force_exit = true;
#ifdef DICE_DEBUG_MSG
    inputParams->set("debug_msg_on",true);
    DEBUG_MSG("debugging messages are on");
#else
    inputParams->set("debug_msg_on",false);
    std::cout << "debugging messages are off" << std::endl;
#endif
    return inputParams;
  }
  // Determine if tracklib is available
  if(cmp.option_exists("--tracklib")){
    force_exit = true;
#ifdef DICE_ENABLE_TRACKLIB
    inputParams->set("tracklib_on",true);
    DEBUG_MSG("tracklib is on");
#else
    inputParams->set("tracklib_on",false);
    DEBUG_MSG("tracklib is off");
#endif
    return inputParams;
  }

  // Generate input file templates and exit
  const std::string &generate_name = cmp.get_option("-g");
  if (!generate_name.empty()){
    if(proc_rank==0) DEBUG_MSG("Generating input file templates using prefix: " << generate_name);
    generate_template_input_files(generate_name);
    force_exit = true;
    return inputParams;
  }

  const std::string &input_file = cmp.get_option("-i");
  if (input_file.empty()){
    std::cout << "Error: The XML input file must be specified on the command line with -i <filename>.xml" << std::endl;
    std::cout << "       (To generate template input files, specify -g [file_prefix] on the command line)" << std::endl;
    exit(1);
  }
  if(proc_rank==0) DEBUG_MSG("Using input file: " << input_file);

  Teuchos::Ptr<Teuchos::ParameterList> inputParamsPtr(inputParams.get());
  Teuchos::updateParametersFromXmlFile(input_file, inputParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(inputParams==Teuchos::null,std::runtime_error,"");

  // Print timing statistics?
  if(cmp.option_exists("-t")){
    inputParams->set(DICe::print_timing,true);
  }

  // Print subset locations and exit?
  if(cmp.option_exists("--ss_locs")){
    inputParams->set(DICe::print_subset_locations_and_exit,true);
  }

  // Print timing statistics?
  if(cmp.option_exists("-s")){
    inputParams->set(DICe::print_stats,true);
  }

  Analysis_Type analysis_type = LOCAL_DIC;
  if(inputParams->isParameter(DICe::mesh_size)||inputParams->isParameter(DICe::mesh_file)){
    DEBUG_MSG("Using GLOBAL DIC formulation since mesh_size or mesh_file parameter was specified.");
    analysis_type = GLOBAL_DIC;
  }
  else if(inputParams->get(DICe::use_tracklib,false)){
    DEBUG_MSG("Using TRACKLIB formulation since use_tracklib was specified.");
    analysis_type = TRACKLIB;
  }

  // Test the input values
  std::vector<std::pair<std::string,std::string> > required_params;
  required_params.push_back(std::pair<std::string,std::string>(DICe::image_folder,"string"));
  if(analysis_type==GLOBAL_DIC){
    required_params.push_back(std::pair<std::string,std::string>(DICe::output_folder,"string"));
    //required_params.push_back(std::pair<std::string,std::string>(DICe::mesh_size,"double"));
    //required_params.push_back(std::pair<std::string,std::string>(DICe::image_edge_buffer_size,"int"));
  }
  if(analysis_type==LOCAL_DIC){
    required_params.push_back(std::pair<std::string,std::string>(DICe::output_folder,"string"));
  }
  if(analysis_type==TRACKLIB){
    required_params.push_back(std::pair<std::string,std::string>(DICe::output_folder,"string"));
  }

  // make sure that a subset file was defined:
  bool required_param_missing = false;
  for(size_t i=0;i<required_params.size();++i){
    if(!inputParams->isParameter(required_params[i].first)){
      std::cout << "Error: The parameter " << required_params[i].first << " of type " <<
          required_params[i].second << " must be defined in " << input_file << std::endl;
      std::cout << "<Parameter name=\"" << required_params[i].first << "\" type=\"" <<
          required_params[i].second << "\" value=\"<value>\" />" << std::endl;
      required_param_missing = true;
    }
  }
  // require either the step_size or subset_file
  if(analysis_type==LOCAL_DIC){
    if(!inputParams->isParameter(DICe::subset_file)&&!inputParams->isParameter(DICe::step_size)){
      std::cout << "Error: The parameter " << DICe::subset_file << " of type string or " <<
          DICe::step_size << " of type int must be defined in " << input_file << std::endl;
      required_param_missing = true;
    }
  }
  else if(analysis_type==GLOBAL_DIC){ // GLOBAL DIC
    std::vector<std::string> invalid_params;
    invalid_params.push_back(DICe::step_size);
    invalid_params.push_back(DICe::subset_size);
    for(size_t i=0;i<invalid_params.size();++i){
      if(inputParams->isParameter(invalid_params[i])){
        std::cout << "Error: The parameter " << invalid_params[i] <<
            " is not valid for this global DIC analysis" << std::endl;
        required_param_missing = true;
      }
    }
    if(!inputParams->isParameter(DICe::subset_file)&&!inputParams->isParameter(DICe::mesh_file)){
      std::cout << "Error: The parameter " << DICe::subset_file << " or " << DICe::mesh_file << " of type string must be defined for global DIC in " << input_file << std::endl;
      required_param_missing = true;
    }
  }
  if(!inputParams->isParameter(DICe::reference_image_index)&&!inputParams->isParameter(DICe::reference_image)&&!inputParams->isParameter(DICe::cine_file)&&!inputParams->isParameter(DICe::netcdf_file)){
    std::cout << "Error: Either the parameter " << DICe:: reference_image_index << " or " <<
        DICe::reference_image << " or " << DICe::cine_file << " or " << DICe::netcdf_file << " needs to be specified in " << input_file << std::endl;
    required_param_missing = true;
  }
  // specifying a simple two image correlation
  if(inputParams->isParameter(DICe::reference_image)){
    if(inputParams->isParameter(DICe::end_image_index)){
      std::cout << "Error: The parameter " << DICe::end_image_index <<
          " cannot be specified for a simple two image correlation (denoted by using the reference_image param) in " << input_file << std::endl;
      required_param_missing = true;
    }
    if(inputParams->isParameter(DICe::num_file_suffix_digits)){
      std::cout << "Error: The parameter " << DICe::num_file_suffix_digits <<
          " cannot be specified for a simple two image correlation (denoted by using the reference_image param) in " << input_file << std::endl;
      required_param_missing = true;
    }
    if(inputParams->isParameter(DICe::image_file_extension)){
      std::cout << "Error: The parameter " << DICe::image_file_extension <<
          " cannot be specified for a simple two image correlation (denoted by using the reference_image param) in " << input_file << std::endl;
      required_param_missing = true;
    }
    if(inputParams->isParameter(DICe::image_file_prefix)){
      std::cout << "Error: The parameter " << DICe::image_file_prefix <<
          " cannot be specified for a simple two image correlation (denoted by using the reference_image param) in " << input_file << std::endl;
      required_param_missing = true;
    }
  }
  // specifying image sequence
  if((inputParams->isParameter(DICe::reference_image_index)&&inputParams->isParameter(DICe::reference_image))||
      (inputParams->isParameter(DICe::reference_image_index)&&inputParams->isParameter(DICe::deformed_images))){
    std::cout << "Error: Cannot specify both " << DICe:: reference_image_index <<
        " and (" << DICe::reference_image << " or " << DICe::deformed_images << ") in " << input_file << std::endl;
    required_param_missing = true;
  }
  if(inputParams->isParameter(DICe::reference_image_index)){
    if(!inputParams->isParameter(DICe::end_image_index)){
      std::cout << "Error: The parameter " << DICe::end_image_index << " of type int must be defined in " << input_file << std::endl;
      required_param_missing = true;
    }
    if(!inputParams->isParameter(DICe::num_file_suffix_digits)){
      std::cout << "Error: The parameter " << DICe::num_file_suffix_digits << " of type int must be defined in " << input_file << std::endl;
      required_param_missing = true;
    }
    if(!inputParams->isParameter(DICe::image_file_extension)){
      std::cout << "Error: The parameter " << DICe::image_file_extension << " of string must be defined in " << input_file << std::endl;
      required_param_missing = true;
    }
    if(!inputParams->isParameter(DICe::image_file_prefix)){
      std::cout << "Error: The parameter " << DICe::image_file_prefix << " of string must be defined in " << input_file << std::endl;
      required_param_missing = true;
    }
  }

  if(required_param_missing) exit(1);

  std::string output_folder = inputParams->get<std::string>(DICe::output_folder);
  if(proc_rank==0) DEBUG_MSG("Attempting to create directory : " << output_folder);
  create_directory(output_folder);

  return inputParams;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> read_input_params(const std::string & paramFileName){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Parsing input parameters from file: " << paramFileName);
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> paramsPtr(params.get());
  Teuchos::updateParametersFromXmlFile(paramFileName, paramsPtr);
  return params;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> read_correlation_params(const std::string & paramFileName){

  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Parsing correlation parameters from file: " << paramFileName);
  Teuchos::RCP<Teuchos::ParameterList> stringParams = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> stringParamsPtr(stringParams.get());
  Teuchos::updateParametersFromXmlFile(paramFileName, stringParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(stringParams==Teuchos::null,std::runtime_error,"");

  // The string params are what as given by the user. These are string values because things like correlation method
  // are not recognized by the xml parser. Only the string values need to be converted into DICe types.
  Teuchos::RCP<Teuchos::ParameterList> diceParams = Teuchos::rcp( new Teuchos::ParameterList() );
  // iterate through the params and copy over all non-string types:
  for(Teuchos::ParameterList::ConstIterator it=stringParams->begin();it!=stringParams->end();++it){
    std::string paramName = it->first;
    if(proc_rank==0) DEBUG_MSG("Found user specified correlation parameter: " << paramName);
    to_lower(paramName); // string parameter names are lower case
    if(DICe::is_string_param(paramName)){
      if(proc_rank==0) DEBUG_MSG("This is a string parameter so it may need to be translated");
      // make sure it's one of the valid entries below:
      if(paramName == DICe::correlation_routine){
        diceParams->set(DICe::correlation_routine,DICe::string_to_correlation_routine(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::interpolation_method){
        diceParams->set(DICe::interpolation_method,DICe::string_to_interpolation_method(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::shape_function_type){
        diceParams->set(DICe::shape_function_type,DICe::string_to_shape_function_type(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::gradient_method){
        diceParams->set(DICe::gradient_method,DICe::string_to_gradient_method(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::optimization_method){
        diceParams->set(DICe::optimization_method,DICe::string_to_optimization_method(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::projection_method){
        diceParams->set(DICe::projection_method,DICe::string_to_projection_method(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::global_formulation){
        diceParams->set(DICe::global_formulation,DICe::string_to_global_formulation(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::global_solver){
        diceParams->set(DICe::global_solver,DICe::string_to_global_solver(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::initialization_method){
        diceParams->set(DICe::initialization_method,DICe::string_to_initialization_method(
          stringParams->get<std::string>(it->first)));
      }
      else if(paramName == DICe::cross_initialization_method){
        diceParams->set(DICe::cross_initialization_method,DICe::string_to_initialization_method(
          stringParams->get<std::string>(it->first)));
      }
      else{
        if(proc_rank==0) DEBUG_MSG("Not a string parameter that needs to be translated");
        diceParams->setEntry(it->first,it->second);
      }
    }
    else{
      if(proc_rank==0) DEBUG_MSG("Not a string parameter, so passing without translating");
     diceParams->setEntry(it->first,it->second);
    }
  }
  return diceParams;
}

DICE_LIB_DLL_EXPORT
void decipher_image_file_names(Teuchos::RCP<Teuchos::ParameterList> params,
  std::vector<std::string> & image_files,
  std::vector<std::string> & stereo_image_files,
  int_t & frame_id_start,
  int_t & num_frames,
  int_t & frame_skip){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  frame_id_start = 0;
  num_frames = 1;
  frame_skip = 1;

  // The reference image will be the first image in the vector, the rest are the deformed
  image_files.clear();
  stereo_image_files.clear();
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::image_folder),std::runtime_error,
    "Error, image folder was not specified");
  std::string folder = params->get<std::string>(DICe::image_folder);
  // Requires that the user add the appropriate slash or backslash as denoted in the template file creator
  if(proc_rank==0) DEBUG_MSG("Image folder prefix: " << folder );

  // User specified a ref and def image alone (not a sequence)
  if(params->isParameter(DICe::reference_image)){
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image_index),std::runtime_error,
      "Error, cannot specify reference_image_index and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::start_image_index),std::runtime_error,
      "Error, cannot specify start_image_index and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::end_image_index),std::runtime_error,
      "Error, cannot specify end_image_index and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::image_file_prefix),std::runtime_error,
      "Error, cannot specify image_file_prefix and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::image_file_extension),std::runtime_error,
      "Error, cannot specify image_file_prefix and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::num_file_suffix_digits),std::runtime_error,
      "Error, cannot specify image_file_prefix and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_left_suffix),std::runtime_error,
      "Error, cannot specify stereo_left_suffix and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_right_suffix),std::runtime_error,
      "Error, cannot specify stereo_right_suffix and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::cine_file),std::runtime_error,
      "Error, cannot specify cine_file and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::netcdf_file),std::runtime_error,
      "Error, cannot specify netcdf_file and reference_image");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_cine_file),std::runtime_error,
      "Error, cannot specify stereo_cine_file and reference_image");

    std::string ref_file_name = params->get<std::string>(DICe::reference_image);
    std::stringstream ref_name;
    ref_name << folder << ref_file_name;
    image_files.push_back(ref_name.str());
    if(params->isParameter(DICe::stereo_reference_image)){
      std::string stereo_ref_file_name = params->get<std::string>(DICe::stereo_reference_image);
      std::stringstream stereo_ref_name;
      stereo_ref_name << folder << stereo_ref_file_name;
      stereo_image_files.push_back(stereo_ref_name.str());
    }

    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::deformed_images),std::runtime_error,
      "Error, the deformed images were not specified");
    // create a sublist of the deformed images:
    Teuchos::ParameterList def_image_sublist = params->sublist(DICe::deformed_images);
    // iterate the sublis and add the params to the output params:
    for(Teuchos::ParameterList::ConstIterator it=def_image_sublist.begin();it!=def_image_sublist.end();++it){
      const bool active = def_image_sublist.get<bool>(it->first);
      if(active){
        std::stringstream def_name;
        def_name << folder << it->first;
        image_files.push_back(def_name.str());
      }
    }
    if(params->isParameter(DICe::stereo_reference_image)){
      TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::stereo_deformed_images),std::runtime_error,
        "Error, the stereo deformed images were not specified");
      Teuchos::ParameterList stereo_def_image_sublist = params->sublist(DICe::stereo_deformed_images);
      // iterate the sublis and add the params to the output params:
      for(Teuchos::ParameterList::ConstIterator it=stereo_def_image_sublist.begin();it!=stereo_def_image_sublist.end();++it){
        const bool active = stereo_def_image_sublist.get<bool>(it->first);
        if(active){
          std::stringstream def_name;
          def_name << folder << it->first;
          stereo_image_files.push_back(def_name.str());
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(stereo_image_files.size()!=image_files.size(),std::runtime_error,
        "Error, the number of deformed images and stereo deformed images must be the same");
    }
    num_frames = image_files.size()-1; // the first is the ref image
  }
  else if(params->isParameter(DICe::cine_file)){
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image),std::runtime_error,
      "Error, cannot specify reference_image and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::deformed_images),std::runtime_error,
      "Error, cannot specify deformed_images and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_reference_image),std::runtime_error,
      "Error, cannot specify stereo_reference_image and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_deformed_images),std::runtime_error,
      "Error, cannot specify stereo_deformed_images and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image_index),std::runtime_error,
      "Error, cannot specify reference_image_index and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::end_image_index),std::runtime_error,
      "Error, cannot specify end_image_index and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::start_image_index),std::runtime_error,
      "Error, cannot specify start_image_index and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::image_file_prefix),std::runtime_error,
      "Error, cannot specify image_file_prefix and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::image_file_extension),std::runtime_error,
      "Error, cannot specify image_file_prefix and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::num_file_suffix_digits),std::runtime_error,
      "Error, cannot specify image_file_prefix and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_left_suffix),std::runtime_error,
      "Error, cannot specify stereo_left_suffix and cine_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_right_suffix),std::runtime_error,
      "Error, cannot specify stereo_right_suffix and cine_file");

//    image_files.push_back(DICe::cine_file);
//    image_files.push_back(DICe::cine_file);
//    if(params->isParameter(DICe::stereo_cine_file)){
//      stereo_image_files.push_back(DICe::cine_file);
//      stereo_image_files.push_back(DICe::cine_file);
//    }
    std::stringstream cine_name;
    std::string cine_file_name = params->get<std::string>(DICe::cine_file);
    cine_name << params->get<std::string>(DICe::image_folder) << cine_file_name;
    Teuchos::RCP<std::ostream> bhs = Teuchos::rcp(new Teuchos::oblackholestream); // outputs nothing
    // read the image data for a frame
    const int_t num_images = DICe::utils::cine_file_frame_count(cine_name.str());
    const int_t first_frame_index = DICe::utils::cine_file_first_frame_id(cine_name.str());
    //TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::cine_ref_index),std::runtime_error,
    //  "Error, the reference index for the cine file has not been specified");
    const int_t cine_ref_index = params->get<int_t>(DICe::cine_ref_index,first_frame_index);
    const int_t cine_start_index = params->get<int_t>(DICe::cine_start_index,first_frame_index);
    const int_t cine_end_index = params->get<int_t>(DICe::cine_end_index,first_frame_index + num_images -1);
    frame_skip = params->get<int_t>(DICe::cine_skip_index,1);
    TEUCHOS_TEST_FOR_EXCEPTION(cine_start_index > cine_end_index,std::invalid_argument,"Error, the cine start index is > the cine end index");
    TEUCHOS_TEST_FOR_EXCEPTION(cine_start_index < first_frame_index,std::invalid_argument,"Error, the cine start index is < the first frame index");
    TEUCHOS_TEST_FOR_EXCEPTION(cine_ref_index > cine_end_index,std::invalid_argument,"Error, the cine ref index is > the cine end index");
    TEUCHOS_TEST_FOR_EXCEPTION(cine_ref_index < first_frame_index,std::invalid_argument,"Error, the cine ref index is < the first frame index");
    TEUCHOS_TEST_FOR_EXCEPTION(cine_end_index < cine_start_index,std::invalid_argument,"Error, the cine end index is < the cine start index");
    TEUCHOS_TEST_FOR_EXCEPTION(cine_end_index < cine_ref_index,std::invalid_argument,"Error, the cine end index is < the ref index");
    TEUCHOS_TEST_FOR_EXCEPTION(cine_end_index >= first_frame_index + num_images,std::invalid_argument,"Error, the cine end index is >= first frame index + num_frames");
    // check if the reference frame should be averaged:
    int_t num_avg_frames = -1;
    if(params->isParameter(DICe::time_average_cine_ref_frame))
      num_avg_frames = params->get<int_t>(DICe::time_average_cine_ref_frame,1);
    // strip the .cine part from the end of the cine file:
    std::string trimmed_cine_name = cine_name.str();
    const std::string ext(".cine");
	std::string lower_case_trimmed_cine_ext(trimmed_cine_name.substr(trimmed_cine_name.size() - ext.size()));
	to_lower(lower_case_trimmed_cine_ext);
    if(trimmed_cine_name.size() > ext.size() && lower_case_trimmed_cine_ext == ".cine")
    {
       trimmed_cine_name = trimmed_cine_name.substr(0, trimmed_cine_name.size() - ext.size());
    }
	else
	{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid cine file: " << cine_file_name); 
    }
    std::stringstream ref_cine_ss;
    ref_cine_ss << trimmed_cine_name << "_";
    if(num_avg_frames>0)
      ref_cine_ss << "avg" << cine_ref_index << "to" << cine_ref_index + num_avg_frames;
    else
      ref_cine_ss << cine_ref_index;

    ref_cine_ss << ".cine";
    frame_id_start = cine_start_index;
    num_frames = std::floor((cine_end_index-cine_start_index)/frame_skip) + 1;
    image_files.resize(num_frames+1); // the additional one is the reference image
    image_files[0] = ref_cine_ss.str();
    int_t current_index = 1;

    for(int_t i=cine_start_index;i<=cine_end_index;i+=frame_skip){
      std::stringstream def_cine_ss;
      def_cine_ss << trimmed_cine_name << "_" << i << ".cine";
      image_files[current_index++] = def_cine_ss.str();
    }
    if(params->isParameter(DICe::stereo_cine_file)){
      std::stringstream stereo_cine_name;
      std::string stereo_cine_file_name = params->get<std::string>(DICe::stereo_cine_file);
      stereo_cine_name << params->get<std::string>(DICe::image_folder) << stereo_cine_file_name;
      // strip the .cine part from the end of the cine file:
      std::string stereo_trimmed_cine_name = stereo_cine_name.str();
	  std::string lower_case_trimmed_cine_ext(stereo_trimmed_cine_name.substr(stereo_trimmed_cine_name.size() - ext.size()));
	  to_lower(lower_case_trimmed_cine_ext);
      if(stereo_trimmed_cine_name.size() > ext.size() && lower_case_trimmed_cine_ext == ".cine" )
      {
         stereo_trimmed_cine_name = stereo_trimmed_cine_name.substr(0, stereo_trimmed_cine_name.size() - ext.size());
      }else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid stereo cine file: " << stereo_cine_file_name);
      }
      std::stringstream stereo_ref_cine_ss;
      stereo_ref_cine_ss << stereo_trimmed_cine_name << "_";
      if(num_avg_frames>0)
        stereo_ref_cine_ss << "avg" << cine_ref_index << "to" << cine_ref_index + num_avg_frames;
      else
        stereo_ref_cine_ss << cine_ref_index;
      stereo_ref_cine_ss << ".cine";
      stereo_image_files.resize(num_frames+1); // additional one is for the ref image
      stereo_image_files[0] = stereo_ref_cine_ss.str();
      int_t current_stereo_index = 1;
      for(int_t i=cine_start_index;i<=cine_end_index;i+=frame_skip){
        std::stringstream stereo_def_cine_ss;
        stereo_def_cine_ss << stereo_trimmed_cine_name << "_" << i << ".cine";
        stereo_image_files[current_stereo_index++] = stereo_def_cine_ss.str();
      }
    }
  } // end cine file
#if DICE_ENABLE_NETCDF
  else if(params->isParameter(DICe::netcdf_file)){
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image),std::runtime_error,
      "Error, cannot specify reference_image and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::deformed_images),std::runtime_error,
      "Error, cannot specify deformed_images and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_reference_image),std::runtime_error,
      "Error, cannot specify stereo_reference_image and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_deformed_images),std::runtime_error,
      "Error, cannot specify stereo_deformed_images and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image_index),std::runtime_error,
      "Error, cannot specify reference_image_index and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::end_image_index),std::runtime_error,
      "Error, cannot specify end_image_index and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::start_image_index),std::runtime_error,
      "Error, cannot specify start_image_index and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::image_file_prefix),std::runtime_error,
      "Error, cannot specify image_file_prefix and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::image_file_extension),std::runtime_error,
      "Error, cannot specify image_file_prefix and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::num_file_suffix_digits),std::runtime_error,
      "Error, cannot specify image_file_prefix and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_left_suffix),std::runtime_error,
      "Error, cannot specify stereo_left_suffix and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_right_suffix),std::runtime_error,
      "Error, cannot specify stereo_right_suffix and netcdf_file");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::cine_file),std::runtime_error,
      "Error, cannot specify cine_file and netcdf_file");

    std::stringstream netcdf_name;
    std::string netcdf_file_name = params->get<std::string>(DICe::netcdf_file);
    netcdf_name << params->get<std::string>(DICe::image_folder) << netcdf_file_name;
    Teuchos::RCP<std::ostream> bhs = Teuchos::rcp(new Teuchos::oblackholestream); // outputs nothing

    Teuchos::RCP<DICe::netcdf::NetCDF_Reader> netcdf_reader = Teuchos::rcp(new DICe::netcdf::NetCDF_Reader());
    int_t netcdf_width = 0;
    int_t netcdf_height = 0;
    int_t netcdf_num_time_steps = 0;
    netcdf_reader->get_image_dimensions(netcdf_name.str(),netcdf_width,netcdf_height,netcdf_num_time_steps);
    TEUCHOS_TEST_FOR_EXCEPTION(netcdf_num_time_steps <=0, std::runtime_error,"");

    // strip the .nc part from the end of the file_name:
    std::string trimmed_name = netcdf_name.str();
    const std::string ext(".nc");
    if(trimmed_name.size() > ext.size() && trimmed_name.substr(trimmed_name.size() - ext.size()) == ".nc" )
    {
       trimmed_name = trimmed_name.substr(0, trimmed_name.size() - ext.size());
    }else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid netcdf file: " << netcdf_file_name);
    }
    std::stringstream ref_netcdf_ss;
    ref_netcdf_ss << trimmed_name << "_frame_0.nc"; // always use the 0th frame for the reference for netcdf files
    image_files.resize(netcdf_num_time_steps+1);
    image_files[0] = ref_netcdf_ss.str();
    int_t current_index = 1;
    for(int_t i=0;i<netcdf_num_time_steps;i++){
      std::stringstream def_netcdf_ss;
      def_netcdf_ss << trimmed_name << "_frame_" << i << ".nc";
      image_files[current_index++] = def_netcdf_ss.str();
    }
    // TODO add stereo netcdf files processed in batch
    num_frames = image_files.size()-1;
  } // end netcdf file
#endif
  // User specified an image sequence:
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image),std::runtime_error,
      "Error, cannot specify reference_image and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::deformed_images),std::runtime_error,
      "Error, cannot specify deformed_images and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_reference_image),std::runtime_error,
      "Error, cannot specify stereo_reference_image and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_deformed_images),std::runtime_error,
      "Error, cannot specify stereo_deformed_images and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::reference_image),std::runtime_error,
      "Error, cannot specify reference_image and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::deformed_images),std::runtime_error,
      "Error, cannot specify deformed_images and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_reference_image),std::runtime_error,
      "Error, cannot specify stereo_reference_image and reference_image_index");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::stereo_deformed_images),std::runtime_error,
      "Error, cannot specify stereo_deformed_images and reference_image_index");

    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::reference_image_index),std::runtime_error,
      "Error, the reference image index was not specified");
    // pull the parameters out
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::end_image_index),std::runtime_error,
      "Error, the end image index was not specified");
    const int_t lastImageIndex = params->get<int_t>(DICe::end_image_index);
    if(params->isParameter(DICe::skip_image_index))
      frame_skip = params->get<int_t>(DICe::skip_image_index,1);

    // single camera only has a prefix
    // stereo can have a prefix with left and right suffix or left and right prefix with no suffix
    const bool has_prefix = params->isParameter(DICe::image_file_prefix);
    std::string file_suffix = params->get<std::string>(DICe::file_suffix,"");
    const bool has_stereo_prefix = params->isParameter(DICe::stereo_left_file_prefix) &&
        params->isParameter(DICe::stereo_left_file_prefix);
    TEUCHOS_TEST_FOR_EXCEPTION(!has_prefix&&!has_stereo_prefix,std::runtime_error,
      "either the image_file_prefix or stereo_left_file_prefix\n"
        " and stereo_right_file_prefix parameters must be set");
    TEUCHOS_TEST_FOR_EXCEPTION(has_prefix&&has_stereo_prefix,std::runtime_error,"");
    const bool has_stereo_suffix = params->isParameter(DICe::stereo_left_suffix) &&
        params->isParameter(DICe::stereo_right_suffix);
    TEUCHOS_TEST_FOR_EXCEPTION(has_stereo_suffix&&has_stereo_prefix,std::runtime_error,
      "stereo files are indicated with a prefix or a suffix, not both");
    const bool is_stereo = has_stereo_prefix || has_stereo_suffix;
    std::string left_prefix="", right_prefix="";
    if(has_prefix){
      left_prefix = params->get<std::string>(DICe::image_file_prefix);
      right_prefix = left_prefix;
    }else{
      left_prefix = params->get<std::string>(DICe::stereo_left_file_prefix);
      right_prefix = params->get<std::string>(DICe::stereo_right_file_prefix);
    }
    std::string left_suffix="", right_suffix="";
    if(has_stereo_suffix){
      left_suffix = params->get<std::string>(DICe::stereo_left_suffix);
      right_suffix = params->get<std::string>(DICe::stereo_right_suffix);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::image_file_extension),std::runtime_error,
      "Error, the image file extension was not specified");
    const std::string fileType = params->get<std::string>(DICe::image_file_extension);
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::num_file_suffix_digits),std::runtime_error,
      "Error, the number of file suffix digits was not specified");
    const int_t digits = params->get<int_t>(DICe::num_file_suffix_digits);
    TEUCHOS_TEST_FOR_EXCEPTION(digits<=0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::reference_image_index),std::runtime_error,
      "Error, the reference image index was not specified");
    const int_t refId = params->get<int_t>(DICe::reference_image_index);
    TEUCHOS_TEST_FOR_EXCEPTION(lastImageIndex < refId,std::runtime_error,"Error invalid reference image index");
    frame_id_start = refId;
    if(params->isParameter(DICe::start_image_index)){
      frame_id_start = params->get<int_t>(DICe::start_image_index);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(frame_id_start > lastImageIndex,std::runtime_error,"Error invalid start image index");
    TEUCHOS_TEST_FOR_EXCEPTION(frame_id_start < 0,std::runtime_error,"Error invalid start image index");
    num_frames = std::floor((lastImageIndex - frame_id_start)/frame_skip) + 1;
    TEUCHOS_TEST_FOR_EXCEPTION(num_frames<=0,std::runtime_error,"");

    // determine the reference image
    int_t number = refId;
    int_t refDig = 0;
    if(refId==0)
      refDig = 1;
    else{
      while (number) {number /= 10; refDig++;}
    }
    std::stringstream left_ref_name, right_ref_name;
    left_ref_name << folder << left_prefix;
    right_ref_name << folder << right_prefix;
    if(digits > 1){
      for(int_t i=0;i<digits - refDig;++i){
        left_ref_name << "0";
        right_ref_name << "0";
      }
    }
    left_ref_name << refId << left_suffix << file_suffix << fileType;
    right_ref_name << refId << right_suffix << file_suffix << fileType;
    image_files.push_back(left_ref_name.str());
    if(is_stereo){
      stereo_image_files.push_back(right_ref_name.str());
    }

    // determine the deformed images
    for(int_t i=frame_id_start;i<=lastImageIndex;i+=frame_skip){
      std::stringstream left_def_name, right_def_name;
      left_def_name << folder << left_prefix;
      right_def_name << folder << right_prefix;
      int_t tmpNum = i;
      int_t defDig = 0;
      if(tmpNum==0)
        defDig = 1;
      else{
        while (tmpNum) {tmpNum /= 10; defDig++;}
      }
      if(digits > 1)
        for(int_t j=0;j<digits - defDig;++j){
          left_def_name << "0";
          right_def_name << "0";
        }
      left_def_name << i << left_suffix << file_suffix << fileType;
      right_def_name << i << right_suffix << file_suffix << fileType;
      image_files.push_back(left_def_name.str());
      if(is_stereo){
        stereo_image_files.push_back(right_def_name.str());
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(is_stereo && image_files.size()!=stereo_image_files.size(),std::runtime_error,"");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(image_files.size()<=1,std::runtime_error,"");
}

/// returns a string with only the name, no extension or directory
DICE_LIB_DLL_EXPORT
std::string file_name_no_dir_or_extension(const std::string & file_name) {
  size_t lastindex = file_name.find_last_of(".");
  size_t last_slash_index = file_name.find_last_of("/");
  if(last_slash_index==std::string::npos)
    last_slash_index = file_name.find_last_of("\\");
  if(last_slash_index==std::string::npos)
    last_slash_index = -1;
  return file_name.substr(last_slash_index+1, lastindex-last_slash_index-1);
}

DICE_LIB_DLL_EXPORT
void generate_template_input_files(const std::string & file_prefix){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("generate_template_input_files() was called.");
  std::stringstream fileNameInput;
  fileNameInput << file_prefix << "_input.xml";
  const std::string inputFile = fileNameInput.str();
  std::stringstream fileNameParams;
  fileNameParams << file_prefix << "_params.xml";
  const std::string paramsFile = fileNameParams.str();
  if(proc_rank==0) DEBUG_MSG("Input file: " << fileNameInput.str() << " Params file: " << fileNameParams.str());

  // clear the files if they exist
  initialize_xml_file(fileNameInput.str());
  initialize_xml_file(fileNameParams.str());

  // write the input file parameters:

  write_xml_comment(inputFile,"Note: this template assumes local DIC algorithm to be used");
  write_xml_string_param(inputFile,DICe::output_folder,"<path>",false);
  write_xml_comment(inputFile,"Note: the output folder needs the trailing slash \"/\" or backslash \"\\\"");
  write_xml_string_param(inputFile,DICe::output_prefix,"<prefix>",true);
  write_xml_comment(inputFile,"Optional prefix to use in the output file name (default is \"DICe_solution\"");
  write_xml_string_param(inputFile,DICe::image_folder,"<path>",false);
  write_xml_comment(inputFile,"Note: the image folder needs the trailing slash \"/\" or backslash \"\\\"");
  write_xml_string_param(inputFile,DICe::correlation_parameters_file,paramsFile,true);
  write_xml_comment(inputFile,"The user can set specific correlation parameters in the file above");
  write_xml_comment(inputFile,"These parameters are activated by uncommenting the correlation_parameters_file option");
  write_xml_string_param(inputFile,DICe::physics_parameters_file,paramsFile,true);
  write_xml_comment(inputFile,"The user can set specific parameters for the physics involved if integrated DIC is used (experimental and not default)");
  write_xml_comment(inputFile,"These parameters are activated by uncommenting the physics_parameters_file option and if the dice_integrated executable is used");
  write_xml_size_param(inputFile,DICe::subset_size,"<value>",false);
  write_xml_size_param(inputFile,DICe::step_size,"<value>",false);
  write_xml_comment(inputFile,"Note: if a subset file is used to below to specify subset centroids, this option should not be used");
  write_xml_bool_param(inputFile,DICe::separate_output_file_for_each_subset,"false",false);
  write_xml_comment(inputFile,"Write a separate output file for each subset with all frames in that file (default is to write one file per frame with all subsets)");
  write_xml_bool_param(inputFile,DICe::create_separate_run_info_file,"false",false);
  write_xml_comment(inputFile,"Write a separate output file that has the header information rather than place it at the top of the output files");
  write_xml_string_param(inputFile,DICe::subset_file,"<path>");
  write_xml_comment(inputFile,"Optional file to specify the coordinates of the subset centroids (cannot be used with step_size param)");
  write_xml_comment(inputFile,"The subset file should be space separated (no commas) with one integer value for the number of subsets on the first line");
  write_xml_comment(inputFile,"and a set of global x and y coordinates for each subset centroid, one point per line.");
  write_xml_comment(inputFile,"There are two ways to specify the deformed images, first by listing them or by providing tokens to create a file sequence (see below)");
  write_xml_string_param(inputFile,DICe::reference_image,"<file_name>",false);
  write_xml_comment(inputFile,"If the images are not grayscale, they will be automatically converted to 8-bit grayscale.");
  write_xml_param_list_open(inputFile,DICe::deformed_images,false);
  write_xml_bool_param(inputFile,"<file_name>","true",false);
  write_xml_param_list_close(inputFile,false);
  write_xml_comment(inputFile,"The correlation evaluation order of the deformed image files will be according the order in the list above.");
  write_xml_comment(inputFile,"The boolean value activates or deactivates the image.");
  write_xml_comment(inputFile,"");
  write_xml_comment(inputFile,"Consider instead an image sequence of files named with the pattern Image_0001.tif, Image_0002.tif, ..., Image_1000.tif");
  write_xml_comment(inputFile,"For an image sequence, remove the two options above (reference_image and deformed_image) and use the following:");
  write_xml_size_param(inputFile,DICe::reference_image_index,"<value>");
  write_xml_comment(inputFile,"The index of the file to use as the reference image (no preceeding zeros). For the example above this would be \"1\" if the first image should be used");
  write_xml_size_param(inputFile,DICe::end_image_index);
  write_xml_comment(inputFile,"The index of the last image in the sequence to analyze. For the example above this would be 1000");
  write_xml_size_param(inputFile,DICe::num_file_suffix_digits);
  write_xml_comment(inputFile,"The number of digits in the file suffix. For the example above this would be 4");
  write_xml_string_param(inputFile,DICe::image_file_extension);
  write_xml_comment(inputFile,"The file extension type. For the example above this would be \".tif\" (Currently, only .tif or .tiff files are allowed.)");
  write_xml_string_param(inputFile,DICe::image_file_prefix);
  write_xml_comment(inputFile,"The tag at the front of the file names. For the example above this would be \"Image_\"");
  write_xml_comment(inputFile,"");
  write_xml_comment(inputFile,"Another option is to read frames from a cine file. In that case, the following parameters are used");
  write_xml_comment(inputFile,"Note that all of the values for indexing are in terms of the cine file's first frame number (which may be negative due to trigger)");
  write_xml_comment(inputFile,"For example, if the trigger image number is -100, the ref_index would be -100 as well as the start index");
  write_xml_comment(inputFile,"The end index would be the trigger index + the number of frames to analyze, the default is the entire video.");
  write_xml_string_param(inputFile,DICe::cine_file);
  write_xml_size_param(inputFile,DICe::cine_ref_index);
  write_xml_size_param(inputFile,DICe::cine_start_index);
  write_xml_size_param(inputFile,DICe::cine_end_index);

  // write the correlation parameters

  write_xml_comment(paramsFile,"Uncomment lines below that begin with \"<Parameter name=...\"\n to specify that parameter. Otherwise, default values will be used");
  for(int_t i=0;i<DICe::num_valid_correlation_params;++i){
    if(valid_correlation_params[i].expose_to_user_ == false) continue;
    // write comment with all possible options
    if(valid_correlation_params[i].type_==STRING_PARAM){
      write_xml_comment(paramsFile,"");
      write_xml_string_param(paramsFile,valid_correlation_params[i].name_);
      write_xml_comment(paramsFile,valid_correlation_params[i].desc_);
      write_xml_comment(paramsFile,"Possible options include:");
      const char ** possibleValues = valid_correlation_params[i].stringNamePtr_;
      for(int_t j=0;j<valid_correlation_params[i].size_;++j){
        write_xml_comment(paramsFile,possibleValues[j]);
      }
    }
    else if(valid_correlation_params[i].type_==SCALAR_PARAM){
      write_xml_comment(paramsFile,"");
      write_xml_real_param(paramsFile,valid_correlation_params[i].name_);
      write_xml_comment(paramsFile,valid_correlation_params[i].desc_);
    }
    else if(valid_correlation_params[i].type_==SIZE_PARAM){
      write_xml_comment(paramsFile,"");
      write_xml_size_param(paramsFile,valid_correlation_params[i].name_);
      write_xml_comment(paramsFile,valid_correlation_params[i].desc_);
    }
    else if(valid_correlation_params[i].type_==BOOL_PARAM){
      write_xml_comment(paramsFile,"");
      write_xml_bool_param(paramsFile,valid_correlation_params[i].name_);
      write_xml_comment(paramsFile,valid_correlation_params[i].desc_);
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: Unknown parameter type");
    }
  }
  // Write a sample output spec
  write_xml_comment(paramsFile,"");
  write_xml_comment(paramsFile,"");
  write_xml_comment(paramsFile,"Uncomment the ParameterList below to request specific fields in the output files.");
  write_xml_comment(paramsFile,"The order in the list represents the column order in the data output.");
  write_xml_param_list_open(paramsFile,DICe::output_spec);
  for(int_t i=0;i<DICe::field_enums::num_fields_defined;++i){
    std::stringstream iToStr;
    iToStr << i;
    write_xml_bool_param(paramsFile,DICe::field_enums::fs_spec_vec[i].get_name_label(),"true");
  }
  write_xml_param_list_close(paramsFile);

  finalize_xml_file(fileNameInput.str());
  finalize_xml_file(fileNameParams.str());
}


DICE_LIB_DLL_EXPORT
void to_upper(std::string &s){
  std::transform(s.begin(), s.end(),s.begin(), ::toupper);
}

DICE_LIB_DLL_EXPORT
void to_lower(std::string &s){
  std::transform(s.begin(), s.end(),s.begin(), ::tolower);
}

DICE_LIB_DLL_EXPORT
std::istream & safeGetline(std::istream& is, std::string& t)
{
    t.clear();
    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.
    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();
    for(;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if(sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if(t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}

DICE_LIB_DLL_EXPORT
std::vector<std::string> tokenize_line(std::istream &dataFile,
  const std::string & delim,
  const bool capitalize){
//  static int_t MAX_CHARS_PER_LINE = 512;
  static int_t MAX_TOKENS_PER_LINE = 100;

  std::vector<std::string> tokens(MAX_TOKENS_PER_LINE,"");

  // read an entire line into memory
  std::string buf_str;
  safeGetline(dataFile, buf_str);
  char *buf = new char[buf_str.length() + 1];
  strcpy(buf, buf_str.c_str());

  // parse the line into blank-delimited tokens
  int_t n = 0; // a for-loop index

  // parse the line
  char * token;
  token = strtok(buf, delim.c_str());
  tokens[0] = token ? token : ""; // first token
  if(capitalize)
    to_upper(tokens[0]);
  bool first_char_is_pound = tokens[0].find("#") == 0;
  if (tokens[0] != "" && tokens[0]!=parser_comment_char && !first_char_is_pound) // zero if line is blank or starts with a comment char
  {
    for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
    {
      token = strtok(0, delim.c_str());
      tokens[n] = token ? token : ""; // subsequent tokens
      if (tokens[n]=="") break; // no more tokens
      if(capitalize)
        to_upper(tokens[n]); // convert the string to upper case
    }
  }
  tokens.resize(n);
  delete [] buf;

  return tokens;

}

DICE_LIB_DLL_EXPORT
bool is_number(const std::string& s)
{
  std::string::const_iterator it = s.begin();
  while (it != s.end() && (std::isdigit(*it) || *it=='+' || *it=='-' || *it=='e' || *it=='E' || *it=='.'))
    ++it;
  return !s.empty() && it == s.end();
}


DICE_LIB_DLL_EXPORT
const Teuchos::RCP<Subset_File_Info> read_subset_file(const std::string & fileName,
  const int_t width,
  const int_t height){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  // here is where the subset defs are read and parsed
  Teuchos::RCP<Subset_File_Info> info = Teuchos::rcp(new Subset_File_Info());
  // NOTE: assumes 2D coordinates
  const int_t dim = 2;
  std::fstream dataFile(fileName.c_str(), std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(), std::runtime_error, "Error, the subset file does not exist: " << fileName);

  if(proc_rank==0) DEBUG_MSG("Reading the subset file " << fileName);

  bool coordinates_defined = false;
  bool conformal_subset_defined = false;
  bool roi_defined = false;
  int_t num_roi = 0;
  int_t num_motion_windows_defined = 0;

  // read each line of the file
   while (!dataFile.eof())
   {
     std::vector<std::string> tokens = tokenize_line(dataFile);
     if(tokens.size()==0) continue;
     for (size_t i = 0; i < tokens.size(); i++)
       if(proc_rank==0) DEBUG_MSG("Tokens[" << i << "] = " << tokens[i]);
     if(tokens.size()<2){
       std::cout << "Error reading subset file, invalid entry: " << fileName << " "  << std::endl;
       for (size_t i = 0; i < tokens.size(); i++)
         std::cout << "Tokens[" << i << "] = " << tokens[i] << std::endl;
       std::cout << std::endl;
       TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
     }
     if(tokens[0]==parser_begin){ // An input block is being defined
       if(tokens[1]==parser_subset_coordinates){
         if(proc_rank==0) DEBUG_MSG("Reading coordinates from subset file");
         coordinates_defined = true;
         // read more lines until parser end is reached
         while(!dataFile.eof()){
           std::vector<std::string> block_tokens = tokenize_line(dataFile);
           if(block_tokens.size()==0) continue; // blank line or comment
           else if(block_tokens[0]==parser_end) break; // end of the list
           else if(is_number(block_tokens[0])){ // set of coordinates
             if(block_tokens.size()<2){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid coordinate (not enough values)" << fileName);}
             info->coordinates_vector->push_back(strtod(block_tokens[0].c_str(),NULL));
             if(block_tokens[1]==parser_comment_char){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid coordinate (not enough values)" << fileName);}
             info->coordinates_vector->push_back(strtod(block_tokens[1].c_str(),NULL));
             info->neighbor_vector->push_back(-1); // neighbor_id
           }
           else{ // or error
             TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error parsing subset coordinates: " << fileName << " "  << block_tokens[0]);
           }
         } // while loop
         // check for valid coordinates
         TEUCHOS_TEST_FOR_EXCEPTION(info->coordinates_vector->size()<2,std::runtime_error,"");
         TEUCHOS_TEST_FOR_EXCEPTION(info->coordinates_vector->size()/2!=info->neighbor_vector->size(),std::runtime_error,"");
         TEUCHOS_TEST_FOR_EXCEPTION(info->coordinates_vector->size()%2!=0,std::runtime_error,"");
         for(int_t i=0;i<(int_t)info->coordinates_vector->size()/dim;++i){
           if((*info->coordinates_vector)[i*dim]<0||((*info->coordinates_vector)[i*dim]>=width&&width!=-1)){
             TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid subset coordinate in " << fileName << " x: "  << (*info->coordinates_vector)[i*dim]);
           }
           if((*info->coordinates_vector)[i*dim+1]<0||((*info->coordinates_vector)[i*dim+1]>=height&&height!=-1)){
             TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid subset coordinate in " << fileName << " y: "  << (*info->coordinates_vector)[i*dim+1]);
           }
           if(proc_rank==0) DEBUG_MSG("Subset coord: (" << (*info->coordinates_vector)[i*dim] << "," << (*info->coordinates_vector)[i*dim+1] << ")");
         }
       }
       else if(tokens[1]==parser_region_of_interest){
         if(proc_rank==0) DEBUG_MSG("Reading region of interest");
         DICe::multi_shape boundary_multi_shape;
         DICe::multi_shape excluded_multi_shape;
         info->type=REGION_OF_INTEREST_INFO;
         roi_defined = true;
         while(!dataFile.eof()){
           std::vector<std::string> block_tokens = tokenize_line(dataFile);
           if(block_tokens.size()==0) continue; // blank line or comment
           else if(block_tokens[0]==parser_end) break; // end of the defs
           // force the triangulation to be a regular grid rather than delaunay
           else if(block_tokens[0]==parser_use_regular_grid){
             info->use_regular_grid = true;
           }
           else if(block_tokens[0]==parser_enforce_lagrange_bc){
             info->enforce_lagrange_bc = true;
           }
           else if(block_tokens[0]==parser_ic_value_x){
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<=1,std::runtime_error,"");
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[1]),std::runtime_error,"");
             info->ic_value_x = strtod(block_tokens[1].c_str(),NULL);
             DEBUG_MSG("found region of interest IC value x: " << info->ic_value_x);
           }
           else if(block_tokens[0]==parser_ic_value_y){
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<=1,std::runtime_error,"");
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[1]),std::runtime_error,"");
             info->ic_value_y = strtod(block_tokens[1].c_str(),NULL);
             DEBUG_MSG("found region of interest IC value x: " << info->ic_value_y);
           }
           // BOUNDARY CONDITIONS
           else if(block_tokens[0]==parser_dirichlet_bc){
             if(proc_rank==0) DEBUG_MSG("Reading dirichlet boundary condition ");
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<7,std::runtime_error,
               "Error, not enough values specified for dirichlet bc: DIRICHLET_BC BOUNDARY/EXCLUDED SHAPE_ID VERTEX_ID VERTEX_ID COMPONENT <VALUE_X VALUE_Y / USE_SUBSETS SUBSET_SIZE>." );
             std::string region_type;
             if(block_tokens[1]==parser_boundary){
               if(proc_rank==0) DEBUG_MSG("Region: boundary");
               region_type=parser_boundary;
             }
             else if(block_tokens[1]==parser_excluded){
               if(proc_rank==0) DEBUG_MSG("Region: excluded");
               region_type=parser_excluded;
             }
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[2]) || !is_number(block_tokens[3]) || !is_number(block_tokens[4]),std::runtime_error,"");
             Boundary_Condition_Def bc_def;
             bc_def.shape_id_ = strtol(block_tokens[2].c_str(),NULL,0);
             bc_def.left_vertex_id_ = strtol(block_tokens[3].c_str(),NULL,0);
             bc_def.right_vertex_id_ = strtol(block_tokens[4].c_str(),NULL,0);
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[5]),std::runtime_error,"");
             bc_def.comp_ = strtol(block_tokens[5].c_str(),NULL,0);
             if(is_number(block_tokens[6])){
               bc_def.has_value_ = true;
               bc_def.use_subsets_ = false;
               bc_def.value_ = strtod(block_tokens[6].c_str(),NULL);
             }
             else if(block_tokens[6]==parser_use_subsets){
               bc_def.has_value_ = false;
               bc_def.use_subsets_ = true;
               bc_def.subset_size_ = strtol(block_tokens[7].c_str(),NULL,0);
             }
             else{
               TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
             }
             DEBUG_MSG("Shape id: " << bc_def.shape_id_ << " left vertex: " << bc_def.left_vertex_id_ << " right vertex: "<< bc_def.right_vertex_id_
               << " component " << bc_def.comp_ << " has value " << bc_def.has_value_ << " value " << bc_def.value_
               << " use subsets " << bc_def.use_subsets_ << " subset size " << bc_def.subset_size_);
             info->boundary_condition_defs->push_back(bc_def);
           }
           else if(block_tokens[0]==parser_neumann_bc){
             if(proc_rank==0) DEBUG_MSG("Reading neumann boundary condition ");
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<5,std::runtime_error,
               "Error, not enough values specified for Neumann bc: NEUMANN_BC BOUNDARY/EXCLUDED SHAPE_ID VERTEX_ID VERTEX_ID." );
             std::string region_type;
             if(block_tokens[1]==parser_boundary){
               if(proc_rank==0) DEBUG_MSG("Region: boundary");
               region_type=parser_boundary;
             }
             else if(block_tokens[1]==parser_excluded){
               if(proc_rank==0) DEBUG_MSG("Region: excluded");
               region_type=parser_excluded;
             }
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[2]) || !is_number(block_tokens[3]) || !is_number(block_tokens[4]),std::runtime_error,"");
             Boundary_Condition_Def bc_def;
             bc_def.shape_id_ = strtol(block_tokens[2].c_str(),NULL,0);
             bc_def.left_vertex_id_ = strtol(block_tokens[3].c_str(),NULL,0);
             bc_def.right_vertex_id_ = strtol(block_tokens[4].c_str(),NULL,0);
             bc_def.has_value_=false;
             bc_def.use_subsets_=false;
             bc_def.is_neumann_=true;
             DEBUG_MSG("Shape id: " << bc_def.shape_id_ << " left vertex: " << bc_def.left_vertex_id_ << " right vertex: "<< bc_def.right_vertex_id_
               << " has value " << bc_def.has_value_ << " value " << bc_def.value_
               << " use subsets " << bc_def.use_subsets_ << " subset size " << bc_def.subset_size_);
             info->boundary_condition_defs->push_back(bc_def);
           }
           // SHAPES
           else if(block_tokens[0]==parser_begin){
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<2,std::runtime_error,"");
             if(block_tokens[1]==parser_boundary){
               if(proc_rank==0) DEBUG_MSG("Reading boundary def");
               boundary_multi_shape = read_shapes(dataFile);
               if(boundary_multi_shape.size()<=0){
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: cannot define a boundary with zero shapes " << fileName);
               }
             }
             else if(block_tokens[1]==parser_excluded){
               if(proc_rank==0) DEBUG_MSG("Reading excluded def");
               excluded_multi_shape = read_shapes(dataFile);
               if(excluded_multi_shape.size()<=0){
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: cannot define an exclusion with zero shapes " << fileName);
               }
             }
             else if(block_tokens[1]==parser_seed){
               bool has_location = false;
               bool has_disp_values = false;
               int_t seed_loc_x = -1;
               int_t seed_loc_y = -1;
               scalar_t seed_disp_x = 0.0;
               scalar_t seed_disp_y = 0.0;
               scalar_t seed_normal_strain_x = 0.0;
               scalar_t seed_normal_strain_y = 0.0;
               scalar_t seed_shear_strain = 0.0;
               scalar_t seed_rotation = 0.0;
               if(proc_rank==0) DEBUG_MSG("Reading seed information");
               while(!dataFile.eof()){
                 std::vector<std::string> seed_tokens = tokenize_line(dataFile);
                 if(seed_tokens.size()==0) continue; // blank line or comment
                 else if(seed_tokens[0]==parser_end) break; // end of the defs
                 else if(seed_tokens[0]==parser_location){
                   TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=2,std::runtime_error,"Error, not enough values specified for seed location (x and y are needed)." );
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1]) || !is_number(seed_tokens[2]),std::runtime_error,"");
                   seed_loc_x = strtol(seed_tokens[1].c_str(),NULL,0);
                   seed_loc_y = strtol(seed_tokens[2].c_str(),NULL,0);
                   if(proc_rank==0) DEBUG_MSG("Seed location " << seed_loc_x << " " << seed_loc_y);
                   has_location = true;
                   info->size_map->insert(std::pair<int_t,std::pair<int_t,int_t> >(num_roi,std::pair<int_t,int_t>(seed_loc_x,seed_loc_y)));
                 }
                 else if(seed_tokens[0]==parser_displacement){
                   TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=2,std::runtime_error,
                     "Error, not enough values specified for seed displacement (x and y are needed)." );
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1])||!is_number(seed_tokens[2]),
                     std::runtime_error,"");
                   seed_disp_x = strtod(seed_tokens[1].c_str(),NULL);
                   seed_disp_y = strtod(seed_tokens[2].c_str(),NULL);
                   if(proc_rank==0) DEBUG_MSG("Seed displacement " << seed_disp_x << " " << seed_disp_y);
                   has_disp_values = true;
                   info->displacement_map->insert(std::pair<int_t,std::pair<scalar_t,scalar_t> >(num_roi,std::pair<scalar_t,scalar_t>(seed_disp_x,seed_disp_y)));
                 }
                 else if(seed_tokens[0]==parser_normal_strain){
                   TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=2,std::runtime_error,
                     "Error, not enough values specified for seed normal_strain (x and y are needed)." );
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1])||!is_number(seed_tokens[2]),
                     std::runtime_error,"");
                   seed_normal_strain_x = strtod(seed_tokens[1].c_str(),NULL);
                   seed_normal_strain_y = strtod(seed_tokens[2].c_str(),NULL);
                   if(proc_rank==0) DEBUG_MSG("Seed normal strain " << seed_normal_strain_x << " " << seed_normal_strain_y);
                   info->normal_strain_map->insert(std::pair<int_t,std::pair<scalar_t,scalar_t> >(num_roi,std::pair<scalar_t,scalar_t>(seed_normal_strain_x,seed_normal_strain_y)));
                 }
                 else if(seed_tokens[0]==parser_shear_strain){
                   TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=1,std::runtime_error,
                     "Error, not enough values specified for seed shear_strain." );
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1])||!is_number(seed_tokens[2]),
                     std::runtime_error,"");
                   seed_shear_strain = strtod(seed_tokens[1].c_str(),NULL);
                   if(proc_rank==0) DEBUG_MSG("Seed shear strain " << seed_shear_strain);
                   info->shear_strain_map->insert(std::pair<int_t,scalar_t>(num_roi,seed_shear_strain));
                 }
                 else if(seed_tokens[0]==parser_rotation){
                   TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=1,std::runtime_error,
                     "Error, not enough values specified for seed rotation." );
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1]),
                     std::runtime_error,"");
                   seed_rotation = strtod(seed_tokens[1].c_str(),NULL);
                   if(proc_rank==0) DEBUG_MSG("Seed rotation " << seed_rotation);
                   info->rotation_map->insert(std::pair<int_t,scalar_t>(num_roi,seed_rotation));
                 }
                 else{
                   TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized command in seed block " << tokens[0]);
                 }
               }
               TEUCHOS_TEST_FOR_EXCEPTION(!has_location,std::runtime_error,"Error, seed must have location specified");
               TEUCHOS_TEST_FOR_EXCEPTION(!has_disp_values,std::runtime_error,"Error, seed must have displacement guess specified");
             }
             else{
               TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error," Error parsing subset file " << fileName << " "  << block_tokens[1]);
             }
           }
           // ERROR
           else{ // or error
             TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error in parsing conformal subset def, unkown block command: " << block_tokens[0]);
           }
         } // while loop
         DICe::Conformal_Area_Def conformal_area_def(boundary_multi_shape,excluded_multi_shape);
         info->conformal_area_defs->insert(std::pair<int_t,DICe::Conformal_Area_Def>(num_roi,conformal_area_def));
         num_roi++;
       }
       else if(tokens[1]==parser_conformal_subset){
         int_t subset_id = -1;
         bool has_seed = false;
         scalar_t seed_disp_x = 0.0;
         scalar_t seed_disp_y = 0.0;
         scalar_t seed_normal_strain_x = 0.0;
         scalar_t seed_normal_strain_y = 0.0;
         scalar_t seed_shear_strain = 0.0;
         scalar_t seed_rotation = 0.0;
         std::vector<int_t> blocking_ids;
         bool force_simplex = false;
         DICe::multi_shape boundary_multi_shape;
         DICe::multi_shape excluded_multi_shape;
         DICe::multi_shape obstructed_multi_shape;
         if(proc_rank==0) DEBUG_MSG("Reading conformal subset def");
         conformal_subset_defined = true;
         bool has_path_file = false;
         bool skip_solve = false;
         std::vector<int_t> skip_solve_ids;
         bool use_optical_flow = false;
         Motion_Window_Params motion_window_params;
         bool test_for_motion = false;
         bool has_motion_window = false;
         std::string path_file_name;
         while(!dataFile.eof()){
           std::streampos pos = dataFile.tellg();
           std::vector<std::string> block_tokens = tokenize_line(dataFile);
           if(block_tokens.size()==0) continue; // blank line or comment
           else if(block_tokens[0]==parser_end) break; // end of the defs
           // SUBSET ID
           else if(block_tokens[0]==parser_subset_id){
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<2,std::runtime_error,"");
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[1]),std::runtime_error,"");
             subset_id = strtol(block_tokens[1].c_str(),NULL,0);
             if(proc_rank==0) DEBUG_MSG("Conformal subset id: " << subset_id);
           }
           else if(block_tokens[0]==parser_test_for_motion){
             test_for_motion = true;
             motion_window_params.use_motion_detection_ = true;
             DEBUG_MSG("Conformal subset will test for motion");
           }
           else if(block_tokens[0]==parser_motion_window){
             has_motion_window = true;
             if(block_tokens.size()==2){
               TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[1]),std::runtime_error,"");
               motion_window_params.use_subset_id_ = strtol(block_tokens[1].c_str(),NULL,0);
               if(proc_rank==0) DEBUG_MSG("Conformal subset using the motion window defined by subset " <<
                 motion_window_params.use_subset_id_);
             }
             else{
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<5,std::invalid_argument,"TEST_FOR_MOTION requires at least 4 arguments \n"
                 "usage: TEST_FOR_MOTION <origin_x> <origin_y> <width> <heigh> [tol], if tol is not set it will be computed automatically based on the first frame");
             for(int_t m=1;m<5;++m){TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[m]),std::invalid_argument,
               "Error, these parameters should be numbers here.");}
             motion_window_params.start_x_ = strtol(block_tokens[1].c_str(),NULL,0);
             motion_window_params.start_y_ = strtol(block_tokens[2].c_str(),NULL,0);
             motion_window_params.end_x_ = strtol(block_tokens[3].c_str(),NULL,0);
             motion_window_params.end_y_ = strtol(block_tokens[4].c_str(),NULL,0);
             motion_window_params.sub_image_id_ = num_motion_windows_defined++;
             if(block_tokens.size()>5)
               motion_window_params.tol_ = strtod(block_tokens[5].c_str(),NULL);
             if(proc_rank==0) DEBUG_MSG("Conformal subset motion window"
                 " start x: " << motion_window_params.start_x_ << " start y: " << motion_window_params.start_y_ <<
                 " end x: " << motion_window_params.end_x_ << " end y: " << motion_window_params.end_y_ <<
                 " tolerance: " << motion_window_params.tol_ << " sub_image_id: " << motion_window_params.sub_image_id_);
             }
           }
           else if(block_tokens[0]==parser_skip_solve){
             DEBUG_MSG("Reading skip solve");
             skip_solve = true;
             // need to re-read the line again without converting to capital case
             // see if the second argument is a string file_name, if so read the file
             if(block_tokens.size()>1){
               if(!is_number(block_tokens[1])){
                 dataFile.seekg(pos,std::ios::beg);
                 std::vector<std::string> block_tokens = tokenize_line(dataFile," ",false);
                 std::string skip_solves_file_name = block_tokens[1];
                 if(proc_rank==0) DEBUG_MSG("Skip solves file name: " << skip_solves_file_name);
                 std::fstream skip_file(skip_solves_file_name.c_str(), std::ios_base::in);
                 TEUCHOS_TEST_FOR_EXCEPTION(!skip_file.good(),std::runtime_error,"Error invalid skip file");
                 int_t id = 0;
                 // TODO add some error checking for one value per line and ensure integer values
                 while (skip_file >> id)
                   skip_solve_ids.push_back(id);
               }
               else{
                 // all other numbers are ids to turn solving on or off
                 for(size_t id=1;id<block_tokens.size();++id){
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[id]),std::runtime_error,"");
                   DEBUG_MSG("Skip solve id : " << block_tokens[id]);
                   skip_solve_ids.push_back(strtol(block_tokens[id].c_str(),NULL,0));
                 }
               }
             }
             // sort the vector
             std::sort(skip_solve_ids.begin(),skip_solve_ids.end());
           }
           else if(block_tokens[0]==parser_use_optical_flow){
             use_optical_flow = true;
           }
           else if(block_tokens[0]==parser_force_simplex){
             DEBUG_MSG("Forcing simplex method for this subset");
             force_simplex = true;
           }
           else if(block_tokens[0]==parser_use_path_file){
             // need to re-read the line again without converting to capital case
             dataFile.seekg(pos, std::ios::beg);
             std::vector<std::string> block_tokens = tokenize_line(dataFile," ",false);
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<2,std::runtime_error,"");
             TEUCHOS_TEST_FOR_EXCEPTION(is_number(block_tokens[1]),std::runtime_error,"");
             path_file_name = block_tokens[1];
             if(proc_rank==0) DEBUG_MSG("Path file name: " << path_file_name);
             has_path_file = true;
           }
           // SEEDS
           else if(block_tokens[1]==parser_seed){
             TEUCHOS_TEST_FOR_EXCEPTION(has_seed,std::runtime_error,
               "Error, only one seed allowed per conformal subset");
             bool has_disp_values = false;
             has_seed = true;
             if(proc_rank==0) DEBUG_MSG("Reading seed information for conformal subset");
             while(!dataFile.eof()){
               std::vector<std::string> seed_tokens = tokenize_line(dataFile);
               if(seed_tokens.size()==0) continue; // blank line or comment
               else if(seed_tokens[0]==parser_end) break; // end of the defs
               else if(seed_tokens[0]==parser_location){
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, location cannot be specified for a seed in a conformal subset, "
                     "the location is the subset centroid automatically");
               }
               else if(seed_tokens[0]==parser_displacement){
                 TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=2,std::runtime_error,
                   "Error, not enough values specified for seed displacement (x and y are needed)." );
                 TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1])||!is_number(seed_tokens[2]),
                   std::runtime_error,"");
                 seed_disp_x = strtod(seed_tokens[1].c_str(),NULL);
                 seed_disp_y = strtod(seed_tokens[2].c_str(),NULL);
                 if(proc_rank==0) DEBUG_MSG("Seed displacement " << seed_disp_x << " " << seed_disp_y);
                 has_disp_values = true;
               }
               else if(seed_tokens[0]==parser_normal_strain){
                 TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=2,std::runtime_error,
                   "Error, not enough values specified for seed normal strain (x and y are needed)." );
                 TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1])||!is_number(seed_tokens[2]),
                   std::runtime_error,"");
                 seed_normal_strain_x = strtod(seed_tokens[1].c_str(),NULL);
                 seed_normal_strain_y = strtod(seed_tokens[2].c_str(),NULL);
                 if(proc_rank==0) DEBUG_MSG("Seed Normal Strain " << seed_normal_strain_x << " " << seed_normal_strain_y);
               }
               else if(seed_tokens[0]==parser_shear_strain){
                 TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=1,std::runtime_error,
                   "Error, a value must be specified for shear strain." );
                 TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1])||!is_number(seed_tokens[2]),std::runtime_error,"");
                 seed_shear_strain = strtod(seed_tokens[1].c_str(),NULL);
                 if(proc_rank==0) DEBUG_MSG("Seed Shear Strain " << seed_shear_strain);
               }
               else if(seed_tokens[0]==parser_rotation){
                 TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=1,std::runtime_error,
                   "Error, a value must be specified for rotation." );
                 TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1]),
                   std::runtime_error,"");
                 seed_rotation = strtod(seed_tokens[1].c_str(),NULL);
                 if(proc_rank==0) DEBUG_MSG("Seed Rotation " << seed_rotation);
               }
               else{
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized command in seed block " << tokens[0]);
               }
             }
             TEUCHOS_TEST_FOR_EXCEPTION(!has_disp_values,std::runtime_error,"Error, seed must have at least a displacement guess specified");
           }
           // SHAPES
           else if(block_tokens[0]==parser_begin){
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<2,std::runtime_error,"");
             if(block_tokens[1]==parser_boundary){
               if(proc_rank==0) DEBUG_MSG("Reading boundary def");
               boundary_multi_shape = read_shapes(dataFile);
               if(boundary_multi_shape.size()<=0){
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: cannot define a boundary with zero shapes " << fileName);
               }
             }
             else if(block_tokens[1]==parser_excluded){
               if(proc_rank==0) DEBUG_MSG("Reading excluded def");
               excluded_multi_shape = read_shapes(dataFile);
               if(excluded_multi_shape.size()<=0){
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: cannot define an exclusion with zero shapes " << fileName);
               }
             }
             else if(block_tokens[1]==parser_obstructed){
               if(proc_rank==0) DEBUG_MSG("Reading obstructed def");
               obstructed_multi_shape = read_shapes(dataFile);
               if(obstructed_multi_shape.size()<=0){
                 TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: cannot define an obstruction with zero shapes " << fileName);
               }
             }
             else if(block_tokens[1]==parser_blocking_subsets){
               if(proc_rank==0) DEBUG_MSG("Reading blocking subsets");
               while(!dataFile.eof()){
                 std::vector<std::string> id_tokens = tokenize_line(dataFile);
                 if(id_tokens.size()==0) continue;
                 if(id_tokens[0]==parser_end) break;
                 TEUCHOS_TEST_FOR_EXCEPTION(!is_number(id_tokens[0]),std::runtime_error,"");
                 blocking_ids.push_back(strtol(id_tokens[0].c_str(),NULL,0));
               }
               for(size_t i=0;i<blocking_ids.size();++i)
                 if(proc_rank==0) DEBUG_MSG("Subset is blocked by " << blocking_ids[i]);
             }
             else{
               TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error," Error parsing subset file " << fileName << " "  << block_tokens[1]);
             }
           }
           // ERROR
           else{ // or error
             TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error in parsing conformal subset def, unkown block command: " << block_tokens[0]);
           }
         } // while loop
         // put the multishapes into a subset def:
         if(proc_rank==0) DEBUG_MSG("Adding " << boundary_multi_shape.size() << " shape(s) to the boundary def ");
         if(proc_rank==0) DEBUG_MSG("Adding " << excluded_multi_shape.size() << " shape(s) to the exclusion def ");
         if(proc_rank==0) DEBUG_MSG("Adding " << obstructed_multi_shape.size() << " shape(s) to the obstruction def ");
         DICe::Conformal_Area_Def conformal_area_def(boundary_multi_shape,excluded_multi_shape,obstructed_multi_shape);
         if(subset_id<0){
           std::cout << "Error, invalid subset id for conformal subset (or subset id was not defined with \"SUBSET_ID <id>\"): " << subset_id << std::endl;
         }
         info->conformal_area_defs->insert(std::pair<int_t,DICe::Conformal_Area_Def>(subset_id,conformal_area_def));
         info->id_sets_map->insert(std::pair<int_t,std::vector<int_t> >(subset_id,blocking_ids));
         if(force_simplex)
           info->force_simplex->insert(subset_id);
         if(has_seed){
           info->seed_subset_ids->insert(std::pair<int_t,int_t>(subset_id,subset_id)); // treating each conformal subset as an roi TODO fix this awkwardness
           info->displacement_map->insert(std::pair<int_t,std::pair<scalar_t,scalar_t> >(subset_id,std::pair<scalar_t,scalar_t>(seed_disp_x,seed_disp_y)));
           info->normal_strain_map->insert(std::pair<int_t,std::pair<scalar_t,scalar_t> >(subset_id,std::pair<scalar_t,scalar_t>(seed_normal_strain_x,seed_normal_strain_y)));
           info->shear_strain_map->insert(std::pair<int_t,scalar_t>(subset_id,seed_shear_strain));
           info->rotation_map->insert(std::pair<int_t,scalar_t>(subset_id,seed_rotation));
         }
         if(has_path_file)
           info->path_file_names->insert(std::pair<int_t,std::string>(subset_id,path_file_name));
         if(use_optical_flow)
           info->optical_flow_flags->insert(std::pair<int_t,bool>(subset_id,true));
         if(skip_solve)
           info->skip_solve_flags->insert(std::pair<int_t,std::vector<int> >(subset_id,skip_solve_ids));
         if(test_for_motion){ // make sure if motion detection is on, there is a corresponding motion window
           TEUCHOS_TEST_FOR_EXCEPTION(!has_motion_window,std::runtime_error,"Error, cannot test for motion without defining a motion window for subset " << subset_id);
         }
         if(has_motion_window)
           info->motion_window_params->insert(std::pair<int_t,Motion_Window_Params>(subset_id,motion_window_params));
       }  // end conformal subset def
       else{
         std::cout << "Error: Unkown block command in " << fileName << std::endl;
         std::cout << "       " << tokens[1] << std::endl;
         TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
       }
     }
   }
  dataFile.close();
  if(roi_defined){
    TEUCHOS_TEST_FOR_EXCEPTION(coordinates_defined || conformal_subset_defined,std::runtime_error,"Error, if a region of interest in defined, the coordinates"
        " cannot be specified, nor can conformal subset definitions.");
  }

  // now that all the motion windows have been defined, set the sub_image_id for those that use another subsets motion window
  if(info->motion_window_params->size()>0){
    info->num_motion_windows = num_motion_windows_defined;
    std::map<int_t,Motion_Window_Params>::iterator map_it=info->motion_window_params->begin();
    for(;map_it!=info->motion_window_params->end();++map_it){
      if(map_it->second.use_subset_id_!=-1){
        TEUCHOS_TEST_FOR_EXCEPTION(info->motion_window_params->find(map_it->second.use_subset_id_)==info->motion_window_params->end(),
          std::runtime_error,"Error, invalid subset id for motion window use_subset_id");
        map_it->second.sub_image_id_ = info->motion_window_params->find(map_it->second.use_subset_id_)->second.sub_image_id_;
      }
    }
    // check that all subsets have a motion window defined:
    const int_t num_subsets = info->coordinates_vector->size()/dim;
    for(int_t i=0;i<num_subsets;++i){
      TEUCHOS_TEST_FOR_EXCEPTION(info->motion_window_params->find(i)==info->motion_window_params->end(),std::runtime_error,
        "Error, if one motion window is defined, motion windows must be defined for all subsets");
      DEBUG_MSG("Subset " << i << " points to sub image id " << info->motion_window_params->find(i)->second.sub_image_id_);
      TEUCHOS_TEST_FOR_EXCEPTION(info->motion_window_params->find(i)->second.sub_image_id_>= num_motion_windows_defined,std::runtime_error,
        "Error, invalid sub image id");
    }
  }

  return info;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Circle> read_circle(std::fstream &dataFile){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Reading a circle");
  int_t cx = -1;
  int_t cy = -1;
  scalar_t radius = -1.0;
  while(!dataFile.eof()){
    std::vector<std::string> tokens = tokenize_line(dataFile);
    if(tokens.size()==0) continue; // comment or blank line
    if(tokens[0]==parser_end) break;
    else if(tokens[0]==parser_center){
      if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define circle center point ");}
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
        "Error, both tokens should be a number");
      cx = strtol(tokens[1].c_str(),NULL,0);
      cy = strtol(tokens[2].c_str(),NULL,0);
    }
    else if(tokens[0]==parser_radius){
      if(tokens.size()<2){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define circle radius ");}
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]),std::runtime_error,"Error, token should be a number");
      radius = strtod(tokens[1].c_str(),NULL);
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid token in circle definition " << tokens[0]);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(radius==-1.0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cx==-1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cy==-1,std::runtime_error,"");
  if(proc_rank==0) DEBUG_MSG("Creating a circle with center " << cx << " " << cy << " and radius " << radius);
  Teuchos::RCP<DICe::Circle> shape = Teuchos::rcp(new DICe::Circle(cx,cy,radius));
  return shape;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::Rectangle> read_rectangle(std::fstream &dataFile){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Reading a rectangle");
  int_t cx = -1;
  int_t cy = -1;
  int_t width = -1.0;
  int_t height = -1.0;
  int_t upper_left_x = -1;
  int_t upper_left_y = -1;
  int_t lower_right_x = -1;
  int_t lower_right_y = -1;
  bool has_upper_left_lower_right = false;
  while(!dataFile.eof()){
    std::vector<std::string> tokens = tokenize_line(dataFile);
    if(tokens.size()==0) continue; // comment or blank line
    if(tokens[0]==parser_end) break;
    if(tokens[0]==parser_center||tokens[0]==parser_width||tokens[0]==parser_height){
      if(tokens[0]==parser_center){
        if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define circle center point ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
          "Error, both tokens should be a number");
        cx = strtol(tokens[1].c_str(),NULL,0);
        cy = strtol(tokens[2].c_str(),NULL,0);
      }
      else if(tokens[0]==parser_width){
        if(tokens.size()<2){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define the width ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]),std::runtime_error,"Error, token should be a number");
        width = strtod(tokens[1].c_str(),NULL);
      }
      else if(tokens[0]==parser_height){
        if(tokens.size()<2){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define the height ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]),std::runtime_error,"Error, token should be a number");
        height = strtod(tokens[1].c_str(),NULL);
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid token in rectangle definition by center, width, height " << tokens[0]);
      }
    }
    else if(tokens[0]==parser_upper_left||tokens[0]==parser_lower_right){
      if(tokens[0]==parser_upper_left){
        if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define rectangle upper left ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
          "Error, both tokens should be numbers");
        has_upper_left_lower_right = true;
        upper_left_x = strtol(tokens[1].c_str(),NULL,0);
        upper_left_y = strtol(tokens[2].c_str(),NULL,0);
      }
      else if(tokens[0]==parser_lower_right){
        if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define rectangle lower right ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
          "Error, both tokens should be a number");
        has_upper_left_lower_right = true;
        lower_right_x = strtol(tokens[1].c_str(),NULL,0);
        lower_right_y = strtol(tokens[2].c_str(),NULL,0);
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid token in rectangle definition by upper left, lower right " << tokens[0]);
      }
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid token in rectangle definition " << tokens[0]);
    }
  }
  if(has_upper_left_lower_right){
    if(proc_rank==0) DEBUG_MSG("Rectangle has upper left_x " << upper_left_x << " upper_left_y " << upper_left_y <<
      " lower_right_x " << lower_right_x << " lower_right_y " << lower_right_y);
    TEUCHOS_TEST_FOR_EXCEPTION(upper_left_x <0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(upper_left_y <0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(lower_right_x <0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(lower_right_y <0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(lower_right_x < upper_left_x,std::runtime_error,"Error: Rectangle inverted or zero width");
    TEUCHOS_TEST_FOR_EXCEPTION(lower_right_y < upper_left_y,std::runtime_error,"Error: Rectangle inverted or zero height");
    width = lower_right_x - upper_left_x;
    height = lower_right_y - upper_left_y;
    cx = width/2 + upper_left_x;
    cy = height/2 + upper_left_y;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(width<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(height<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cx<0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cy<0,std::runtime_error,"");
  if(proc_rank==0) DEBUG_MSG("Creating a rectangle with center " << cx << " " << cy << " width " << width << " height " << height);
  Teuchos::RCP<DICe::Rectangle> shape = Teuchos::rcp(new DICe::Rectangle(cx,cy,width,height));
  return shape;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::Polygon> read_polygon(std::fstream &dataFile){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Reading a Polygon");
  std::vector<int_t> vertices_x;
  std::vector<int_t> vertices_y;
  while(!dataFile.eof()){
    std::vector<std::string> tokens = tokenize_line(dataFile);
    if(tokens.size()==0) continue; // comment or blank line
    if(tokens[0]==parser_end) break;
    TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()<2,std::runtime_error,"");
    // only other valid option is BEGIN VERTICES
    TEUCHOS_TEST_FOR_EXCEPTION(tokens[0]!=parser_begin,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(tokens[1]!=parser_vertices,std::runtime_error,"");
    // read the vertices
    while(!dataFile.eof()){
      std::vector<std::string> vertex_tokens = tokenize_line(dataFile);
      if(vertex_tokens.size()==0)continue;
      if(vertex_tokens[0]==parser_end) break;
      TEUCHOS_TEST_FOR_EXCEPTION(vertex_tokens.size()<2,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(vertex_tokens[0]),std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(vertex_tokens[1]),std::runtime_error,"");
      vertices_x.push_back(strtol(vertex_tokens[0].c_str(),NULL,0));
      vertices_y.push_back(strtol(vertex_tokens[1].c_str(),NULL,0));
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(vertices_x.empty(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(vertices_y.empty(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(vertices_x.size()!=vertices_y.size(),std::runtime_error,"");
  if(proc_rank==0) DEBUG_MSG("Creating a polygon with " << vertices_x.size() << " vertices");
  for(size_t i=0;i<vertices_x.size();++i){
    if(proc_rank==0) DEBUG_MSG("vx " << vertices_x[i] << " vy " << vertices_y[i] );
  }
  Teuchos::RCP<DICe::Polygon> shape = Teuchos::rcp(new DICe::Polygon(vertices_x,vertices_y));
  return shape;
}

DICE_LIB_DLL_EXPORT
multi_shape read_shapes(std::fstream & dataFile){
  DICe::multi_shape multi_shape;
  while(!dataFile.eof()){
    std::vector<std::string> shape_tokens = tokenize_line(dataFile);
    if(shape_tokens.size()==0)continue;
    if(shape_tokens[0]==parser_end) break;
    TEUCHOS_TEST_FOR_EXCEPTION(shape_tokens.size()<2,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(shape_tokens[0]!=parser_begin,std::runtime_error,"");
    if(shape_tokens[1]==parser_circle){
      Teuchos::RCP<DICe::Circle> shape = read_circle(dataFile);
      multi_shape.push_back(shape);
    }
    else if(shape_tokens[1]==parser_polygon){
      Teuchos::RCP<DICe::Polygon> shape = read_polygon(dataFile);
      multi_shape.push_back(shape);
    }
    else if(shape_tokens[1]==parser_rectangle){
      Teuchos::RCP<DICe::Rectangle> shape = read_rectangle(dataFile);
      multi_shape.push_back(shape);
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized shape : " << shape_tokens[1]);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(multi_shape.size()<=0,std::runtime_error,"");
  return multi_shape;
}


}// End DICe Namespace
