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

#include <DICe_Parser.h>
#include <DICe_XMLUtils.h>
#include <DICe_ParameterUtilities.h>
#include <DICe.h>

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#ifndef DICE_DISABLE_BOOST_FILESYSTEM
  #include <boost/filesystem.hpp>
#endif
  #include <boost/program_options.hpp>
namespace po = boost::program_options;

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

  // Declare the supported options.
  po::options_description desc("Allowed options");

  desc.add_options()("help,h", "produce help message")
                       ("verbose,v","Output log to screen")
                       ("version","Output version information to screen")
                       ("timing,t","Print timing statistics to screen")
                       ("input,i",po::value<std::string>(),"XML input file name <filename>.xml")
                       ("generate,g",po::value<std::string>()->implicit_value("dice"),"Create XML input file templates")
                       ("stats,s","Print field statistics to screen")
                       ;

  // Parse the command line options
  po::variables_map vm;
  // TODO add graceful exception catching for unrecognized options
  // currently the executable crashes with a malloc error that is a little
  // misleading
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  Teuchos::RCP<std::ostream> bhs = Teuchos::rcp(new Teuchos::oblackholestream); // outputs nothing
  outStream = Teuchos::rcp(&std::cout, false);
  // Print info to screen
  if(!vm.count("verbose") || proc_rank!=0){
    outStream = bhs;
  }

  // Handle version requests
  if(vm.count("version")){
    if(proc_rank==0)
      print_banner();
    force_exit = true;
    return inputParams;
  }

  // Handle help requests
  if(vm.count("help")){
    print_banner();
    std::cout << desc << std::endl;
    force_exit = true;
    return inputParams;
  }

  // Generate input file templates and exit
  if(vm.count("generate")){
    std::string templatePrefix = vm["generate"].as<std::string>();
    if(proc_rank==0) DEBUG_MSG("Generating input file templates using prefix: " << templatePrefix);
    generate_template_input_files(templatePrefix);
    force_exit = true;
    return inputParams;
  }

  std::string input_file;
  if(vm.count("input")){
    input_file = vm["input"].as<std::string>();
  }
  else{
    std::cout << "Error: The XML input file must be specified on the command line with -i <filename>.xml" << std::endl;
    std::cout << "       (To generate template input files, specify -g [file_prefix] on the command line)" << std::endl;
    exit(1);
  }
  if(proc_rank==0) DEBUG_MSG("Using input file: " << input_file);

  Teuchos::Ptr<Teuchos::ParameterList> inputParamsPtr(inputParams.get());
  Teuchos::updateParametersFromXmlFile(input_file, inputParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(inputParams==Teuchos::null,std::runtime_error,"");

  // Print timing statistics?
  if(vm.count("timing")){
    inputParams->set(DICe::print_timing,true);
  }

  // Print timing statistics?
  if(vm.count("stats")){
    inputParams->set(DICe::print_stats,true);
  }

  Analysis_Type analysis_type = LOCAL_DIC;
  if(inputParams->isParameter(DICe::mesh_size)){
    DEBUG_MSG("Using GLOBAL DIC formulation since mesh_size parameter was specified.");
    analysis_type = GLOBAL_DIC;
  }

  // Test the input values
  std::vector<std::pair<std::string,std::string> > required_params;
  required_params.push_back(std::pair<std::string,std::string>(DICe::image_folder,"string"));
  if(analysis_type==GLOBAL_DIC){
    required_params.push_back(std::pair<std::string,std::string>(DICe::output_folder,"string"));
    required_params.push_back(std::pair<std::string,std::string>(DICe::mesh_size,"double"));
    //required_params.push_back(std::pair<std::string,std::string>(DICe::image_edge_buffer_size,"int"));
  }
  if(analysis_type==LOCAL_DIC){
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
  else{ // GLOBAL DIC
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
    if(!inputParams->isParameter(DICe::subset_file)){
      std::cout << "Error: The parameter " << DICe::subset_file << " of type string must be defined  for global DIC in " << input_file << std::endl;
      required_param_missing = true;
    }
  }
  if(!inputParams->isParameter(DICe::reference_image_index)&&!inputParams->isParameter(DICe::reference_image)&&!inputParams->isParameter(DICe::cine_file)){
    std::cout << "Error: Either the parameter " << DICe:: reference_image_index << " or " <<
        DICe::reference_image << " or " << DICe::cine_file << " needs to be specified in " << input_file << std::endl;
    required_param_missing = true;
  }
  // specifying a simple two image correlation
  if(inputParams->isParameter(DICe::reference_image)){
    if(inputParams->isParameter(DICe::last_image_index)){
      std::cout << "Error: The parameter " << DICe::last_image_index <<
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
    if(!inputParams->isParameter(DICe::last_image_index)){
      std::cout << "Error: The parameter " << DICe::last_image_index << " of type int must be defined in " << input_file << std::endl;
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

  // create the output folder if it does not exist
#ifdef DICE_DISABLE_BOOST_FILESYSTEM
  std::cout << "** WARNING: Boost filesystem has been disabled so files will be output to current execution directory " << std::endl;
#else
  std::string output_folder = inputParams->get<std::string>(DICe::output_folder);
  if(proc_rank==0) DEBUG_MSG("Attempting to create directory : " << output_folder);
  boost::filesystem::path dir(output_folder);
  if(boost::filesystem::create_directory(dir)) {
    if(proc_rank==0) DEBUG_MSG("Directory " << output_folder << " was successfully created");
  }
#endif
  return inputParams;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Teuchos::ParameterList> read_physics_params(const std::string & paramFileName){

  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Parsing physics parameters from file: " << paramFileName);
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
    if(proc_rank==0) DEBUG_MSG("Found user specified physics parameter: " << paramName);
    stringToLower(paramName); // string parameter names are lower case
    diceParams->setEntry(it->first,it->second);
  }
  return diceParams;
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
    stringToLower(paramName); // string parameter names are lower case
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
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile);
    if(tokens.size()==0) continue; // comment or blank line
    if(tokens[0]==parser_end) break;
    else if(tokens[0]==parser_center){
      if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define circle center point ");}
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
        "Error, both tokens should be a number");
      cx = atoi(tokens[1].c_str());
      cy = atoi(tokens[2].c_str());
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
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile);
    if(tokens.size()==0) continue; // comment or blank line
    if(tokens[0]==parser_end) break;
    if(tokens[0]==parser_center||tokens[0]==parser_width||tokens[0]==parser_height){
      if(tokens[0]==parser_center){
        if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define circle center point ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
          "Error, both tokens should be a number");
        cx = atoi(tokens[1].c_str());
        cy = atoi(tokens[2].c_str());
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
        upper_left_x = atoi(tokens[1].c_str());
        upper_left_y = atoi(tokens[2].c_str());
      }
      else if(tokens[0]==parser_lower_right){
        if(tokens.size()<3){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, not enough values to define rectangle lower right ");}
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[1]) || !is_number(tokens[2]),std::runtime_error,
          "Error, both tokens should be a number");
        has_upper_left_lower_right = true;
        lower_right_x = atoi(tokens[1].c_str());
        lower_right_y = atoi(tokens[2].c_str());
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
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile);
    if(tokens.size()==0) continue; // comment or blank line
    if(tokens[0]==parser_end) break;
    TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()<2,std::runtime_error,"");
    // only other valid option is BEGIN VERTICES
    TEUCHOS_TEST_FOR_EXCEPTION(tokens[0]!=parser_begin,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(tokens[1]!=parser_vertices,std::runtime_error,"");
    // read the vertices
    while(!dataFile.eof()){
      Teuchos::ArrayRCP<std::string> vertex_tokens = tokenize_line(dataFile);
      if(vertex_tokens.size()==0)continue;
      if(vertex_tokens[0]==parser_end) break;
      TEUCHOS_TEST_FOR_EXCEPTION(vertex_tokens.size()<2,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(vertex_tokens[0]),std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(vertex_tokens[1]),std::runtime_error,"");
      vertices_x.push_back(atoi(vertex_tokens[0].c_str()));
      vertices_y.push_back(atoi(vertex_tokens[1].c_str()));
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
    Teuchos::ArrayRCP<std::string> shape_tokens = tokenize_line(dataFile);
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


DICE_LIB_DLL_EXPORT
Teuchos::ArrayRCP<std::string> tokenize_line(std::fstream &dataFile,
  const std::string & delim,
  const bool capitalize){
  static int_t MAX_CHARS_PER_LINE = 512;
  static int_t MAX_TOKENS_PER_LINE = 20;

  Teuchos::ArrayRCP<std::string> tokens(MAX_TOKENS_PER_LINE,"");

  // read an entire line into memory
  char * buf = new char[MAX_CHARS_PER_LINE];
  dataFile.getline(buf, MAX_CHARS_PER_LINE);

  // parse the line into blank-delimited tokens
  int_t n = 0; // a for-loop index

  // parse the line
  char * token;
  token = strtok(buf, delim.c_str());
  tokens[0] = token ? token : ""; // first token
  if(capitalize)
    stringToUpper(tokens[0]);
  bool first_char_is_pound = tokens[0].find("#") == 0;
  if (tokens[0] != "" && tokens[0]!=parser_comment_char && !first_char_is_pound) // zero if line is blank or starts with a comment char
  {
    for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
    {
      token = strtok(0, delim.c_str());
      tokens[n] = token ? token : ""; // subsequent tokens
      if (tokens[n]=="") break; // no more tokens
      if(capitalize)
        stringToUpper(tokens[n]); // convert the string to upper case
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
     Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile);
     if(tokens.size()==0) continue;
     for (int i = 0; i < tokens.size(); i++)
       if(proc_rank==0) DEBUG_MSG("Tokens[" << i << "] = " << tokens[i]);
     if(tokens.size()<2){
       std::cout << "Error reading subset file, invalid entry: " << fileName << " "  << std::endl;
       for (int i = 0; i < tokens.size(); i++)
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
           Teuchos::ArrayRCP<std::string> block_tokens = tokenize_line(dataFile);
           if(block_tokens.size()==0) continue; // blank line or comment
           else if(block_tokens[0]==parser_end) break; // end of the list
           else if(is_number(block_tokens[0])){ // set of coordinates
             if(block_tokens.size()<2){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid coordinate (not enough values)" << fileName);}
             info->coordinates_vector->push_back(std::atoi(block_tokens[0].c_str()));
             if(block_tokens[1]==parser_comment_char){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid coordinate (not enough values)" << fileName);}
             info->coordinates_vector->push_back(std::atoi(block_tokens[1].c_str()));
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
           Teuchos::ArrayRCP<std::string> block_tokens = tokenize_line(dataFile);
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
             bc_def.shape_id_ = atoi(block_tokens[2].c_str());
             bc_def.left_vertex_id_ = atoi(block_tokens[3].c_str());
             bc_def.right_vertex_id_ = atoi(block_tokens[4].c_str());
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[5]),std::runtime_error,"");
             bc_def.comp_ = atoi(block_tokens[5].c_str());
             if(is_number(block_tokens[6])){
               bc_def.has_value_ = true;
               bc_def.use_subsets_ = false;
               bc_def.value_ = strtod(block_tokens[6].c_str(),NULL);
             }
             else if(block_tokens[6]==parser_use_subsets){
               bc_def.has_value_ = false;
               bc_def.use_subsets_ = true;
               bc_def.subset_size_ = atoi(block_tokens[7].c_str());
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
             bc_def.shape_id_ = atoi(block_tokens[2].c_str());
             bc_def.left_vertex_id_ = atoi(block_tokens[3].c_str());
             bc_def.right_vertex_id_ = atoi(block_tokens[4].c_str());
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
                 Teuchos::ArrayRCP<std::string> seed_tokens = tokenize_line(dataFile);
                 if(seed_tokens.size()==0) continue; // blank line or comment
                 else if(seed_tokens[0]==parser_end) break; // end of the defs
                 else if(seed_tokens[0]==parser_location){
                   TEUCHOS_TEST_FOR_EXCEPTION(seed_tokens.size()<=2,std::runtime_error,"Error, not enough values specified for seed location (x and y are needed)." );
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(seed_tokens[1]) || !is_number(seed_tokens[2]),std::runtime_error,"");
                   seed_loc_x = atoi(seed_tokens[1].c_str());
                   seed_loc_y = atoi(seed_tokens[2].c_str());
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
           Teuchos::ArrayRCP<std::string> block_tokens = tokenize_line(dataFile);
           if(block_tokens.size()==0) continue; // blank line or comment
           else if(block_tokens[0]==parser_end) break; // end of the defs
           // SUBSET ID
           else if(block_tokens[0]==parser_subset_id){
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<2,std::runtime_error,"");
             TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[1]),std::runtime_error,"");
             subset_id = atoi(block_tokens[1].c_str());
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
               motion_window_params.use_subset_id_ = atoi(block_tokens[1].c_str());
               if(proc_rank==0) DEBUG_MSG("Conformal subset using the motion window defined by subset " <<
                 motion_window_params.use_subset_id_);
             }
             else{
             TEUCHOS_TEST_FOR_EXCEPTION(block_tokens.size()<5,std::invalid_argument,"TEST_FOR_MOTION requires at least 4 arguments \n"
                 "usage: TEST_FOR_MOTION <origin_x> <origin_y> <width> <heigh> [tol], if tol is not set it will be computed automatically based on the first frame");
             for(int_t m=1;m<5;++m){TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[m]),std::invalid_argument,
               "Error, these parameters should be numbers here.");}
             motion_window_params.start_x_ = atoi(block_tokens[1].c_str());
             motion_window_params.start_y_ = atoi(block_tokens[2].c_str());
             motion_window_params.end_x_ = atoi(block_tokens[3].c_str());
             motion_window_params.end_y_ = atoi(block_tokens[4].c_str());
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
                 Teuchos::ArrayRCP<std::string> block_tokens = tokenize_line(dataFile," ",false);
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
                 for(int_t id=1;id<block_tokens.size();++id){
                   TEUCHOS_TEST_FOR_EXCEPTION(!is_number(block_tokens[id]),std::runtime_error,"");
                   DEBUG_MSG("Skip solve id : " << block_tokens[id]);
                   skip_solve_ids.push_back(atoi(block_tokens[id].c_str()));
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
             Teuchos::ArrayRCP<std::string> block_tokens = tokenize_line(dataFile," ",false);
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
               Teuchos::ArrayRCP<std::string> seed_tokens = tokenize_line(dataFile);
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
                 Teuchos::ArrayRCP<std::string> id_tokens = tokenize_line(dataFile);
                 if(id_tokens.size()==0) continue;
                 if(id_tokens[0]==parser_end) break;
                 TEUCHOS_TEST_FOR_EXCEPTION(!is_number(id_tokens[0]),std::runtime_error,"");
                 blocking_ids.push_back(atoi(id_tokens[0].c_str()));
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
           info->skip_solve_flags->insert(std::pair<int_t,std::vector<int>>(subset_id,skip_solve_ids));
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
void decipher_image_file_names(Teuchos::RCP<Teuchos::ParameterList> params,
  std::vector<std::string> & image_files,
  std::vector<std::string> & stereo_image_files){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

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
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::last_image_index),std::runtime_error,
      "Error, cannot specify last_image_index and reference_image");
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
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::last_image_index),std::runtime_error,
      "Error, cannot specify last_image_index and cine_file");
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
    // flag that this analysis is a cine file
    // the first two strings will be "cine_file"
    image_files.push_back(DICe::cine_file);
    image_files.push_back(DICe::cine_file);
    if(params->isParameter(DICe::stereo_cine_file)){
      stereo_image_files.push_back(DICe::cine_file); // push back cine_file not stereo_cine_file because this is the flag that main uses to denote cine input
      stereo_image_files.push_back(DICe::cine_file);
    }
  }
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
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::last_image_index),std::runtime_error,
      "Error, the last image index was not specified");
    const int_t lastImageIndex = params->get<int_t>(DICe::last_image_index);
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::image_file_prefix),std::runtime_error,
      "Error, the image file prefix was not specified");
    const std::string prefix = params->get<std::string>(DICe::image_file_prefix);
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
    const int_t numImages = lastImageIndex - refId + 1;
    TEUCHOS_TEST_FOR_EXCEPTION(numImages<=0,std::runtime_error,"");

    std::string stereo_left_suffix = "";
    std::string stereo_right_suffix = "";
    const bool is_stereo = params->isParameter(DICe::stereo_left_suffix) || params->isParameter(DICe::stereo_right_suffix);
    if(is_stereo){
      TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::stereo_left_suffix),std::runtime_error,"Error, for stereo the stereo_left_suffix parameter must be set");
      TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::stereo_right_suffix),std::runtime_error,"Error, for stereo the stereo_right_suffix parameter must be set");
      stereo_left_suffix = params->get<std::string>(DICe::stereo_left_suffix);
      stereo_right_suffix = params->get<std::string>(DICe::stereo_right_suffix);
    }

    // determine the reference image
    int_t number = refId;
    int_t refDig = 0;
    if(refId==0)
      refDig = 1;
    else{
      while (number) {number /= 10; refDig++;}
    }
    std::stringstream ref_name;
    ref_name << folder << prefix;
    if(digits > 1){
      for(int_t i=0;i<digits - refDig;++i) ref_name << "0";
    }
    ref_name << refId << stereo_left_suffix << fileType;
    image_files.push_back(ref_name.str());
    if(is_stereo){
      std::stringstream stereo_ref_name;
      stereo_ref_name << folder << prefix;
      if(digits > 1){
        for(int_t i=0;i<digits - refDig;++i) stereo_ref_name << "0";
      }
      stereo_ref_name << refId << stereo_right_suffix << fileType;
      stereo_image_files.push_back(stereo_ref_name.str());
    }

    // determine the deformed images
    for(int_t i=0;i<numImages;++i){
      std::stringstream def_name;
      def_name << folder << prefix;
      int_t tmpNum = refId+i;
      int_t defDig = 0;
      if(tmpNum==0)
        defDig = 1;
      else{
        while (tmpNum) {tmpNum /= 10; defDig++;}
      }
      if(digits > 1)
        for(int_t j=0;j<digits - defDig;++j) def_name << "0";
      def_name << refId+i << stereo_left_suffix << fileType;
      image_files.push_back(def_name.str());
      if(is_stereo){
        std::stringstream stereo_def_name;
        stereo_def_name << folder << prefix;
        if(digits > 1)
          for(int_t j=0;j<digits - defDig;++j) stereo_def_name << "0";
        stereo_def_name << refId+i << stereo_right_suffix << fileType;
        stereo_image_files.push_back(stereo_def_name.str());
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(is_stereo && image_files.size()!=stereo_image_files.size(),std::runtime_error,"");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(image_files.size()<=1,std::runtime_error,"");
}

DICE_LIB_DLL_EXPORT
bool valid_correlation_point(const int_t x_coord,
  const int_t y_coord,
  const int_t width,
  const int_t height,
  const int_t subset_size,
  std::set<std::pair<int_t,int_t> > & coords,
  std::set<std::pair<int_t,int_t> > & excluded_coords){

  // need to check if the point is interior to the image by at least one subset_size
  if(x_coord<subset_size-1) return false;
  if(x_coord>width-subset_size) return false;
  if(y_coord<subset_size-1) return false;
  if(y_coord>height-subset_size) return false;

  // only the centroid has to be inside the ROI
  if(coords.find(std::pair<int_t,int_t>(y_coord,x_coord))==coords.end()) return false;

  static std::vector<int_t> corners_x(4);
  static std::vector<int_t> corners_y(4);
  corners_x[0] = x_coord - subset_size/2;  corners_y[0] = y_coord - subset_size/2;
  corners_x[1] = corners_x[0]+subset_size; corners_y[1] = corners_y[0];
  corners_x[2] = corners_x[0]+subset_size; corners_y[2] = corners_y[0] + subset_size;
  corners_x[3] = corners_x[0];             corners_y[3] = corners_y[0] + subset_size;
  // check four points to see if any of the corners fall in an excluded region
  bool all_corners_in = true;
  for(int_t i=0;i<4;++i){
    if(excluded_coords.find(std::pair<int_t,int_t>(corners_y[i],corners_x[i]))!=excluded_coords.end())
      all_corners_in = false;
  }
  return all_corners_in;

}

DICE_LIB_DLL_EXPORT
void
create_regular_grid_of_correlation_points(std::vector<int_t> & correlation_points,
  std::vector<int_t> & neighbor_ids,
  Teuchos::RCP<Teuchos::ParameterList> params,
  const int_t width, const int_t height,
  Teuchos::RCP<DICe::Subset_File_Info> subset_file_info){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Creating a grid of regular correlation points");
  // note: assumes two dimensional
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::step_size),std::runtime_error,
    "Error, the step size was not specified");
  const int_t step_size = params->get<int_t>(DICe::step_size);
  // set up the control points
  TEUCHOS_TEST_FOR_EXCEPTION(step_size<=0,std::runtime_error,"Error: step size is <= 0");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::subset_size),std::runtime_error,
    "Error, the subset size was not specified");
  const int_t subset_size = params->get<int_t>(DICe::subset_size);
  correlation_points.clear();
  neighbor_ids.clear();

  bool seed_was_specified = false;
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > roi_defs;
  if(subset_file_info!=Teuchos::null){
    if(subset_file_info->conformal_area_defs!=Teuchos::null){
      if(proc_rank==0) DEBUG_MSG("Using ROIs from a subset file");
      roi_defs = subset_file_info->conformal_area_defs;
      if(proc_rank==0) DEBUG_MSG("create_regular_grid_of_correlation_points(): user requested " << roi_defs->size() <<  " ROI(s)");
      seed_was_specified = subset_file_info->size_map->size() > 0;
      if(seed_was_specified){
        TEUCHOS_TEST_FOR_EXCEPTION(subset_file_info->size_map->size()!=subset_file_info->displacement_map->size(),
          std::runtime_error,"Error the number of displacement guesses and seed locations must be the same");
      }
    }
    else{
      if(proc_rank==0) DEBUG_MSG("create_regular_grid_of_correlation_points(): subset file exists, but has no ROIs");
    }
  }
  if(roi_defs==Teuchos::null){ // wasn't populated above so create a dummy roi with the whole image:
    if(proc_rank==0) DEBUG_MSG("create_regular_grid_of_correlation_points(): creating dummy ROI of the entire image");
    Teuchos::RCP<DICe::Rectangle> rect = Teuchos::rcp(new DICe::Rectangle(width/2,height/2,width,height));
    DICe::multi_shape boundary_multi_shape;
    boundary_multi_shape.push_back(rect);
    DICe::Conformal_Area_Def conformal_area_def(boundary_multi_shape);
    roi_defs = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def> ());
    roi_defs->insert(std::pair<int_t,DICe::Conformal_Area_Def>(0,conformal_area_def));
  }


  // if no ROI is specified, the whole image is the ROI

  int_t current_subset_id = 0;

  std::map<int_t,DICe::Conformal_Area_Def>::iterator map_it=roi_defs->begin();
  for(;map_it!=roi_defs->end();++map_it){

    std::set<std::pair<int_t,int_t> > coords;
    std::set<std::pair<int_t,int_t> > excluded_coords;
    // collect the coords of all the boundary shapes
    for(size_t i=0;i<map_it->second.boundary()->size();++i){
      std::set<std::pair<int_t,int_t> > shapeCoords = (*map_it->second.boundary())[i]->get_owned_pixels();
      coords.insert(shapeCoords.begin(),shapeCoords.end());
    }
    // collect the coords of all the exclusions
    if(map_it->second.has_excluded_area()){
      for(size_t i=0;i<map_it->second.excluded_area()->size();++i){
        std::set<std::pair<int_t,int_t> > shapeCoords = (*map_it->second.excluded_area())[i]->get_owned_pixels();
        excluded_coords.insert(shapeCoords.begin(),shapeCoords.end());
      }
    }
    int_t num_rows = 0;
    int_t num_cols = 0;
    int_t seed_row = 0;
    int_t seed_col = 0;
    int_t x_coord = subset_size-1;
    int_t y_coord = subset_size-1;
    int_t seed_location_x = 0;
    int_t seed_location_y = 0;
    int_t seed_subset_id = -1;

    bool this_roi_has_seed = false;
    if(seed_was_specified){
      if(subset_file_info->size_map->find(map_it->first)!=subset_file_info->size_map->end()){
        seed_location_x = subset_file_info->size_map->find(map_it->first)->second.first;
        seed_location_y = subset_file_info->size_map->find(map_it->first)->second.second;
        if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " seed x: " << seed_location_x << " seed_y: " << seed_location_y);
        this_roi_has_seed = true;
      }
    }
    while(x_coord < width - subset_size) {
      if(x_coord + step_size/2 < seed_location_x) seed_col++;
      x_coord+=step_size;
      num_cols++;
    }
    while(y_coord < height - subset_size) {
      if(y_coord + step_size/2 < seed_location_y) seed_row++;
      y_coord+=step_size;
      num_rows++;
    }
    if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " has " << num_rows << " rows and " << num_cols << " cols, seed row " << seed_row << " seed col " << seed_col);
    //if(seed_was_specified&&this_roi_has_seed){
      x_coord = subset_size-1 + seed_col*step_size;
      y_coord = subset_size-1 + seed_row*step_size;
    if(valid_correlation_point(x_coord,y_coord,width,height,subset_size,coords,excluded_coords)){
      correlation_points.push_back(x_coord);
      correlation_points.push_back(y_coord);
      //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding seed correlation point " << x_coord << " " << y_coord);
      if(seed_was_specified&&this_roi_has_seed){
        seed_subset_id = current_subset_id;
        subset_file_info->seed_subset_ids->insert(std::pair<int_t,int_t>(seed_subset_id,map_it->first));
      }
      neighbor_ids.push_back(-1); // seed point cannot have a neighbor
      current_subset_id++;
    }

    // snake right from seed
    const int_t right_start_subset_id = current_subset_id;
    int_t direction = 1; // sign needs to be flipped if the seed row is the first row
    int_t row = seed_row;
    int_t col = seed_col;
    while(row>=0&&row<num_rows){
      if((direction==1&&row+1>=num_rows)||(direction==-1&&row-1<0)){
        direction *= -1;
        col++;
      }else{
        row += direction;
      }
      if(col>=num_cols)break;
      x_coord = subset_size - 1 + col*step_size;
      y_coord = subset_size - 1 + row*step_size;
      if(valid_correlation_point(x_coord,y_coord,width,height,subset_size,coords,excluded_coords)){
        correlation_points.push_back(x_coord);
        correlation_points.push_back(y_coord);
        //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding snake right correlation point " << x_coord << " " << y_coord);
        if(current_subset_id==right_start_subset_id)
          neighbor_ids.push_back(seed_subset_id);
        else
          neighbor_ids.push_back(current_subset_id - 1);
        current_subset_id++;
      }  // end valid point
    }  // end snake right
    // snake left from seed
    const int_t left_start_subset_id = current_subset_id;
    direction = -1;
    row = seed_row;
    col = seed_col;
    while(row>=0&&row<num_rows){
      if((direction==1&&row+1>=num_rows)||(direction==-1&&row-1<0)){
        direction *= -1;
        col--;
      }
      else{
        row += direction;
      }
      if(col<0)break;
      x_coord = subset_size - 1 + col*step_size;
      y_coord = subset_size - 1 + row*step_size;
      if(valid_correlation_point(x_coord,y_coord,width,height,subset_size,coords,excluded_coords)){
        correlation_points.push_back(x_coord);
        correlation_points.push_back(y_coord);
        //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding snake left correlation point " << x_coord << " " << y_coord);
        if(current_subset_id==left_start_subset_id)
          neighbor_ids.push_back(seed_subset_id);
        else
          neighbor_ids.push_back(current_subset_id-1);
        current_subset_id++;
      }  // valid point
    }  // end snake left
  }  // conformal area map
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
  write_xml_size_param(inputFile,DICe::last_image_index);
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
  for(int_t i=0;i<MAX_FIELD_NAME;++i){
    std::stringstream iToStr;
    iToStr << i;
    write_xml_bool_param(paramsFile,to_string(static_cast<Field_Name>(i)),"true");
  }
  write_xml_param_list_close(paramsFile);

  finalize_xml_file(fileNameInput.str());
  finalize_xml_file(fileNameParams.str());
}

}// End DICe Namespace
