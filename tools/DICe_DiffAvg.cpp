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

#include <DICe.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cassert>

using namespace std;
using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);
  int_t errorFlag  = 0;

  // read the parameters from the input file:
  if(argc!=2){ // executable and input file
    std::cout << "Usage: DICe_Diff_Avg <parameters_file> " << std::endl;
    exit(1);
  }

  const std::string params_file = argv[1];
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> paramsPtr(params.get());
  Teuchos::updateParametersFromXmlFile(params_file, paramsPtr);
  assert(params!=Teuchos::null);

  // check that all the required params are given:
  assert(params->isParameter("input_file"));
  const std::string input_file = params->get<std::string>("input_file");
  assert(params->isParameter("num_header_rows"));
  const int_t num_header_rows = params->get<int_t>("num_header_rows");
  assert(params->isParameter("collect_averages_for_unique_values_in_column"));
  const int_t coord_col = params->get<int_t>("collect_averages_for_unique_values_in_column");
  assert(params->isParameter("compute_average_of_column"));
  const int_t data_col = params->get<int_t>("compute_average_of_column");
  assert(params->isParameter("command_file_name"));
  const std::string command_file_name = params->get<std::string>("command_file_name");
  assert(params->isParameter("command_num_header_rows"));
  const int_t command_num_header_rows = params->get<int_t>("command_num_header_rows");
  assert(params->isParameter("command_coord_col"));
  const int_t command_coord_col = params->get<int_t>("command_coord_col");
  assert(params->isParameter("command_value_col"));
  const int_t command_data_col = params->get<int_t>("command_value_col");
  assert(params->isParameter("output_file_name"));
  const std::string output_file = params->get<std::string>("output_file_name");
  const bool verbose = params->get<bool>("verbose",false);
  if(verbose)
    outStream = Teuchos::rcp(&std::cout, false);
  const double rel_tol = params->get<double>("relative_tolerance",1.0E-6);
  const double compare_factor = params->get<double>("compare_factor",1.0);

  params->print(*outStream);

  // Input values will be averaged for all entries in the given coord column, for example if coord col is 0 (x coordinate)
  // the values will be summed over all y values for the given x and divided by the number of y values
  // The command data can have more points than the input data, but values should
  // be available in the command data at every point that is in the input data (TODO add interpolation)

  *outStream << "File name " << input_file << std::endl;
  *outStream << "Command File name " << command_file_name << std::endl;
  *outStream << "Num header rows: " << num_header_rows << std::endl;
  *outStream << "Num command header rows: " << command_num_header_rows << std::endl;
  *outStream << "Coord column: " << coord_col << std::endl;
  *outStream << "Data column: " << data_col << std::endl;
  *outStream << "Command coord column: " << command_coord_col << std::endl;
  *outStream << "Command data column: " << command_data_col << std::endl;
  vector <vector <string> > results;
  ifstream file(input_file.c_str());
  assert(file.good());
  // get rid of the header rows:
  string s;
  for(int_t i=0;i<num_header_rows;++i){
    getline(file,s);
  }
  // read the data
  while (file)
  {
    if (!getline( file, s )) break;
    istringstream ss( s );
    vector <string> record;
    while (ss)
    {
      string s;
      if (!getline( ss, s, ',' )) break;
      record.push_back( s );
    }
    results.push_back( record );
  }
  if (!file.eof())
  {
    cerr << "EOF Error Ocurred !\n";
  }
  *outStream << "Number of columns in data: " << results[0].size() << std::endl;
  *outStream << "Number of rows in data: " << results.size() << std::endl;
  file.close();


  // sort the data according to either x or y:
  std::map<int_t,std::vector<work_t> > sortedMap;
  for(size_t row=0;row<results.size();++row){
    work_t coordReal = strtod(results[row][coord_col].c_str(),NULL);
    int_t coord = static_cast<int_t>(coordReal);
    //int_t coord = strtol(results[row][coord_col].c_str(),NULL,0);
    work_t value = strtod(results[row][data_col].c_str(),NULL);
    if(sortedMap.find(coord)==sortedMap.end()){
      std::vector<work_t> tmp_vec;
      sortedMap.insert(std::pair<int_t,std::vector<work_t> >(coord,tmp_vec));
    }
    sortedMap.find(coord)->second.push_back(value);
  }
  // average the values
  std::vector<int_t> coords;
  std::vector<work_t> avg_values;
  std::vector<work_t> std_dev_values;
  std::vector<work_t> max_values;
  std::vector<work_t> min_values;
  std::map<int_t,std::vector<work_t> > ::iterator map_it = sortedMap.begin();
  for(;map_it!=sortedMap.end();++map_it){
    int_t num_values = map_it->second.size();
    assert(num_values!=0);
    int_t coord = map_it->first;
    work_t avg_value = 0.0;
    for(int_t i=0;i<num_values;++i){
      avg_value += map_it->second[i];
    }
    avg_value /= num_values;
    coords.push_back(coord);
    avg_values.push_back(avg_value*compare_factor);
  }

  map_it = sortedMap.begin();
  int_t v_index = 0;
  for(;map_it!=sortedMap.end();++map_it){
    int_t num_values = map_it->second.size();
    work_t std_dev = 0.0;
    work_t min_value = map_it->second[0];
    work_t max_value = map_it->second[0];
    for(int_t i=0;i<num_values;++i){
      if(map_it->second[i] > max_value)
        max_value = map_it->second[i];
      if(map_it->second[i] < min_value)
        min_value = map_it->second[i];
      std_dev += (map_it->second[i] - avg_values[v_index])*(map_it->second[i] - avg_values[v_index]);
    }
    std_dev /= num_values;
    std_dev = std::sqrt(std_dev);
    std_dev_values.push_back(std_dev);
    max_values.push_back(max_value);
    min_values.push_back(min_value);
    v_index++;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(std_dev_values.size()!=avg_values.size(),std::runtime_error,"Error, these two vectors should be the same size.");

  // command data

  vector <vector <string> > command;
  ifstream command_file(command_file_name.c_str());
  assert(command_file.good());
  // get rid of the header rows:
  for(int_t i=0;i<command_num_header_rows;++i){
    getline(command_file,s);
  }
  // read the data
  while (command_file)
  {
    if (!getline( command_file, s )) break;
    istringstream ss( s );
    vector <string> record;
    while (ss)
    {
      string s;
      if (!getline( ss, s, ',' )) break;
      record.push_back( s );
    }
    command.push_back( record );
  }
  if (!command_file.eof())
  {
    cerr << "Command EOF Error Ocurred !\n";
  }
  *outStream << "Number of columns in command: " << command[0].size() << std::endl;
  *outStream << "Number of rows in command: " << command.size() << std::endl;
  if(command[0].size()!=command[command.size()-1].size()){
    cerr << "The last row does not have the right number of columns" << std::endl;
    assert(false);
  }
  command_file.close();

  // sort the data according to either x or y:
  std::map<int_t,work_t> commandMap;
  for(size_t row=0;row<command.size();++row){
    work_t coordReal = strtod(command[row][command_coord_col].c_str(),NULL);
    int_t coord = static_cast<int_t>(coordReal);
    //int_t coord = strtol(command[row][command_coord_col].c_str(),NULL,0);
    work_t value = strtod(command[row][command_data_col].c_str(),NULL);
    commandMap.insert(std::pair<int_t,work_t >(coord,value));
  }
  *outStream << " The command map has " << commandMap.size() << " entries" << std::endl;

  // test the values:

  work_t avg_diff = 0.0;
  std::vector<work_t> command_values(coords.size());
  std::vector<work_t> diff_values(coords.size());
  for(size_t i=0;i<avg_values.size();++i){
    if(commandMap.find(coords[i])==commandMap.end()){
      std::cout << " Averaged values for coordinate " << coords[i] << " are in the input, but not in the command file " << std::endl;
      assert(false);
    }
    work_t command_val = commandMap.find(coords[i])->second;
    diff_values[i] = avg_values[i] - command_val;
    command_values[i] = command_val;
    avg_diff += (avg_values[i] - command_val)*(avg_values[i] - command_val);
  }
  avg_diff = std::sqrt(avg_diff);
  *outStream << " Difference: " << avg_diff << " over " << coords.size() << " points " << std::endl;
  if(avg_diff > rel_tol){
    *outStream << " Error: Difference is too large " << std::endl;
    errorFlag++;
  }

  // write averaged output to file:

  std::FILE * filePtr = fopen(output_file.c_str(),"w"); // overwrite the file if it exists
  for(size_t row=0;row<coords.size();++row){
    fprintf(filePtr,"%i %4.4E %4.4E %4.4E %4.4E %4.4E %4.4E",coords[row],avg_values[row],command_values[row],diff_values[row],std_dev_values[row],max_values[row],min_values[row]);
    fprintf(filePtr,"\n");
  }
  fclose(filePtr);

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

