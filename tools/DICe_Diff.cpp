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

/*! \file  DICe_Diff.cpp
    \brief Utility for diffing DICe output files (text and numeric)
*/

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Parser.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <cassert>
#include <map>
#include <fstream>

using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage ./DICe_Diff <infileA> <infileB or base name for parallel> [-t <tol>] [-f <value>] [-v] [-n] [-p <count>]
  /// Note: if this is a parallel comparison, this exec assumes that the first file is serail (contains all data points)
  /// the second file listed is the one that is split among several processors so only the base name is specified and the
  /// rest of the name is determined based on the number of procs

  DICe::initialize(argc, argv);

  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);
  int_t errorFlag  = 0;
  scalar_t relTol = 1.0E-6;
  scalar_t floor = 0.0;
  bool use_floor = false;
  std::string delimiter = " ,\r";
  bool numerical_values_only = false;

  if(argc==2){
    std::string help = argv[1];
    if(help=="-h"){
      std::cout << " DICe_Diff (compares two DICe output files, numerical and string values) " << std::endl;
      std::cout << " Syntax: DICe_Diff <fileA> <fileB> [options] " << std::endl;
      std::cout << " Options: -h show help message " << std::endl;
      std::cout << "          -v verbose " << std::endl;
      std::cout << "          -t <tol> relative tolerance " << std::endl;
      std::cout << "          -f <value> floor, values below the floor in the gold file will not be tested" << std::endl;
      std::cout << "          -n numerical values only" << std::endl;
      std::cout << "          -p <count> parallel output number of processors" << std::endl;
      exit(0);
    }
  }

  if(argc < 3) {
    printf("You must provide two files to compare\n");
    exit(1);
  }

  DEBUG_MSG("User specified " << argc << " arguments");
  for(int_t i=0;i<argc;++i){
    DEBUG_MSG(argv[i]);
  }

  std::string masthead = "DIGITAL"; // the masthead line will have a different git sha1 for each file so skip it

  // TODO add delimeter option

  int_t num_procs = 1;
  for(int_t i=3;i<argc;++i){
    std::string arg = argv[i];
    if(arg=="-v"){
      outStream = Teuchos::rcp(&std::cout, false);
    }
    else if(arg=="-n"){
      numerical_values_only = true;
    }
    else if(arg=="-t"){
      assert(argc>i+1 && "Error, tolerance must be specified for -t option");
      relTol = strtod(argv[i+1],NULL);
      i++;
    }
    else if(arg=="-p"){
      assert(argc>i+1 && "Error, count must be specified for -p option");
      num_procs = atoi(argv[i+1]);
      i++;
    }
    else if(arg=="-f"){
      assert(argc>i+1 && "Error, floor value must be specified for -f option");
      floor = strtod(argv[i+1],NULL);
      use_floor = true;
      i++;
    }
    else{
      std::cout << "Error, unrecognized option: " << argv[i] << std::endl;
      assert(false);
    }
  }
  std::string fileA = argv[1];
  std::string fileB = argv[2];
  *outStream << "File A:                " << fileA << std::endl;
  *outStream << "File B (or base name): " << fileB << std::endl;
  *outStream << "Relative Tol:          " << relTol << std::endl;
  *outStream << "Floor value:           " << floor << " active " << use_floor << std::endl;
  *outStream << "Number of processors:  " << num_procs << std::endl;

  if(num_procs > 1){
    // read all of the number line from A and store in a map
    std::map<int_t,std::vector<std::string> > fileASolutions;
    std::fstream dataFileA(fileA.c_str(), std::ios_base::in | std::ios_base::binary);
    while (!dataFileA.eof())
    {
      std::vector<std::string> tokens = DICe::tokenize_line(dataFileA,delimiter);
      if(tokens.size()==0) continue;
      if(!DICe::is_number(tokens[0])) continue;
      // check that the first number is an integer (presumed an id)
      // if ids have been omitted this will fail
      const scalar_t remainder = strtod(tokens[0].c_str(),NULL) - std::floor(strtod(tokens[0].c_str(),NULL));
      TEUCHOS_TEST_FOR_EXCEPTION(remainder!=0.0,std::runtime_error,
        "Error, first column in the output file must be the subset or node id "
        "(cannot ommit the id in the output parameters to compare parallel files)");
      fileASolutions.insert(std::pair<int_t,std::vector<std::string> >(std::atoi(tokens[0].c_str()),tokens));
    }
    dataFileA.close();
    // now that the offsets are set up, compare the files one processor chunk at a time
    std::set<int_t> compared_ids;
    for(int_t i=0;i<num_procs;++i){
      std::stringstream name;
      name << fileB << "." << num_procs << "." << i << ".txt";
      // read the number of lines in each file:
      std::fstream dataFileB(name.str().c_str(), std::ios_base::in | std::ios_base::binary);
      assert(dataFileB.good());
      int_t par_line = 0;
      while (!dataFileB.eof())
      {
        bool line_diff = false;
        std::vector<int_t> badTokenIds;
        std::vector<std::string> badTokenTypes;
        std::vector<std::string> tokensB = DICe::tokenize_line(dataFileB,delimiter);
        if(tokensB.size()==0) continue;
        if(!DICe::is_number(tokensB[0])) continue;
        const scalar_t remainder = strtod(tokensB[0].c_str(),NULL) - std::floor(strtod(tokensB[0].c_str(),NULL));
        TEUCHOS_TEST_FOR_EXCEPTION(remainder!=0.0,std::runtime_error,
          "Error, first column in the parallel output file must be the subset or node id "
          "(cannot ommit the id in the output parameters to compare parallel files)");
        const int_t subset_id = std::atoi(tokensB[0].c_str());
        // find that row in the saved data:
        TEUCHOS_TEST_FOR_EXCEPTION(fileASolutions.find(subset_id)==fileASolutions.end(),std::runtime_error,
          "Error could not find parallel subset " << subset_id << " in serial file");
        std::vector<std::string> tokensA = fileASolutions.find(subset_id)->second;
        compared_ids.insert(subset_id);
        if(tokensB.size()==0) break;
        bool read_error = false;
        assert(tokensA.size()!=0);
        assert(tokensB.size()!=0);
        read_error = !DICe::is_number(tokensA[0]);
        read_error = !DICe::is_number(tokensB[0]);
        read_error = tokensA.size()!=tokensB.size();
        if(read_error){
          *outStream << "Error, output files are not compatible (read error)" << std::endl;
          errorFlag++;
          break;
        }
        for(size_t i=0;i<tokensA.size();++i){
          // number
          if(DICe::is_number(tokensA[i])){
            assert(DICe::is_number(tokensB[i]));
            scalar_t valA = strtod(tokensA[i].c_str(),NULL);
            scalar_t valB = strtod(tokensB[i].c_str(),NULL);
            scalar_t diff = valA == 0.0 ? std::abs((valA - valB)/valA) : std::abs(valA - valB);
            const bool tiny = (std::abs(valA) + std::abs(valB) < 1.0E-8);
            const bool below_floor = std::abs(valA) < floor && valA!=0.0;
            if(!below_floor||!use_floor){
              if(!tiny && diff > relTol){
                line_diff = true;
                badTokenIds.push_back(i);
                badTokenTypes.push_back("n");
              }
            }
          }
          // string
          else{
            assert(!DICe::is_number(tokensB[i]));
            if(tokensA[i]!=tokensB[i] && !numerical_values_only)
              line_diff = true;
            badTokenIds.push_back(i);
            badTokenTypes.push_back("s");
          }
        } // end token iteration
        if(line_diff){
          errorFlag++;
          *outStream << "< " << par_line << " (";
          for(size_t i=0;i<badTokenIds.size();++i){
            *outStream << badTokenIds[i] << "[" << badTokenTypes[i] << "] ";
          }
          *outStream  << "): ";
          for(size_t i=0;i<tokensA.size();++i)
            *outStream << tokensA[i] << " ";
          *outStream << std::endl;
          *outStream << "> " << par_line << " (";
          for(size_t i=0;i<badTokenIds.size();++i){
            *outStream << badTokenIds[i] << "[" << badTokenTypes[i] << "] ";
          }
          *outStream  << "): ";
          for(size_t i=0;i<tokensB.size();++i)
            *outStream << tokensB[i] << " ";
          *outStream << std::endl;
        } // end line diff
        par_line++;
      } // end dataFileB.eof() loop
      *outStream << "proc " << i << " number of lines compared " << par_line << std::endl;
      assert(par_line>0);
      dataFileB.close();
    } // end of parallel loop
    // check that all the ids were compared
    std::map<int_t,std::vector<std::string> >::iterator it=fileASolutions.begin();
    std::map<int_t,std::vector<std::string> >::iterator it_end=fileASolutions.end();
    bool missing_value = false;
    for(;it!=it_end;++it){
      if(compared_ids.find(it->first)==compared_ids.end())missing_value = true;
    }
    if(missing_value){
      *outStream << "Error, some ids in the serial output file were not present in any of the parallel output files" << std::endl;
      errorFlag++;
    }
  }
  else{
    // read the two files line by line and compare both
    // (white space is ignored)
    std::fstream dataFileA(fileA.c_str(), std::ios_base::in | std::ios_base::binary);
    assert(dataFileA.good());
    std::fstream dataFileB(fileB.c_str(), std::ios_base::in | std::ios_base::binary);
    assert(dataFileB.good());

    // read each line of the file
    int_t line = 0;
    while (!dataFileA.eof())
    {
      bool line_diff = false;
      std::vector<int_t> badTokenIds;
      std::vector<std::string> badTokenTypes;
      if(dataFileB.eof()) {
        *outStream << "Error, File A has more lines than FileB " << std::endl;
        errorFlag++;
        break;
      }
      std::vector<std::string> tokensA = DICe::tokenize_line(dataFileA,delimiter);
      std::vector<std::string> tokensB = DICe::tokenize_line(dataFileB,delimiter);
      if(tokensA.size()>=2){
        if(tokensA[1].find(masthead)!=std::string::npos){ // skip the masthead
          line++;
          continue;
        }
      }
      if(tokensA.size()!=tokensB.size()){
        *outStream << "Error, Different number of tokens per line A:" << tokensA.size() << " B: " << tokensB.size() << std::endl;
        errorFlag++;
        break;
      }
      for(size_t i=0;i<tokensA.size();++i){
        // number
        if(DICe::is_number(tokensA[i])){
          assert(DICe::is_number(tokensB[i]));
          scalar_t valA = strtod(tokensA[i].c_str(),NULL);
          scalar_t valB = strtod(tokensB[i].c_str(),NULL);
          scalar_t diff = std::abs((valA - valB)/valA);
          const bool tiny = (std::abs(valA) + std::abs(valB) < 1.0E-8);
          const bool below_floor = std::abs(valA) < floor;
          if(!below_floor||!use_floor){
            if(!tiny && diff > relTol){
              line_diff = true;
              badTokenIds.push_back(i);
              badTokenTypes.push_back("n");
            }
          }
        }
        // string
        else{
          assert(!DICe::is_number(tokensB[i]));
          if(tokensA[i]!=tokensB[i] && !numerical_values_only)
            line_diff = true;
          badTokenIds.push_back(i);
          badTokenTypes.push_back("s");
        }
      }
      if(line_diff){
        errorFlag++;
        *outStream << "< " << line << " (";
        for(size_t i=0;i<badTokenIds.size();++i){
          *outStream << badTokenIds[i] << "[" << badTokenTypes[i] << "] ";
        }
        *outStream  << "): ";
        for(size_t i=0;i<tokensA.size();++i)
          *outStream << tokensA[i] << " ";
        *outStream << std::endl;
        *outStream << "> " << line << " (";
        for(size_t i=0;i<badTokenIds.size();++i){
          *outStream << badTokenIds[i] << "[" << badTokenTypes[i] << "] ";
        }
        *outStream  << "): ";
        for(size_t i=0;i<tokensB.size();++i)
          *outStream << tokensB[i] << " ";
        *outStream << std::endl;
      }
      line++;
    }
    dataFileA.close();
    dataFileB.close();
  } // end serial comparison
  DICe::finalize();

  if (errorFlag != 0){
    std::cout << "End Result: TEST FAILED\n";
    exit(1);
  }
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

