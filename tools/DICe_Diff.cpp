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

/*! \file  DICe_Diff.cpp
    \brief Utility for diffing DICe output files (text and numeric)
*/

#include <DICe.h>
#include <DICe_Parser.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <cassert>

using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage ./DICe_Diff <infileA> <infileB> [-t <tol>] [-v] [-n]

  DICe::initialize(argc, argv);

  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);
  int_t errorFlag  = 0;
  scalar_t relTol = 1.0E-6;
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
      std::cout << "          -n numerical values only" << std::endl;
      exit(0);
    }
  }

  if(argc < 3) {
    printf("You must provide two files to compare\n");
    exit(0);
  }

  DEBUG_MSG("User specified " << argc << " arguments");
  for(int_t i=0;i<argc;++i){
    DEBUG_MSG(argv[i]);
  }

  std::string masthead = "DIGITAL"; // the masthead line will have a different git sha1 for each file so skip it

  // TODO add delimeter option

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
    else{
      std::cout << "Error, unrecognized option: " << argv[i] << std::endl;
      assert(false);
    }
  }
  std::string fileA = argv[1];
  std::string fileB = argv[2];
  *outStream << "File A: " << fileA << std::endl;
  *outStream << "File B: " << fileB << std::endl;
  *outStream << "Relative Tol: " << relTol << std::endl;

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
    Teuchos::ArrayRCP<std::string> tokensA = DICe::tokenize_line(dataFileA,delimiter);
    Teuchos::ArrayRCP<std::string> tokensB = DICe::tokenize_line(dataFileB,delimiter);
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
    for(int_t i=0;i<tokensA.size();++i){
      // number
      if(DICe::is_number(tokensA[i])){
        assert(DICe::is_number(tokensB[i]));
        scalar_t valA = strtod(tokensA[i].c_str(),NULL);
        scalar_t valB = strtod(tokensB[i].c_str(),NULL);
        scalar_t diff = std::abs((valA - valB)/valA);
        const bool tiny = (std::abs(valA) + std::abs(valB) < 1.0E-8);
        if(!tiny && diff > relTol){
          line_diff = true;
          badTokenIds.push_back(i);
          badTokenTypes.push_back("n");
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
      for(int_t i=0;i<tokensA.size();++i)
        *outStream << tokensA[i] << " ";
      *outStream << std::endl;
      *outStream << "> " << line << " (";
      for(size_t i=0;i<badTokenIds.size();++i){
        *outStream << badTokenIds[i] << "[" << badTokenTypes[i] << "] ";
      }
      *outStream  << "): ";
      for(int_t i=0;i<tokensB.size();++i)
        *outStream << tokensB[i] << " ";
      *outStream << std::endl;
    }
    line++;
  }

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

