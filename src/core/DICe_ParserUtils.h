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

#ifndef DICE_PARSERUTILS_H
#define DICE_PARSERUTILS_H

#include <DICe.h>

#include <vector>

namespace DICe {

/// Convert a string to all upper case
DICE_LIB_DLL_EXPORT
void to_upper(std::string &s);

/// Convert a string to all lower case
DICE_LIB_DLL_EXPORT
void to_lower(std::string &s);

DICE_LIB_DLL_EXPORT
/// safe getline that works with windows and linux line endings
/// \param is input stream
/// \param t [out] string from getline
std::istream& safeGetline(std::istream& is, std::string& t);

/// \brief Turns a string read from getline() into tokens
/// \param dataFile fstream file to read line from (assumed to be open)
/// \param delim Delimiter character
/// \param capitalize true if the tokens should be automatically capitalized
DICE_LIB_DLL_EXPORT
std::vector<std::string> tokenize_line(std::istream &dataFile,
  const std::string & delim=" \t",
  const bool capitalize = true);

/// \brief Determines if a string is a number
/// \param s Input string
DICE_LIB_DLL_EXPORT
bool is_number(const std::string& s);

/// Parser string
const char* const parser_comment_char = "#";

}// End DICe Namespace

#endif
