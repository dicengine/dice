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

#include <DICe_ParserUtils.h>

#include <iostream>
#include <fstream>
#include <cctype>
#include <string>
#include <cstring>
#include <algorithm>

namespace DICe {

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

}// End DICe Namespace
