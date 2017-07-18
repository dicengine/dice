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

#include <DICe_XMLUtils.h>

#include <iostream>
#include <fstream>
#include <cctype>
#include <string>

namespace DICe {

void initialize_xml_file(const std::string & file_name){
  std::ofstream file;
  file.open(file_name.c_str());
  file << "<ParameterList>" << std::endl;
  file.close();
}

void finalize_xml_file(const std::string & file_name){
  std::ofstream file;
  file.open(file_name.c_str(),std::ios::app);
  file << "</ParameterList>" << std::endl;
  file.close();
}

void write_xml_comment(const std::string & file_name, const std::string & comment){
  std::ofstream file;
  file.open(file_name.c_str(),std::ios::app);
  file << "<!-- " << comment << " -->" << std::endl;
  file.close();
}

void write_xml_param_list_open(const std::string & file_name, const std::string & name, const bool commented){
  std::ofstream file;
  file.open(file_name.c_str(),std::ios::app);
  if(commented){
    file << "<!-- ";
  }
  file << "<ParameterList name=\"" << name << "\">";
  if(commented)
    file << "-->";
  file << std::endl;
  file.close();
}

void write_xml_param_list_close(const std::string & file_name, const bool commented){
  std::ofstream file;
  file.open(file_name.c_str(),std::ios::app);
  if(commented){
    file << "<!-- ";
  }
  file << "</ParameterList>";
  if(commented)
    file << "-->";
  file << std::endl;
  file.close();
}

void write_xml_param(const std::string & file_name, const std::string & name, const std::string & type, const std::string & value,
  const bool commented){
  std::ofstream file;
  file.open(file_name.c_str(),std::ios::app);
  if(commented){
    file << "<!-- ";
  }
  file << "<Parameter name=\"" << name << "\" type=\""<< type << "\" value=\"" << value << "\" />  ";
  if(commented)
    file << "-->";
  file << std::endl;
  file.close();
}

void write_xml_string_param(const std::string & file_name, const std::string & name, const std::string & value,
  const bool commented){
  write_xml_param(file_name,name,"string",value,commented);
}
void write_xml_real_param(const std::string & file_name, const std::string & name, const std::string & value,
  const bool commented){
  write_xml_param(file_name,name,"double",value,commented);
}
void write_xml_size_param(const std::string & file_name, const std::string & name, const std::string & value,
  const bool commented){
  write_xml_param(file_name,name,"int",value,commented);
}
void write_xml_bool_param(const std::string & file_name, const std::string & name, const std::string & value,
  const bool commented){
  write_xml_param(file_name,name,"bool",value,commented);
}
void write_xml_bool_literal_param(const std::string & file_name, const std::string & name, const bool value,
  const bool commented){
    if(value)
        write_xml_param(file_name,name,"bool","true",commented);
    else
        write_xml_param(file_name,name,"bool","false",commented);
}

}// End DICe Namespace
