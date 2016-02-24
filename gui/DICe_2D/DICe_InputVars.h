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

#ifndef DICE_INPUTVARS_H
#define DICE_INPUTVARS_H

#include <QFileInfo>

namespace DICe{

namespace gui{

/// \class Input_Vars
/// \brief Container singleton class to hold the information from the GUI
/// as to the input file names, output folders, etc.
class Input_Vars{
public:
  /// return an instance of the singleton
  static Input_Vars * instance(){
    if(!input_vars_ptr_){
      input_vars_ptr_ = new Input_Vars;
    }
    return input_vars_ptr_;
  }

  /// sets the file name for the reference image
  /// \param file_name the file name to set
  void set_ref_file_info(const QFileInfo & file_info){
    ref_file_info_ = file_info;
  }

  /// returns the reference file name
  QFileInfo get_ref_file_info(){
    return ref_file_info_;
  }

  /// returns a pointer to the deformed images list
  QStringList * get_def_file_list(){
    return &def_file_list_;
  }

private:
  /// constructor
  Input_Vars(){};
  /// copy constructor
  Input_Vars(Input_Vars const&);
  /// asignment operator
  void operator=(Input_Vars const &);

  /// reference image file name
  QFileInfo ref_file_info_;
  QStringList def_file_list_;


  /* ----------------------- */
  /// singleton pointer
  static Input_Vars * input_vars_ptr_;

};

} // gui namespace

} // DICe namespace

#endif // DICE_INPUTVARS_H
