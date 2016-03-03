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
#include <iostream>
#include <DICe_Parser.h>

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

  /// flag if the ref name has been set
  bool has_ref_file(){
      return !(ref_file_info_.fileName().size()==0);
  }

  /// flag if the ref name has been set
  bool has_def_files(){
      return def_file_list_.size()>0;
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

  /// decrement the last element of the roi vertex vectors
  void decrement_vertex_vector(){
      if(roi_vertex_vectors_.size()>0)
          roi_vertex_vectors_.removeLast();
  }
  /// decrement the last element of the roi vertex vectors
  void decrement_excluded_vertex_vector(){
      if(roi_excluded_vertex_vectors_.size()>0)
          roi_excluded_vertex_vectors_.removeLast();
  }

  /// append a vector of vertices to the set
  /// \param vertex_vector a QList containing the points of the vertices
  void append_vertex_vector(QList<QPoint> vertex_vector){
      std::cout << "Appending vertex vector ";
      for(QList<QPoint>::iterator i=vertex_vector.begin();i<vertex_vector.end();++i)
          std::cout << " " << i->x() << " " << i->y();
      std::cout << std::endl;
      roi_vertex_vectors_.append(vertex_vector);
  }

  /// append a vector of vertices to the set
  /// \param vertex_vector a QList containing the points of the vertices
  void append_excluded_vertex_vector(QList<QPoint> vertex_vector){
      std::cout << "Appending excluded vertex vector ";
      for(QList<QPoint>::iterator i=vertex_vector.begin();i<vertex_vector.end();++i)
          std::cout << " " << i->x() << " " << i->y();
      std::cout << std::endl;
      roi_excluded_vertex_vectors_.append(vertex_vector);
  }

  /// return a pointer to the roi vertex vectors
  QList<QList <QPoint> > * get_roi_vertex_vectors(){
      return & roi_vertex_vectors_;
  }

  /// return a pointer to the roi vertex vectors
  QList<QList <QPoint> > * get_roi_excluded_vertex_vectors(){
      return & roi_excluded_vertex_vectors_;
  }

  /// display the list of roi vertices
  void display_roi_vertices(){
      int shape_it = 0;
      for(QList<QList<QPoint> >::iterator it=roi_vertex_vectors_.begin();
          it!=roi_vertex_vectors_.end();++it){
          std::cout << "Shape " << shape_it;
          for(QList<QPoint>::iterator cit=it->begin();cit!=it->end();++cit)
              std::cout << " x " << cit->x() << " y " << cit->y();
          std::cout << std::endl;
          shape_it++;
      }
  }

  /// set the working directory
  void set_working_dir(const QString & dir){
      working_dir_ = dir;
  }

  /// set the subset size
  void set_subset_size(const int size){
      subset_size_ = size;
  }

  /// set the step size
  void set_step_size(const int size){
      subset_size_ = size;
  }

  /// write the input file
  void write_input_file(){
      std::stringstream input_file_ss;
#ifdef WIN32
      input_file_ss << working_dir_.toStdString() << "\\" << "input.xml";
#else
      input_file_ss << working_dir_.toStdString() << "/" << "input.xml";
 #endif
      std::string inputFile = input_file_ss.str();
      std::cout << "DICe::Input_Vars::instance(): writing input xml file: " << inputFile << std::endl;
      DICe::initialize_xml_file(inputFile);

      DICe::write_xml_comment(inputFile,"Auto generated input file from DICe GUI");

      DICe::write_xml_string_param(inputFile,DICe::output_folder,working_dir_.toStdString(),false);
      DICe::write_xml_string_param(inputFile,DICe::image_folder,"",false);
      DICe::write_xml_string_param(inputFile,DICe::correlation_parameters_file,"params.xml",true);

      std::stringstream subsetSizeSS;
      subsetSizeSS << subset_size_;
      DICe::write_xml_size_param(inputFile,DICe::subset_size,subsetSizeSS.str(),false);
      std::stringstream stepSizeSS;
      stepSizeSS << step_size_;
      DICe::write_xml_size_param(inputFile,DICe::step_size,stepSizeSS.str(),false);

      DICe::write_xml_bool_param(inputFile,DICe::separate_output_file_for_each_subset,"false",false);

      DICe::write_xml_bool_param(inputFile,DICe::create_separate_run_info_file,"true",false);

      // TODO add subset file later ...
      //DICe::write_xml_string_param(inputFile,DICe::subset_file,"<path>");

      DICe::write_xml_string_param(inputFile,DICe::reference_image,ref_file_info_.filePath().toStdString(),false);

      DICe::write_xml_param_list_open(inputFile,DICe::deformed_images,false);
      // create an entry for all def images here:
      for(QStringList::iterator it=def_file_list_.begin();it!=def_file_list_.end();++it){
        DICe::write_xml_bool_param(inputFile,it->toStdString(),"true",false);
      }
      DICe::write_xml_param_list_close(inputFile,false);

      DICe::finalize_xml_file(inputFile);
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
  QString working_dir_;

  /// ROIs
  ///
  /// Boundary
  QList<QList <QPoint> > roi_vertex_vectors_;
  /// Excluded
  QList<QList <QPoint> > roi_excluded_vertex_vectors_;

  /// subset size
  int subset_size_;

  /// step size
  int step_size_;

  /* ----------------------- */
  /// singleton pointer
  static Input_Vars * input_vars_ptr_;

};

} // gui namespace

} // DICe namespace

#endif // DICE_INPUTVARS_H
