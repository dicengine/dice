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

  /// return a pointer to the roi vertex vectors
  QList<QList <QPoint> > * boundaryShapes(){
      return & boundaryShapes_;
  }

  /// return a pointer to the roi vertex vectors
  QList<QList <QPoint> > * excludedShapes(){
      return & excludedShapes_;
  }

  /// set the working directory
  void set_working_dir(const QString & dir){
      working_dir_ = dir;
  }

  /// set the inteprolation method
  void set_interpolation_method(const std::string & method){
      interp_method_str_ = method;
  }

  /// set the optimization method
  void set_optimization_method(const std::string & method){
      opt_method_str_ = method;
  }

  /// set the initialization method
  void set_initialization_method(const std::string & method){
      init_method_str_ = method;
  }

  /// set the subset size
  void set_subset_size(const int size){
      subset_size_ = size;
  }

  /// set the step size
  void set_step_size(const int size){
      step_size_ = size;
  }

  /// set the shape functions
  void set_enable_translation(const bool flag){
      enable_translation_ = flag;
  }

  /// set the shape functions
  void set_enable_rotation(const bool flag){
      enable_rotation_ = flag;
  }

  /// set the shape functions
  void set_enable_normal_strain(const bool flag){
      enable_normal_strain_ = flag;
  }

  /// set the shape functions
  void set_enable_shear_strain(const bool flag){
      enable_shear_strain_ = flag;
  }

  /// return the name of the input file
  std::string input_file_name(){
      return inputFile;
  }

  /// set the output fields
  void set_output_fields(std::vector<std::string> & vec){
      output_fields_ = vec;
  }

  /// write the input file
  void write_input_file(){
      std::stringstream input_file_ss;
      std::stringstream params_file_ss;
      std::stringstream subset_file_ss;
      std::stringstream working_dir_ss;
#ifdef WIN32
      working_dir_ss << working_dir_.toStdString() << "\\results\\";
      input_file_ss << working_dir_.toStdString() << "\\" << "input.xml";
      params_file_ss << working_dir_.toStdString() << "\\" << "params.xml";
      subset_file_ss << working_dir_.toStdString() << "\\" << "subset_defs.txt";
#else
      working_dir_ss << working_dir_.toStdString() << "/results/";
      input_file_ss << working_dir_.toStdString() << "/" << "input.xml";
      params_file_ss << working_dir_.toStdString() << "/" << "params.xml";
      subset_file_ss << working_dir_.toStdString() << "/" << "subset_defs.txt";
#endif
      paramsFile = params_file_ss.str();
      inputFile = input_file_ss.str();
      subsetFile = subset_file_ss.str();

      std::cout << "DICe::Input_Vars::instance(): writing input xml file: " << inputFile << std::endl;
      DICe::initialize_xml_file(inputFile);

      DICe::write_xml_comment(inputFile,"Auto generated input file from DICe GUI");

      DICe::write_xml_string_param(inputFile,DICe::output_folder,working_dir_ss.str(),false);
      DICe::write_xml_string_param(inputFile,DICe::image_folder,"",false);
      DICe::write_xml_string_param(inputFile,DICe::correlation_parameters_file,paramsFile,false);

      std::stringstream subsetSizeSS;
      subsetSizeSS << subset_size_;
      DICe::write_xml_size_param(inputFile,DICe::subset_size,subsetSizeSS.str(),false);
      std::stringstream stepSizeSS;
      stepSizeSS << step_size_;
      DICe::write_xml_size_param(inputFile,DICe::step_size,stepSizeSS.str(),false);

      DICe::write_xml_bool_param(inputFile,DICe::separate_output_file_for_each_subset,"false",false);

      DICe::write_xml_bool_param(inputFile,DICe::create_separate_run_info_file,"true",false);

      if(boundaryShapes_.size()>0)
          DICe::write_xml_string_param(inputFile,DICe::subset_file,subsetFile,false);

      DICe::write_xml_string_param(inputFile,DICe::reference_image,ref_file_info_.filePath().toStdString(),false);

      DICe::write_xml_param_list_open(inputFile,DICe::deformed_images,false);
      // create an entry for all def images here:
      for(QStringList::iterator it=def_file_list_.begin();it!=def_file_list_.end();++it){
        DICe::write_xml_bool_param(inputFile,it->toStdString(),"true",false);
      }
      DICe::write_xml_param_list_close(inputFile,false);

      DICe::finalize_xml_file(inputFile);
  }

  /// write the analysis parameters to a file
  void write_params_file(){
      std::stringstream params_file_ss;
#ifdef WIN32
      params_file_ss << working_dir_.toStdString() << "\\" << "params.xml";
#else
      params_file_ss << working_dir_.toStdString() << "/" << "params.xml";
#endif
      paramsFile = params_file_ss.str();
      std::cout << "DICe::Input_Vars::instance(): writing parameters xml file: " << paramsFile << std::endl;
      DICe::initialize_xml_file(paramsFile);
      DICe::write_xml_comment(paramsFile,"Auto generated params file from DICe GUI");

      DICe::write_xml_string_param(paramsFile,DICe::interpolation_method,interp_method_str_,false);
      DICe::write_xml_string_param(paramsFile,DICe::optimization_method,opt_method_str_,false);
      DICe::write_xml_string_param(paramsFile,DICe::initialization_method,init_method_str_,false);

      DICe::write_xml_bool_literal_param(paramsFile,DICe::enable_translation,enable_translation_,false);
      DICe::write_xml_bool_literal_param(paramsFile,DICe::enable_rotation,enable_rotation_,false);
      DICe::write_xml_bool_literal_param(paramsFile,DICe::enable_normal_strain,enable_normal_strain_,false);
      DICe::write_xml_bool_literal_param(paramsFile,DICe::enable_shear_strain,enable_shear_strain_,false);

      // set the default delimiter
      DICe::write_xml_string_param(paramsFile,DICe::output_delimiter,",",false);

      // write the output fields
      write_xml_param_list_open(paramsFile,DICe::output_spec,false);
      for(size_t i=0;i<output_fields_.size();++i){
        write_xml_bool_param(paramsFile,output_fields_[i],"true",false);
      }
      write_xml_param_list_close(paramsFile,false);

      DICe::finalize_xml_file(paramsFile);
  }

  /// write the subset.txt file
  void write_subset_file(){
      if(boundaryShapes_.size()<=0) return;

      std::stringstream subset_file_ss;
#ifdef WIN32
      subset_file_ss << working_dir_.toStdString() << "\\" << "subset_defs.txt";
#else
      subset_file_ss << working_dir_.toStdString() << "/" << "subset_defs.txt";
#endif
      subsetFile = subset_file_ss.str();
      std::cout << "DICe::Input_Vars::instance(): writing subset file: " << subsetFile << std::endl;

      // Write the ROIs to file:
      std::ofstream file;
      file.open(subsetFile.c_str());
      file << "# Auto generated subset file from DICe GUI" << std::endl;
      file << "begin region_of_interest" << std::endl;
      file << "  begin boundary" << std::endl;
      // write all the boundary shapes
      for(QList<QList<QPoint> >::iterator it=boundaryShapes_.begin();it!=boundaryShapes_.end();++it){
          file << "    begin polygon" << std::endl;
          file << "      begin vertices" << std::endl;
          for(QList<QPoint>::iterator pit=it->begin();pit!=it->end();++pit){
              file << "        " << pit->x() << " " << pit->y() << std::endl;
          }
          file << "      end vertices" << std::endl;
          file << "    end polygon" << std::endl;
      }
      file << "  end boundary" << std::endl;
      if(excludedShapes_.size()>0){
          file << "  begin excluded" << std::endl;
          for(QList<QList<QPoint> >::iterator it=excludedShapes_.begin();it!=excludedShapes_.end();++it){
              file << "    begin polygon" << std::endl;
              file << "      begin vertices" << std::endl;
              for(QList<QPoint>::iterator pit=it->begin();pit!=it->end();++pit){
                  file << "        " << pit->x() << " " << pit->y() << std::endl;
              }
              file << "      end vertices" << std::endl;
              file << "    end polygon" << std::endl;
          }
          file << "  end excluded" << std::endl;
      }
      file << "end region_of_interest" << std::endl;
      file.close();
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
  QList<QList <QPoint> > boundaryShapes_;
  /// Excluded
  QList<QList <QPoint> > excludedShapes_;

  /// subset size
  int subset_size_;

  /// step size
  int step_size_;

  /// string parameters
  std::string interp_method_str_;
  std::string init_method_str_;
  std::string opt_method_str_;

  /// bool params
  bool enable_translation_;
  bool enable_normal_strain_;
  bool enable_shear_strain_;
  bool enable_rotation_;

  /// File names
  std::string inputFile;
  std::string paramsFile;
  std::string subsetFile;

  /// output fields
  std::vector<std::string> output_fields_;

  /* ----------------------- */
  /// singleton pointer
  static Input_Vars * input_vars_ptr_;

};

} // gui namespace

} // DICe namespace

#endif // DICE_INPUTVARS_H
