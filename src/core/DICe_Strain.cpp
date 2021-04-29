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
#include <DICe_MeshIOUtils.h>
#include <DICe_Parser.h>
#include <DICe_MeshIO.h>
#include <DICe_Schema.h>
#include <DICe_PostProcessor.h>
#include <DICe_Image.h>

#include <Teuchos_RCP.hpp>

#include <fstream>

using namespace DICe;

int main(int argc, char *argv[]) {
  try{
    DICe::initialize(argc,argv);

    Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);

    print_banner();

    // command line options
    bool use_nonlocal = false;
    int_t num_time_steps = 0;
    std::string output_pre = "DICe_strain_solution";
    *outStream << "number of args:     " << argc << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(argc<3,std::runtime_error,
      "Invalid command line. Syntax: dice_strain source_file_1 ... source_file_n strain_window_size_in_pixels [target_locations_file] [ouput_prefix] [1: for use_nonlocal_formulation]");
    std::vector<std::string> source_files;
    for(int_t i=0;i<argc-1;++i){
      std::string res_file = argv[i+1];
      if(is_number(argv[i+1])) break;
      *outStream << "source file:        " << res_file << std::endl;
      source_files.push_back(res_file);
    }
    std::string target_file = source_files[0];
    const int_t sg_size = std::strtol(argv[1 + source_files.size()],NULL,0);
    *outStream << "strain gauge size:  " << sg_size << " pixels" << std::endl;
    std::string opt_arg_1 = "";
    std::string opt_arg_2 = "";
    std::string opt_arg_3 = "";
    if(argc>(int_t)source_files.size()+2){
      opt_arg_1 = argv[source_files.size()+2];
      if(argc>(int_t)source_files.size()+3){
        opt_arg_2 = argv[source_files.size()+3];
        if(argc>(int_t)source_files.size()+4){
          opt_arg_3 = argv[source_files.size()+4];
        }
      }
      if(is_number(opt_arg_3)){
        use_nonlocal = true;
      }
      if(is_number(opt_arg_1)){
        use_nonlocal = true;
      }else{
        const std::string text_ext(".txt");
        const std::string csv_ext(".csv");
        if(opt_arg_1.find(text_ext)!=std::string::npos||opt_arg_1.find(csv_ext)!=std::string::npos){
          TEUCHOS_TEST_FOR_EXCEPTION(opt_arg_2.find(text_ext)!=std::string::npos||opt_arg_2.find(csv_ext)!=std::string::npos,
            std::runtime_error,"Error can't have .txt in output_prefix");
          target_file = opt_arg_1;
          if(argc>(int_t)source_files.size()+3){
            if(is_number(opt_arg_2)){
              use_nonlocal = true;
            }
            else{
              output_pre = opt_arg_2;
            }
          }
        }
        else{ // its a output prefix
          output_pre = opt_arg_1;
        }
      }
    }
    *outStream << "target file:        " << target_file << std::endl;
    *outStream << "using prefix:       " << output_pre << std::endl;
    // determine the number of time steps
    const std::string exo_ext(".e");
    bool exodus_format = false;
    std::string disp_field_name = "NOT_FOUND";
    if(source_files[0].find(exo_ext)!=std::string::npos){
      num_time_steps = DICe::mesh::read_exodus_num_steps(source_files[0]);
      std::vector<std::string> field_names = DICe::mesh::read_exodus_field_names(source_files[0]);
      for(size_t i=0;i<field_names.size();++i){
        if(field_names[i]=="DISPLACEMENT_X"||field_names[i]=="DISPLACEMENT_Y"){
          disp_field_name = "DISPLACEMENT";
        }
        if(field_names[i]=="SUBSET_DISPLACEMENT_X"||field_names[i]=="SUBSET_DISPLACEMENT_Y"){
          disp_field_name = "SUBSET_DISPLACEMENT";
        }
      }
      *outStream << "source format:      exodus" << std::endl;
      exodus_format = true;
    }else{
      num_time_steps = source_files.size();
      *outStream << "source format:      text" << std::endl;
      // read the first line of the source file to get the field names
      std::fstream dataFile(source_files[0].c_str(), std::ios_base::in);
      TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading file " << source_files[0]);
      std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");
      for(size_t i=0;i<tokens.size();++i){
        if(tokens[i]=="DISPLACEMENT_X"||tokens[i]=="DISPLACEMENT_Y"){
          disp_field_name = "DISPLACEMENT";
        }
        if(tokens[i]=="SUBSET_DISPLACEMENT_X"||tokens[i]=="SUBSET_DISPLACEMENT_Y"){
          disp_field_name = "SUBSET_DISPLACEMENT";
        }
      }
      dataFile.close();
    }
    *outStream <<   "disp field:         " << disp_field_name << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(disp_field_name=="NOT_FOUND",std::runtime_error,"");

    // create an importer to read the files and project the points to the target locations
    Teuchos::RCP<DICe::mesh::Importer_Projector> importer = Teuchos::rcp(new DICe::mesh::Importer_Projector(source_files[0],target_file));
    DEBUG_MSG("DICe_Strain(): number of source points: " << importer->num_source_pts());
    DEBUG_MSG("DICe_Strain(): number of target points: " << importer->num_target_pts());

    // create a schema to manage the results:
    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set(DICe::output_delimiter,",");
    params->set(DICe::omit_output_row_id,true);
    params->set(DICe::sort_txt_output,true);
    params->set(DICe::output_prefix,output_pre);
    Teuchos::RCP<Teuchos::ParameterList> output_sublist = Teuchos::rcp(new Teuchos::ParameterList());
    output_sublist->set("COORDINATE_X",true);
    output_sublist->set("COORDINATE_Y",true);
    output_sublist->set("DISPLACEMENT_X",true);
    output_sublist->set("DISPLACEMENT_Y",true);
    if(use_nonlocal){
      output_sublist->set("NLVC_STRAIN_XX",true);
      output_sublist->set("NLVC_STRAIN_YY",true);
      output_sublist->set("NLVC_STRAIN_XY",true);
    }
    else{
      output_sublist->set("VSG_STRAIN_XX",true);
      output_sublist->set("VSG_STRAIN_YY",true);
      output_sublist->set("VSG_STRAIN_XY",true);
    }
    output_sublist->set("SIGMA",true);
    output_sublist->set("MATCH",true);
    Teuchos::RCP<Teuchos::ParameterList> post_sublist = Teuchos::rcp(new Teuchos::ParameterList());
    if(use_nonlocal){
      post_sublist->set(DICe::horizon_diameter_in_pixels,sg_size);
      params->set(DICe::post_process_nlvc_strain,*post_sublist);
    }
    else{
      post_sublist->set(DICe::strain_window_size_in_pixels,sg_size);
      params->set(DICe::post_process_vsg_strain,*post_sublist);
    }
    params->set(DICe::output_spec,*output_sublist);


    const int_t dummy_subset_size = 25;
    Teuchos::ArrayRCP<scalar_t> target_pts_x(importer->num_target_pts(),0.0);
    Teuchos::ArrayRCP<scalar_t> target_pts_y(importer->num_target_pts(),0.0);
    for(int_t i=0;i<importer->num_target_pts();++i){
      target_pts_x[i] = (*importer->target_pts_x())[i];
      target_pts_y[i] = (*importer->target_pts_y())[i];
    }
    Teuchos::RCP<Image> ref_img = Teuchos::rcp(new Image(100,100,0));
    Teuchos::RCP<Schema> schema = Teuchos::rcp(new Schema(target_pts_x,target_pts_y,dummy_subset_size,Teuchos::null,Teuchos::null,params));
    schema->set_ref_image(ref_img);
    schema->set_def_image(ref_img);
    // initialize the schema with the subset coordinates
    schema->set_frame_range(0,num_time_steps,1);

    for(int_t time_step=0;time_step<num_time_steps;++time_step){
      schema->update_frame_id();

      // fill the displacement values of the schema:
      Teuchos::RCP<MultiField> disp_x = schema->mesh()->get_field(DICe::field_enums::SUBSET_DISPLACEMENT_X_FS);
      Teuchos::RCP<MultiField> disp_y = schema->mesh()->get_field(DICe::field_enums::SUBSET_DISPLACEMENT_Y_FS);
      TEUCHOS_TEST_FOR_EXCEPTION(disp_x->get_map()->get_num_global_elements()!=importer->num_target_pts(),std::runtime_error,"");

      std::vector<scalar_t> proj_disp_x(importer->num_target_pts(),0.0);
      std::vector<scalar_t> proj_disp_y(importer->num_target_pts(),0.0);

      if(exodus_format)
        importer->import_vector_field(source_files[0],disp_field_name,proj_disp_x,proj_disp_y,time_step);
      else
        importer->import_vector_field(source_files[time_step],disp_field_name,proj_disp_x,proj_disp_y);

      for(int_t i=0;i<importer->num_target_pts();++i){
        disp_x->local_value(i) = proj_disp_x[i];
        disp_y->local_value(i) = proj_disp_y[i];
      }
      //disp_x->describe();

      //schema->mesh()->print_field_stats();

      // FIXME: assumes post processor is of length 1 and the first entry is the VSG_STRAIN post processor
      if(time_step==0) (*schema->post_processors())[0]->pre_execution_tasks();
      (*schema->post_processors())[0]->execute(ref_img,ref_img);

      schema->write_output("./",output_pre,false,true);

      schema->mesh()->print_field_stats();
    }

    DICe::finalize();
  }
  catch(std::exception & e){
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
