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
#include <DICe_Parser.h>
#include <DICe_Image.h>
#include <DICe_ImageIO.h>
#include <DICe_Schema.h>
#include <DICe_Triangulation.h>

#include <boost/timer.hpp>

#if DICE_MPI
#  include <mpi.h>
#endif

using namespace DICe;

int main(int argc, char *argv[]) {
  boost::timer total_t;
  try{
    DICe::initialize(argc,argv);

    //const int_t dim = 2;         // Assumes 2D images
    int_t proc_size = 1;
    int_t proc_rank = 0;
#if DICE_MPI
    MPI_Comm_size(MPI_COMM_WORLD,&proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

    // Command line options

    if(proc_rank==0)  DEBUG_MSG("Parsing command line options");
    Teuchos::RCP<std::ostream> outStream;
    bool force_exit = false;
    Teuchos::RCP<Teuchos::ParameterList> input_params = DICe::parse_command_line(argc,argv,force_exit,outStream);
    if(force_exit){
      // exit gracefully:
      DICe::finalize();
      if(input_params->isParameter("debug_msg_on")){
        if(input_params->get<bool>("debug_msg_on")){
          return 1;
        }
      }
      return 0;
    }
    *outStream << "Input Parameters: " << std::endl;
    input_params->print(*outStream);
    *outStream << "\n--- Input read successfully ---\n" << std::endl;

    // correlation parameters

    bool is_error_est_run = false;
    Teuchos::RCP<Teuchos::ParameterList> correlation_params;
    if(input_params->isParameter(DICe::correlation_parameters_file)){
      const std::string paramsFileName = input_params->get<std::string>(DICe::correlation_parameters_file);
      correlation_params = DICe::read_correlation_params(paramsFileName);
      *outStream << "User specified correlation Parameters: " << std::endl;
      correlation_params->print(*outStream);
      is_error_est_run = correlation_params->get<bool>(DICe::estimate_resolution_error,false);
      if(is_error_est_run){
        // force the computing of the image laplacian for the reference image:
        correlation_params->set(DICe::compute_laplacian_image,true);
      }
      *outStream << "\n--- Correlation parameters read successfully ---\n" << std::endl;
    }
    else{
      *outStream << "Correlation parameters not specified by user" << std::endl;
    }

    // if the mesh size was specified in the input params set the use_global_dic flag
    if(input_params->isParameter(DICe::mesh_size)){
#ifdef DICE_ENABLE_GLOBAL
      if(correlation_params==Teuchos::null) correlation_params = Teuchos::rcp(new Teuchos::ParameterList());
      correlation_params->set(DICe::use_global_dic,true);
#else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, global method requested, but code"
          " was not built with DICE_ENABLE_GLOBAL");
#endif
    }

    // decipher the image file names (note: zero entry is the reference image):

    std::vector<std::string> image_files;
    std::vector<std::string> stereo_image_files;
    DICe::decipher_image_file_names(input_params,image_files,stereo_image_files);
    const bool is_stereo = stereo_image_files.size() > 0;

    const int_t num_frames = image_files.size()-1;
    int_t first_frame_id = 0;
    int_t image_width = 0;
    int_t image_height = 0;
    const bool is_cine = utils::image_file_type(image_files[0].c_str()) == CINE;
    bool filter_failed_pixels = false;
    if(is_cine){
      int_t s_id = 0, e_id = 0;
      bool is_avg = false;
      utils::cine_index(image_files[0].c_str(),s_id,e_id,is_avg);
      first_frame_id = s_id;
      filter_failed_pixels = correlation_params->get<bool>(DICe::filter_failed_cine_pixels,false);
      utils::Image_Reader_Cache::instance().set_filter_failed_pixels(filter_failed_pixels);
    }
    else  // non-cine input
    {
      if(input_params->isParameter(DICe::start_image_index)){
        first_frame_id = input_params->get<int_t>(DICe::start_image_index);
      }else if(input_params->isParameter(DICe::reference_image_index)){
        first_frame_id = input_params->get<int_t>(DICe::reference_image_index);
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(num_frames<=0,std::runtime_error,"");
    *outStream << "Reference image: " << image_files[0] << std::endl;
    for(int_t i=1;i<=num_frames;++i){
      if(i==10&&num_frames!=10) *outStream << "..." << std::endl;
      else if(i>10&&i<num_frames) continue;
      else
        *outStream << "Deformed image: " << image_files[i] << std::endl;
    }
    *outStream << "\n--- List of images constructed successfuly ---\n" << std::endl;

    // get width and heigh of reference image to use in setting up the subets
    utils::read_image_dimensions(image_files[0].c_str(),image_width,image_height);
    *outStream << "Image dimensions: " << image_width << " x " << image_height << std::endl;

    // set up output files
    std::string output_folder = input_params->get<std::string>(DICe::output_folder);
    const bool separate_output_file_for_each_subset = input_params->get<bool>(DICe::separate_output_file_for_each_subset,false);
    if(separate_output_file_for_each_subset){
      *outStream << "Output will be written to separate output files for each subset" << std::endl;
    }
    else{
      *outStream << "Output will be written to one file per frame with all subsets included" << std::endl;
    }
    const bool separate_header_file = input_params->get<bool>(DICe::create_separate_run_info_file,false);
    if(separate_header_file){
      *outStream << "Execution information will be written to a separate file (not placed in the output headers)" << std::endl;
    }

    // create schemas:
    Teuchos::RCP<DICe::Schema> schema = Teuchos::rcp(new DICe::Schema(input_params,correlation_params));
    Teuchos::RCP<DICe::Schema> stereo_schema;
    // let the schema know how many images there are in the sequence and the first frame id:
    schema->set_frame_range(first_frame_id,num_frames);

    if(input_params->get<bool>(DICe::print_subset_locations_and_exit,false)){
      // write out the subset locations for left camera and exit
      Teuchos::RCP<MultiField> coords = schema->mesh()->get_field(DICe::field_enums::INITIAL_COORDINATES_FS);
      const int_t ss_locs_size = proc_rank==0 ?
          coords->get_map()->get_num_global_elements() : 0; // both x and y coords
      Teuchos::Array<int_t> owned_ids(ss_locs_size);
      for(int_t i=0;i<ss_locs_size;++i)
          owned_ids[i] = i;
      Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, owned_ids,0,*schema->mesh()->get_comm()));
      Teuchos::RCP<MultiField> all_on_zero_coords = Teuchos::rcp( new MultiField(zero_map,1,true));
      // all to zero gather of coordinates
      MultiField_Exporter exporter(*zero_map,*coords->get_map());
      // export the field to zero
      all_on_zero_coords->do_import(coords,exporter);
      //all_on_zero_coords->describe();
      if(proc_rank==0){
        std::FILE * ssFile = fopen("subset_locs.txt","w");
        for(int_t i=0;i<ss_locs_size/2;++i){
          fprintf(ssFile,"%i %i\n",(int_t)all_on_zero_coords->local_value(i*2+0),(int_t)all_on_zero_coords->local_value(i*2+1));
        }
        fclose(ssFile);
      }
      DICe::finalize();
      return 0;
    }

    if(schema->analysis_type()==LOCAL_DIC){
      *outStream << "Number of global subsets: " << schema->global_num_subsets() << std::endl;
      for(int_t i=0;i<schema->local_num_subsets();++i){
        if(i==10&&schema->local_num_subsets()!=11) *outStream << "..." << std::endl;
        else if(i>10&&i<schema->local_num_subsets()-1) continue;
        else
          *outStream << "Proc 0: subset global id: " << schema->subset_global_id(i) << " global coordinates (" << schema->local_field_value(i,DICe::field_enums::SUBSET_COORDINATES_X_FS) <<
          "," << schema->local_field_value(i,DICe::field_enums::SUBSET_COORDINATES_Y_FS) << ")" << std::endl;
      }
      *outStream << std::endl;
    } // end local dic
    else if(schema->analysis_type()==GLOBAL_DIC){
#ifdef DICE_ENABLE_GLOBAL
      *outStream << "Global formulation mesh stats:" << std::endl;
      *outStream << "Mesh size:          " << schema->global_algorithm()->mesh_size() << std::endl;
      *outStream << "Number of nodes:    " << schema->global_algorithm()->mesh()->num_nodes() << std::endl;
      *outStream << "Number of elements: " << schema->global_algorithm()->mesh()->num_elem() << std::endl;
#endif
    }  // end global dic
    std::string file_prefix = input_params->get<std::string>(DICe::output_prefix,"DICe_solution");
    std::string stereo_file_prefix = input_params->get<std::string>(DICe::output_prefix,"DICe_solution");
    stereo_file_prefix += "_stereo";

    // if the user selects predict_resolution_error option, an error analysis is performed and the actual analysis is skipped
    if(is_error_est_run){
      std::string resolution_output_folder;
#if defined(WIN32)
      resolution_output_folder = ".\\";
#else
      resolution_output_folder + "./";
#endif
      if(input_params->isParameter(DICe::resolution_output_folder))
        resolution_output_folder = input_params->get<std::string>(DICe::resolution_output_folder);
      file_prefix = "DICe_error_estimation_solution";
      if(input_params->isParameter(DICe::output_prefix)){
        if(input_params->get<std::string>(DICe::output_prefix)!="DICe_solution")
          file_prefix = input_params->get<std::string>(DICe::output_prefix);
      }
      schema->update_extents();
      schema->set_ref_image(image_files[0]);
      schema->set_def_image(image_files[0]);
      schema->estimate_resolution_error(correlation_params,output_folder,resolution_output_folder,file_prefix,outStream);
      DICe::finalize();
      return 0;
    }

    TEUCHOS_TEST_FOR_EXCEPTION(is_stereo&&!input_params->isParameter(DICe::calibration_parameters_file),std::runtime_error,
      "Error, calibration_parameters_file required for stereo");
    Teuchos::RCP<DICe::Triangulation> triangulation;
    if(input_params->isParameter(DICe::calibration_parameters_file)){
      const std::string cal_file_name = input_params->get<std::string>(DICe::calibration_parameters_file);
      triangulation = Teuchos::rcp(new DICe::Triangulation(cal_file_name));
      *outStream << "\n--- Calibration parameters read successfully ---\n" << std::endl;
    }
    else{
      *outStream << "Calibration parameters not specified by user" << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(is_stereo&&triangulation==Teuchos::null,std::runtime_error,
      "Error, triangulation should be instantiated at this point");

    // if this is a stereo analysis do the initial cross correlation:
    scalar_t cross_corr_time = 0.0;
    if(is_stereo){
      boost::timer t;
      TEUCHOS_TEST_FOR_EXCEPTION(schema->analysis_type()==GLOBAL_DIC,std::runtime_error,"Error, global stereo not enabled yet");
      *outStream << "Processing cross correlation between left and right images" << std::endl;
      schema->initialize_cross_correlation(triangulation,input_params); // images don't need to be loaded by here they get loaded in this routine based on the input params
      schema->update_extents(true);
      schema->set_ref_image(image_files[0]);
      schema->set_def_image(stereo_image_files[0]);
      if(schema->use_nonlinear_projection()){
        schema->project_right_image_into_left_frame(triangulation,false);
      }
      schema->execute_cross_correlation();
      schema->save_cross_correlation_fields();
      stereo_schema = Teuchos::rcp(new DICe::Schema(input_params,correlation_params,schema));
      stereo_schema->update_extents();
      stereo_schema->set_ref_image(stereo_image_files[0]);
      assert(stereo_schema!=Teuchos::null);
      //if(stereo_schema->use_nonlinear_projection())
      //  stereo_schema->project_right_image_into_left_frame(triangulation,true);
      stereo_schema->set_frame_range(first_frame_id,num_frames);
      // go ahead and set up the model coordinates field
      schema->execute_triangulation(triangulation,stereo_schema);
      cross_corr_time = t.elapsed();
    } // end is stereo
    else{ // only the ref image needs to be set
      schema->update_extents();
      schema->set_ref_image(image_files[0]);
    }

    // iterate through the images and perform the correlation:
    scalar_t corr_time = 0.0;
    scalar_t elapsed_time = 0.0;
    scalar_t write_time = 0.0;
    scalar_t max_time = 0.0;
    scalar_t min_time = 1.0E10;
    scalar_t avg_time = 0.0;
    bool failed_step = false;

    for(int_t image_it=1;image_it<=num_frames;++image_it){
      *outStream << "Processing frame: " << image_it << " of " << num_frames << ", " << image_files[image_it] << std::endl;
      if(schema->use_incremental_formulation()&&image_it>1){
        schema->set_ref_image(schema->def_img());
      }
      schema->update_extents();
      schema->set_def_image(image_files[image_it]);
      if(is_stereo){
        if(stereo_schema->use_incremental_formulation()&&image_it>1){
          stereo_schema->set_ref_image(stereo_schema->def_img());
        }
        stereo_schema->update_extents();
        stereo_schema->set_def_image(stereo_image_files[image_it]);
        //if(stereo_schema->use_nonlinear_projection())
        //  stereo_schema->project_right_image_into_left_frame(triangulation,false);
      }
      { // start the timer
        boost::timer t;

        int_t corr_error = schema->execute_correlation();
        if(corr_error)
          failed_step = true;
        if(is_stereo){
          corr_error = stereo_schema->execute_correlation();
          if(corr_error)
            failed_step = true;
          schema->execute_triangulation(triangulation,stereo_schema);
        }
        schema->execute_post_processors();
        // timing info
        elapsed_time = t.elapsed();
        if(elapsed_time>max_time)max_time = elapsed_time;
        if(elapsed_time<min_time)min_time = elapsed_time;
        corr_time += elapsed_time;
      }

      // write the output
      boost::timer write_t;
      const bool no_text_output = input_params->get<bool>(DICe::no_text_output_files,false);
      schema->write_output(output_folder,file_prefix,separate_output_file_for_each_subset,separate_header_file,no_text_output);
      schema->post_execution_tasks();
      // print the timing data with or without verbose flag
      if(input_params->get<bool>(DICe::print_stats,false)){
        schema->mesh()->print_field_stats();
      }
      //if(subset_info->conformal_area_defs!=Teuchos::null&&image_it==1){
      //  schema->write_control_points_image("RegionOfInterest");
      //}
      if(is_stereo){
        if(input_params->get<bool>(DICe::output_stereo_files,false)){
          stereo_schema->write_output(output_folder,stereo_file_prefix,separate_output_file_for_each_subset,separate_header_file,no_text_output);
        }
        stereo_schema->post_execution_tasks();
      }
      write_time = write_t.elapsed();
    } // image loop

    schema->write_stats(output_folder,file_prefix);
    if(is_stereo)
      stereo_schema->write_stats(output_folder,stereo_file_prefix);

    avg_time = corr_time / num_frames;

    if(failed_step)
      *outStream << "\n--- Failed Step Occurred ---\n" << std::endl;
    else
      *outStream << "\n--- Successful Completion ---\n" << std::endl;

    scalar_t total_time = total_t.elapsed();

    // output timing
    // print the timing data with or without verbose flag
    if(input_params->get<bool>(DICe::print_timing,false)){
      std::cout << "Total time:                " << total_time << std::endl;
      std::cout << "Cross correlation time:    " << cross_corr_time << std::endl;
      std::cout << "Correlation time:          " << corr_time << std::endl;
      std::cout << "  Avgerage time per image: " << avg_time << std::endl;
      std::cout << "  Max time per image:      " << max_time << std::endl;
      std::cout << "  Min time per image:      " << min_time << std::endl;
      std::cout << "Write output time:         " << write_time << std::endl;
    }
    //  write the time output to file:
    std::stringstream timeFileName;
    timeFileName << output_folder << "timing."<< proc_size << "." << proc_rank << ".txt";
    std::FILE * timeFile = fopen(timeFileName.str().c_str(),"w");
    fprintf(timeFile,"TOTAL CROSS_CORR CORR AVERAGE_PER_IMAGE MAX_PER_IMAGE MIN_PER_IMAGE WRITE\n");
    fprintf(timeFile,"%4.4E %4.4E %4.4E %4.4E %4.4E %4.4E %4.4E\n",total_time,cross_corr_time,corr_time,avg_time,max_time,min_time,write_time);
    fclose(timeFile);

    DICe::finalize();
  }
  catch(std::exception & e){
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
