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

#include <DICe.h>
#include <DICe_Parser.h>
#include <DICe_Image.h>
#include <DICe_ImageIO.h>
#include <DICe_Schema.h>
#include <DICe_Cine.h>
#include <DICe_Triangulation.h>

#include <boost/timer.hpp>

#if DICE_MPI
#  include <mpi.h>
#endif

using namespace DICe;

int main(int argc, char *argv[]) {
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
      return 0;
    }
    *outStream << "Input Parameters: " << std::endl;
    input_params->print(*outStream);
    *outStream << "\n--- Input read successfully ---\n" << std::endl;

    // correlation parameters

    Teuchos::RCP<Teuchos::ParameterList> correlation_params;
    if(input_params->isParameter(DICe::correlation_parameters_file)){
      const std::string paramsFileName = input_params->get<std::string>(DICe::correlation_parameters_file);
      correlation_params = DICe::read_correlation_params(paramsFileName);
      *outStream << "User specified correlation Parameters: " << std::endl;
      correlation_params->print(*outStream);
      *outStream << "\n--- Correlation parameters read successfully ---\n" << std::endl;
    }
    else{
      *outStream << "Correlation parameters not specified by user" << std::endl;
    }
    Teuchos::RCP<DICe::Triangulation> triangulation;
    if(input_params->isParameter(DICe::calibration_parameters_file)){
      const std::string cal_file_name = input_params->get<std::string>(DICe::calibration_parameters_file);
      triangulation = Teuchos::rcp(new DICe::Triangulation(cal_file_name));
      *outStream << "\n--- Calibration parameters read successfully ---\n" << std::endl;
    }
    else{
      *outStream << "Calibration parameters not specified by user" << std::endl;
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
    if(is_stereo){
      TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::calibration_parameters_file),std::runtime_error,
        "Error, calibration_parameters_file required for stereo");
      TEUCHOS_TEST_FOR_EXCEPTION(triangulation==Teuchos::null,std::runtime_error,
        "Error, triangulation should be instantiated at this point");
    }
    int_t num_images = 0;
    int_t cine_ref_index = -1;
    int_t cine_start_index = -1;
    int_t cine_end_index = -1;
    int_t image_width = 0;
    int_t image_height = 0;
    int_t first_frame_index = 1;
    bool is_cine = false;
    bool filter_failed_pixels = false;
    Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader;
    Teuchos::RCP<DICe::cine::Cine_Reader> stereo_cine_reader;
    if(image_files[0]==DICe::cine_file){
      is_cine = true;
      // read the file_name from the input_parasm
      TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::cine_file),std::runtime_error,
        "Error, the file name of the cine file has not been specified");
      std::string cine_file_name = input_params->get<std::string>(DICe::cine_file);
      std::string stereo_cine_file_name;
      if(is_stereo){
        TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::stereo_cine_file),std::runtime_error,
          "Error, the file name of the cine file has not been specified");
        stereo_cine_file_name = input_params->get<std::string>(DICe::stereo_cine_file);
      }
      // add the directory to the name:
      std::stringstream cine_name;
      std::stringstream stereo_cine_name;
      cine_name << input_params->get<std::string>(DICe::image_folder) << cine_file_name;
      if(is_stereo)
        *outStream << "stereo left cine file name: " << cine_name.str() << std::endl;
      else
        *outStream << "cine file name: " << cine_name.str() << std::endl;
      stereo_cine_name << input_params->get<std::string>(DICe::image_folder) << stereo_cine_file_name;
      if(is_stereo)
      *outStream << "stereo right cine file name: " << stereo_cine_name.str() << std::endl;
      // read the cine header info:
      filter_failed_pixels = correlation_params->get<bool>(DICe::filter_failed_cine_pixels,false);
      cine_reader = Teuchos::rcp(new DICe::cine::Cine_Reader(cine_name.str(),outStream.getRawPtr(),filter_failed_pixels));
      if(is_stereo)
        stereo_cine_reader = Teuchos::rcp(new DICe::cine::Cine_Reader(stereo_cine_name.str(),outStream.getRawPtr(),filter_failed_pixels));
      // read the image data for a frame
      num_images = cine_reader->num_frames();
      image_width = cine_reader->width();
      image_height = cine_reader->height();
      first_frame_index = cine_reader->first_image_number();
      *outStream << "number of frames in cine file: " << num_images << std::endl;
      //TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::cine_ref_index),std::runtime_error,
      //  "Error, the reference index for the cine file has not been specified");
      cine_ref_index = input_params->get<int_t>(DICe::cine_ref_index,first_frame_index);
      *outStream << "cine ref index: " << cine_ref_index << std::endl;
      cine_start_index = input_params->get<int_t>(DICe::cine_start_index,first_frame_index);
      *outStream << "cine start index: " << cine_start_index << std::endl;
      cine_end_index = input_params->get<int_t>(DICe::cine_end_index,first_frame_index + num_images -1);
      *outStream << "cine end index: " << cine_end_index << std::endl;
      num_images = cine_end_index - cine_start_index + 1;
      *outStream << "number of frames to analyze: " << num_images << std::endl;

      // sanity checks
      if(is_stereo){
        TEUCHOS_TEST_FOR_EXCEPTION(cine_reader->num_frames()!=stereo_cine_reader->num_frames(),std::runtime_error,
          "Error, incompatible left and right cine files, num frames left " << cine_reader->num_frames() << " num frames right " << stereo_cine_reader->num_frames());
        TEUCHOS_TEST_FOR_EXCEPTION(image_width!=stereo_cine_reader->width(),std::runtime_error,
          "Error, incompatible left and right cine files, image width left " << image_width << " image width right " << stereo_cine_reader->width());
        TEUCHOS_TEST_FOR_EXCEPTION(image_height!=stereo_cine_reader->height(),std::runtime_error,
          "Error, incompatible left and right cine files, image height left " << image_height << " image height right " << stereo_cine_reader->height());
      }
      TEUCHOS_TEST_FOR_EXCEPTION(cine_start_index > cine_end_index,std::invalid_argument,"Error, the cine start index is > the cine end index");
      TEUCHOS_TEST_FOR_EXCEPTION(cine_start_index < first_frame_index,std::invalid_argument,"Error, the cine start index is < the first frame index");
      TEUCHOS_TEST_FOR_EXCEPTION(cine_ref_index > cine_end_index,std::invalid_argument,"Error, the cine ref index is > the cine end index");
      TEUCHOS_TEST_FOR_EXCEPTION(cine_ref_index < first_frame_index,std::invalid_argument,"Error, the cine ref index is < the first frame index");
      TEUCHOS_TEST_FOR_EXCEPTION(cine_end_index < cine_start_index,std::invalid_argument,"Error, the cine end index is < the cine start index");
      TEUCHOS_TEST_FOR_EXCEPTION(cine_end_index < cine_ref_index,std::invalid_argument,"Error, the cine end index is < the ref index");

      // convert the cine ref, start and end index to the DICe indexing, not cine indexing
      cine_start_index = cine_start_index - first_frame_index;
      cine_ref_index = cine_ref_index - first_frame_index;
      cine_end_index = cine_end_index - first_frame_index;

      *outStream << "\n--- Cine file information read successfuly ---\n" << std::endl;
    }
    else  // non-cine input
    {
      num_images = image_files.size() - 1; // the first file is the reference image
      first_frame_index = input_params->get<int_t>(DICe::reference_image_index,0)+1;
      TEUCHOS_TEST_FOR_EXCEPTION(num_images<=0,std::runtime_error,"");
      *outStream << "Reference image: " << image_files[0] << std::endl;
      for(int_t i=1;i<=num_images;++i){
        if(i==10&&num_images!=10) *outStream << "..." << std::endl;
        else if(i>10&&i<num_images) continue;
        else
          *outStream << "Deformed image: " << image_files[i] << std::endl;
      }
      *outStream << "\n--- List of images constructed successfuly ---\n" << std::endl;
      // get width and heigh of reference image to use in setting up the subets
      utils::read_image_dimensions(image_files[0].c_str(),image_width,image_height);
    }  // end non-cine input
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

    // create a schema:
    Teuchos::RCP<DICe::Schema> schema;
    if(is_cine){
      // read in the reference image from the cine file and create the schema:
      Teuchos::RCP<DICe::Image> ref_image;
      if(input_params->isParameter(DICe::time_average_cine_ref_frame)){
        const int_t num_avg_frames = input_params->get<int_t>(DICe::time_average_cine_ref_frame,1);
        ref_image = cine_reader->get_average_frame(cine_ref_index,cine_ref_index+num_avg_frames,true,filter_failed_pixels,correlation_params);
      }
      else{
        ref_image = cine_reader->get_frame(cine_ref_index,true,filter_failed_pixels,correlation_params);
      }
      schema = Teuchos::rcp(new DICe::Schema(ref_image,ref_image,correlation_params));
    }
    else{
      const std::string ref_image_string = image_files[0];
      schema = Teuchos::rcp(new DICe::Schema(ref_image_string,ref_image_string,correlation_params));
    }

    schema->set_first_frame_index(cine_start_index + first_frame_index);

    schema->initialize(input_params);

    bool has_motion_windows = false;
    if(schema->analysis_type()==LOCAL_DIC){
      has_motion_windows = schema->motion_window_params()->size()>0;
      *outStream << "Number of global subsets: " << schema->global_num_subsets() << std::endl;
      for(int_t i=0;i<schema->local_num_subsets();++i){
        if(i==10&&schema->local_num_subsets()!=11) *outStream << "..." << std::endl;
        else if(i>10&&i<schema->local_num_subsets()-1) continue;
        else
          *outStream << "Proc 0: subset global id: " << schema->subset_global_id(i) << " global coordinates (" << schema->local_field_value(i,COORDINATE_X) <<
          "," << schema->local_field_value(i,COORDINATE_Y) << ")" << std::endl;
      }
      *outStream << std::endl;
    }
    else if(schema->analysis_type()==GLOBAL_DIC){
#ifdef DICE_ENABLE_GLOBAL
      *outStream << "Using qaudratic tri 6 elements" << std::endl;
      *outStream << "Mesh size:          " << schema->global_algorithm()->mesh_size() << std::endl;
      *outStream << "Number of nodes:    " << schema->global_algorithm()->mesh()->num_nodes() << std::endl;
      *outStream << "Number of elements: " << schema->global_algorithm()->mesh()->num_elem() << std::endl;
#endif
    }

    // let the schema know how many images there are in the sequence:
    schema->set_num_image_frames(num_images);
    std::string file_prefix = input_params->get<std::string>(DICe::output_prefix,"DICe_solution");

    // if the user selects predict_resolution_error option, an error analysis is performed and the actual analysis is skipped
    if(correlation_params->get<bool>(DICe::estimate_resolution_error,false)){
      file_prefix = "DICe_error_estimation_solution";
      const scalar_t min_period = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_min_period,25);
      const scalar_t max_period = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_max_period,100);
      const scalar_t period_factor = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_period_factor,0.5);
      const scalar_t min_amp = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_min_amplitude,0.5);
      const scalar_t max_amp = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_max_amplitude,4.0);
      const scalar_t amp_step = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_amplitude_step,0.5);
      const scalar_t speckle_size = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_speckle_size,-1.0);
      const scalar_t noise_percent = correlation_params->get<scalar_t>(DICe::estimate_resolution_error_noise_percent,-1.0);
      schema->estimate_resolution_error(speckle_size,noise_percent,min_period,max_period,period_factor,min_amp,max_amp,amp_step,output_folder,file_prefix,outStream);
      DICe::finalize();
      return 0;
    }

    // iterate through the images and perform the correlation:
    scalar_t total_time = 0.0;
    scalar_t elapsed_time = 0.0;
    scalar_t max_time = 0.0;
    scalar_t min_time = 1.0E10;
    scalar_t avg_time = 0.0;
    bool failed_step = false;
    // TODO find a more straightforward way to do the indexing
    const int_t start_frame = cine_start_index==-1 ? 1 : cine_start_index;
    const int_t end_frame = cine_end_index==-1 ? num_images : cine_end_index;
    for(int_t image_it=start_frame;image_it<=end_frame;++image_it){
      if(is_cine){
        *outStream << "Processing Image: " << image_it - start_frame + 1 << " of " << num_images << " frame id: " << first_frame_index + image_it << std::endl;
        if(has_motion_windows&&schema->analysis_type()!=GLOBAL_DIC){
          std::map<int_t,Motion_Window_Params>::iterator map_it = schema->motion_window_params()->begin();
          for(;map_it!=schema->motion_window_params()->end();++map_it){
            if(schema->subset_local_id(map_it->first)<0) continue;
            DEBUG_MSG("[PROC " << proc_rank << "] Reading motion window for subset " << map_it->first);
            if(map_it->second.use_subset_id_!=-1&&schema->subset_local_id(map_it->second.use_subset_id_)>=0) continue;
            const int_t use_subset_id = map_it->second.use_subset_id_!=-1 ? map_it->second.use_subset_id_ : map_it->first;
            const int_t sub_image_id = map_it->second.sub_image_id_;
            DEBUG_MSG("[PROC " << proc_rank << "] Reading motion window sub_image " << sub_image_id);
            const int_t start_x = schema->motion_window_params()->find(use_subset_id)->second.start_x_;
            const int_t end_x = schema->motion_window_params()->find(use_subset_id)->second.end_x_;
            const int_t start_y = schema->motion_window_params()->find(use_subset_id)->second.start_y_;
            const int_t end_y = schema->motion_window_params()->find(use_subset_id)->second.end_y_;
            Teuchos::RCP<DICe::Image> def_img = cine_reader->get_frame(image_it,start_x,start_y,end_x,end_y,true,filter_failed_pixels,correlation_params);
            schema->set_def_image(def_img,sub_image_id);
            if(image_it==start_frame){ // initially populate the previous frame
              schema->set_prev_image(def_img,sub_image_id);
            }
          }
        } // end has_motion_windows
        else{
          Teuchos::RCP<DICe::Image> def_image = cine_reader->get_frame(image_it,true,filter_failed_pixels,correlation_params);
          if(schema->use_incremental_formulation()){
            schema->set_ref_image(schema->def_img());
          }
          schema->set_def_image(def_image);
        }
      } // end is_cine
      else{
        const std::string def_image_string = image_files[image_it];
        *outStream << "Processing Image: " << image_it << " of " << num_images << ", " << def_image_string << std::endl;
        if(schema->use_incremental_formulation()){
          schema->set_ref_image(schema->def_img());
        }
        schema->set_def_image(def_image_string);
      }
      { // start the timer
        boost::timer t;

        int_t corr_error = schema->execute_correlation();
        if(corr_error)
          failed_step = true;

        // timing info
        elapsed_time = t.elapsed();
        if(elapsed_time>max_time)max_time = elapsed_time;
        if(elapsed_time<min_time)min_time = elapsed_time;
        total_time += elapsed_time;
      }

      // write the output
      schema->write_output(output_folder,file_prefix,separate_output_file_for_each_subset,separate_header_file);
      //if(subset_info->conformal_area_defs!=Teuchos::null&&image_it==1){
      //  schema->write_control_points_image("RegionOfInterest");
      //}
      schema->post_execution_tasks();

      // print the timing data with or without verbose flag
      if(input_params->get<bool>(DICe::print_stats,false)){
        schema->mesh()->print_field_stats();
      }
    } // image loop

    schema->write_stats(output_folder,file_prefix);

    avg_time = total_time / num_images;

    if(failed_step)
      *outStream << "\n--- Failed Step Occurred ---\n" << std::endl;
    else
      *outStream << "\n--- Successful Completion ---\n" << std::endl;

    // output timing

    // print the timing data with or without verbose flag
    if(input_params->get<bool>(DICe::print_timing,false)){
      std::cout << "Total time:              " << total_time << std::endl;
      std::cout << "Avgerage time per image: " << avg_time << std::endl;
      std::cout << "Max time per image:      " << max_time << std::endl;
      std::cout << "Min time per image:      " << min_time << std::endl;
    }
    //  write the time output to file:
    std::stringstream timeFileName;
    timeFileName << output_folder << "timing."<< proc_size << "." << proc_rank << ".txt";
    std::FILE * timeFile = fopen(timeFileName.str().c_str(),"w");
    fprintf(timeFile,"TOTAL AVERAGE_PER_IMAGE MAX_PER_IMAGE MIN_PER_IMAGE\n");
    fprintf(timeFile,"%4.4E %4.4E %4.4E %4.4E\n",total_time,avg_time,max_time,min_time);
    fclose(timeFile);

    DICe::finalize();
  }
  catch(std::exception & e){
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
