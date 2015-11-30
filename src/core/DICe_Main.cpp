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
#include <DICe_Schema.h>
#include <DICe_Cine.h>
#include <DICe_Tiff.h>

#include <boost/timer.hpp>

#if DICE_MPI
#  include <mpi.h>
#endif

using namespace DICe;

int main(int argc, char *argv[]) {

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
  Teuchos::RCP<Teuchos::ParameterList> input_params = DICe::parse_command_line(argc,argv,outStream);
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

  // decypher the image file names (note: zero entry is the reference image):

  // TODO some error checking to prevent the wrong image type (jpg, text, ...)
  std::vector<std::string> image_files = DICe::decypher_image_file_names(input_params);

  int_t num_images = 0;
  int_t cine_ref_index = -1;
  int_t cine_start_index = -1;
  int_t cine_end_index = -1;
  int_t image_width = 0;
  int_t image_height = 0;
  bool is_cine = false;
  Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader;
  if(image_files[0]==DICe::cine_file){
    is_cine = true;
    // read the file_name from the input_parasm
    assert(input_params->isParameter(DICe::cine_file));
    std::string cine_file_name = input_params->get<std::string>(DICe::cine_file);
    *outStream << "cine file name: " << cine_file_name << std::endl;
    // add the directory to the name:
    std::stringstream cine_name;
    cine_name << input_params->get<std::string>(DICe::image_folder) << cine_file_name;
    *outStream << "cine file name: " << cine_name.str() << std::endl;
    // read the cine header info:
    cine_reader = Teuchos::rcp(new DICe::cine::Cine_Reader(cine_name.str(),outStream.getRawPtr()));
    // read the image data for a frame
    num_images = cine_reader->num_frames();
    image_width = cine_reader->width();
    image_height = cine_reader->height();
    *outStream << "number of frames in cine file: " << num_images << std::endl;
    assert(input_params->isParameter(DICe::cine_ref_index));
    cine_ref_index = input_params->get<int_t>(DICe::cine_ref_index);
    *outStream << "cine ref index: " << cine_ref_index << std::endl;
    cine_start_index = input_params->get<int_t>(DICe::cine_start_index,cine_ref_index);
    *outStream << "cine start index: " << cine_start_index << std::endl;
    cine_end_index = input_params->get<int_t>(DICe::cine_end_index,num_images-2);
    num_images = cine_end_index - cine_start_index + 1;
    *outStream << "cine end index: " << cine_end_index << std::endl;
    *outStream << "\n--- Cine file information read successfuly ---\n" << std::endl;
  }
  else
  {
    num_images = image_files.size() - 1; // the first file is the reference image
    assert(num_images>0);
    *outStream << "Reference image: " << image_files[0] << std::endl;
    for(int_t i=1;i<=num_images;++i){
      if(i==10&&num_images!=10) *outStream << "..." << std::endl;
      else if(i>10&&i<num_images) continue;
      else
        *outStream << "Deformed image: " << image_files[i] << std::endl;
    }
    *outStream << "\n--- List of images constructed successfuly ---\n" << std::endl;
    // get width and heigh of reference image to use in setting up the subets
    utils::read_tiff_image_dimensions(image_files[0].c_str(),image_width,image_height);
  }
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

  // create a schema:
  Teuchos::RCP<DICe::Schema> schema;
  if(is_cine){
    // read in the reference image from the cine file and create the schema:
    Teuchos::RCP<DICe::Image> ref_image = cine_reader->get_frame(cine_ref_index,correlation_params);
    schema = Teuchos::rcp(new DICe::Schema(ref_image,ref_image,correlation_params));
  }
  else{
    const std::string ref_image_string = image_files[0];
    schema = Teuchos::rcp(new DICe::Schema(ref_image_string,ref_image_string,correlation_params));
  }

  schema->initialize(input_params);

  *outStream << "Number of subsets: " << schema->data_num_points() << std::endl;
  for(int_t i=0;i<schema->data_num_points();++i){
    if(i==10&&schema->data_num_points()!=11) *outStream << "..." << std::endl;
    else if(i>10&&i<schema->data_num_points()-1) continue;
    else
      *outStream << "Subset: " << i << " global coordinates (" << schema->field_value(i,COORDINATE_X) <<
        "," << schema->field_value(i,COORDINATE_Y) << ")" << std::endl;
  }
  *outStream << std::endl;

  // let the schema know how many images there are in the sequence:
  schema->set_num_image_frames(num_images);

  // iterate through the images and perform the correlation:

  scalar_t total_time = 0.0;
  scalar_t elapsed_time = 0.0;
  scalar_t max_time = 0.0;
  scalar_t min_time = 1.0E10;
  scalar_t avg_time = 0.0;
  std::string file_prefix = input_params->get<std::string>(DICe::output_prefix,"DICe_solution");
  // TODO find a more straightforward way to do the indexing
  const int_t start_frame = cine_start_index==-1 ? 1 : cine_start_index;
  const int_t end_frame = cine_end_index==-1 ? num_images : cine_end_index;
  for(int_t image_it=start_frame;image_it<=end_frame;++image_it){
    if(is_cine){
      Teuchos::RCP<DICe::Image> def_image = cine_reader->get_frame(image_it,correlation_params);
      *outStream << "Processing Image: " << image_it - start_frame + 1 << " of " << num_images << " frame id: " << image_it << std::endl;
      schema->set_def_image(def_image);
    }
    else{
      const std::string def_image_string = image_files[image_it];
      *outStream << "Processing Image: " << image_it << " of " << num_images << ", " << def_image_string << std::endl;
      schema->set_def_image(def_image_string);
    }

    { // start the timer
      boost::timer t;

      schema->execute_correlation();

      // timing info
      elapsed_time = t.elapsed();
      if(elapsed_time>max_time)max_time = elapsed_time;
      if(elapsed_time<min_time)min_time = elapsed_time;
      total_time += elapsed_time;
    }

    // write the output
    schema->write_output(output_folder,file_prefix,separate_output_file_for_each_subset);
    //if(subset_info->conformal_area_defs!=Teuchos::null&&image_it==1){
    //  schema->write_control_points_image("RegionOfInterest");
    //}

  } // image loop

  avg_time = total_time / num_images;

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

  return 0;
}

