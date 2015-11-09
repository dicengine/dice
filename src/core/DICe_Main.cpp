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

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

using namespace DICe;

int main(int argc, char *argv[]) {

  const int_t dim = 2;         // Assumes 2D images

  std::stringstream banner;
  banner << "--- Digital Image Correlation Engine (DICe), Copyright 2015 Sandia Corporation ---" << std::endl << std::endl;
  std::string bannerStr = banner.str();

  int_t proc_size = 1;
  int_t proc_rank = 0;
#ifdef HAVE_MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&proc_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
  if(proc_rank==0) DEBUG_MSG("Code was compiled with MPI enabled");
#else
  DEBUG_MSG("Code was compiled with MPI disabled");
#endif

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  // Command line options

  if(proc_rank==0)  DEBUG_MSG("Parsing command line options");
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::RCP<Teuchos::ParameterList> input_params = DICe::parse_command_line(argc,argv,outStream,bannerStr);
  *outStream << banner.str();
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
  //DICe::cine::Cine_Header cine_header;
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

  // subset locations:

  // if the subset locations are specified in an input file, read them in (else they will be defined later)
  Teuchos::RCP<std::vector<int_t> > subset_centroids;
  Teuchos::RCP<std::vector<int_t> > neighbor_ids;
  Teuchos::RCP<DICe::Subset_File_Info> subset_info;
  int_t step_size = -1;
  int_t subset_size = -1;
  int_t num_subsets = -1;
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs;
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > blocking_subset_ids;
  const bool has_subset_file = input_params->isParameter(DICe::subset_file);
  DICe::Subset_File_Info_Type subset_info_type = DICe::SUBSET_INFO;
  if(has_subset_file){
    std::string fileName = input_params->get<std::string>(DICe::subset_file);
    subset_info = DICe::read_subset_file(fileName,image_width,image_height);
    subset_info_type = subset_info->type;
  }
  if(!has_subset_file || subset_info_type==DICe::REGION_OF_INTEREST_INFO){
    assert(input_params->isParameter(DICe::step_size));
    step_size = input_params->get<int_t>(DICe::step_size);
    *outStream << "Correlation point centroids were not specified by the user. \nThey will be evenly distrubed in the region"
        " of interest with separation (step_size) of " << step_size << " pixels." << std::endl;
    subset_centroids = Teuchos::rcp(new std::vector<int_t>());
    neighbor_ids = Teuchos::rcp(new std::vector<int_t>());
    DICe::create_regular_grid_of_correlation_points(*subset_centroids,*neighbor_ids,input_params,image_width,image_height,subset_info);
    num_subsets = subset_centroids->size()/dim; // divide by three because the stride is x y neighbor_id
    assert(neighbor_ids->size()==subset_centroids->size()/2);
    assert(input_params->isParameter(DICe::subset_size)); // required for all square subsets case
    subset_size = input_params->get<int_t>(DICe::subset_size);
  }
  else{
    assert(subset_info!=Teuchos::null);
    subset_centroids = subset_info->data_vector;
    neighbor_ids = subset_info->neighbor_vector;
    conformal_area_defs = subset_info->conformal_area_defs;
    blocking_subset_ids = subset_info->data_map;
    num_subsets = subset_info->data_vector->size()/dim;
    if((int_t)subset_info->conformal_area_defs->size()<num_subsets){
      // Only require this if not all subsets are conformal:
      assert(input_params->isParameter(DICe::subset_size));
      subset_size = input_params->get<int_t>(DICe::subset_size);
    }
  }
  assert(subset_centroids->size()>0);
  assert(num_subsets>0);
  *outStream << "\n--- Correlation point centroids read sucessfuly ---\n" << std::endl;

  *outStream << "Number of subsets: " << num_subsets << std::endl;
  for(int_t i=0;i<num_subsets;++i){
    if(i==10&&num_subsets!=11) *outStream << "..." << std::endl;
    else if(i>10&&i<num_subsets-1) continue;
    else
      *outStream << "Subset: " << i << " global coordinates (" << (*subset_centroids)[i*dim+0] <<
        "," << (*subset_centroids)[i*dim+1] << ")" << std::endl;
  }
  *outStream << std::endl;

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
  // set the blocking subset ids if they exist
  schema->set_obstructing_subset_ids(blocking_subset_ids);
  // initialize the schema
  schema->initialize(num_subsets,subset_size,conformal_area_defs,neighbor_ids);
  schema->set_step_size(step_size); // this is done just so the step_size appears in the output file header (it's not actually used)
  // let the schema know how many images there are in the sequence:
  schema->set_num_image_frames(num_images);

  // populate the fields:

  // set the coordinates for the subsets:
  // all other values are initiliazed to zero
  for(int_t i=0;i<num_subsets;++i){
    schema->field_value(i,DICe::COORDINATE_X) = (*subset_centroids)[i*dim + 0];
    schema->field_value(i,DICe::COORDINATE_Y) = (*subset_centroids)[i*dim + 1];
  }
  // set the seed value if they exist
  if(subset_info!=Teuchos::null){
    if(subset_info->path_file_names->size()>0){
      schema->set_path_file_names(subset_info->path_file_names);
    }
    if(subset_info->skip_solve_flags->size()>0){
      schema->set_skip_solve_flags(subset_info->skip_solve_flags);
    }
    if(subset_info->motion_window_params->size()>0){
      schema->set_motion_window_params(subset_info->motion_window_params);
    }
    if(subset_info->seed_subset_ids->size()>0){
      //schema->has_seed(true);
      assert(subset_info->displacement_map->size()>0);
      std::map<int_t,int_t>::iterator it=subset_info->seed_subset_ids->begin();
      for(;it!=subset_info->seed_subset_ids->end();++it){
        const int_t subset_id = it->first;
        const int_t roi_id = it->second;
        assert(subset_info->displacement_map->find(roi_id)!=subset_info->displacement_map->end());
        schema->field_value(subset_id,DICe::DISPLACEMENT_X) = subset_info->displacement_map->find(roi_id)->second.first;
        schema->field_value(subset_id,DICe::DISPLACEMENT_Y) = subset_info->displacement_map->find(roi_id)->second.second;
        if(proc_rank==0) DEBUG_MSG("Seeding the displacement solution for subset " << subset_id << " with ux: " <<
          schema->field_value(subset_id,DICe::DISPLACEMENT_X) << " uy: " << schema->field_value(subset_id,DICe::DISPLACEMENT_Y));
        if(subset_info->normal_strain_map->find(roi_id)!=subset_info->normal_strain_map->end()){
          schema->field_value(subset_id,DICe::NORMAL_STRAIN_X) = subset_info->normal_strain_map->find(roi_id)->second.first;
          schema->field_value(subset_id,DICe::NORMAL_STRAIN_Y) = subset_info->normal_strain_map->find(roi_id)->second.second;
          if(proc_rank==0) DEBUG_MSG("Seeding the normal strain solution for subset " << subset_id << " with ex: " <<
            schema->field_value(subset_id,DICe::NORMAL_STRAIN_X) << " ey: " << schema->field_value(subset_id,DICe::NORMAL_STRAIN_Y));
        }
        if(subset_info->shear_strain_map->find(roi_id)!=subset_info->shear_strain_map->end()){
          schema->field_value(subset_id,DICe::SHEAR_STRAIN_XY) = subset_info->shear_strain_map->find(roi_id)->second;
          if(proc_rank==0) DEBUG_MSG("Seeding the shear strain solution for subset " << subset_id << " with gamma_xy: " <<
            schema->field_value(subset_id,DICe::SHEAR_STRAIN_XY));
        }
        if(subset_info->rotation_map->find(roi_id)!=subset_info->rotation_map->end()){
          schema->field_value(subset_id,DICe::ROTATION_Z) = subset_info->rotation_map->find(roi_id)->second;
          if(proc_rank==0) DEBUG_MSG("Seeding the rotation solution for subset " << subset_id << " with theta_z: " <<
            schema->field_value(subset_id,DICe::ROTATION_Z));
        }
      }
    }
  }

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
      *outStream << "Processing Image: " << image_it - start_frame + 1 << " of " << num_images << " frame: " << image_it << std::endl;
      schema->set_def_image(def_image);
    }
    else{
      const std::string def_image_string = image_files[image_it];
      *outStream << "Processing Image: " << image_it << " of " << num_images << ", " << def_image_string << std::endl;
      schema->set_def_image(def_image_string);
    }

    { // start the timer
      //boost::timer t;

      schema->execute_correlation();

      // timing info
      //elapsed_time = t.elapsed();
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
  //if(input_params->get<bool>(DICe::print_timing,false)){
  //  std::cout << "Total time:              " << total_time << std::endl;
  //  std::cout << "Avgerage time per image: " << avg_time << std::endl;
  //  std::cout << "Max time per image:      " << max_time << std::endl;
  //  std::cout << "Min time per image:      " << min_time << std::endl;
  //}
  // write the time output to file:
  std::stringstream timeFileName;
  timeFileName << output_folder << "timing."<< proc_size << "." << proc_rank << ".txt";
  std::FILE * timeFile = fopen(timeFileName.str().c_str(),"w");
  fprintf(timeFile,"TOTAL AVERAGE_PER_IMAGE MAX_PER_IMAGE MIN_PER_IMAGE\n");
  fprintf(timeFile,"%4.4E %4.4E %4.4E %4.4E\n",total_time,avg_time,max_time,min_time);
  fclose(timeFile);

#ifdef HAVE_MPI
  (void) MPI_Finalize ();
#endif

  // finalize kokkos
  Kokkos::finalize();

  return 0;
}

