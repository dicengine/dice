// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
#include <DICe_Image.h>
#include <DICe_ImageIO.h>
#include <DICe_Subset.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

using namespace DICe;

// Usage DICe_PerformanceFunctors [<num_image_sizes> <num_time_samples> <num_thread_teams>]

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  Teuchos::RCP<Teuchos::Time> read_time  = Teuchos::TimeMonitor::getNewCounter("read time");
  Teuchos::RCP<Teuchos::Time> grad_time  = Teuchos::TimeMonitor::getNewCounter("grad time");
  Teuchos::RCP<Teuchos::Time> filter_time  = Teuchos::TimeMonitor::getNewCounter("filter time");
  Teuchos::RCP<Teuchos::Time> subset_time  = Teuchos::TimeMonitor::getNewCounter("subset construct time");
  Teuchos::RCP<Teuchos::Time> subset_ref_time  = Teuchos::TimeMonitor::getNewCounter("subset set ref");
  Teuchos::RCP<Teuchos::Time> subset_def_bilinear_time  = Teuchos::TimeMonitor::getNewCounter("subset set def bilinear");
  Teuchos::RCP<Teuchos::Time> subset_def_keys_time  = Teuchos::TimeMonitor::getNewCounter("subset set def keys");
  Teuchos::RCP<Teuchos::Time> subset_mean_time  = Teuchos::TimeMonitor::getNewCounter("subset mean");
  Teuchos::RCP<Teuchos::Time> corr_time  = Teuchos::TimeMonitor::getNewCounter("correlate");

  // only print output if args are given (for testing the output is quiet)
  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);
  if(argc>1) // anything but the default cases, writes output to screen
    outStream = Teuchos::rcp(&std::cout, false);

  *outStream << "--- Begin performance test ---" << std::endl;

  // optional argument for the number of image sizes
  int_t num_img_sizes = 5;
  if(argc>1) num_img_sizes = std::strtol(argv[1],NULL,0);
  assert(num_img_sizes!=0);
  int_t num_time_samples = 5;
  if(argc>2) num_time_samples = std::strtol(argv[2],NULL,0);

  // optional argument for the number of thread teams:

//  int_t num_thread_teams = -1;
//  if(argc>3) num_thread_teams = std::strtol(argv[3],NULL,0);
//  *outStream << "number of thread teams:   " << num_thread_teams << " (-1 means thread teams not used)" << std::endl;
//  const bool use_hierarchical = num_thread_teams > 0;
//  *outStream << "hierarchical parallelism: " << use_hierarchical << std::endl;

  // create a vector of image sizes to use
  int_t width = 0;
  int_t height = 0;
  std::vector<int_t> widths(num_img_sizes,0);
  std::vector<int_t> heights(num_img_sizes,0);
  const std::string file_name = "./images/UberImage.tif";

  // read the image dimensions from file:
  utils::read_image_dimensions(file_name.c_str(),width,height);
  *outStream << "master image dims:        " << width << " x " << height << std::endl;
  *outStream << "number of image sizes:    " << num_img_sizes << std::endl;
  const int_t w_step = width/num_img_sizes;
  const int_t h_step = height/num_img_sizes;
  for(int_t i=0;i<num_img_sizes;++i){
    widths[i] = (i+1)*w_step;
    heights[i] = (i+1)*h_step;
  }

  //Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  //params->set(DICe::gauss_filter_mask_size,13);
  Teuchos::RCP<Local_Shape_Function> map = shape_function_factory();
  map->insert_motion(1.25,-1.456,0.001);
//  (*map)[DOF_EX] = 0.002;
//  (*map)[DOF_EY] = 0.001;
//  (*map)[DOF_GXY] = 0.0;
  Teuchos::RCP<Subset> subset;
  const int_t subset_edge_buffer = 10; // must be larger than the deformations above

  std::vector<scalar_t> sizes(num_img_sizes,0.0);
  std::vector<scalar_t> read_times(num_img_sizes,0.0);
  std::vector<scalar_t> grad_times(num_img_sizes,0.0);
  std::vector<scalar_t> filter_times(num_img_sizes,0.0);
  std::vector<scalar_t> sub_construct_times(num_img_sizes,0.0);
  std::vector<scalar_t> sub_init_times(num_img_sizes,0.0);
  std::vector<scalar_t> sub_bilinear_times(num_img_sizes,0.0);
  std::vector<scalar_t> sub_keys_times(num_img_sizes,0.0);
  std::vector<scalar_t> mean_times(num_img_sizes,0.0);
  std::vector<scalar_t> corr_times(num_img_sizes,0.0);
  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());

  // num timing samples loop
  for(int_t time_sample=0;time_sample<num_time_samples;++time_sample){
    *outStream << "\n%%% time sample " << time_sample << std::endl;
    // image size loop:
    for(int_t size_it=0;size_it<num_img_sizes;++size_it){
      const int_t w_it = widths[size_it];
      const int_t h_it = heights[size_it];
      imgParams->set(DICe::subimage_width,w_it);
      imgParams->set(DICe::subimage_height,h_it);

      *outStream << "---------------------------------------------------------------------------" << std::endl;
      *outStream << "image size: " << w_it << " x " << h_it << std::endl;
      if(time_sample==0) sizes[size_it] = w_it*h_it;

      Teuchos::RCP<Image> img;
      // read image
      *outStream << "reading the image" << std::endl;
      {
        Teuchos::TimeMonitor read_time_monitor(*read_time);
        img = Teuchos::rcp(new Image(file_name.c_str(),imgParams));
      }

      // gradient
      *outStream << "computing the image gradients" << std::endl;
      {
        Teuchos::TimeMonitor grad_time_monitor(*grad_time);
//        if(use_hierarchical){
//          img->compute_gradients(true,num_thread_teams);
//        }
//        else{
          img->compute_gradients();
//        }
      }

      // image filter
      *outStream << "computing convolution filter" << std::endl;
      {
        Teuchos::TimeMonitor filter_time_monitor(*filter_time);
//        if(use_hierarchical){
//          img->gauss_filter(true,num_thread_teams);
//        }
//        else{
          img->gauss_filter();
//        }
      }

      // create a subset
      *outStream << "creating a subset of size " << w_it-subset_edge_buffer << " x " << h_it-subset_edge_buffer << std::endl;
      {
        Teuchos::TimeMonitor subset_time_monitor(*subset_time);
        subset = Teuchos::rcp(new Subset(w_it/2,h_it/2,w_it-subset_edge_buffer,h_it-subset_edge_buffer));
      }

      // initialize the reference values (copy, no deformation involved)
      *outStream << "initializing the reference intensities" << std::endl;
      {
        Teuchos::TimeMonitor subset_ref_time_monitor(*subset_ref_time);
        subset->initialize(img);
      }

      // initialize the deformed values (mapping involved as well as interpolation, cant separate these)
      *outStream << "initializing the deformed intensities (bilinear)" << std::endl;
      {
        Teuchos::TimeMonitor subset_def_biliear_time_monitor(*subset_def_bilinear_time);
        subset->initialize(img,DEF_INTENSITIES,map,BILINEAR);
      }

      *outStream << "re-initializing the deformed intensities (Keys fourth-order)" << std::endl;
      {
        Teuchos::TimeMonitor subset_def_keys_time_monitor(*subset_def_keys_time);
        subset->initialize(img,DEF_INTENSITIES,map,KEYS_FOURTH);
      }

      // mean value
      *outStream << "computing the mean subset value" << std::endl;
      {
        Teuchos::TimeMonitor subset_mean_time_monitor(*subset_mean_time);
        subset->mean(REF_INTENSITIES);
      }

      // correlation
      *outStream << "computing the correlation between the ref and def intensities" << std::endl;
      {
        Teuchos::TimeMonitor corr_time_monitor(*corr_time);
        subset->gamma();
      }

      Teuchos::TimeMonitor::summarize(*outStream,false,true,false/*zero timers*/);

    } // end size loop

  } // end time sample loop

  *outStream << "\n\nTiming summary: " << std::endl;


  std::cout << "End Result: TEST PASSED\n";

  *outStream << "--- End performance test ---" << std::endl;

  DICe::finalize();

  return 0;
}

