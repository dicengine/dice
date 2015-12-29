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
#include <DICe_Image.h>
#include <DICe_ImageIO.h>
#include <DICe_Subset.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <boost/timer/timer.hpp>

#include <iostream>

using namespace DICe;
using namespace boost::timer;

// Usage DICe_PerformanceFunctors [<num_image_sizes> <num_time_samples> <num_thread_teams>]

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);
  if(argc>1) // anything but the default cases, writes output to screen
    outStream = Teuchos::rcp(&std::cout, false);

  *outStream << "--- Begin performance test ---" << std::endl;

  // optional argument for the number of image sizes
  int_t num_img_sizes = 5;
  if(argc>1) num_img_sizes = std::atoi(argv[1]);
  int_t num_time_samples = 5;
  if(argc>2) num_time_samples = std::atoi(argv[2]);

  // optional argument for the number of thread teams:

  int_t num_thread_teams = -1;
  if(argc>3) num_thread_teams = std::atoi(argv[3]);
  *outStream << "number of thread teams:   " << num_thread_teams << " (-1 means thread teams not used)" << std::endl;
  const bool use_hierarchical = num_thread_teams > 0;
  *outStream << "hierarchical parallelism: " << use_hierarchical << std::endl;

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

  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::gauss_filter_mask_size,13);
  Teuchos::RCP<std::vector<scalar_t> > map = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
  (*map)[DISPLACEMENT_X] = 1.25; // TODO randomize deformation map
  (*map)[DISPLACEMENT_Y] = -1.456; // needs to remain inside the current image
  (*map)[ROTATION_Z] = 0.001;
  (*map)[NORMAL_STRAIN_X] = 0.002;
  (*map)[NORMAL_STRAIN_Y] = 0.001;
  (*map)[SHEAR_STRAIN_XY] = 0.0;
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


  // num timing samples loop
  for(int_t time_sample=0;time_sample<num_time_samples;++time_sample){
    *outStream << "\n%%% time sample " << time_sample << std::endl;
    // image size loop:
    for(int_t size_it=0;size_it<num_img_sizes;++size_it){
      const int_t w_it = widths[size_it];
      const int_t h_it = heights[size_it];
      *outStream << "---------------------------------------------------------------------------" << std::endl;
      *outStream << "image size: " << w_it << " x " << h_it << std::endl;
      if(time_sample==0) sizes[size_it] = w_it*h_it;

      Teuchos::RCP<Image> img;
      // read image
      *outStream << "reading the image" << std::endl;
      cpu_timer read_timer;
      {
        read_timer.start();
        img = Teuchos::rcp(new Image(file_name.c_str(),0,0,w_it,h_it));
        read_timer.stop();
      }
      *outStream << "** read time" << read_timer.format();
      read_times[size_it] += ((scalar_t)(read_timer.elapsed().wall)/1000000000)/num_time_samples;

      // gradient
      *outStream << "computing the image gradients" << std::endl;
      cpu_timer grad_timer;
      {
        if(use_hierarchical){
          grad_timer.start();
          img->compute_gradients(true,num_thread_teams);
          grad_timer.stop();
        }
        else{
          grad_timer.start();
          img->compute_gradients();
          grad_timer.stop();
        }
      }
      *outStream << "** grad time" << grad_timer.format();
      grad_times[size_it] += ((scalar_t)(grad_timer.elapsed().wall)/1000000000)/num_time_samples;

      // image filter
      *outStream << "computing convolution filter" << std::endl;
      cpu_timer filter_timer;
      {
        if(use_hierarchical){
          filter_timer.start();
          img->gauss_filter(true,num_thread_teams);
          filter_timer.stop();
        }
        else{
          filter_timer.start();
          img->gauss_filter();
          filter_timer.stop();
        }
      }
      *outStream << "** filter time" << filter_timer.format();
      filter_times[size_it] += ((scalar_t)(filter_timer.elapsed().wall)/1000000000)/num_time_samples;

      // create a subset
      *outStream << "creating a subset of size " << w_it-subset_edge_buffer << " x " << h_it-subset_edge_buffer << std::endl;
      cpu_timer subset_construct_timer;
      {
        subset_construct_timer.start();
        subset = Teuchos::rcp(new Subset(w_it/2,h_it/2,w_it-subset_edge_buffer,h_it-subset_edge_buffer));
        subset_construct_timer.stop();
      }
      *outStream <<"** subset constructor time" << subset_construct_timer.format();
      sub_construct_times[size_it] += ((scalar_t)(subset_construct_timer.elapsed().wall)/1000000000)/num_time_samples;

      // initialize the reference values (copy, no deformation involved)
      *outStream << "initializing the reference intensities" << std::endl;
      cpu_timer subset_init_ref_timer;
      {
        subset_init_ref_timer.start();
        subset->initialize(img);
        subset_init_ref_timer.stop();
      }
      *outStream <<"** subset init ref time" << subset_init_ref_timer.format();
      sub_init_times[size_it] += ((scalar_t)(subset_init_ref_timer.elapsed().wall)/1000000000)/num_time_samples;

      // initialize the deformed values (mapping involved as well as interpolation, cant separate these)
      *outStream << "initializing the deformed intensities (bilinear)" << std::endl;
      cpu_timer subset_init_def_bilinear_timer;
      {
        subset_init_def_bilinear_timer.start();
        subset->initialize(img,DEF_INTENSITIES,map,BILINEAR);
        subset_init_def_bilinear_timer.stop();
      }
      *outStream <<"** subset init def bilinear time" << subset_init_def_bilinear_timer.format();
      sub_bilinear_times[size_it] += ((scalar_t)(subset_init_def_bilinear_timer.elapsed().wall)/1000000000)/num_time_samples;

      *outStream << "re-initializing the deformed intensities (Keys fourth-order)" << std::endl;
      cpu_timer subset_init_def_keys_timer;
      {
        subset_init_def_keys_timer.start();
        subset->initialize(img,DEF_INTENSITIES,map,KEYS_FOURTH);
        subset_init_def_keys_timer.stop();
      }
      *outStream <<"** subset init def Keys time" << subset_init_def_keys_timer.format();
      sub_keys_times[size_it] += ((scalar_t)(subset_init_def_keys_timer.elapsed().wall)/1000000000)/num_time_samples;

      // mean value
      *outStream << "computing the mean subset value" << std::endl;
      cpu_timer subset_mean_timer;
      {
        subset_mean_timer.start();
        subset->mean(REF_INTENSITIES);
        subset_mean_timer.stop();
      }
      *outStream << "** subset mean intensity time" << subset_mean_timer.format();
      mean_times[size_it] += ((scalar_t)(subset_mean_timer.elapsed().wall)/1000000000)/num_time_samples;

      // correlation
      *outStream << "computing the correlation between the ref and def intensities" << std::endl;
      cpu_timer corr_timer;
      {
        corr_timer.start();
        subset->gamma();
        corr_timer.stop();
      }
      *outStream << "** correlation time" << corr_timer.format();
      corr_times[size_it] += ((scalar_t)(corr_timer.elapsed().wall)/1000000000)/num_time_samples;

    } // end size loop

  } // end time sample loop

  *outStream << "\n\nTiming summary: " << std::endl;
  *outStream << std::setw(15) << "size" <<
      std::setw(15) << "read"<<
      std::setw(15) << "grad" <<
      std::setw(15) << "filter" <<
      std::setw(15) << "subset" <<
      std::setw(15) << "init ref" <<
      std::setw(15) << "bilinear" <<
      std::setw(15) << "keys" <<
      std::setw(15) << "mean" <<
      std::setw(15) << "correlation" <<
      std::endl;
  for(int_t i=0;i<num_img_sizes;++i){
    *outStream << std::setw(15) <<
        sizes[i] << std::setw(15) <<
        read_times[i] << std::setw(15) <<
        grad_times[i] << std::setw(15) <<
        filter_times[i] << std::setw(15) <<
        sub_construct_times[i] << std::setw(15) <<
        sub_init_times[i] << std::setw(15) <<
        sub_bilinear_times[i] << std::setw(15) <<
        sub_keys_times[i] << std::setw(15) <<
        mean_times[i] << std::setw(15) <<
        corr_times[i] << std::setw(15) <<
        std::endl;
  }

  std::cout << "End Result: TEST PASSED\n";

  *outStream << "--- End performance test ---" << std::endl;

  DICe::finalize();

  return 0;
}

