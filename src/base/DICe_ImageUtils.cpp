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

#include <DICe_ImageUtils.h>
#include <DICe_Image.h>

#include <random>

namespace DICe {

void apply_transform(Teuchos::RCP<Image> image_in,
  Teuchos::RCP<Image> image_out,
  const int_t cx,
  const int_t cy,
  Teuchos::RCP<const std::vector<scalar_t> > deformation){
  const int_t width = image_in->width();
  const int_t height = image_in->height();
  TEUCHOS_TEST_FOR_EXCEPTION(width!=image_out->width(),std::runtime_error,"Dimensions must be the same");
  TEUCHOS_TEST_FOR_EXCEPTION(height!=image_out->height(),std::runtime_error,"Dimensions must be the same");
  TEUCHOS_TEST_FOR_EXCEPTION(deformation==Teuchos::null,std::runtime_error,"");
  const scalar_t u = (*deformation)[DISPLACEMENT_X];
  const scalar_t v = (*deformation)[DISPLACEMENT_Y];
  const scalar_t t = (*deformation)[ROTATION_Z];
  const scalar_t ex = (*deformation)[NORMAL_STRAIN_X];
  const scalar_t ey = (*deformation)[NORMAL_STRAIN_Y];
  const scalar_t g = (*deformation)[SHEAR_STRAIN_XY];
  const scalar_t cos_t = std::cos(t);
  const scalar_t sin_t = std::sin(t);
  const scalar_t CX = cx + u;
  const scalar_t CY = cy + v;
  scalar_t dX=0.0, dY=0.0;
  scalar_t Dx=0.0, Dy=0.0;
  scalar_t mapped_x=0.0, mapped_y=0.0;
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      dX = x - CX;
      dY = y - CY;
      Dx = (1.0-ex)*dX - g*dY;
      Dy = (1.0-ey)*dY - g*dX;
      mapped_x = cos_t*Dx - sin_t*Dy - u + CX;
      mapped_y = sin_t*Dx + cos_t*Dy - v + CY;
      image_out->intensities()[y*width+x] = image_in->interpolate_keys_fourth(mapped_x,mapped_y);
    }// x
  }// y
}

void SinCos_Image_Deformer::compute_deformation(const scalar_t & coord_x,
  const scalar_t & coord_y,
  scalar_t & bx,
  scalar_t & by){
  assert(period_>0.0);
  const scalar_t beta = DICE_TWOPI*(1.0/period_);
  bx = 0.5*amplitude_ + sin(beta*coord_x)*cos(beta*coord_y)*0.5*amplitude_;
  by = 0.5*amplitude_ - cos(beta*coord_x)*sin(beta*coord_y)*0.5*amplitude_;
}

void SinCos_Image_Deformer::compute_deriv_deformation(const scalar_t & coord_x,
  const scalar_t & coord_y,
  scalar_t & bxx,
  scalar_t & bxy,
  scalar_t & byx,
  scalar_t & byy){
  bxx = 0.0;
  bxy = 0.0;
  byx = 0.0;
  byy = 0.0;
  const scalar_t beta = DICE_TWOPI*(1.0/period_);
  bxx = beta*cos(beta*coord_x)*cos(beta*coord_y)*0.5*amplitude_;
  bxy = -beta*sin(beta*coord_x)*sin(beta*coord_y)*0.5*amplitude_;
  byx = beta*sin(beta*coord_x)*sin(beta*coord_y)*0.5*amplitude_;
  byy = -beta*cos(beta*coord_x)*cos(beta*coord_y)*0.5*amplitude_;
}

void SinCos_Image_Deformer::compute_displacement_error(const scalar_t & coord_x,
  const scalar_t & coord_y,
  const scalar_t & sol_x,
  const scalar_t & sol_y,
  scalar_t & error_x,
  scalar_t & error_y,
  const bool use_mag,
  const bool relative){

  scalar_t out_x = 0.0;
  scalar_t out_y = 0.0;
  compute_deformation(coord_x,coord_y,out_x,out_y);
  if(use_mag){
    error_x = (sol_x - out_x)*(sol_x - out_x);
    error_y = (sol_y - out_y)*(sol_y - out_y);
  }else{
    error_x = sol_x - out_x;
    error_y = sol_y - out_y;
  }
  if(relative){
    error_x /= amplitude_/200.0; // convert to percent (multiplied by two to account for top and bottom roll-off)
    error_y /= amplitude_/200.0;
  }
}


void SinCos_Image_Deformer::compute_lagrange_strain(const scalar_t & coord_x,
  const scalar_t & coord_y,
  scalar_t & strain_xx,
  scalar_t & strain_xy,
  scalar_t & strain_yy){

  scalar_t out_xx = 0.0;
  scalar_t out_xy = 0.0;
  scalar_t out_yx = 0.0;
  scalar_t out_yy = 0.0;
  compute_deriv_deformation(coord_x,coord_y,out_xx,out_xy,out_yx,out_yy);

  strain_xx = 0.5*(2.0*out_xx + out_xx*out_xx + out_yx*out_yx);
  strain_xy = 0.5*(out_xy + out_yx + out_xy*out_xy + out_yx*out_yx);
  strain_yy = 0.5*(2.0*out_yy + out_yy*out_yy + out_xy*out_xy);
}


void SinCos_Image_Deformer::compute_lagrange_strain_error(const scalar_t & coord_x,
  const scalar_t & coord_y,
  const scalar_t & sol_xx,
  const scalar_t & sol_xy,
  const scalar_t & sol_yy,
  scalar_t & error_xx,
  scalar_t & error_xy,
  scalar_t & error_yy,
  const bool use_mag,
  const bool relative){

  scalar_t strain_xx = 0.0;
  scalar_t strain_xy = 0.0;
  scalar_t strain_yy = 0.0;

  compute_lagrange_strain(coord_x,coord_y,strain_xx,strain_xy,strain_yy);

  if(use_mag){
    error_xx = (sol_xx - strain_xx)*(sol_xx - strain_xx);
    error_xy = (sol_xy - strain_xy)*(sol_xy - strain_xy);
    error_yy = (sol_yy - strain_yy)*(sol_yy - strain_yy);
  }else{
    error_xx = sol_xx - strain_xx;
    error_xy = sol_xy - strain_xy;
    error_yy = sol_yy - strain_yy;
  }
  if(relative){
    scalar_t rel = (1.0/period_)*DICE_TWOPI*amplitude_/200.0;
    error_xx /= rel;
    error_xy /= rel;
    error_yy /= rel;
  }
}

Teuchos::RCP<Image>
SinCos_Image_Deformer::deform_image(Teuchos::RCP<Image> ref_image){
  const int_t w = ref_image->width();
  const int_t h = ref_image->height();
//  // Note: uses 5 x 5 point sampling grid to evaluate the deformed intensity
//  const int_t num_pts = 5;
//  static scalar_t coeffs[5] = {0.0014,0.1574,0.62825,0.1574,0.0014};
//  // Note: uses 11 x 11 point sampling grid to evaluate the deformed intensity
////  static scalar_t coeffs[11] =
////  {0.0001,0.0017,0.0168,0.0870,
////    0.2328,0.3231,0.2328,
////    0.0870,0.0168,0.0017,0.0001};
//  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
//  scalar_t bx=0.0,by=0.0;
//  for(int_t j=0;j<h;++j){
//    for(int_t i=0;i<w;++i){
//      scalar_t avg_intens = 0.0;
//      for(int_t oy=0;oy<num_pts;++oy){
//        const scalar_t sample_y = j - 0.5*(num_pts-1)/num_pts + oy/num_pts;
//        for(int_t ox=0;ox<num_pts;++ox){
//          const scalar_t sample_x = i - 0.5*(num_pts-1)/num_pts + ox/num_pts;
//          const scalar_t weight = coeffs[ox]*coeffs[oy];
//          compute_deformation(sample_x,sample_y,bx,by);
//          scalar_t intens = ref_image->interpolate_keys_fourth(sample_x-bx,sample_y-by);
//          avg_intens += weight*intens;
//        } // end super pixel ox
//      } // end super pixel oy
//      def_intens[j*w+i] = avg_intens;
//    } // end pixel i
//  } // ens pixel j


  // Note: uses 5 x 5 point sampling grid to evaluate the deformed intensity
  const int_t num_pts = 5;
  static scalar_t offsets_x[5] = {0.0,-0.5,0.5,0.5,-0.5};
  static scalar_t offsets_y[5] = {0.0,-0.5,-0.5,0.5,0.5};
  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
  scalar_t bx=0.0,by=0.0;
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      scalar_t avg_intens = 0.0;
      for(int_t pt=0;pt<num_pts;++pt){
        const scalar_t sample_x = i - offsets_x[pt];
        const scalar_t sample_y = j - offsets_y[pt];
        compute_deformation(sample_x,sample_y,bx,by);
        scalar_t intens = ref_image->interpolate_keys_fourth(sample_x-bx,sample_y-by);
        avg_intens += intens;
      } // end avg points
      def_intens[j*w+i] = avg_intens/num_pts;
    } // end pixel i
  } // ens pixel j

  // no weighted average ...
//  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
//  scalar_t bx=0.0,by=0.0;
//  for(int_t j=0;j<h;++j){
//    for(int_t i=0;i<w;++i){
//      compute_deformation(i,j,bx,by);
//      scalar_t intens = ref_image->interpolate_keys_fourth(i-bx,j-by);
//      def_intens[j*w+i] = intens;
//    } // end pixel i
//  } // ens pixel j
  Teuchos::RCP<Image> def_img = Teuchos::rcp(new Image(w,h,def_intens));
  return def_img;
}

void compute_roll_off_stats(const scalar_t & period,
  const scalar_t & img_w,
  const scalar_t & img_h,
  Teuchos::RCP<MultiField> & coords,
  Teuchos::RCP<MultiField> & disp,
  Teuchos::RCP<MultiField> & exact_disp,
  Teuchos::RCP<MultiField> & disp_error,
  scalar_t & peaks_avg_error_x,
  scalar_t & peaks_std_dev_error_x,
  scalar_t & peaks_avg_error_y,
  scalar_t & peaks_std_dev_error_y){

  // Send the whole field to procesor 0
  MultiField_Comm comm = coords->get_map()->get_comm();
  const int_t p_rank = comm.get_rank();

  // create all on zero map
  const int_t num_values = coords->get_map()->get_num_global_elements();
  Teuchos::Array<int_t> all_on_zero_ids;
  if(p_rank==0){
    all_on_zero_ids.resize(num_values);
    for(int_t i=0;i<num_values;++i)
       all_on_zero_ids[i] = i;
  }
  Teuchos::RCP<MultiField_Map> all_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, all_on_zero_ids, 0, comm));
  Teuchos::RCP<MultiField> all_on_zero_coords = Teuchos::rcp( new MultiField(all_on_zero_map,1,true));
  Teuchos::RCP<MultiField> all_on_zero_disp = Teuchos::rcp( new MultiField(all_on_zero_map,1,true));
  Teuchos::RCP<MultiField> all_on_zero_exact_disp = Teuchos::rcp( new MultiField(all_on_zero_map,1,true));
  Teuchos::RCP<MultiField> all_on_zero_disp_error = Teuchos::rcp( new MultiField(all_on_zero_map,1,true));

  // create exporter with the map
  MultiField_Exporter exporter(*all_on_zero_map,*coords->get_map());
  // export the field to zero
  all_on_zero_coords->do_import(coords,exporter);
  all_on_zero_disp->do_import(disp,exporter);
  all_on_zero_exact_disp->do_import(exact_disp,exporter);
  all_on_zero_disp_error->do_import(disp_error,exporter);

  // analyze stats
  //TEUCHOS_TEST_FOR_EXCEPTION(disp->get_map()->get_num_local_elements()!=disp->get_map()->get_num_global_elements(),
  //  std::runtime_error,"Error, this method is not implemented for parallel yet");
  const int_t num_approx_points = all_on_zero_coords->get_map()->get_num_local_elements()/2;

  std::vector<computed_point> approx_points(num_approx_points);
  for(int_t i=0;i<num_approx_points;++i){
    const scalar_t x = all_on_zero_coords->local_value(i*2+0);
    const scalar_t y = all_on_zero_coords->local_value(i*2+1);
    approx_points[i].x_ = x;
    approx_points[i].y_ = y;
    approx_points[i].bx_ = all_on_zero_disp->local_value(i*2+0);
    approx_points[i].by_ = all_on_zero_disp->local_value(i*2+1);
    approx_points[i].sol_bx_ = all_on_zero_exact_disp->local_value(i*2+0);
    approx_points[i].sol_by_ = all_on_zero_exact_disp->local_value(i*2+1);
    approx_points[i].error_bx_ = all_on_zero_disp_error->local_value(i*2+0);
    approx_points[i].error_by_ = all_on_zero_disp_error->local_value(i*2+1);
  }

  scalar_t avg_error_x = 0.0;
  scalar_t std_dev_x = 0.0;
  scalar_t avg_error_y = 0.0;
  scalar_t std_dev_y = 0.0;
  if(p_rank==0){
    // now sort the approximate points to pick out the peaks
    int_t num_peaks = ((int_t)(img_w/period)-2)*((int_t)(img_h/period)-2);
    if(num_peaks > 100) num_peaks=100;
    TEUCHOS_TEST_FOR_EXCEPTION(num_peaks<=5,std::runtime_error,"");
    // bx-sort
    std::sort(std::begin(approx_points), std::end(approx_points), [](const computed_point& a, const computed_point& b)
      {
      // sort based on bx alone (assumes that by is a peak at the same location
      return a.bx_ > b.bx_;
      });
    // compute the average of the top num_peaks error
    for(int_t i=0;i<num_peaks;++i){
      //*outStream << "peak: " << i << " command bx: " << approx_points[i].sol_bx_ << " comp bx: " << approx_points[i].bx_ <<  " rel error bx: " << approx_points[i].error_bx_ << "%" <<  std::endl;
      avg_error_x += std::abs(approx_points[i].error_bx_);
    }
    avg_error_x /= num_peaks;
    for(int_t i=0;i<num_peaks;++i){
      std_dev_x += (approx_points[i].error_bx_-avg_error_x)*(approx_points[i].error_bx_-avg_error_x);
    }
    std_dev_x /= num_peaks;
    std_dev_x = std::sqrt(std_dev_x);
    // by-sort
    std::sort(std::begin(approx_points), std::end(approx_points), [](const computed_point& a, const computed_point& b)
      {
      // sort based on bx alone (assumes that by is a peak at the same location
      return a.by_ > b.by_;
      });
    // compute the average of the top num_peaks error
    for(int_t i=0;i<num_peaks;++i){
      //*outStream << "peak: " << i << " command by: " << approx_points[i].sol_by_ << " rel error by: " << approx_points[i].error_by_ << std::endl;
      avg_error_y += std::abs(approx_points[i].error_by_);
    }
    avg_error_y /= num_peaks;
    for(int_t i=0;i<num_peaks;++i){
      std_dev_y += (approx_points[i].error_by_-avg_error_y)*(approx_points[i].error_by_-avg_error_y);
    }
    std_dev_y /= num_peaks;
    std_dev_y = std::sqrt(std_dev_y);
  }
  // communicate results back to dist procs
  Teuchos::Array<int_t> results_on_zero_ids;
  if(p_rank==0){
    results_on_zero_ids.resize(4);
    for(int_t i=0;i<4;++i)
       results_on_zero_ids[i] = i;
  }
  // create a field to send out the results to the other procs:
  Teuchos::RCP<MultiField_Map> results_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, results_on_zero_ids, 0, comm));
  Teuchos::RCP<MultiField> results_on_zero_field = Teuchos::rcp( new MultiField(results_on_zero_map,1,true));
  // set the values on zero
  if(p_rank==0){
    results_on_zero_field->local_value(0) = avg_error_x;
    results_on_zero_field->local_value(1) = std_dev_x;
    results_on_zero_field->local_value(2) = avg_error_y;
    results_on_zero_field->local_value(3) = std_dev_y;
  }

  // create the all on all map
  Teuchos::Array<int_t> results_on_all_ids;
  results_on_all_ids.resize(4);
  for(int_t i=0;i<4;++i)
    results_on_all_ids[i] = i;

  // create a field to send out the results to the other procs:
  Teuchos::RCP<MultiField_Map> results_on_all_map = Teuchos::rcp (new MultiField_Map(-1, results_on_all_ids, 0, comm));
  Teuchos::RCP<MultiField> results_on_all_field = Teuchos::rcp( new MultiField(results_on_all_map,1,true));

  // create exporter with the map
  MultiField_Exporter exporter_res(*results_on_all_map,*results_on_zero_field->get_map());
  // export the field to zero
  results_on_all_field->do_import(results_on_zero_field,exporter_res);

  if(p_rank==0){
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(0)!=avg_error_x,std::runtime_error,"Error miscommunication of field stats across processors");
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(1)!=std_dev_x,std::runtime_error,"Error miscommunication of field stats across processors");
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(2)!=avg_error_y,std::runtime_error,"Error miscommunication of field stats across processors");
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(3)!=std_dev_y,std::runtime_error,"Error miscommunication of field stats across processors");
  }else{
    avg_error_x = results_on_all_field->local_value(0);
    std_dev_x = results_on_all_field->local_value(1);
    avg_error_y = results_on_all_field->local_value(2);
    std_dev_y = results_on_all_field->local_value(3);
  }
  peaks_avg_error_x = avg_error_x;
  peaks_avg_error_y = avg_error_y;
  peaks_std_dev_error_x = std_dev_x;
  peaks_std_dev_error_y = std_dev_y;
}

Teuchos::RCP<Image> create_synthetic_speckle_image(const int_t w,
  const int_t h,
  const scalar_t & speckle_size,
  const Teuchos::RCP<Teuchos::ParameterList> & params){

  const scalar_t period = speckle_size * 2;
  assert(period > 0.0);
  const scalar_t freq = 1.0/period;
  const scalar_t gamma = freq*DICE_TWOPI;
  const intensity_t mag = 255.0*0.5;

  Teuchos::ArrayRCP<intensity_t> intensities(w*h,0.0);
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      intensities[y*w+x] = mag + mag*std::cos(gamma*x)*std::cos(gamma*y);
    }
  }
  Teuchos::RCP<Image> img = Teuchos::rcp(new Image(w,h,intensities,params));
  return img;
}

void add_noise_to_image(Teuchos::RCP<Image> & image,
  const scalar_t & noise_percent){

  // convert noise_percent to counts:
  // rip through the image and find the max intensity
  scalar_t max_intensity = 0.0;
  for(int_t i=0;i<image->width()*image->height();++i){
    if((*image)(i)>max_intensity)
      max_intensity = (*image)(i);
  }
  const scalar_t std_dev = noise_percent*0.01*max_intensity;
  DEBUG_MSG("add_noise_to_image(): max intensity:    " << max_intensity << " counts");
  DEBUG_MSG("add_noise_to_image(): std dev of noise: " << std_dev << " counts");
  std::default_random_engine generator;
  std::normal_distribution<intensity_t> distribution(0.0,std_dev);
  for(int_t i=0;i<image->width()*image->height();++i){
    image->intensities()[i] += distribution(generator);
  }
}


}// End DICe Namespace
