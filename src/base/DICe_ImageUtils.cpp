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
#include <DICe_Schema.h>
#include <DICe_PostProcessor.h>

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
  // pattern repeats every 500 pixels
  // TODO should this depend on the image dims?
  const scalar_t L = 500.0;
  bx = 0.0;
  by = 0.0;
//  for(int_t i=0;i<num_steps_;++i){
//    const scalar_t beta = (i+1)*DICE_PI/L;
//    bx += sin(beta*coord_x)*cos(beta*coord_y)*0.5/(i+1);
//    by += -cos(beta*coord_x)*sin(beta*coord_y)*0.5/(i+1);
  //  }
  const scalar_t beta = (num_steps_+1)*DICE_PI/L;
  bx += sin(beta*coord_x)*cos(beta*coord_y);
  by += -cos(beta*coord_x)*sin(beta*coord_y);
}

void SinCos_Image_Deformer::compute_deriv_deformation(const scalar_t & coord_x,
  const scalar_t & coord_y,
  scalar_t & bxx,
  scalar_t & bxy,
  scalar_t & byx,
  scalar_t & byy){
  // pattern repeats every 500 pixels
  // TODO should this depend on the image dims?
  const scalar_t L = 500.0;
  bxx = 0.0;
  bxy = 0.0;
  byx = 0.0;
  byy = 0.0;
//  for(int_t i=0;i<num_steps_;++i){
//    const scalar_t beta = (i+1)*DICE_PI/L;
//    bxx += beta*cos(beta*coord_x)*cos(beta*coord_y)*0.5/(i+1);
//    bxy += -beta*sin(beta*coord_x)*sin(beta*coord_y)*0.5/(i+1);
//    byx += beta*sin(beta*coord_x)*sin(beta*coord_y)*0.5/(i+1);
//    byy += -beta*cos(beta*coord_x)*cos(beta*coord_y)*0.5/(i+1);
//  }
  const scalar_t beta = (num_steps_+1)*DICE_PI/L;
  bxx = beta*cos(beta*coord_x)*cos(beta*coord_y);
  bxy = -beta*sin(beta*coord_x)*sin(beta*coord_y);
  byx = beta*sin(beta*coord_x)*sin(beta*coord_y);
  byy = -beta*cos(beta*coord_x)*cos(beta*coord_y);
}

void SinCos_Image_Deformer::compute_displacement_error(const scalar_t & coord_x,
  const scalar_t & coord_y,
  const scalar_t & sol_x,
  const scalar_t & sol_y,
  scalar_t & error_x,
  scalar_t & error_y){

  scalar_t out_x = 0.0;
  scalar_t out_y = 0.0;
  compute_deformation(coord_x,coord_y,out_x,out_y);
  error_x = (sol_x - out_x)*(sol_x - out_x);
  error_y = (sol_y - out_y)*(sol_y - out_y);
}

void SinCos_Image_Deformer::compute_deriv_error(const scalar_t & coord_x,
  const scalar_t & coord_y,
  const scalar_t & sol_x,
  const scalar_t & sol_y,
  scalar_t & error_x,
  scalar_t & error_y){

  scalar_t out_xx = 0.0;
  scalar_t out_xy = 0.0;
  scalar_t out_yx = 0.0;
  scalar_t out_yy = 0.0;
  compute_deriv_deformation(coord_x,coord_y,out_xx,out_xy,out_yx,out_yy);
  error_x = (sol_x - out_xx)*(sol_x - out_xx);
  error_y = (sol_y - out_yy)*(sol_y - out_yy);
}



Teuchos::RCP<Image>
SinCos_Image_Deformer::deform_image(Teuchos::RCP<Image> ref_image){
  const int_t w = ref_image->width();
  const int_t h = ref_image->height();
  // Note: uses 5 x 5 point sampling grid to evaluate the deformed intensity
  const int_t num_pts = 5;
  static scalar_t coeffs[5] = {0.0014,0.1574,0.62825,0.1574,0.0014};
  // Note: uses 11 x 11 point sampling grid to evaluate the deformed intensity
//  static scalar_t coeffs[11] =
//  {0.0001,0.0017,0.0168,0.0870,
//    0.2328,0.3231,0.2328,
//    0.0870,0.0168,0.0017,0.0001};
  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
  scalar_t bx=0.0,by=0.0;
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      scalar_t avg_intens = 0.0;
      for(int_t oy=0;oy<num_pts;++oy){
        const scalar_t sample_y = j - 0.5*(num_pts-1)/num_pts + oy/num_pts;
        for(int_t ox=0;ox<num_pts;++ox){
          const scalar_t sample_x = i - 0.5*(num_pts-1)/num_pts + ox/num_pts;
          const scalar_t weight = coeffs[ox]*coeffs[oy];
          compute_deformation(sample_x,sample_y,bx,by);
          scalar_t intens = ref_image->interpolate_keys_fourth(sample_x-bx,sample_y-by);
          avg_intens += weight*intens;
        } // end super pixel ox
      } // end super pixel oy
      def_intens[j*w+i] = avg_intens;
    } // end pixel i
  } // ens pixel j
  Teuchos::RCP<Image> def_img = Teuchos::rcp(new Image(w,h,def_intens));
  return def_img;
}

void
SinCos_Image_Deformer::estimate_resolution_error(DICe::Schema * schema,
  const int num_steps,
  std::string & output_folder,
  std::string & prefix,
  Teuchos::RCP<std::ostream> & outStream){

  std::stringstream result_stream;
  // create an image deformer class
  DEBUG_MSG("SinCos_Image_Deformer::estimate_resolution_error(): Estimating resolution error with num_steps = " << num_steps);

  for(int_t step=0;step<num_steps;++step){
    DEBUG_MSG("Processing step " << step);
    num_steps_ = step;
    Teuchos::RCP<Image> def_img = deform_image(schema->ref_img());
    std::stringstream sincos_name;
    sincos_name << "sincos_iamge_" << step << ".tif";
    def_img->write(sincos_name.str());

    // set the deformed image for the schema
    schema->set_def_image(def_img);
    int_t corr_error = schema->execute_correlation();
    DEBUG_MSG("Error prediction step correlation return value " << corr_error);
    schema->write_output(output_folder,prefix,false,true);
    schema->post_execution_tasks();

    // compute the error:
    scalar_t h1x_error = 0.0;
    scalar_t h1y_error = 0.0;
    scalar_t hinfx_error = 0.0;
    scalar_t hinfy_error = 0.0;
    scalar_t avgx_error = 0.0;
    scalar_t avgy_error = 0.0;
    for(int_t i=0;i<schema->data_num_points();++i){
      const scalar_t x = schema->field_value(i,DICe::COORDINATE_X);
      const scalar_t y = schema->field_value(i,DICe::COORDINATE_Y);
      const scalar_t u = schema->field_value(i,DICe::DISPLACEMENT_X);
      const scalar_t v = schema->field_value(i,DICe::DISPLACEMENT_Y);
      scalar_t e_x;
      scalar_t e_y;
      compute_displacement_error(x,y,u,v,e_x,e_y);
      h1x_error += e_x;
      h1y_error += e_y;
      avgx_error += std::sqrt(e_x);
      avgy_error += std::sqrt(e_y);
      if(e_x > hinfx_error) hinfx_error = e_x;
      if(e_y > hinfy_error) hinfy_error = e_y;
    }
    h1x_error = std::sqrt(h1x_error);
    h1y_error = std::sqrt(h1y_error);
    hinfx_error = std::sqrt(hinfx_error);
    hinfy_error = std::sqrt(hinfy_error);
    avgx_error /= schema->data_num_points();
    avgy_error /= schema->data_num_points();
    scalar_t std_dev_x = 0.0;
    scalar_t std_dev_y = 0.0;
    for(int_t i=0;i<schema->data_num_points();++i){
      const scalar_t x = schema->field_value(i,DICe::COORDINATE_X);
      const scalar_t y = schema->field_value(i,DICe::COORDINATE_Y);
      const scalar_t u = schema->field_value(i,DICe::DISPLACEMENT_X);
      const scalar_t v = schema->field_value(i,DICe::DISPLACEMENT_Y);
      scalar_t e_x;
      scalar_t e_y;
      compute_displacement_error(x,y,u,v,e_x,e_y);
      e_x = std::sqrt(e_x);
      e_y = std::sqrt(e_y);
      std_dev_x += (e_x - avgx_error)*(e_x - avgx_error);
      std_dev_y += (e_y - avgy_error)*(e_y - avgy_error);
    }
    std_dev_x = std::sqrt(1.0/schema->data_num_points()*std_dev_x);
    std_dev_y = std::sqrt(1.0/schema->data_num_points()*std_dev_y);
    result_stream << "Step: " << std::setw(3) << step << " ERRORS H_1 x: " << std::setw(8) << h1x_error << " H_1 y: " << std::setw(8) << h1y_error <<
        " H_inf x: " << std::setw(8) << hinfx_error << " H_inf error y: " << std::setw(8) << hinfy_error << " std_dev x: " << std::setw(8) << std_dev_x <<
        " std_dev y: " << std::setw(8) << std_dev_y << std::endl;

    scalar_t sh1x_error = 0.0;
    scalar_t sh1y_error = 0.0;
    scalar_t shinfx_error = 0.0;
    scalar_t shinfy_error = 0.0;
    scalar_t savgx_error = 0.0;
    scalar_t savgy_error = 0.0;
    // FIXME assumes that the first post_processor gives the strain values
    TEUCHOS_TEST_FOR_EXCEPTION(schema->post_processors()->size()<=0,std::runtime_error,"Error, VSG post processor not enabled, but needs to be for image deformer error calcs");
    for(int_t i=0;i<schema->data_num_points();++i){
      const scalar_t x = schema->field_value(i,DICe::COORDINATE_X);
      const scalar_t y = schema->field_value(i,DICe::COORDINATE_Y);
      const scalar_t exx = (*schema->post_processors())[0]->field_value(i,vsg_strain_xx);
      const scalar_t eyy = (*schema->post_processors())[0]->field_value(i,vsg_strain_yy);
      scalar_t e_x;
      scalar_t e_y;
      compute_deriv_error(x,y,exx,eyy,e_x,e_y);
      sh1x_error += e_x;
      sh1y_error += e_y;
      savgx_error += std::sqrt(e_x);
      savgy_error += std::sqrt(e_y);
      if(e_x > shinfx_error) shinfx_error = e_x;
      if(e_y > shinfy_error) shinfy_error = e_y;
    }
    sh1x_error = std::sqrt(sh1x_error);
    sh1y_error = std::sqrt(sh1y_error);
    shinfx_error = std::sqrt(shinfx_error);
    shinfy_error = std::sqrt(shinfy_error);
    savgx_error /= schema->data_num_points();
    savgy_error /= schema->data_num_points();
    scalar_t std_dev_sx = 0.0;
    scalar_t std_dev_sy = 0.0;
    for(int_t i=0;i<schema->data_num_points();++i){
      const scalar_t x = schema->field_value(i,DICe::COORDINATE_X);
      const scalar_t y = schema->field_value(i,DICe::COORDINATE_Y);
      const scalar_t exx = (*schema->post_processors())[0]->field_value(i,vsg_strain_xx);
      const scalar_t eyy = (*schema->post_processors())[0]->field_value(i,vsg_strain_yy);
      scalar_t e_x;
      scalar_t e_y;
      compute_deriv_error(x,y,exx,eyy,e_x,e_y);
      e_x = std::sqrt(e_x);
      e_y = std::sqrt(e_y);
      std_dev_sx += (e_x - savgx_error)*(e_x - savgx_error);
      std_dev_sy += (e_y - savgy_error)*(e_y - savgy_error);
    }
    std_dev_sx = std::sqrt(1.0/schema->data_num_points()*std_dev_sx);
    std_dev_sy = std::sqrt(1.0/schema->data_num_points()*std_dev_sy);
    result_stream << "Step: " << std::setw(3) << step << " STRAIN ERRORS H_1 x: " << std::setw(8) << sh1x_error << " H_1 y: " << std::setw(8) << sh1y_error <<
        " H_inf x: " << std::setw(8) << shinfx_error << " H_inf error y: " << std::setw(8) << shinfy_error << " std_dev x: " << std::setw(8) << std_dev_sx <<
        " std_dev y: " << std::setw(8) << std_dev_sy << std::endl;
  }

  schema->write_stats(output_folder,prefix);

  // write the results to the .info file
  std::stringstream infoName;
  infoName << output_folder << prefix << ".info";
  std::FILE * infoFilePtr = fopen(infoName.str().c_str(),"a");
  fprintf(infoFilePtr,"***\n");
  fprintf(infoFilePtr,"*** Displacement and Strain Error Estimates \n");
  fprintf(infoFilePtr,"***\n");
  fprintf(infoFilePtr,result_stream.str().c_str());
  fprintf(infoFilePtr,"***\n");
  fprintf(infoFilePtr,"***\n");
  fprintf(infoFilePtr,"***\n");
  fclose(infoFilePtr);

  *outStream << result_stream.str();
}



}// End DICe Namespace
