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
#include <DICe_Image.h>
#include <DICe_Shape.h>
#include <DICe_LocalShapeFunction.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create an image from file:
  *outStream << "creating an image from a tiff file " << std::endl;
  Teuchos::RCP<Image> img = Teuchos::rcp(new Image("./images/ImageA.tif"));
  //img->write("outImageA.tif");
  if(img->width()!=2048){
    *outStream << "Error, the image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(img->height()!=589){
    *outStream << "Error, the image height is not correct" << std::endl;
    errorFlag +=1;
  }

  // test the mask values for an image:
  *outStream << "testing image mask" << std::endl;
  // create a mask using a simple rectangle
  Teuchos::RCP<DICe::Rectangle> rect1 = Teuchos::rcp(new DICe::Rectangle(1024,294,401,200));
  Teuchos::RCP<DICe::Rectangle> rect2 = Teuchos::rcp(new DICe::Rectangle(1024,294,101,50));
  DICe::multi_shape boundary;
  boundary.push_back(rect1);
  DICe::multi_shape excluded;
  excluded.push_back(rect2);
  DICe::Conformal_Area_Def area_def(boundary,excluded);
  img->create_mask(area_def,true);

  // create an image of the mask
  Teuchos::ArrayRCP<intensity_t> mask_values(img->height()*img->width(),0.0);
  for(int_t y=0;y<img->height();++y)
    for(int_t x=0;x<img->width();++x)
      mask_values[y*img->width()+x] = img->mask(x,y);
  Image mask(img->width(),img->height(),mask_values);
  //mask.write("mask_d.rawi");
  // compare with the saved mask file
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> mask_exact = Teuchos::rcp(new Image("./images/mask_d.rawi"));
#else
  Teuchos::RCP<Image> mask_exact = Teuchos::rcp(new Image("./images/mask.rawi"));
#endif
  const scalar_t mask_diff = mask.diff(mask_exact);
  *outStream << "mask value diff: " << mask_diff << std::endl;
  const scalar_t mask_tol = 1.0E-2;
  if(mask_diff > mask_tol){
    *outStream << "Error, mask values not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "image mask values have been checked" << std::endl;

  // capture a portion of an image from file
  *outStream << "creating an image from a portion of a tiff file " << std::endl;
  Image sub_img("./images/ImageA.tif",100,100,300,200);
  //sub_img.write("outSubImageA.tif");
  if(sub_img.width()!=300){
    *outStream << "Error, the sub image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(sub_img.height()!=200){
    *outStream << "Error, the sub image height is not correct" << std::endl;
    errorFlag +=1;
  }
  // the pixel values from the sub image should line up with the image above, if given the right coordinates
  bool intensity_match_error = false;
  for(int_t y=0;y<sub_img.height();++y){
    for(int_t x=0;x<sub_img.width();++x){
      if(sub_img(x,y)!=(*img)(x+sub_img.offset_x(),y+sub_img.offset_y()))
        intensity_match_error = true;
    }
  }
  if(intensity_match_error){
    *outStream << "Error, the intensities for the sub image do not match the global image" << std::endl;
    errorFlag+=1;
  }

  *outStream << "creating a sub-image" << std::endl;
  // purposefully making the image extend beyond the bounds of the input image
  Teuchos::RCP<Image> portion = Teuchos::rcp(new Image(img,img->width()/2,img->height()/2,img->width(),img->height()));
  //portion->write("portion_d.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> portion_exact = Teuchos::rcp(new Image("./images/portion_d.rawi"));
#else
  Teuchos::RCP<Image> portion_exact = Teuchos::rcp(new Image("./images/portion.rawi"));
#endif
  scalar_t portion_diff = portion_exact->diff(portion);
  if(portion_diff > mask_tol){
    *outStream << "Error, the portion image is not correct" << std::endl;
    errorFlag++;
  }

  // create an image from an array
  *outStream << "creating an image from an array" << std::endl;
  const int_t array_w = 30;
  const int_t array_h = 20;
  intensity_t * intensities = new intensity_t[array_w*array_h];
  scalar_t * gx = new scalar_t[array_w*array_h];
  scalar_t * gy = new scalar_t[array_w*array_h];
  // populate the intensities with a sin/cos function
  for(int_t y=0;y<array_h;++y){
    for(int_t x=0;x<array_w;++x){
      intensities[y*array_w+x] = 255*std::cos(x/(4*DICE_PI))*std::sin(y/(4*DICE_PI));
      gx[y*array_w+x] = -255*(1/(4*DICE_PI))*std::sin(x/(4*DICE_PI))*std::sin(y/(4*DICE_PI));
      gy[y*array_w+x] = 255*(1/(4*DICE_PI))*std::cos(x/(4*DICE_PI))*std::cos(y/(4*DICE_PI));
    }
  }
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::compute_image_gradients,true);
  Image array_img(intensities,array_w,array_h,params);

  *outStream << "creating an image from an array RCP" << std::endl;
  Teuchos::ArrayRCP<intensity_t> intensityRCP(intensities,0,array_w*array_h,false);
  Image rcp_img(array_w,array_h,intensityRCP);

  bool intensity_value_error = false;
  bool grad_x_error = false;
  bool grad_y_error = false;
  scalar_t grad_tol = 1.0E-3;
  for(int_t y=2;y<array_h-2;++y){
    for(int_t x=2;x<array_w-2;++x){
      if(intensities[y*array_w+x] != array_img(x,y))
        intensity_value_error = true;
      //std::cout << " grad x " << gx[y*array_w+x] << " computed " << array_img.grad_x(x,y) << std::endl;
      if(std::abs(gx[y*array_w+x] - array_img.grad_x(x,y)) > grad_tol)
        grad_x_error = true;
      if(std::abs(gy[y*array_w+x] - array_img.grad_y(x,y)) > grad_tol)
        grad_y_error = true;
    }
  }
  if(intensity_value_error){
    *outStream << "Error, the intensity values are wrong" << std::endl;
    errorFlag++;
  }
  if(grad_x_error){
    *outStream << "Error, the flat x-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  if(grad_y_error){
    *outStream << "Error, the flat y-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "flat image gradients have been checked" << std::endl;

  grad_x_error = false;
  grad_y_error = false;
  // check the hierarchical gradients:
  const int_t team_size = 256;
  array_img.compute_gradients(true,team_size);
  for(int_t y=2;y<array_h-2;++y){
    for(int_t x=2;x<array_w-2;++x){
      if(std::abs(gx[y*array_w+x] - array_img.grad_x(x,y)) > grad_tol)
        grad_x_error = true;
      if(std::abs(gy[y*array_w+x] - array_img.grad_y(x,y)) > grad_tol)
        grad_y_error = true;
    }
  }
  if(grad_x_error){
    *outStream << "Error, hierarchical x-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  if(grad_y_error){
    *outStream << "Error, the hierarchical y-gradient values are wrong" << std::endl;
    errorFlag++;
  }
  *outStream << "hierarchical image gradients have been checked" << std::endl;

  // create an image with a teuchos array and compare the values
  bool rcp_error = false;
  for(int_t y=0;y<array_h;++y){
    for(int_t x=0;x<array_w;++x){
      if(rcp_img(x,y) != array_img(x,y))
        rcp_error = true;
    }
  }
  if(rcp_error){
    *outStream << "Error, the rcp image was not created correctly" << std::endl;
    errorFlag++;
  }

  delete[] intensities;
  delete[] gx;
  delete[] gy;

  // test image transform:
  *outStream << "testing an image transform" << std::endl;
#ifdef DICE_USE_DOUBLE
  Image baboon("./images/baboon_d.rawi");
#else
  Image baboon("./images/baboon.rawi");
#endif
  //Image baboon("./images/baboon.tif");
  //baboon.write("baboon_d.rawi");
  const int_t cx = 100;
  const int_t cy = 100;
  const scalar_t u = 25.0;
  const scalar_t v = -15.2;
  const scalar_t theta = 22.8*DICE_PI/180.0;

  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory();
  shape_function->insert_motion(u,v,theta);
  Teuchos::RCP<Image> trans_baboon = baboon.apply_transformation(shape_function,cx,cy,false);
  //trans_baboon->write("baboon_trans.tiff");
  //baboon.write("baboon.tiff");
  //trans_baboon->write("baboon_trans_d.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> trans_baboon_exact = Teuchos::rcp(new Image("./images/baboon_trans_d.rawi"));
#else
  Teuchos::RCP<Image> trans_baboon_exact = Teuchos::rcp(new Image("./images/baboon_trans.rawi"));
#endif
  //trans_baboon_exact->write("baboon_trans_exact.tif");
  const scalar_t diff_trans_baboon = trans_baboon->diff(trans_baboon_exact);
  *outStream << "image diff trans baboon: " << diff_trans_baboon << std::endl;
  const scalar_t diff_tol = 0.1;
  if(diff_trans_baboon > diff_tol){
    *outStream << "Error, the transformed image does not have the right intensities" << std::endl;
    errorFlag++;
  }

  *outStream << "testing 180 rotation in place" << std::endl;
  Teuchos::RCP<Image> baboon_180 = baboon.apply_rotation(ONE_HUNDRED_EIGHTY_DEGREES);
  //baboon_180->write("baboon_180_d.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> baboon_180_exact = Teuchos::rcp(new Image("./images/baboon_180_d.rawi"));
#else
  Teuchos::RCP<Image> baboon_180_exact = Teuchos::rcp(new Image("./images/baboon_180.rawi"));
#endif
  const scalar_t diff_180 = baboon_180->diff(baboon_180_exact);
  if(diff_180 > diff_tol){
    *outStream << "Error, the 180 degree transformed image does not have the right intensities." << std::endl;
    errorFlag++;
  }
  *outStream << "the transformation method for an image has been checked" << std::endl;

  // test 90 and 270 degree rotation

  *outStream << "testing 90 and 270 degree rotations" << std::endl;
  Teuchos::RCP<Image> img_0_deg = Teuchos::rcp(new Image("./images/ImageB.tif"));
  Teuchos::RCP<Image> img_90_deg = img_0_deg->apply_rotation(NINTY_DEGREES);
  //img_90_deg->write("img_90_d.rawi");
  //img_90_deg->write("img_90.rawi");
  Teuchos::RCP<Image> img_270_deg = img_0_deg->apply_rotation(TWO_HUNDRED_SEVENTY_DEGREES);
  //img_270_deg->write("img_270_d.rawi");
  //img_270_deg->write("img_270.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> img_90_exact = Teuchos::rcp(new Image("./images/img_90_d.rawi"));
#else
  Teuchos::RCP<Image> img_90_exact = Teuchos::rcp(new Image("./images/img_90.rawi"));
#endif
  const scalar_t diff_90 = img_90_deg->diff(img_90_exact);
  if(diff_90 > 1.0E-4){
    *outStream << "Error, the 90 degree transformed image does not have the right intensities." << std::endl;
    errorFlag++;
  }
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> img_270_exact = Teuchos::rcp(new Image("./images/img_270_d.rawi"));
#else
  Teuchos::RCP<Image> img_270_exact = Teuchos::rcp(new Image("./images/img_270.rawi"));
#endif
  const scalar_t diff_270 = img_270_deg->diff(img_270_exact);
  if(diff_270 > 1.0E-4){
    *outStream << "Error, the 270 degree transformed image does not have the right intensities." << std::endl;
    errorFlag++;
  }
  *outStream << "rotations have been tested" << std::endl;

  // test filtering an image:

  *outStream << "creating gauss filtered image filter size 5: outFilter5.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_5_params = rcp(new Teuchos::ParameterList());
  filter_5_params->set(DICe::gauss_filter_images,true);
  filter_5_params->set(DICe::gauss_filter_mask_size,5);
  Teuchos::RCP<Image> filter_5_img = Teuchos::rcp(new Image("./images/ImageB.tif",filter_5_params));
  //filter_5_img->write("outFilter5_d.rawi");
  //filter_5_img.write("outFilter5.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> filter_5_exact = Teuchos::rcp(new Image("./images/outFilter5_d.rawi"));
#else
  Teuchos::RCP<Image> filter_5_exact = Teuchos::rcp(new Image("./images/outFilter5.rawi"));
#endif
  const scalar_t diff_5 = filter_5_exact->diff(filter_5_img);
  if(diff_5 > diff_tol){
    *outStream << "Error, the 5 pixel filter image does not have the right intensities." << std::endl;
    errorFlag++;
  }

  *outStream << "creating gauss filtered image filter size 7: outFilter7.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_7_params = rcp(new Teuchos::ParameterList());
  filter_7_params->set(DICe::gauss_filter_images,true);
  filter_7_params->set(DICe::gauss_filter_mask_size,7);
  Teuchos::RCP<Image> filter_7_img = Teuchos::rcp(new Image("./images/ImageB.tif",filter_7_params));
  //filter_7_img->write("outFilter7_d.rawi");
  //filter_7_img.write("outFilter7.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> filter_7_exact = Teuchos::rcp(new Image("./images/outFilter7_d.rawi"));
#else
  Teuchos::RCP<Image> filter_7_exact = Teuchos::rcp(new Image("./images/outFilter7.rawi"));
#endif
  const scalar_t diff_7 = filter_7_exact->diff(filter_7_img);
  if(diff_7 > diff_tol){
    *outStream << "Error, the 7 pixel filter image does not have the right intensities." << std::endl;
    errorFlag++;
  }

  *outStream << "creating gauss filtered image filter size 9: outFilter9.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_9_params = rcp(new Teuchos::ParameterList());
  filter_9_params->set(DICe::gauss_filter_images,true);
  filter_9_params->set(DICe::gauss_filter_mask_size,9);
  Teuchos::RCP<Image> filter_9_img = Teuchos::rcp(new Image("./images/ImageB.tif",filter_9_params));
  //filter_9_img->write("outFilter9_d.rawi");
  //filter_9_img.write("outFilter9.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> filter_9_exact = Teuchos::rcp(new Image("./images/outFilter9_d.rawi"));
#else
  Teuchos::RCP<Image> filter_9_exact = Teuchos::rcp(new Image("./images/outFilter9.rawi"));
#endif
  const scalar_t diff_9 = filter_9_exact->diff(filter_9_img);
  if(diff_9 > diff_tol){
    *outStream << "Error, the 9 pixel filter image does not have the right intensities." << std::endl;
    errorFlag++;
  }

  *outStream << "creating gauss filtered image filter size 11: outFilter11.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_11_params = rcp(new Teuchos::ParameterList());
  filter_11_params->set(DICe::gauss_filter_images,true);
  filter_11_params->set(DICe::gauss_filter_mask_size,11);
  Teuchos::RCP<Image> filter_11_img = Teuchos::rcp(new Image("./images/ImageB.tif",filter_11_params));
  //filter_11_img->write("outFilter11_d.rawi");
  //filter_11_img.write("outFilter11.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> filter_11_exact = Teuchos::rcp(new Image("./images/outFilter11_d.rawi"));
#else
  Teuchos::RCP<Image> filter_11_exact = Teuchos::rcp(new Image("./images/outFilter11.rawi"));
#endif
  const scalar_t diff_11 = filter_11_exact->diff(filter_11_img);
  if(diff_11 > diff_tol){
    *outStream << "Error, the 11 pixel filter image does not have the right intensities." << std::endl;
    errorFlag++;
  }

  *outStream << "creating gauss filtered image filter size 13: outFilter13.tif " << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_13_params = rcp(new Teuchos::ParameterList());
  filter_13_params->set(DICe::gauss_filter_images,true);
  filter_13_params->set(DICe::gauss_filter_mask_size,13);
  Teuchos::RCP<Image> filter_13_img = Teuchos::rcp(new Image("./images/ImageB.tif",filter_13_params));
  //filter_13_img->write("outFilter13_d.rawi");
  //filter_13_img.write("outFilter13.rawi");
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> filter_13_exact = Teuchos::rcp(new Image("./images/outFilter13_d.rawi"));
#else
  Teuchos::RCP<Image> filter_13_exact = Teuchos::rcp(new Image("./images/outFilter13.rawi"));
#endif
  const scalar_t diff_13 = filter_13_exact->diff(filter_13_img);
  if(diff_13 > diff_tol){
    *outStream << "Error, the 13 pixel filter image does not have the right intensities." << std::endl;
    errorFlag++;
  }

  *outStream << "creating gauss filtered image filter size 13 (with hierarchical parallelism) to compare to flat filter functor" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> filter_h_params = rcp(new Teuchos::ParameterList());
  filter_h_params->set(DICe::gauss_filter_images,true);
  filter_h_params->set(DICe::gauss_filter_use_hierarchical_parallelism,true);
  filter_h_params->set(DICe::gauss_filter_team_size,256);
  filter_h_params->set(DICe::gauss_filter_mask_size,13);
  Image filter_h_img("./images/ImageB.tif",filter_h_params);

  bool filter_error = false;
  for(int_t y=0;y<filter_h_img.height();++y){
    for(int_t x=0;x<filter_h_img.width();++x){
      if(std::abs(filter_h_img(x,y) - (*filter_13_img)(x,y)) > grad_tol)
        filter_error = true;
    }
  }
  if(filter_error){
    *outStream << "Error, the filtered image using the flat functor does not match the hierarchical one" << std::endl;
    errorFlag++;
  }
  *outStream << "hierarchical image filter has been checked" << std::endl;

  bool exception_thrown = false;

#if DICE_JPEG
  // create an image from jpeg file:
  *outStream << "creating an image from a jpeg file " << std::endl;
  Teuchos::RCP<Image> img_jpg = Teuchos::rcp(new Image("./images/ImageB.jpg"));
  //img_jpg->write("JpegImageB_d.rawi");
  //img_jpg->write("JpegImageB.rawi");
  if(img_jpg->width()!=240){
    *outStream << "Error, the jpeg image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(img_jpg->height()!=161){
    *outStream << "Error, the jpeg image height is not correct" << std::endl;
    errorFlag +=1;
  }
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> img_jpg_exact = Teuchos::rcp(new Image("./images/JpegImageB_d.rawi"));
#else
  Teuchos::RCP<Image> img_jpg_exact = Teuchos::rcp(new Image("./images/JpegImageB.rawi"));
#endif
  const scalar_t diff_jpg = img_jpg_exact->diff(img_jpg);
  if(diff_jpg > diff_tol){
    *outStream << "Error, the jpeg image does not have the right intensities." << std::endl;
    errorFlag++;
 }
  *outStream << "creating a sub portion image from a jpeg file " << std::endl;
  Teuchos::RCP<Image> img_sub_jpg = Teuchos::rcp(new Image("./images/ImageB.jpg",10,15,20,30));
  //img_sub_jpg->write("JpegImageBSub_d.rawi");
  //img_sub_jpg->write("JpegImageBSub.rawi");
  if(img_sub_jpg->width()!=20){
    *outStream << "Error, the jpeg sub image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(img_sub_jpg->height()!=30){
    *outStream << "Error, the jpeg sub image height is not correct" << std::endl;
    errorFlag +=1;
  }
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> img_jpg_sub_exact = Teuchos::rcp(new Image("./images/JpegImageBSub_d.rawi"));
#else
  Teuchos::RCP<Image> img_jpg_sub_exact = Teuchos::rcp(new Image("./images/JpegImageBSub.rawi"));
#endif
  const scalar_t diff_jpg_sub = img_jpg_sub_exact->diff(img_sub_jpg);
  if(diff_jpg_sub > diff_tol){
    *outStream << "Error, the jpeg sub image does not have the right intensities." << std::endl;
    errorFlag++;
  }
#else
  // test that unsupported file formats throw an exception
  exception_thrown = false;
  try{
    DICe::Image jpg("./images/invalid.jpg");
  }
  catch (const std::exception &e){
    exception_thrown = true;
    *outStream << ".jpg throws an exception as expected" << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for unsupported .jpg file format, but was not" << std::endl;
    errorFlag++;
  }
#endif
#if DICE_PNG
  // create an image from jpeg file:
  *outStream << "creating an image from a png file " << std::endl;
  Teuchos::RCP<Image> img_png = Teuchos::rcp(new Image("./images/ImageB.png"));
  //img_png->write("PngImageB_d.rawi");
  //img_png->write("PngImageB.rawi");
  if(img_png->width()!=240){
    *outStream << "Error, the png image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(img_png->height()!=161){
    *outStream << "Error, the png image height is not correct" << std::endl;
    errorFlag +=1;
  }
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> img_png_exact = Teuchos::rcp(new Image("./images/PngImageB_d.rawi"));
#else
  Teuchos::RCP<Image> img_png_exact = Teuchos::rcp(new Image("./images/PngImageB.rawi"));
#endif
  const scalar_t diff_png = img_png_exact->diff(img_png);
  if(diff_png > diff_tol){
    *outStream << "Error, the png image does not have the right intensities." << std::endl;
    errorFlag++;
  }
  *outStream << "creating a sub portion image from a png file " << std::endl;
  Teuchos::RCP<Image> img_sub_png = Teuchos::rcp(new Image("./images/ImageB.jpg",10,15,20,30));
  //img_sub_png->write("PngImageBSub_d.rawi");
  //img_sub_png->write("PngImageBSub.rawi");
  if(img_sub_png->width()!=20){
    *outStream << "Error, the png sub image width is not correct" << std::endl;
    errorFlag +=1;
  }
  if(img_sub_png->height()!=30){
    *outStream << "Error, the png sub image height is not correct" << std::endl;
    errorFlag +=1;
  }
#ifdef DICE_USE_DOUBLE
  Teuchos::RCP<Image> img_png_sub_exact = Teuchos::rcp(new Image("./images/PngImageBSub_d.rawi"));
#else
  Teuchos::RCP<Image> img_png_sub_exact = Teuchos::rcp(new Image("./images/PngImageBSub.rawi"));
#endif
  const scalar_t diff_png_sub = img_png_sub_exact->diff(img_sub_png);
  if(diff_png_sub > diff_tol){
    *outStream << "Error, the png sub image does not have the right intensities." << std::endl;
    errorFlag++;
  }
#else
  // test that unsupported file formats throw an exception
  exception_thrown = false;
  try{
    DICe::Image png("./images/invalid.png");
  }
  catch (const std::exception &e){
    exception_thrown = true;
    *outStream << ".png throws an exception as expected" << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for invalid .png file format, but was not" << std::endl;
    errorFlag++;
  }
#endif
  // test that unsupported file formats throw an exception
  try{
    DICe::Image bmp("./images/invalid.bmp");
  }
  catch (const std::exception &e){
    exception_thrown = true;
    *outStream << ".bmp throws an exception as expected" << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for invalid .bmp file format, but was not" << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

