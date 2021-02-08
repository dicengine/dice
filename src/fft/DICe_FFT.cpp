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

#include <DICe_FFT.h>

#include <Teuchos_ArrayRCP.hpp>

#include <cassert>

namespace DICe {

DICE_LIB_DLL_EXPORT
void
complex_divide(kiss_fft_cpx * lhs,
  kiss_fft_cpx * rhs,
  const int_t size)
{
  for(int_t i=0;i<size;++i){

    work_t e = 0.0;
    work_t f = 0.0;
    if( fabs(rhs[i].i)<fabs(rhs[i].r) )
    {
        e = rhs[i].r==0.0?0.0:rhs[i].i/rhs[i].r;
        f = rhs[i].r+rhs[i].i*e;
        lhs[i].r = (lhs[i].r+lhs[i].i*e)/f;
        lhs[i].i = (lhs[i].i-lhs[i].r*e)/f;
    }
    else
    {
        e = rhs[i].i==0.0?0.0:rhs[i].r/rhs[i].i;
        f = rhs[i].i+rhs[i].r*e;
        lhs[i].r = (lhs[i].i+lhs[i].r*e)/f;
        lhs[i].i = (-lhs[i].r+lhs[i].i*e)/f;
    }
  }
}

DICE_LIB_DLL_EXPORT
void
complex_divide(work_t & result_r,
  work_t & result_i,
  const work_t & a_r,
  const work_t & a_i,
  const work_t & b_r,
  const work_t & b_i)
{
  work_t e = 0.0;
  work_t f = 0.0;
  if( fabs(b_i)<fabs(b_r) )
  {
    e = b_r==0.0?0.0:b_i/b_r;
    f = b_r+b_i*e;
    result_r = (a_r+a_i*e)/f;
    result_i = (a_i-a_r*e)/f;
  }
  else
  {
    e = b_i==0.0?0.0:b_r/b_i;
    f = b_i+b_r*e;
    result_r = (a_i+a_r*e)/f;
    result_i = (-a_r+a_i*e)/f;
  }
}

DICE_LIB_DLL_EXPORT
void
complex_multiply(work_t & result_r,
  work_t & result_i,
  const work_t & a_r,
  const work_t & a_i,
  const work_t & b_r,
  const work_t & b_i){
  result_r = a_r*b_r - a_i*b_i;
  result_i = a_i*b_r + a_r*b_i;
}

DICE_LIB_DLL_EXPORT
void
complex_abs(work_t & result,
  const work_t & a_r,
  const work_t & a_i){
  result = std::sqrt(a_r*a_r + a_i*a_i);
}

DICE_LIB_DLL_EXPORT
work_t
phase_correlate_x_y(Teuchos::RCP<Image> image_a,
  Teuchos::RCP<Image> image_b,
  work_t & u_x,
  work_t & u_y,
  const bool convert_to_r_theta){

  const int_t w = image_a->width();
  const int_t h = image_a->height();
  assert(image_b->width()==w && "Error: images must be the same dims");
  assert(image_b->height()==h && "Error: images must be the same dims");
  assert(w>8);
  assert(h>8);

  // test that the images don't have the same intensities, if so return 0,0
  // This hopefully removes the false 0,0 peak
  // TODO address this more formally
  work_t diff_tol = 50.0;
  const work_t diff = image_a->diff(image_b);
  DEBUG_MSG("phase_correlate_x_y(): image diff test result: " << diff);
  if(diff < diff_tol){
    DEBUG_MSG("phase_correlate_x_y(): skipping phase correlation because images are the same.");
    u_x = 0.0;
    u_y = 0.0;
    return -1.0;
  }

  // fft of image a
  Teuchos::ArrayRCP<work_t> a_r,a_i;
  DICe::image_fft(image_a,a_r,a_i,0);

  // fft of image b
  Teuchos::ArrayRCP<work_t> b_r,b_i;
  DICe::image_fft(image_b,b_r,b_i,0);

  // conjugate of image b
  for(int_t i=0;i<w*h;++i)
    b_i[i] = -b_i[i];

  // FFTR = FFT1 .* FFT2
  Teuchos::ArrayRCP<work_t> FFTR_r(w*h,0.0), FFTR_i(w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    complex_multiply(FFTR_r[i],FFTR_i[i],a_r[i],a_i[i],b_r[i],b_i[i]);

  // compute abs(FFTR)
  Teuchos::ArrayRCP<work_t> FFTR_abs(w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    complex_abs(FFTR_abs[i],FFTR_r[i],FFTR_i[i]);

  //FFTRN = FFTR / (abs(FT1 .* FFT2))
  Teuchos::ArrayRCP<work_t> FFTRN_r(w*h,0.0), FFTRN_i(w*h,0.0),zero(w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    complex_divide(FFTRN_r[i],FFTRN_i[i],FFTR_r[i],FFTR_i[i],FFTR_abs[i],zero[i]);

  // result = inverse FFTRN
  array_2d_fft_in_place(w,h,FFTRN_r,FFTRN_i,1);

//  std::cout << " FFTRN " << std::endl;
//  for(int_t i=0;i<h;++i){
//    for(int_t j=0;j<w;++j){
//      std::cout << FFTRN_r[i*w+j] << "+"<< FFTRN_i[i*w+j] << "j " << std::endl;
//    }
//    std::cout << std::endl;
//  }

  // find max of result
  work_t max_real = 0.0;
  // find the max of all pixels aside from (0,0)
  work_t next_real = 0.0;
  u_x = 0.0;
  u_y = 0.0;
  work_t next_x = 0;
  work_t next_y = 0;
  work_t max_complex = 0.0;

  work_t test_real = 0.0;
  work_t test_complex = 0.0;
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      test_real = std::abs(FFTRN_r[y*w+x]);
      test_complex = std::abs(FFTRN_i[y*w+x]);
      if(test_real > max_real){
        max_real = test_real;
        u_x = x;
        u_y = y;
      }
      if(test_complex > max_complex){
        max_complex = test_complex;
      }
      if(x==0&&y==0) continue;
      if(test_real > next_real){
        next_real = test_real;
        next_x = x;
        next_y = y;
      }
    }
  }
  //std::cout << " max real " << max_real << " u_x " << u_x << " u_y " << u_y << " max_complex  " <<
  //max_complex << " next_real " << next_real << " nx " << next_x << " ny " << next_y << std::endl;
  assert(max_complex <= 1.0E-5);
  // deal with aliasing (which causes a false peak at 0,0)
  // TODO find a better approach to this using pre-whitening, etc.
  if(u_x!=next_x&&next_x>1){
    u_x = next_x;
    max_real = next_real;
  }
  if(u_y!=next_y&&next_y>1){
    u_y = next_y;
    max_real = next_real;
  }

//  perc_90_max_real = 0.0;
//  int_t num_above = 0;
//  for(int_t i=0;i<w*h;++i){
//    test_real = std::abs(FFTRN_r[i]);
//    //std::cout << i << " " << test_real << std::endl;
//    if(test_real > 0.1*max_real){
//      num_above++;
//    }
//  }
//  perc_90_max_real = (work_t)num_above/(w*h);

  // convert back to image coordinates
  if(u_x >= w/2)
    u_x = w - u_x;
  else
    u_x = -u_x;

  if(u_y >= h/2)
    u_y = h - u_y;
  else
    u_y = -u_y;

  if(convert_to_r_theta){
    assert(w!=0);
    assert(h!=0);
    const work_t t_size = DICE_TWOPI/w;
    const work_t r_size = std::sqrt((0.5*w)*(0.5*w) + (0.5*h)*(0.5*h)) / h;
    u_x *= t_size; // u_x is actually radius
    u_y *= r_size; //u_y is theta
  }

//  // draw FFT image
//  Teuchos::ArrayRCP<work_t> out_intens(w*h,0.0);
//  // scale from 0 to 255;
//  const work_t factor = 255.0 / max_real;
//  for(int_t i=0;i<w*h;++i){
//    out_intens[i] = std::abs(FFTRN_r[i]) * factor;
//  }
//  Image out_image(w,h,out_intens);
//  std::stringstream name;
//  name << "FFT_out_" << image_a->file_name() << ".tif";
//  out_image.write(name.str());
  return max_real;
}

DICE_LIB_DLL_EXPORT
void
phase_correlate_row(Teuchos::RCP<Image> image_a,
  Teuchos::RCP<Image> image_b,
  const int_t row_id,
  work_t & u,
  const bool convert_to_theta){

  assert(image_a->width()==image_b->width());
  assert(image_a->height()==image_b->height());
  int_t w = image_a->width();
  assert(w>1);

  // compute the hamming filter
  Teuchos::ArrayRCP<work_t> x_ham(w,0.0);
  for(int_t i=0;i<w;++i)
    x_ham[i] = 0.54 - 0.46*std::cos(DICE_TWOPI*i/(w-1));

  // compute th fft of image a's row
  Teuchos::ArrayRCP<work_t> a_real(w,0.0), a_complex(w,0.0);
  kiss_fft_cpx * img_row = new kiss_fft_cpx[w];
  kiss_fft_cfg cfg_ffx = kiss_fft_alloc(w,0,0,0);
  // fft the rows
  for(int_t x=0;x<w;++x){
    img_row[x].r=(*image_a)(x,row_id)*x_ham[x];
    img_row[x].i=0;
  }
  kiss_fft(cfg_ffx,img_row,img_row);
  for(int_t x=0;x<w;++x){
    a_real[x] = img_row[x].r;
    a_complex[x] = img_row[x].i;
  }
  // compute the fft of b's row:
  Teuchos::ArrayRCP<work_t> b_real(w,0.0), b_complex(w,0.0);
  for(int_t x=0;x<w;++x){
    img_row[x].r=(*image_b)(x,row_id)*x_ham[x];
    img_row[x].i=0;
  }
  kiss_fft(cfg_ffx,img_row,img_row);
  for(int_t x=0;x<w;++x){
    b_real[x] = img_row[x].r;
    b_complex[x] = img_row[x].i;
  }
  free(cfg_ffx);


  // conjugate of image b
  for(int_t i=0;i<w;++i)
    b_complex[i] = -b_complex[i];

  // FFTR = FFT1 .* FFT2
  Teuchos::ArrayRCP<work_t> FFTR_r(w,0.0), FFTR_i(w,0.0);
  for(int_t i=0;i<w;++i)
    complex_multiply(FFTR_r[i],FFTR_i[i],a_real[i],a_complex[i],b_real[i],b_complex[i]);

  // compute abs(FFTR)
  Teuchos::ArrayRCP<work_t> FFTR_abs(w,0.0);
  for(int_t i=0;i<w;++i)
    complex_abs(FFTR_abs[i],FFTR_r[i],FFTR_i[i]);

  //FFTRN = FFTR / (abs(FT1 .* FFT2))
  Teuchos::ArrayRCP<work_t> FFTRN_r(w,0.0), FFTRN_i(w,0.0),zero(w,0.0);
  for(int_t i=0;i<w;++i)
    complex_divide(FFTRN_r[i],FFTRN_i[i],FFTR_r[i],FFTR_i[i],FFTR_abs[i],zero[i]);

  kiss_fft_cfg cfg_ffxi = kiss_fft_alloc(w,1,0,0);
  // fft the rows
  for(int_t x=0;x<w;++x){
    img_row[x].r=FFTRN_r[x];
    img_row[x].i=FFTRN_i[x];
  }
  kiss_fft(cfg_ffxi,img_row,img_row);
  for(int_t x=0;x<w;++x){
    FFTRN_r[x] = img_row[x].r;
    FFTRN_i[x] = img_row[x].i;
  }
  free(cfg_ffxi);
  delete[] img_row;

  // find the max and convert to theta if necessary
  u = 0;
  work_t max_value = 0.0;
  work_t value = 0.0;
  for(int_t i=0;i<w;++i){
    value = std::abs(FFTRN_r[i]);
    if(value > max_value){
      max_value = value;
      u = i;
    }
  }
  if(convert_to_theta){
    u *= DICE_TWOPI/w;
  }
}


// TODO address why we're using 2pi for the range on theta when the angle
// is only determined up to +/- pi (should we be preventing ambiguity here?)

template <typename S>
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image_<S>>
polar_transform(Teuchos::RCP<Image_<S>> image,
  bool high_pass_filter){
  const int_t w = image->width();
  assert(w>0);
  const work_t w_2 = 0.5*w;
  const int_t h = image->height();
  assert(h>0);
  const work_t h_2 = 0.5*h;

  assert(w>0);
  assert(h>0);
  // whatever is passed in for the output array RPC, it gets written over
  Teuchos::ArrayRCP<S> output(w*h,0);
  const work_t t_size = DICE_TWOPI/w;
  const work_t r_size = high_pass_filter ? 0.25*w/h : std::sqrt(w_2*w_2 + h_2*h_2)/h;
  int_t x1 = 0;
  int_t x2 = 0;
  int_t y1 = 0;
  int_t y2 = 0;
  int_t y_size = h;

  //std::cout << " x will range from 0 to " << w << " and y from 0 to " << y_size << std::endl;
  //std::cout << " rsize " << r_size << " tsize " << t_size << std::endl;
  work_t r = 0.0;
  work_t t = 0.0;
  for(int_t y=0;y<y_size;++y){
    r = (y+0.5)*r_size;
    for(int_t x=0;x<w;++x){
      t = (x+0.5)*t_size;
      // convert r,t into x and y coordinates
      work_t X = 0.5*w + r*std::cos(t);
      work_t Y = h_2 + r*std::sin(t);
      x1 = (int_t)X;
      y1 = (int_t)Y;
      if(X>=0&&X<w-2&&Y>=0&&Y<h-2){
        // BILINEAR INTERPOLATION
        // interpolate the image at those points:
        x2 = x1+1;
        y2 = y1+1;
        output[y*w+x] =
            ((*image)(x1,y1)*(x2-X)*(y2-Y)
                +(*image)(x2,y1)*(X-x1)*(y2-Y)
                +(*image)(x2,y2)*(X-x1)*(Y-y1)
                +(*image)(x2,y1)*(x2-X)*(Y-y1));
      }
      else if(X>=0&&X<w&&Y>=0&&Y<h){
        output[y*w+x] = (*image)(x1,y1);
      }
      else{
        output[y*w+x] = 0;
      }
    } // x loop
  } // y loop
  Teuchos::RCP<Image_<S>> out_image = Teuchos::rcp(new Image_<S>(w,h,output));
  return out_image;
}

template
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image> polar_transform(Teuchos::RCP<Image>,bool);
template
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Scalar_Image> polar_transform(Teuchos::RCP<Scalar_Image>,bool);

template <typename S>
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image_<S>>
image_fft(Teuchos::RCP<Image_<S>> image,
  const bool hamming_filter,
  const bool apply_log,
  const work_t scale_factor,
  bool shift,
  const bool high_pass_filter){

  // NOTE: if the high_pass_filter is used,
  // the fft values are automatically shifted:
  if(high_pass_filter)
    shift = true;

  const int_t w = image->width();
  const int_t h = image->height();
  const int_t w_2 = w/2;
  const int_t h_2 = h/2;
  const int_t w_4 = w/4;
  assert(w>0);
  assert(h>0);
  assert(w_2>0);
  Teuchos::ArrayRCP<work_t> real;
  Teuchos::ArrayRCP<work_t> complex;
  // for now, disallow the use of the inverse fft
  image_fft(image,real,complex,0,hamming_filter);
  assert(real.size()==w*h);
  assert(complex.size()==w*h);
  work_t max_mag = std::numeric_limits<work_t>::min();
  work_t min_mag = std::numeric_limits<work_t>::max();

  int_t index = 0;
  Teuchos::ArrayRCP<work_t> mag(w*h,0.0);
  Teuchos::ArrayRCP<S> intensities(w*h,0);
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      mag[index] = std::sqrt(real[index]*real[index] + complex[index]*complex[index]);
      if(apply_log)
        mag[index] = scale_factor*std::log(mag[index]+1);
      if(mag[index]<min_mag) min_mag = mag[index];
      if(mag[index]>max_mag) max_mag = mag[index];
      index++;
    } // x
  }  // y

  // scale the image to fit in 8-bit output range
  assert(max_mag - min_mag > 0);
  work_t factor = max_mag-min_mag==0.0?0.0:255.0/(max_mag - min_mag);
  for(int_t i=0;i<w*h;++i)
    mag[i] = (mag[i]-min_mag)*factor;

  if(shift){
    // now shift the quadrants:
    Teuchos::ArrayRCP<work_t> mag_shift(w*h,0.0);
    int_t xp=0,yp=0;
    for(int_t y=0;y<h;++y){
      for(int_t x=0;x<w;++x){
        if(x<w_2){
          if(y<h_2){
            xp = x + w_2;
            yp = y + h_2;
          }
          else{
            xp = x + w_2;
            yp = y - h_2;
          }
        }
        else{
          if(y<h/2){
            xp = x - w_2;
            yp = y + h_2;
          }
          else{
            xp = x - w_2;
            yp = y - h_2;
          }
        }
        mag_shift[yp*w+xp] = mag[y*w+x];
      } // x loop
    } // y loop
    if(high_pass_filter){
      // compute the radius from the center:
      work_t rad = 0.0;
      int_t dx=0,dy=0;
      for(int_t y=0;y<h;++y){
        dy = y - h_2;
        for(int_t x=0;x<w;++x){
          dx = x - w_2;
          rad = std::sqrt(dx*dx + dy*dy);
          if(rad > w_4)
            mag_shift[y*w+x] = 0.0;
          else
            mag_shift[y*w+x] *= std::cos(DICE_PI*dx/w_2)*std::cos(DICE_PI*dy/w_2);
        } // x loop
      } // y loop
    }
    for(int_t i=0;i<intensities.size();++i)
      intensities[i] = static_cast<S>(mag_shift[i]);
  }
  else{
    for(int_t i=0;i<intensities.size();++i)
      intensities[i] = static_cast<S>(mag[i]);
  }
  return Teuchos::rcp(new Image_<S>(w,h,intensities));
}

template
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image> image_fft(Teuchos::RCP<Image>,const bool,const bool,const work_t,bool,const bool);
template
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Scalar_Image> image_fft(Teuchos::RCP<Scalar_Image>,const bool,const bool,const work_t,bool,const bool);

template <typename S>
DICE_LIB_DLL_EXPORT
void
image_fft(Teuchos::RCP<Image_<S>> image,
  Teuchos::ArrayRCP<work_t> & real,
  Teuchos::ArrayRCP<work_t> & complex,
  const int_t inverse,
  const bool hamming_filter){

  const int_t w = image->width();
  assert(w>1);
  const int_t h = image->height();
  assert(h>1);
  real = Teuchos::ArrayRCP<work_t> (w*h,0.0);
  complex = Teuchos::ArrayRCP<work_t> (w*h,0.0);

  if(hamming_filter){
    Teuchos::ArrayRCP<work_t> x_ham(w,0.0);
    Teuchos::ArrayRCP<work_t> y_ham(h,0.0);
    for(int_t i=0;i<w;++i){
      x_ham[i] = 0.54 - 0.46*std::cos(DICE_TWOPI*i/(w-1));
    }
    for(int_t i=0;i<h;++i){
      y_ham[i] = 0.54 - 0.46*std::cos(DICE_TWOPI*i/(h-1));
    }
    for(int_t y=0;y<h;++y){
      for(int_t x=0;x<w;++x){
        real[y*w+x] = (*image)(x,y)*x_ham[x]*y_ham[y];
      }
    }
  }
  else{
    for(int_t y=0;y<h;++y){
      for(int_t x=0;x<w;++x){
        real[y*w+x] = (*image)(x,y);
      }
    }
  }

  kiss_fft_cpx * img_row = new kiss_fft_cpx[w];
  kiss_fft_cpx * img_col = new kiss_fft_cpx[h];
  kiss_fft_cfg cfg_ffx = kiss_fft_alloc(w,inverse,0,0);
  kiss_fft_cfg cfg_ffy = kiss_fft_alloc(h,inverse,0,0);

  // fft the rows
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      img_row[x].r=real[y*w+x];
      img_row[x].i=0.0;
    }
    kiss_fft(cfg_ffx,img_row,img_row);
    for(int_t x=0;x<w;++x){
      real[y*w+x] = img_row[x].r;
      complex[y*w+x] = img_row[x].i;
    }
  }
  // fft the cols
  for(int_t x=0;x<w;++x){
    for(int_t y=0;y<h;++y){
      img_col[y].r=real[y*w+x];
      img_col[y].i=complex[y*w+x];
    }
    kiss_fft(cfg_ffy,img_col,img_col);
    for(int_t y=0;y<h;++y){
      real[y*w+x] = img_col[y].r;
      complex[y*w+x] = img_col[y].i;
    }
  }
  free(cfg_ffx);
  free(cfg_ffy);
  delete[] img_row;
  delete[] img_col;
};

template
DICE_LIB_DLL_EXPORT
void image_fft(Teuchos::RCP<Image>,Teuchos::ArrayRCP<work_t> &,Teuchos::ArrayRCP<work_t> &,const int_t,const bool);
template
DICE_LIB_DLL_EXPORT
void image_fft(Teuchos::RCP<Scalar_Image>,Teuchos::ArrayRCP<work_t> &,Teuchos::ArrayRCP<work_t> &,const int_t,const bool);

DICE_LIB_DLL_EXPORT
void
array_2d_fft_in_place(const int_t w,
  const int_t h,
  Teuchos::ArrayRCP<work_t> & real,
  Teuchos::ArrayRCP<work_t> & complex,
  const int_t inverse){

  kiss_fft_cpx * img_row = new kiss_fft_cpx[w];
  kiss_fft_cpx * img_col = new kiss_fft_cpx[h];
  kiss_fft_cfg cfg_ffx = kiss_fft_alloc(w,inverse,0,0);
  kiss_fft_cfg cfg_ffy = kiss_fft_alloc(h,inverse,0,0);

  // fft the rows
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      img_row[x].r=real[y*w+x];
      img_row[x].i=complex[y*w+x];
    }
    kiss_fft(cfg_ffx,img_row,img_row);
    for(int_t x=0;x<w;++x){
      real[y*w+x] = img_row[x].r;
      complex[y*w+x] = img_row[x].i;
    }
  }
  // fft the cols
  for(int_t x=0;x<w;++x){
    for(int_t y=0;y<h;++y){
      img_col[y].r=real[y*w+x];
      img_col[y].i=complex[y*w+x];
    }
    kiss_fft(cfg_ffy,img_col,img_col);
    for(int_t y=0;y<h;++y){
      real[y*w+x] = img_col[y].r;
      complex[y*w+x] = img_col[y].i;
    }
  }
  free(cfg_ffx);
  free(cfg_ffy);
  delete[] img_row;
  delete[] img_col;
};

}// End DICe Namespace
