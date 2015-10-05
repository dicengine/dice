// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
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

    scalar_t e = 0.0;
    scalar_t f = 0.0;
    if( fabs(rhs[i].i)<fabs(rhs[i].r) )
    {
        e = rhs[i].i/rhs[i].r;
        f = rhs[i].r+rhs[i].i*e;
        lhs[i].r = (lhs[i].r+lhs[i].i*e)/f;
        lhs[i].i = (lhs[i].i-lhs[i].r*e)/f;
    }
    else
    {
        e = rhs[i].r/rhs[i].i;
        f = rhs[i].i+rhs[i].r*e;
        lhs[i].r = (lhs[i].i+lhs[i].r*e)/f;
        lhs[i].i = (-lhs[i].r+lhs[i].i*e)/f;
    }
  }
}

DICE_LIB_DLL_EXPORT
void
complex_divide(scalar_t & result_r,
  scalar_t & result_i,
  const scalar_t & a_r,
  const scalar_t & a_i,
  const scalar_t & b_r,
  const scalar_t & b_i)
{
  scalar_t e = 0.0;
  scalar_t f = 0.0;
  if( fabs(b_i)<fabs(b_r) )
  {
    e = b_i/b_r;
    f = b_r+b_i*e;
    result_r = (a_r+a_i*e)/f;
    result_i = (a_i-a_r*e)/f;
  }
  else
  {
    e = b_r/b_i;
    f = b_i+b_r*e;
    result_r = (a_i+a_r*e)/f;
    result_i = (-a_r+a_i*e)/f;
  }
}

DICE_LIB_DLL_EXPORT
void
complex_multiply(scalar_t & result_r,
  scalar_t & result_i,
  const scalar_t & a_r,
  const scalar_t & a_i,
  const scalar_t & b_r,
  const scalar_t & b_i){
  result_r = a_r*b_r - a_i*b_i;
  result_i = a_i*b_r + a_r*b_i;
}

DICE_LIB_DLL_EXPORT
void
complex_abs(scalar_t & result,
  const scalar_t & a_r,
  const scalar_t & a_i){
  result = std::sqrt(a_r*a_r + a_i*a_i);
}

DICE_LIB_DLL_EXPORT
void
phase_correlate_x_y(Teuchos::RCP<Image> image_a,
  Teuchos::RCP<Image> image_b,
  scalar_t & u_x,
  scalar_t & u_y,
  const bool convert_to_r_theta){

  const int_t w = image_a->width();
  const int_t h = image_a->height();
  assert(image_b->width()==w && "Error: images must be the same dims");
  assert(image_b->height()==h && "Error: images must be the same dims");
  assert(w>8);
  assert(h>8);

  // fft of image a
  Teuchos::ArrayRCP<scalar_t> a_r,a_i;
  DICe::image_fft(image_a,a_r,a_i,0);

  // fft of image b
  Teuchos::ArrayRCP<scalar_t> b_r,b_i;
  DICe::image_fft(image_b,b_r,b_i,0);

  // conjugate of image b
  for(int_t i=0;i<w*h;++i)
    b_i[i] = -b_i[i];

  // FFTR = FFT1 .* FFT2
  Teuchos::ArrayRCP<scalar_t> FFTR_r(w*h,0.0), FFTR_i(w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    complex_multiply(FFTR_r[i],FFTR_i[i],a_r[i],a_i[i],b_r[i],b_i[i]);

  // compute abs(FFTR)
  Teuchos::ArrayRCP<scalar_t> FFTR_abs(w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    complex_abs(FFTR_abs[i],FFTR_r[i],FFTR_i[i]);

  //FFTRN = FFTR / (abs(FT1 .* FFT2))
  Teuchos::ArrayRCP<scalar_t> FFTRN_r(w*h,0.0), FFTRN_i(w*h,0.0),zero(w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    complex_divide(FFTRN_r[i],FFTRN_i[i],FFTR_r[i],FFTR_i[i],FFTR_abs[i],zero[i]);

  // result = inverse FFTRN
  array_2d_fft_in_place(w,h,FFTRN_r,FFTRN_i,1);

//  std::cout << " FFTRN " << std::endl;
//  for(size_t i=0;i<h;++i){
//    for(size_t j=0;j<w;++j){
//      std::cout << FFTRN_r[i*w+j] << "+"<< FFTRN_i[i*w+j] << "j " << std::endl;
//    }
//    std::cout << std::endl;
//  }

  // find max of result
  scalar_t max_real = 0.0;
  // find the max of all pixels aside from (0,0)
  scalar_t next_real = 0.0;
  u_x = 0.0;
  u_y = 0.0;
  scalar_t next_x = 0;
  scalar_t next_y = 0;
  scalar_t max_complex = 0.0;

  scalar_t test_real = 0.0;
  scalar_t test_complex = 0.0;
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
  assert(max_complex <= 1.0E-6);
  // deal with aliasing (which causes a false peak at 0,0)
  // TODO find a better approach to this using pre-whitening, etc.
  if(u_x!=next_x&&next_x>1)
    u_x = next_x;
  if(u_y!=next_y&&next_y>1)
    u_y = next_y;

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
    const scalar_t t_size = DICE_TWOPI / w;
    const scalar_t r_size = std::sqrt((0.5*w)*(0.5*w) + (0.5*h)*(0.5*h)) / h;
    u_x *= t_size; // u_x is actually radius
    u_y *= r_size; //u_y is theta
  }
//  // draw FFT image
//  Teuchos::ArrayRCP<scalar_t> out_intens(w*h,0.0);
//  // scale from 0 to 255;
//  const scalar_t factor = 255.0 / max_real;
//  for(int_t i=0;i<w*h;++i){
//    out_intens[i] = std::abs(FFTRN_r[i]) * factor;
//  }
//  Image out_image(w,h,out_intens);
//  out_image.write_tiff("FFT_out_image.tif");
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image>
polar_transform(Teuchos::RCP<Image> image){
  const int_t w = image->width();
  const scalar_t w_2 = 0.5*w;
  const int_t h = image->height();
  const scalar_t h_2 = 0.5*h;
  assert(w>0);
  assert(h>0);
  // whatever is passed in for the output array RPC, it gets written over
  Teuchos::ArrayRCP<intensity_t> output = Teuchos::ArrayRCP<intensity_t> (w*h,0.0);
  const scalar_t t_size = DICE_TWOPI / w;
  const scalar_t r_size = std::sqrt(w_2*w_2 + h_2*h_2)/h;
  int_t x1 = 0;
  int_t x2 = 0;
  int_t y1 = 0;
  int_t y2 = 0;
  scalar_t r = 0.0;
  scalar_t t = 0.0;
  for(int_t y=0;y<h;++y){
    r = (y+0.5)*r_size;
    for(int_t x=0;x<w;++x){
      t = (x+0.5)*t_size;
      // convert r,t into x and y coordinates
      scalar_t X = 0.5*w + r*std::cos(t);
      scalar_t Y = 0.5*h + r*std::sin(t);
      if(X>=0&&X<w-2&&Y>=0&&Y<h-2){
        // BILINEAR INTERPOLATION
        // interpolate the image at those points:
        x1 = (int_t)X;
        x2 = x1+1;
        y1 = (int_t)Y;
        y2  = y1+1;
        output[y*w+x] =
            ((*image)(x1,y1)*(x2-X)*(y2-Y)
                +(*image)(x2,y1)*(X-x1)*(y2-Y)
                +(*image)(x2,y2)*(X-x1)*(Y-y1)
                +(*image)(x2,y1)*(x2-X)*(Y-y1));
      }
      else{
        output[y*w+x] = 0;
      }
    } // x loop
  } // y loop
  Teuchos::RCP<Image> out_image = Teuchos::rcp(new Image(w,h,output));
  return out_image;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image>
image_fft(Teuchos::RCP<Image> image,
  const bool hamming_filter,
  const bool apply_log,
  const scalar_t scale_factor,
  const bool shift){

  const int_t w = image->width();
  const int_t h = image->height();
  assert(w>0);
  assert(h>0);
  Teuchos::ArrayRCP<intensity_t> real;
  Teuchos::ArrayRCP<intensity_t> complex;
  // for now, disallow the use of the inverse fft
  image_fft(image,real,complex,0,hamming_filter);
  assert(real.size()==w*h);
  assert(complex.size()==w*h);
  scalar_t max_mag = 0.0;
  scalar_t min_mag = 1.0E12;
  Teuchos::ArrayRCP<intensity_t> mag(w*h,0.0);
  for(size_t i=0;i<w*h;++i){
    mag[i] = std::sqrt(real[i]*real[i] + complex[i]*complex[i]);
    if(apply_log)
      mag[i] = scale_factor*std::log(mag[i]+1);
    if(mag[i]<min_mag) min_mag = mag[i];
    if(mag[i]>max_mag) max_mag = mag[i];
  }

  // scale the image to fit in 8-bit output range
  assert(max_mag - min_mag > 0);
  scalar_t factor = 255.0/(max_mag - min_mag);
  for(size_t i=0;i<w*h;++i)
    mag[i] = (mag[i]-min_mag)*factor;

  if(shift){
    // now shift the quadrants:
    Teuchos::ArrayRCP<intensity_t> mag_shift(w*h,0.0);
    int_t w_2 = w/2;
    int_t h_2 = h/2;
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
    return Teuchos::rcp(new Image(w,h,mag_shift));
  }
  else{
    return Teuchos::rcp(new Image(w,h,mag));
  }
}

DICE_LIB_DLL_EXPORT
void
image_fft(Teuchos::RCP<Image> image,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  const int_t inverse,
  const bool hamming_filter){

  const int_t w = image->width();
  const int_t h = image->height();
  real = Teuchos::ArrayRCP<scalar_t> (w*h,0.0);
  complex = Teuchos::ArrayRCP<scalar_t> (w*h,0.0);

  if(hamming_filter){
    Teuchos::ArrayRCP<scalar_t> x_ham(w,0.0);
    Teuchos::ArrayRCP<scalar_t> y_ham(h,0.0);
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
  //  for(size_t i=0;i<h;++i){
  //    for(size_t j=0;j<w;++j){
  //      std::cout << real[i*w+j] << "+"<< complex[i*w+j] << "j " ;
  //    }
  //    std::cout << std::endl;
  //  }
};

DICE_LIB_DLL_EXPORT
void
array_2d_fft_in_place(const int_t w,
  const int_t h,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
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
