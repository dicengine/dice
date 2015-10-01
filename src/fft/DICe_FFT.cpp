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
  scalar_t & u_y){

  const int_t w = image_a->width();
  const int_t h = image_a->height();
  assert(image_b->width()==w && "Error: images must be the same dims");
  assert(image_b->height()==h && "Error: images must be the same dims");
  assert(w>8);
  assert(h>8);

  // x hamming filter
  Teuchos::ArrayRCP<scalar_t> x_ham(w,0.0);
  Teuchos::ArrayRCP<scalar_t> y_ham(h,0.0);
  for(int_t i=0;i<w;++i){
    x_ham[i] = 0.54 - 0.46*std::cos(DICE_TWOPI*i/(w-1));
  }
  for(int_t i=0;i<h;++i){
    y_ham[i] = 0.54 - 0.46*std::cos(DICE_TWOPI*i/(h-1));
  }

  Teuchos::ArrayRCP<scalar_t> ham_a_intens(w*h,0.0);
  Teuchos::ArrayRCP<scalar_t> ham_b_intens(w*h,0.0);
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      ham_a_intens[y*w+x] = (*image_a)(x,y)*x_ham[x]*y_ham[y];
      ham_b_intens[y*w+x] = (*image_b)(x,y)*x_ham[x]*y_ham[y];
    }
  }

  Teuchos::RCP<DICe::Image> image_ham_a = Teuchos::rcp(new DICe::Image(w,h,ham_a_intens));
  Teuchos::RCP<DICe::Image> image_ham_b = Teuchos::rcp(new DICe::Image(w,h,ham_b_intens));

  // fft of image a
  Teuchos::ArrayRCP<scalar_t> a_r,a_i;
  DICe::image_fft(image_ham_a,a_r,a_i,0);

  // fft of image b
  Teuchos::ArrayRCP<scalar_t> b_r,b_i;
  DICe::image_fft(image_ham_b,b_r,b_i,0);

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
//      std::cout << FFTRN_r[i*w+j] << "+"<< FFTRN_i[i*w+j] << "j " ;
//    }
//    std::cout << std::endl;
//  }
  // find min and max of result
  scalar_t max_real = 0.0;
  u_x = 0.0;
  u_y = 0.0;
  scalar_t max_complex = 0.0;

  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      if(std::abs(FFTRN_r[y*w+x]) > max_real){
        max_real = std::abs(FFTRN_r[y*w+x]);
        u_x = x;
        u_y = y;
      }
      if(std::abs(FFTRN_i[y*w+x]) > max_complex){
        max_complex = std::abs(FFTRN_i[y*w+x]);
      }
    }
  }
  //std::cout << " max real " << max_real << " u_x " << u_x << " u_y " << u_y << " max_complex  " << max_complex << std::endl;
  assert(max_complex <= 1.0E-6);

  // convert back to image coordinates
  if(u_x > w/2)
    u_x = w - u_x;
  else
    u_x = -u_x;

  if(u_y > h/2)
    u_y = h - u_y;
  else
    u_y = -u_y;

//  if(u_x >= w || u_y >= h){ // assume the correlation failed
//    u_x = 0;
//    u_y = 0;
//  }

//  // draw FFT image
//  Teuchos::ArrayRCP<scalar_t> out_intens(w*h,0.0);
//  // scale from 0 to 255;
//  const scalar_t factor = 255.0 / max_real;
//  for(int_t i=0;i<w*h;++i){
//    out_intens[i] = std::abs(FFTRN_r[i]) * factor;
//  }
//  DICe::Image<scalar_t,int_t> out_image(w,h,out_intens);
//  out_image.write("out_image");
}

DICE_LIB_DLL_EXPORT
void
image_fft(Teuchos::RCP<Image> image,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  int_t inverse){

  const int_t w = image->width();
  const int_t h = image->height();

  real = Teuchos::ArrayRCP<scalar_t> (w*h,0.0);
  complex = Teuchos::ArrayRCP<scalar_t> (w*h,0.0);
  for(int_t i=0;i<w*h;++i)
    real[i] = (*image)(i);

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
array_2d_fft_in_place(int_t w,
  int_t h,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  int_t inverse){

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
