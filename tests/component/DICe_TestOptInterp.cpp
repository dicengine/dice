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

#include <boost/timer.hpp>
#include <random>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <iostream>

using namespace DICe;

intensity_t intens(const scalar_t & x, const scalar_t & y){
  static scalar_t gamma = 0.2;
  return (255.0*0.5) + 0.5*255.0*std::sin(gamma*x)*std::cos(gamma*y);
}

inline scalar_t f0(const scalar_t & s){
  return 1.33333333333333*s*s*s - 2.33333333333333*s*s+ 1.0;
}
inline scalar_t f1(const scalar_t & s){
  return -0.58333333333333*s*s*s + 3.0*s*s - 4.91666666666666*s + 2.5;
}
inline scalar_t f2(const scalar_t & s){
  return 0.08333333333333*s*s*s - 0.66666666666666*s*s + 1.75*s - 1.5;
}

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

  // create a bi-linear image from an array
  *outStream << "creating an image from intensity function" << std::endl;
  const int_t w = 1000;
  const int_t h = 500;
  intensity_t * intensities = new intensity_t[w*h];
  // populate the intensities with a sin/cos function
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      intensities[y*w+x] = intens(x,y);
    }
  }
  Teuchos::RCP<Image> img = Teuchos::rcp(new Image(intensities,w,h));
  //array_img->write("opt_interp_ref.tif");

  // do tons of interpolations at random points
  std::default_random_engine generator;
  const scalar_t mean = 0.0;
  const scalar_t std_dev = 0.25;
  std::normal_distribution<scalar_t> distribution(mean,std_dev);
  scalar_t error = 0.0;
  *outStream << "running bilinear" << std::endl;
  {
    boost::timer t;
    for(int_t it=0;it<100;++it){
      for(int_t j=10;j<h-10;++j){
        for(int_t i=10;i<h-10;++i){
          const scalar_t x = i + distribution(generator);
          const scalar_t y = j + distribution(generator);
          const scalar_t value = img->interpolate_bilinear(x,y);
          const scalar_t exact = intens(x,y);
          error += (value - exact)*(value - exact);
        }
      }
    }
    error = std::sqrt(error);
    // timing info
    const scalar_t elapsed_time = t.elapsed();
    *outStream << "error:        " << error << std::endl;
    *outStream << "elapsed time: " << elapsed_time << std::endl;
  }
  error = 0.0;
  *outStream << "running bicubic" << std::endl;
  {
    boost::timer t;
    for(int_t it=0;it<100;++it){
      for(int_t j=10;j<h-10;++j){
        for(int_t i=10;i<h-10;++i){
          const scalar_t x = i + distribution(generator);
          const scalar_t y = j + distribution(generator);
          const scalar_t value = img->interpolate_bicubic(x,y);
          const scalar_t exact = intens(x,y);
          error += (value - exact)*(value - exact);
        }
      }
    }
    error = std::sqrt(error);
    // timing info
    const scalar_t elapsed_time = t.elapsed();
    *outStream << "error:        " << error << std::endl;
    *outStream << "elapsed time: " << elapsed_time << std::endl;
  }
  error = 0.0;
  *outStream << "running keys-fourth" << std::endl;
  {
    boost::timer t;
    for(int_t it=0;it<100;++it){
      for(int_t j=10;j<h-10;++j){
        for(int_t i=10;i<h-10;++i){
          const scalar_t x = i + distribution(generator);
          const scalar_t y = j + distribution(generator);
          const scalar_t value = img->interpolate_keys_fourth(x,y);
          const scalar_t exact = intens(x,y);
          error += (value - exact)*(value - exact);
        }
      }
    }
    error = std::sqrt(error);
    // timing info
    const scalar_t elapsed_time = t.elapsed();
    *outStream << "error:        " << error << std::endl;
    *outStream << "elapsed time: " << elapsed_time << std::endl;
  }
  error = 0.0;
  *outStream << "running keys-opt" << std::endl;
  {
    boost::timer t;
    std::vector<std::vector<intensity_t> > intens_cache(w*h,std::vector<intensity_t>(36,0.0));
    // prefetch the neighbors
//    for(int_t j=2;j<h-3;++j){
//      for(int_t i=2;i<h-3;++i){
//        int_t index = 0;
//        for(int_t m=-2;m<=3;++m){
//          for(int_t n=-2;n<=3;++n){
//            intens_cache[j*w+i][index++] = intensities[(j+m)*w+(i+n)];
//          }
//        }
//      }
//    }

    std::vector<scalar_t> coeffs_x(6,0.0);
    std::vector<scalar_t> coeffs_y(6,0.0);
    scalar_t dx = 0.0;
    scalar_t dy = 0.0;
    scalar_t x = 0.0;
    scalar_t y = 0.0;
    int_t ix = 0;
    int_t iy = 0;
    int_t m=0,n=0;
    intensity_t value = 0.0;
    scalar_t exact = 0.0;
    for(int_t it=0;it<100;++it){
      for(int_t j=10;j<h-10;++j){
        for(int_t i=10;i<h-10;++i){
          x = i + distribution(generator);
          y = j + distribution(generator);
          ix = (int_t)x;
          iy = (int_t)y;
          dx = x - ix;
          dy = y - iy;
          coeffs_x[0] = f2(dx+2.0);
          coeffs_x[1] = f1(dx+1.0);
          coeffs_x[2] = f0(dx);
          coeffs_x[3] = f0(1.0-dx);
          coeffs_x[4] = f1(2.0-dx);
          coeffs_x[5] = f2(3.0-dx);
          coeffs_y[0] = f2(dy+2.0);
          coeffs_y[1] = f1(dy+1.0);
          coeffs_y[2] = f0(dy);
          coeffs_y[3] = f0(1.0-dy);
          coeffs_y[4] = f1(2.0-dy);
          coeffs_y[5] = f2(3.0-dy);
          value = 0.0;
          //int_t index = 0;
          for(m=0;m<6;++m){
            for(n=0;n<6;++n){
              value += coeffs_y[m]*coeffs_x[n]*intensities[(iy-2+m)*w + ix-2+n];
              //value += coeffs_y[m]*coeffs_x[n]*intens_cache[iy*w + ix][index++];
            }
          }
          exact = intens(x,y);
          //std::cout << " exact " << exact << " value " << value << std::endl;
          error += (value - exact)*(value - exact);
        }
      }
    }
    error = std::sqrt(error);
    // timing info
    const scalar_t elapsed_time = t.elapsed();
    *outStream << "error:        " << error << std::endl;
    *outStream << "elapsed time: " << elapsed_time << std::endl;
  }

  delete [] intensities;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

