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

  // set up coeffs for curve fit:
  const int_t num_neigh = 25;
  const scalar_t XtX[6][num_neigh] = {{-0.074286,0.011429,0.04,0.011429,-0.074286,0.011429,0.097143,0.12571,0.097143,0.011429,0.04,0.12571,0.15429,0.12571,0.04,0.011429,0.097143,0.12571,0.097143,0.011429,-0.074286,0.011429,0.04,0.011429,-0.074286},
  {-0.04,-0.04,-0.04,-0.04,-0.04,-0.02,-0.02,-0.02,-0.02,-0.02,0,0,0,0,0,0.02,0.02,0.02,0.02,0.02,0.04,0.04,0.04,0.04,0.04},
  {-0.04,-0.02,0,0.02,0.04,-0.04,-0.02,0,0.02,0.04,-0.04,-0.02,0,0.02,0.04,-0.04,-0.02,0,0.02,0.04,-0.04,-0.02,0,0.02,0.04},
  {0.04,0.02,0,-0.02,-0.04,0.02,0.01,0,-0.01,-0.02,0,0,0,0,0,-0.02,-0.01,0,0.01,0.02,-0.04,-0.02,0,0.02,0.04},
  {0.028571,0.028571,0.028571,0.028571,0.028571,-0.014286,-0.014286,-0.014286,-0.014286,-0.014286,-0.028571,-0.028571,-0.028571,-0.028571,-0.028571,-0.014286,-0.014286,-0.014286,-0.014286,-0.014286,0.028571,0.028571,0.028571,0.028571,0.028571},
  {0.028571,-0.014286,-0.028571,-0.014286,0.028571,0.028571,-0.014286,-0.028571,-0.014286,0.028571,0.028571,-0.014286,-0.028571,-0.014286,0.028571,0.028571,-0.014286,-0.028571,-0.014286,0.028571,0.028571,-0.014286,-0.028571,-0.014286,0.028571}};
  const int_t neigh_index_x[] = {-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};
  const int_t neigh_index_y[] = {-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2};

  // TODO do 1 million interpolations at random points
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
  *outStream << "running new interp" << std::endl;
  {
    boost::timer t;

    // storage for curve fit coeffs
    std::vector<std::vector<scalar_t> > coeffs(w*h,std::vector<scalar_t>(6,0.0));
    // top row is not interpolated
    for(int_t j=0;j<2;++j){
      for(int_t i=0;i<w;++i){
        coeffs[j*w+i][0] = intensities[j*w+i];
      }
    }
    // bottom row is not interpolated
    for(int_t j=h-2;j<h;++j){
      for(int_t i=0;i<w;++i){
        coeffs[j*w+i][0] = intensities[j*w+i];
      }
    }
    // left and right not interpolated
    for(int_t j=0;j<h;++j){
      for(int_t i=0;i<2;++i){
        coeffs[j*w+i][0] = intensities[j*w+i];
      }
      for(int_t i=w-2;i<w;++i){
        coeffs[j*w+i][0] = intensities[j*w+i];
      }
    }
    // set up the rest of the coeffs:
    for(int_t j=2;j<h-2;++j){
      for(int_t i=2;i<h-2;++i){
         for(int_t k=0;k<6;++k){
           for(int_t m=0;m<num_neigh;++m){
             coeffs[j*w+i][k] += XtX[k][m]*intensities[(j+neigh_index_y[m])*w + i+neigh_index_x[m]];
           }
         }
      }
    }
    for(int_t it=0;it<1;++it){
      for(int_t j=10;j<h-10;++j){
        for(int_t i=10;i<h-10;++i){
          const scalar_t x = i + distribution(generator);
          const scalar_t y = j + distribution(generator);
          int_t xi = (int_t)x;
          int_t yi = (int_t)y;
          scalar_t dx = x-xi;
          scalar_t dy = y-yi;
          const scalar_t value = coeffs[yi*w+xi][0] + coeffs[yi*w+xi][1]*dx + coeffs[yi*w+xi][2]*dy + coeffs[yi*w+xi][3]*dx*dy + coeffs[yi*w+xi][4]*dx*dx + coeffs[yi*w+xi][5]*dy*dy;
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

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

