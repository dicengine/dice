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
#include <DICe_Triangulation.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <cassert>

using namespace std;
using namespace DICe;

/// Initializes the cross correlation between two images from the left and right camera, respectively

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // read the parameters from the input file:
  if(argc!=5){
    std::cout << "Usage: DICe_Epiline <x> <y> <img_width> <cal_file> <which_image>" << std::endl;
    exit(1);
  }

  const scalar_t x = std::atof(argv[1]);
  const scalar_t y = std::atof(argv[2]);
  const std::string cal_file = argv[3];
  const int which_image = std::atoi(argv[4]);
  std::string output_file = "./.dice/.epiline.txt";

  std::cout << "DICe_Epiline: converting point " << x << " " << y << " using file " << cal_file << " which image " << which_image << std::endl;

  Teuchos::RCP<DICe::Triangulation> tri = Teuchos::rcp(new DICe::Triangulation(cal_file));
  const int img_width = tri->image_width();
  cv::Mat F = tri->fundamental_matrix();
  cv::Mat lines;
  std::vector<cv::Point2f> points(1);
  points[0] = cv::Point2f(x,y);
  cv::computeCorrespondEpilines(points,which_image,F,lines); // epipolar lines for the subset centroids
  assert(lines.rows==1);
  const float a = lines.at<float>(0,0);
  const float b = lines.at<float>(0,1);
  const float c = lines.at<float>(0,2);
  // compute y0 and y1
  const float epi_y0 = b==0.0?0.0:-1.0*c/b;
  const float epi_y1 = b==0.0?0.0:(-1.0*c-img_width*a)/b;

  std::cout << "DICe_Epiline: image width " << img_width << " a " << a << " b " << b << " c " << c << " y0 " << epi_y0 << " y1 " << epi_y1 << std::endl;

  // write output
  std::FILE * infoFilePtr = fopen(output_file.c_str(),"w");
  fprintf(infoFilePtr,"%f %f\n",epi_y0,epi_y1);
  fclose(infoFilePtr);

  DICe::finalize();
  return 0;
}

