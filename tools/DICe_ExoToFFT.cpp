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

/*! \file  DICe_NetCDFToTiff.cpp
    \brief Utility for exporting NetCDF files to tiff files
*/

#include <DICe.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_PointCloud.h>
#include <DICe_FFT.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <cassert>
#include <string>
#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  std::string delimiter = " ,\r";

  // determine if the second argument is help, file name or folder name
  if(argc!=6){
    std::cout << " DICe_ExoToFFT (exports a serial exodus file to FFT) " << std::endl;
    std::cout << " Syntax: DICe_ExoToFFT <exodus_file_name> <field_name> <num_neigh> <mm_per_pixel> <frequency_thresh>" << std::endl;
    exit(-1);
  }
  std::string exo_name = argv[1];
  std::string field_name = argv[2];
  const int_t num_neigh = std::atoi(argv[3]);
  const scalar_t mm_per_pixel = std::atof(argv[4]);
  const scalar_t freq_thresh = std::atof(argv[5]);

  *outStream << "exodus input file:          " << exo_name << std::endl;
  *outStream << "requested field:            " << field_name << std::endl;
  *outStream << "num neighbors:              " << num_neigh << std::endl;
  *outStream << "mm per pixel conversion:    " << mm_per_pixel << std::endl;
  *outStream << "frequency threshold:        " << freq_thresh << std::endl;

  if(freq_thresh > 0.5){
    std::cout << "Error, frequency threshold is too high (>0.5)" << std::endl;
    exit(-1);
  }

  // read the properties of the exodus file:
  Teuchos::RCP<DICe::mesh::Mesh> mesh = DICe::mesh::read_exodus_mesh(exo_name,"dummy_output_name.e");
  std::vector<std::string> mesh_fields = DICe::mesh::read_exodus_field_names(mesh);
  *outStream << "fields:" << std::endl;
  bool has_requested_field = false;
  for(size_t i=0;i<mesh_fields.size();++i){
    *outStream << mesh_fields[i] << std::endl;
    if(mesh_fields[i] == field_name) has_requested_field = true;
  }
  if(!has_requested_field){
    std::cout << "Error, requested field is not in the exodus file" << std::endl;
    exit(-1);
  }

  int_t num_time_steps = DICe::mesh::read_exodus_num_steps(mesh);
  *outStream << "number of time steps: " << num_time_steps << std::endl;
  DICe::mesh::close_exodus_output(mesh);

  // read the coordinates:
  std::vector<scalar_t> coords_x;
  std::vector<scalar_t> coords_y;
  std::vector<scalar_t> coords_z;
  DICe::mesh::read_exodus_coordinates(exo_name,coords_x,coords_y,coords_z);

  const int_t num_nodes = coords_x.size();
  std::cout << "number of nodes: " << num_nodes << std::endl;
  if(num_nodes<=0){
    std::cout << "Error, invalid number of nodes" << std::endl;
    exit(-1);
  }

  // get stats on the mesh size:
  // assumes a mostly planar geometry
  scalar_t min_x = std::numeric_limits<scalar_t>::max();
  scalar_t max_x = std::numeric_limits<scalar_t>::min();
  scalar_t avg_x = 0.0;
  scalar_t min_y = std::numeric_limits<scalar_t>::max();
  scalar_t max_y = std::numeric_limits<scalar_t>::min();
  scalar_t avg_y = 0.0;
  scalar_t min_z = std::numeric_limits<scalar_t>::max();
  scalar_t max_z = std::numeric_limits<scalar_t>::min();
  scalar_t avg_z = 0.0;

  for(int_t i=0;i<num_nodes;++i){
    avg_x += coords_x[i];
    avg_y += coords_y[i];
    avg_z += coords_z[i];
    if(coords_x[i]<min_x){
      min_x = coords_x[i];
    }
    if(coords_y[i]<min_y){
      min_y = coords_y[i];
    }
    if(coords_z[i]<min_z){
      min_z = coords_z[i];
    }
    if(coords_x[i]>max_x){
      max_x = coords_x[i];
    }
    if(coords_y[i]>max_y){
      max_y = coords_y[i];
    }
    if(coords_z[i]>max_z){
      max_z = coords_z[i];
    }
  }
  avg_x/=num_nodes;
  avg_y/=num_nodes;
  avg_z/=num_nodes;
  std::cout << "geometry stats: " << std::endl;
  std::cout << "min x: " << min_x << std::endl;
  std::cout << "max x: " << max_x << std::endl;
  std::cout << "avg x: " << avg_x << std::endl;
  std::cout << "min y: " << min_y << std::endl;
  std::cout << "max y: " << max_y << std::endl;
  std::cout << "avg y: " << avg_y << std::endl;
  std::cout << "min z: " << min_z << std::endl;
  std::cout << "max z: " << max_z << std::endl;
  std::cout << "avg z: " << avg_z << std::endl;

  // iterate the field values to get the stats:
  scalar_t min_value = std::numeric_limits<scalar_t>::max();
  scalar_t max_value = std::numeric_limits<scalar_t>::min();
  std::vector<scalar_t> values;
  for(int_t step=1;step<=num_time_steps;++step){
    values = mesh::read_exodus_field(exo_name,field_name,step);
    if(values.size()<=0){
      std::cout << "Error, reading field failed" << std::endl;
      exit(-1);
    }
    for(size_t i=0;i<values.size();++i){
      if(values[i]<min_value) min_value = values[i];
      if(values[i]>max_value) max_value = values[i];
    }
  }
  // determine the conversion factor from the field values to image intensity counts:
  const scalar_t range = max_value - min_value;
  const scalar_t counts_per_unit = 255/range;
  // value = (unit - min_unit)*counts_per_unit
  std::cout << "field stats on " << field_name << ": " << std::endl;
  std::cout << "min value: " << min_value << std::endl;
  std::cout << "max value: " << max_value << std::endl;
  std::cout << "counts per unit: " << counts_per_unit << std::endl;

  // set up a nearest neighbor tree to get the avg distance
  Teuchos::RCP<Point_Cloud_3D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_3D<scalar_t>());
  point_cloud->pts.resize(num_nodes);
  for(int_t i=0;i<num_nodes;++i){
    point_cloud->pts[i].x = coords_x[i];
    point_cloud->pts[i].y = coords_y[i];
    point_cloud->pts[i].z = coords_z[i];
  }
  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_3d_t> kd_tree =
      Teuchos::rcp(new kd_tree_3d_t(3 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  *outStream << "kd-tree completed" << std::endl;

  // create a regular grid (in this case, an image)

  // determine the two longest dimensions
  const scalar_t length_x = max_x - min_x;
  const scalar_t length_y = max_y - min_y;
  const scalar_t length_z = max_z - min_z;

  scalar_t min_i = 0.0;
  scalar_t max_i = 0.0;
  scalar_t min_j = 0.0;
  scalar_t max_j = 0.0;
  scalar_t avg_k = 0.0;

  if(length_x<=length_y && length_x<=length_z){
    // x is the thickness dim (i is z, j is y)
    std::cout << "x is the thickness dim" << std::endl;
    avg_k = avg_x;
    min_i = min_z; max_i = max_z;
    min_j = min_y; max_j = max_y;
  }
  else if(length_y<=length_x && length_y<=length_z){
    // y is the thickness dim (i is z, j is x)
    std::cout << "y is the thickness dim" << std::endl;
    avg_k = avg_y;
    min_i = min_z; max_i = max_z;
    min_j = min_x; max_j = max_x;
  }
  else{
    // z is the thickness dim (i is x, j is y)
    std::cout << "z is the thickness dim" << std::endl;
    avg_k = avg_z;
    min_i = min_x; max_i = max_x;
    min_j = min_y; max_j = max_y;
  }

  // determine the image size:
  const scalar_t length_i = max_i - min_i;
  const scalar_t length_j = max_j - min_j;
  const int_t img_w = (int_t)(length_i/mm_per_pixel);
  const int_t img_h = (int_t)(length_j/mm_per_pixel);
  std::cout << "converted image dimensions: " << img_w << " x " << img_h << std::endl;
  if(img_w<50||img_h<50){
    std::cout << "Error, image dimensions too small (is the mm/pixel conversion correct?)" << std::endl;
    exit(-1);
  }

  std::FILE * resultsFilePtr = fopen("results.txt","w");

  // create an image to store the values for each step
  Teuchos::RCP<DICe::Image> image = Teuchos::rcp(new DICe::Image(img_w,img_h,0.0));
  Teuchos::ArrayRCP<intensity_t> intensities = image->intensities();

  scalar_t query_pt[3];
  std::vector<size_t> ret_index(num_neigh);
  std::vector<scalar_t> out_dist_sqr(num_neigh);
  const int_t N = 3;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;

  Teuchos::ArrayRCP<double> u(num_neigh,0.0);
  Teuchos::ArrayRCP<double> X_t_u(N,0.0);
  Teuchos::ArrayRCP<double> coeffs(N,0.0);
  Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
  Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

  for(int_t step=1;step<=num_time_steps;++step){
    std::cout << "*** processing step: " << step << std::endl;
    values = mesh::read_exodus_field(exo_name,field_name,step);
    if(values.size()<=0){
      std::cout << "Error, reading field failed" << std::endl;
      exit(-1);
    }
    if((int_t)values.size()!=num_nodes){
      std::cout << "Error, values vector is the wrong size" << std::endl;
      exit(-1);
    }
    // populate the image values:
    for(int_t px_j=0;px_j<img_h;++px_j){
      for(int_t px_i=0;px_i<img_w;++px_i){
        // clear the storage
        X_t_X.putScalar(0.0);
        for(int_t i=0;i<N;++i){
          X_t_u[i] = 0.0;
          coeffs[i] = 0.0;
        }
        scalar_t my_x = min_i + px_i*mm_per_pixel;
        scalar_t my_y = min_j + px_j*mm_per_pixel;
        query_pt[0] = my_x;
        query_pt[1] = my_y;
        query_pt[2] = avg_k;
        kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
        // iterate the neighbors and fit the data
        for(int_t neigh = 0;neigh<num_neigh; ++neigh){
          const int_t neigh_id = ret_index[neigh];
          u[neigh] = values[neigh_id];
          // set up the X^T matrix
          X_t(0,neigh) = 1.0;
          X_t(1,neigh) = coords_x[neigh_id] - my_x;
          X_t(2,neigh) = coords_y[neigh_id] - my_y;
        }
        // set up X^T*X
        for(int_t k=0;k<N;++k){
          for(int_t m=0;m<N;++m){
            for(int_t j=0;j<num_neigh;++j){
              X_t_X(k,m) += X_t(k,j)*X_t(m,j);
            }
          }
        }
        lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
        lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);
        // compute X^T*u
        for(int_t i=0;i<N;++i){
          for(int_t j=0;j<num_neigh;++j){
            X_t_u[i] += X_t(i,j)*u[j];
          }
        }
        // compute the coeffs
        for(int_t i=0;i<N;++i){
          for(int_t j=0;j<N;++j){
            coeffs[i] += X_t_X(i,j)*X_t_u[j];
          }
        }
        scalar_t value = coeffs[0];
        // convert the value to counts and put it in the image:
        scalar_t count_value = (value - min_value)*counts_per_unit;
        intensities[px_j*img_w+px_i] = count_value;
      } // end image i
    } // end image j
    //image->write("test_image.tif");
    Teuchos::RCP<Image> image_fft = DICe::image_fft(image);
    Teuchos::ArrayRCP<intensity_t> fft_intensities = image_fft->intensities();
    //image_fft->write("test_image_fft.tif");
    // convert the frequency threshold to pixel limits
    const int_t px_thresh_top_i = img_w/2 + (int_t)(freq_thresh*img_w);
    const int_t px_thresh_bottom_i = img_w/2 - (int_t)(freq_thresh*img_w);
    const int_t px_thresh_top_j = img_h/2 + (int_t)(freq_thresh*img_h);
    const int_t px_thresh_bottom_j = img_h/2 - (int_t)(freq_thresh*img_h);
    scalar_t total_power = 0.0;
    scalar_t exceed_power = 0.0;
    for(int_t px_j=0;px_j<img_h;++px_j){
      for(int_t px_i=0;px_i<img_w;++px_i){
        total_power += fft_intensities[px_j*img_w+px_i];
        if(px_j>px_thresh_top_j||px_j<px_thresh_bottom_j||px_i>px_thresh_top_i||px_i<px_thresh_bottom_i)
          exceed_power += fft_intensities[px_j*img_w+px_i];
      }
    }
    const scalar_t ratio = exceed_power / total_power;
    fprintf(resultsFilePtr,"%i,%f,%f\n",step,freq_thresh,ratio);
    std::cout << "*** power ratio: " << ratio << std::endl;
  } // end step loop
  delete [] WORK;
  delete [] IPIV;

  fclose(resultsFilePtr);

  DICe::finalize();

  return 0;
}

