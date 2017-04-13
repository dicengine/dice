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

/*! \file  DICe_NetCDFToTiff.cpp
    \brief Utility for exporting NetCDF files to tiff files
*/

#include <DICe.h>
#include <DICe_NetCDF.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_PointCloud.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <cassert>
#include <string>
#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage 1: ./DICe_ExoToNetCDF <exo_file_name> <input_netcdf_file> <netcdf_output_file_name>
  /// only works for serial exo file, only converts nodal variables


  DICe::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  std::string delimiter = " ,\r";

  // determine if the second argument is help, file name or folder name
  if(argc!=5){
    std::cout << " DICe_ExoToNetCDF (exports a serial exodus file to NetCDF) " << std::endl;
    std::cout << " Syntax: DICe_ExoToNetCDF <exodus_file_name> <netcdf_file_name> <output_prefix (no extension)> <num_neighbors>" << std::endl;
    exit(-1);
  }
  std::string exo_name = argv[1];
  std::string netcdf_input_name = argv[2];
  std::string output_prefix = argv[3];
  const int_t num_neigh = std::atoi(argv[4]);

  *outStream << "exodus input file: " << exo_name << std::endl;
  *outStream << "netcdf input file: " << netcdf_input_name << std::endl;
  *outStream << "output file prefix: " << output_prefix << std::endl;
  *outStream << "num neighbors to use for surface fit: " << num_neigh << std::endl;

  // read the properties of the exodus file:
  Teuchos::RCP<DICe::mesh::Mesh> mesh = DICe::mesh::read_exodus_mesh(exo_name,"dummy_output_name.e");
  std::vector<std::string> mesh_fields = DICe::mesh::read_exodus_field_names(mesh);
  *outStream << "fields:" << std::endl;
  bool has_altitude = false;
  bool has_strain = false;
  bool has_acc_disp = false;
  for(size_t i=0;i<mesh_fields.size();++i){
    *outStream << mesh_fields[i] << std::endl;
    if(mesh_fields[i] == "ALTITUDE_ABOVE_GROUND") has_altitude = true;
    if(mesh_fields[i] == "VSG_STRAIN_XX") has_strain = true;
    if(mesh_fields[i] == "ACCUMULATED_DISP_X") has_acc_disp = true;
  }
  int_t num_time_steps = DICe::mesh::read_exodus_num_steps(mesh);
  *outStream << "number of time steps " << num_time_steps << std::endl;
  DICe::mesh::close_exodus_output(mesh);

  // read the image coordinates and intensities:
  Teuchos::RCP<DICe::netcdf::NetCDF_Reader> netcdf_reader = Teuchos::rcp(new DICe::netcdf::NetCDF_Reader());
  int_t img_w = 0;
  int_t img_h = 0;
  netcdf_reader->get_image_dimensions(netcdf_input_name,img_w,img_h);
  *outStream << "image dimensions; " << img_w << " x " << img_h << std::endl;
  std::vector<intensity_t> intensities(img_w*img_h);
  netcdf_reader->read_netcdf_image(netcdf_input_name.c_str(),&intensities[0]);
  // convert the intensities to floats to save space:
  std::vector<float> intens_float(img_w*img_h);
  for(int_t i=0;i<img_w*img_h;++i)
    intens_float[i] = (float)intensities[i];

  std::vector<std::string> output_field_names;
  output_field_names.push_back("data"); // holds the image intensities
  output_field_names.push_back("SUBSET_DISPLACEMENT_X");
  output_field_names.push_back("SUBSET_DISPLACEMENT_Y");
  output_field_names.push_back("SIGMA");
  if(has_strain){
    output_field_names.push_back("VSG_STRAIN_XX");
    output_field_names.push_back("VSG_STRAIN_YY");
    output_field_names.push_back("VSG_STRAIN_XY");
  }
  if(has_acc_disp){
    output_field_names.push_back("ACCUMULATED_DISP_X");
    output_field_names.push_back("ACCUMULATED_DISP_Y");
  }
  if(has_altitude)
    output_field_names.push_back("ALTITUDE_ABOVE_GROUND");

  // read the subset coordinates:
  std::vector<scalar_t> subset_coords_x = DICe::mesh::read_exodus_field(exo_name,"SUBSET_COORDINATES_X",1);
  std::vector<scalar_t> subset_coords_y = DICe::mesh::read_exodus_field(exo_name,"SUBSET_COORDINATES_Y",1);
  const int_t num_nodes = subset_coords_x.size();
  *outStream << "num subset coorindates: " << num_nodes << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(subset_coords_x.size()!=subset_coords_y.size(),std::runtime_error,"");

  // build a kd tree for the displacement values:
  // create neighborhood lists using nanoflann:
  Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud->pts.resize(num_nodes);
  for(int_t i=0;i<num_nodes;++i){
    point_cloud->pts[i].x = subset_coords_x[i];
    point_cloud->pts[i].y = subset_coords_y[i];
  }
  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_2d_t> kd_tree =
      Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  *outStream << "kd-tree completed" << std::endl;

  // temp storage
  scalar_t query_pt[2];
  std::vector<size_t> ret_index(num_neigh);
  std::vector<scalar_t> out_dist_sqr(num_neigh);
  const int_t N = 3;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  Teuchos::LAPACK<int,double> lapack;

  // iterate each step in the exodus file:
  for(int_t step=0;step<num_time_steps;++step){
    std::stringstream netcdf_name;
    netcdf_name << output_prefix << "_" << step << ".nc";
    *outStream << "processing output file " << netcdf_name.str() << std::endl;
    Teuchos::RCP<DICe::netcdf::NetCDF_Writer> netcdf_writer =
        Teuchos::rcp(new DICe::netcdf::NetCDF_Writer(netcdf_name.str().c_str(),img_w,img_h,output_field_names));
    // add the image intensities to the output file
    netcdf_writer->write_float_array("data",intens_float);

    // get each of the fields from the exodus mesh
    std::vector<std::vector<scalar_t> > exo_fields(output_field_names.size()-1);
    std::vector<std::vector<float> > pixel_fields(output_field_names.size()-1,std::vector<float>(img_w*img_h));
    for(size_t j=0;j<exo_fields.size();++j)
      exo_fields[j] = DICe::mesh::read_exodus_field(exo_name,output_field_names[j+1],step+1);

    // iterate each pixel
    for(int_t y=0;y<img_h;++y){
      for(int_t x=0;x<img_w;++x){
        // determine the nearest num_neigh points to this pixel:
        query_pt[0] = x;
        query_pt[1] = y;
        kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);

        Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
        Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);
        for(int_t j=0;j<num_neigh;++j){
          const int_t neigh_id = ret_index[j];
          X_t(0,j) = 1.0;
          X_t(1,j) = subset_coords_x[neigh_id] - x;
          X_t(2,j) = subset_coords_y[neigh_id] - y;
        }
        // set up X^T*X
        for(int_t k=0;k<N;++k){
          for(int_t m=0;m<N;++m){
            for(int_t j=0;j<num_neigh;++j){
              X_t_X(k,m) += X_t(k,j)*X_t(m,j);
            }
          }
        }
        //X_t_X.print(std::cout);
        // Invert X^T*X
        lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
        lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);

        // iterate over the fields and compute the projected values at the pixels
        for(size_t f=0;f<exo_fields.size();++f){
          Teuchos::ArrayRCP<double> u(num_neigh,0.0);
          if(output_field_names[f+1]=="SIGMA"){ // sigma get the nearest neighbor's value with no interpolation
            pixel_fields[f][y*img_w+x] = exo_fields[f][ret_index[0]];
            continue;
          }
          for(int_t j=0;j<num_neigh;++j){
            const int_t neigh_id = ret_index[j];
            u[j] = exo_fields[f][neigh_id];
          }
          Teuchos::ArrayRCP<double> X_t_u(N,0.0);
          for(int_t i=0;i<N;++i){
            for(int_t j=0;j<num_neigh;++j){
              X_t_u[i] += X_t(i,j)*u[j];
            }
          }
          // compute the coeffs
          Teuchos::ArrayRCP<double> coeffs(N,0.0);
          for(int_t i=0;i<N;++i){
            for(int_t j=0;j<N;++j){
              coeffs[i] += X_t_X(i,j)*X_t_u[j];
            }
          }
          pixel_fields[f][y*img_w+x] = (float)coeffs[0];
        } // end field loop
      } // end x pixel loop
    } // end y pixel loop
    // save off the resulting pixel values
    for(size_t f=0;f<pixel_fields.size();++f){
      netcdf_writer->write_float_array(output_field_names[f+1],pixel_fields[f]);
    }
  } // end step loop

  delete [] WORK;
  delete [] GWORK;
  delete [] IWORK;
  delete [] IPIV;

  DICe::finalize();

  return 0;
}

