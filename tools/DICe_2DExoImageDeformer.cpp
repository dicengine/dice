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

/*! \file  DICe_2DExoImageDeformer.cpp
    \brief Utility to deform images based on a displacment field in an exodus file
*/

#include <DICe.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_PointCloud.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>


#include <exodusII.h>

#include <string>
#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage: ./DICe_2DExoImageDeformer <input.xml>

  DICe::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  std::string delimiter = " ,\r";

  // determine if the second argument is help, file name or folder name
  if(argc!=2){
    std::cout << " DICe_2DExoImageDeformer" << std::endl;
    std::cout << " Syntax: DICe_2DExoImageDeformer <input xml file name>" << std::endl;
    exit(-1);
  }

  // parse the input parameter from the input file

  const std::string input_file = argv[1];
  std::cout << "reading parameters from file: " << input_file << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> paramsPtr(params.get());
  Teuchos::updateParametersFromXmlFile(input_file, paramsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,"");
  params->print(std::cout);
  // ensure all the necessary parameters are there
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("scale_factor"),std::runtime_error,"");
  const work_t scale_factor = params->get<double>("scale_factor");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("model_origin_in_pixels_x"),std::runtime_error,"");
  const int_t ox = params->get<int_t>("model_origin_in_pixels_x");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("model_origin_in_pixels_y"),std::runtime_error,"");
  const int_t oy = params->get<int_t>("model_origin_in_pixels_y");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("num_neighbors"),std::runtime_error,"");
  const int_t num_neigh = params->get<int_t>("num_neighbors");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("exodus_file"),std::runtime_error,"");
  const std::string exo_file = params->get<std::string>("exodus_file");
  //TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("z_value"),std::runtime_error,"");
  //const work_t z_value = params->get<double>("z_value");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("disp_field_name_x"),std::runtime_error,"");
  const std::string disp_field_name_x = params->get<std::string>("disp_field_name_x");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("disp_field_name_y"),std::runtime_error,"");
  const std::string disp_field_name_y = params->get<std::string>("disp_field_name_y");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("input_image_name"),std::runtime_error,"");
  const std::string input_image_name = params->get<std::string>("input_image_name");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("output_image_name"),std::runtime_error,"");
  const std::string output_image_name = params->get<std::string>("output_image_name");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("exodus_step"),std::runtime_error,"");
  const int_t exo_step = params->get<int_t>("exodus_step");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("plot_disp_threshold"),std::runtime_error,"");
  const work_t plot_thresh = params->get<double>("plot_disp_threshold");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("convert_to_eulerian"),std::runtime_error,"");
  const bool convert_to_eulerian = params->get<bool>("convert_to_eulerian");

  // read the exodus file to get the displacement field and coordinates

  std::vector<work_t> coords_x;
  std::vector<work_t> coords_y;
  std::vector<work_t> coords_z;
  DICe::mesh::read_exodus_coordinates(exo_file,coords_x,coords_y,coords_z);
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x.size()!=coords_y.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x.size()!=coords_z.size(),std::runtime_error,"");
  std::cout << "sucessfully read coordinates of size " << coords_x.size() << std::endl;
  std::vector<work_t> exo_ux = DICe::mesh::read_exodus_field(exo_file,disp_field_name_x,exo_step+1);
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x.size()!=exo_ux.size(),std::runtime_error,"");
  std::cout << "sucessfully read displacemet field x of size " << exo_ux.size() << std::endl;
  std::vector<work_t> exo_uy = DICe::mesh::read_exodus_field(exo_file,disp_field_name_y,exo_step+1);
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x.size()!=exo_uy.size(),std::runtime_error,"");
  std::cout << "sucessfully read displacemet field y of size " << exo_uy.size() << std::endl;
  const int_t num_nodes = exo_ux.size();

  Teuchos::RCP<Image> ref_img = Teuchos::rcp(new Image(input_image_name.c_str()));
  const int_t img_w = ref_img->width();
  const int_t img_h = ref_img->height();
  const int_t num_px = img_w*img_h;
  Teuchos::ArrayRCP<work_t> def_intens(num_px,0.0);

  // build a kd tree for the displacement values:
  // create neighborhood lists using nanoflann:
  Teuchos::RCP<Point_Cloud_2D<work_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<work_t>());
  point_cloud->pts.resize(num_nodes);
  for(int_t i=0;i<num_nodes;++i){
    point_cloud->pts[i].x = convert_to_eulerian ? coords_x[i] + exo_ux[i]: coords_x[i];
    point_cloud->pts[i].y = convert_to_eulerian ? coords_y[i] + exo_uy[i]: coords_y[i];
  }

  std::stringstream csv_exo_filename;
  // strip off the extention from the original file name
  size_t lastindex = output_image_name.find_last_of(".");
  std::string rawname = output_image_name.substr(0, lastindex);
  csv_exo_filename << rawname << ".csv";

  // output a csv file with the exodus points
  std::FILE * infoFilePtr = fopen(csv_exo_filename.str().c_str(),"w");
  fprintf(infoFilePtr,"X,Y,Z,U,V\n");
  for(int_t i=0;i<num_nodes;++i){
    fprintf(infoFilePtr,"%4.4E,%4.4E,%4.4E,%4.4E,%4.4E\n",point_cloud->pts[i].x,point_cloud->pts[i].y,coords_z[i],exo_ux[i],exo_uy[i]);
  }
  fclose(infoFilePtr);


  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_2d_t> kd_tree =
      Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  DEBUG_MSG("kd-tree completed");

  // perform a pass to size the neighbor lists
  std::vector<std::vector<int_t> > neighbor_list(num_px);
  std::vector<std::vector<work_t> > neighbor_dist_x(num_px);
  std::vector<std::vector<work_t> > neighbor_dist_y(num_px);
  work_t query_pt[2];
  std::vector<size_t> ret_index(num_neigh);
  std::vector<work_t> out_dist_sqr(num_neigh);
  const int_t N = 3;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  // Note, LAPACK does not allow templating on long int or work_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;

  // output a csv file with the image points
  std::stringstream img_pts_filename;
  img_pts_filename << rawname << "_img.csv";

  std::FILE * imgFilePtr = fopen(img_pts_filename.str().c_str(),"w");
  fprintf(imgFilePtr,"X,Y,Z,U,V\n");

  for(int_t py=0;py<img_h;++py){
    for(int_t px=0;px<img_w;++px){
      // compute the model location for this pixel:
      // remove the model offsets and scale the positions
      work_t mx = (px - ox)*scale_factor;
      work_t my = (oy - py)*scale_factor; // y pixel has to be flipped to match the model coordiantes which are y up instead of y down like the image
      query_pt[0] = mx;
      query_pt[1] = my;
      kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
      work_t ls_ux = 0.0;
      work_t ls_uy = 0.0;
      Teuchos::ArrayRCP<double> u_x(num_neigh,0.0);
      Teuchos::ArrayRCP<double> u_y(num_neigh,0.0);
      Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
      Teuchos::ArrayRCP<double> X_t_u_y(N,0.0);
      Teuchos::ArrayRCP<double> coeffs_x(N,0.0);
      Teuchos::ArrayRCP<double> coeffs_y(N,0.0);
      Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
      Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);
      for(int_t j=0;j<num_neigh;++j){
        const int_t neigh_id = ret_index[j];
        u_x[j] = exo_ux[neigh_id];
        u_y[j] = exo_uy[neigh_id];
        X_t(0,j) = 1.0;
        X_t(1,j) = point_cloud->pts[neigh_id].x - mx;
        X_t(2,j) = point_cloud->pts[neigh_id].y - my;
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
      // compute X^T*u
      for(int_t i=0;i<N;++i){
        for(int_t j=0;j<num_neigh;++j){
          X_t_u_x[i] += X_t(i,j)*u_x[j];
          X_t_u_y[i] += X_t(i,j)*u_y[j];
        }
      }
      // compute the coeffs
      for(int_t i=0;i<N;++i){
        for(int_t j=0;j<N;++j){
          coeffs_x[i] += X_t_X(i,j)*X_t_u_x[j];
          coeffs_y[i] += X_t_X(i,j)*X_t_u_y[j];
        }
      }
      ls_ux = coeffs_x[0];
      ls_uy = coeffs_y[0];
      // convert the displacement back to image coordinates
      const work_t bx = ls_ux / scale_factor;
      const work_t by = ls_uy / scale_factor;
      work_t out_bx = std::abs(ls_ux) < plot_thresh ? ls_ux : 0.0;
      work_t out_by = std::abs(ls_uy) < plot_thresh ? ls_uy : 0.0;
      fprintf(imgFilePtr,"%4.4E,%4.4E,%4.4E,%4.4E,%4.4E\n",mx,my,0.0,out_bx,out_by);

      // apply the displacement to the image
      const work_t intens = ref_img->interpolate_keys_fourth(px-bx,py-by);
      def_intens[py*img_w+px] = intens > 0.0 ? intens : 0.0;
    } // end px
  } // end py

  fclose(imgFilePtr);

  // output the deformed image:
  Teuchos::RCP<Scalar_Image> def_img = Teuchos::rcp(new Scalar_Image(img_w,img_h,def_intens));
  def_img->write(output_image_name);

  delete [] WORK;
  delete [] GWORK;
  delete [] IWORK;
  delete [] IPIV;

  DICe::finalize();

  return 0;
}

