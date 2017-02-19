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

#include <DICe_MeshIO.h>
#include <DICe_MeshIOUtils.h>
#include <DICe_ParserUtils.h>
#include <DICe_PointCloud.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

namespace DICe {
namespace mesh {

Importer_Projector::Importer_Projector(const std::string & source_file_name,
  Teuchos::RCP<DICe::mesh::Mesh> target_mesh):
  projection_required_(true),
  num_neigh_(5){

  const int_t spa_dim = target_mesh->spatial_dimension();
  Teuchos::RCP<MultiField> coords = target_mesh->get_field(DICe::mesh::field_enums::INITIAL_COORDINATES_FS);
  target_pts_x_.resize(target_mesh->num_nodes());
  target_pts_y_.resize(target_mesh->num_nodes());
  for(size_t i=0;i<target_mesh->num_nodes();++i){
    target_pts_x_[i] = coords->local_value(i*spa_dim+0);
    target_pts_y_[i] = coords->local_value(i*spa_dim+1);
  }
  initialize_source_points(source_file_name);
}

Importer_Projector::Importer_Projector(const std::string & source_file_name,
  const std::string & target_file_name):
  projection_required_(true),
  num_neigh_(5){

  // read in the target locations
  read_coordinates(target_file_name,target_pts_x_,target_pts_y_);
  // read in the source points
  // read in the source points
  if(target_file_name==source_file_name){
    projection_required_ = false;
    source_pts_x_ = target_pts_x_;
    source_pts_y_ = target_pts_y_;
  }
  else{
    initialize_source_points(source_file_name);
  }
}

void
Importer_Projector::initialize_source_points(const std::string & source_file_name){

  // only acceptable option here is that the source points come from DICe exodus mesh or a DICe text output file
  // DICe text output file
  const std::string text_ext(".txt");
  const std::string exo_ext(".e");
  // make sure its not a locations file as the input
  if(source_file_name.find(text_ext)!=std::string::npos){
    std::fstream dataFile(source_file_name.c_str(), std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading file " << source_file_name);
    std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");
    TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()<4,std::runtime_error,"Error, invalid source file " << source_file_name << ". must have at least 4 cols, (x,y,u,v)");
    dataFile.close();
  }
  if(source_file_name.find(text_ext)!=std::string::npos||source_file_name.find(exo_ext)!=std::string::npos){
    read_coordinates(source_file_name,source_pts_x_,source_pts_y_);
  }
  // DICe exodus file
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid source file format." << source_file_name);
  }
  projection_required_ = true;

  // determine if both files have the same points:
  if(source_pts_x_.size()==target_pts_x_.size()){
    const scalar_t tol = 1.0E-3;
    scalar_t diff = 0.0;
    for(size_t i=0;i<source_pts_x_.size();++i){
      diff += (std::abs(source_pts_x_[i] - target_pts_x_[i]) + std::abs(source_pts_y_[i] - target_pts_y_[i]));
    }
    DEBUG_MSG("Importer_Projector::Importer_Projector(): difference between source and target points: " << diff);
    if(diff <= tol){
      DEBUG_MSG("Importer_Projector::Importer_Projector(): diff tolerance met, treating as colocated points (no projection will be used)");
      projection_required_ = false;
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(target_pts_x_.size()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(target_pts_y_.size()!=target_pts_x_.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(source_pts_x_.size()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(source_pts_y_.size()!=source_pts_x_.size(),std::runtime_error,"");
  // set up the neighbor lists for projection
  if(projection_required_){
    Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
    point_cloud->pts.resize(source_pts_x_.size());
    for(size_t i=0;i<source_pts_x_.size();++i){
      point_cloud->pts[i].x = source_pts_x_[i];
      point_cloud->pts[i].y = source_pts_y_[i];
    }
    DEBUG_MSG("Importer_Projector::Importer_Projector(): building the kd-tree");
    Teuchos::RCP<kd_tree_2d_t> kd_tree =
        Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
    kd_tree->buildIndex();
    DEBUG_MSG("Importer_Projector::Importer_Projector(): kd-tree completed");
    std::vector<std::pair<size_t,scalar_t> > ret_matches;
    DEBUG_MSG("Importer_Projector::Importer_Projector(): executing neighbor search");
    neighbors_.resize(target_pts_x_.size());
    neighbor_dist_x_.resize(target_pts_x_.size());
    neighbor_dist_y_.resize(target_pts_x_.size());
    for(size_t i=0;i<target_pts_x_.size();++i){
      neighbors_[i].resize(num_neigh_);
      neighbor_dist_x_[i].resize(num_neigh_);
      neighbor_dist_y_[i].resize(num_neigh_);
    }
    std::vector<size_t> ret_index(num_neigh_);
    std::vector<scalar_t> out_dist_sqr(num_neigh_);
    std::vector<scalar_t> query_pt(2,0.0);
    for(size_t i=0;i<target_pts_x_.size();++i){
      query_pt[0] = target_pts_x_[i];
      query_pt[1] = target_pts_y_[i];
      kd_tree->knnSearch(&query_pt[0], num_neigh_, &ret_index[0], &out_dist_sqr[0]);
      for(int_t neigh = 0;neigh<num_neigh_;++neigh){
        neighbors_[i][neigh] = ret_index[neigh];
        neighbor_dist_x_[i][neigh] = source_pts_x_[ret_index[neigh]] - target_pts_x_[i];
        neighbor_dist_y_[i][neigh] = source_pts_y_[ret_index[neigh]] - target_pts_y_[i];
      }
    }
    DEBUG_MSG("Importer_Projector::Importer_Projector(): projection neighborhood has been initialized");
  }
}

void
Importer_Projector::import_vector_field(const std::string & file_name,
  const std::string & field_name,
  std::vector<scalar_t> & field_x,
  std::vector<scalar_t> & field_y,
  const int_t step){

  // read and project (if necessary) the requested field from the source file to the target points
  field_x.clear();
  field_y.clear();
  if(!projection_required_){
    // simple copy operation
    read_vector_field(file_name,field_name,field_x,field_y,step);

    // fields have to be of compatible sizes
    TEUCHOS_TEST_FOR_EXCEPTION(field_x.size()!=target_pts_x_.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(field_y.size()!=target_pts_y_.size(),std::runtime_error,"");
    return;
  }
  else{
    std::vector<scalar_t> source_field_x;
    std::vector<scalar_t> source_field_y;
    read_vector_field(file_name,field_name,source_field_x,source_field_y,step);
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)source_field_x.size()!=num_source_pts(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)source_field_y.size()!=num_source_pts(),std::runtime_error,"");
    field_x.resize(num_target_pts());
    field_y.resize(num_target_pts());

    static int_t N = 3;
    static std::vector<int> IPIV(N+1,0);
    static int_t LWORK = N*N;
    static int_t INFO = 0;
    static std::vector<double> WORK(LWORK,0.0);
    Teuchos::LAPACK<int,double> lapack;

    static Teuchos::ArrayRCP<double> u_x(num_neigh_,0.0);
    static Teuchos::ArrayRCP<double> u_y(num_neigh_,0.0);
    static Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
    static Teuchos::ArrayRCP<double> X_t_u_y(N,0.0);
    static Teuchos::ArrayRCP<double> coeffs_x(N,0.0);
    static Teuchos::ArrayRCP<double> coeffs_y(N,0.0);
    static Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh_, true);
    static Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

    const int_t num_points = target_pts_x_.size();
    TEUCHOS_TEST_FOR_EXCEPTION(num_points<=0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)target_pts_y_.size()!=num_points,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbors_.size()!=num_points,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbor_dist_x_.size()!=num_points,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbor_dist_y_.size()!=num_points,std::runtime_error,"");

    int_t neigh_id = 0;
    for(int_t pt=0;pt<num_points;++pt){
      // clear storage
      for(int_t i=0;i<N;++i){
        X_t_u_x[i] = 0.0;
        X_t_u_y[i] = 0.0;
        coeffs_x[i] = 0.0;
        coeffs_y[i] = 0.0;
        for(int_t j=0;j<N;++j)
          X_t_X(i,j) = 0.0;
        for(int_t j=0;j<num_neigh_;++j)
          X_t(i,j) = 0.0;
      }
      // gather the displacements of the neighbors
      for(int_t j=0;j<num_neigh_;++j){
        neigh_id = neighbors_[pt][j];
        u_x[j] = source_field_x[neigh_id];
        u_y[j] = source_field_y[neigh_id];
      }
      // set up the X^T matrix
      for(int_t j=0;j<num_neigh_;++j){
        X_t(0,j) = 1.0;
        X_t(1,j) = neighbor_dist_x_[pt][j];
        X_t(2,j) = neighbor_dist_y_[pt][j];
      }
      // set up X^T*X
      for(int_t k=0;k<N;++k){
        for(int_t m=0;m<N;++m){
          for(int_t j=0;j<num_neigh_;++j){
            X_t_X(k,m) += X_t(k,j)*X_t(m,j);
          }
        }
      }
      // Invert X^T*X
      try
      {
        lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),&IPIV[0],&INFO);
      }
      catch(std::exception &e){
        DEBUG_MSG( e.what() << '\n');
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      try
      {
        lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),&IPIV[0],&WORK[0],LWORK,&INFO);
      }
      catch(std::exception &e){
        DEBUG_MSG( e.what() << '\n');
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      // compute X^T*u
      for(int_t i=0;i<N;++i){
        for(int_t j=0;j<num_neigh_;++j){
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
      // update the field values
      field_x[pt] = coeffs_x[0];
      field_y[pt] = coeffs_y[0];
    }// end point loop
  } // end projection required
}

void
Importer_Projector::read_vector_field(const std::string & file_name,
  const std::string & field_name,
  std::vector<scalar_t> & field_x,
  std::vector<scalar_t> & field_y,
  const int_t step){

  field_x.clear();
  field_y.clear();

  // automatically append the components to the field name
  const std::string field_name_x = field_name + "_X";
  const std::string field_name_y = field_name + "_Y";

  DEBUG_MSG("Importer_Projector::read_vector_field: reading vector fields " << field_name_x << " and " << field_name_y );

  // determine the file type
  const std::string text_ext(".txt");
  const std::string exo_ext(".e");

  // text file, could be a set of points or DICe output file
  if(file_name.find(text_ext)!=std::string::npos){
    TEUCHOS_TEST_FOR_EXCEPTION(step!=0,std::runtime_error,"Error, cannot specify step!=0 for text input");
    // read the first line of the file
    std::fstream dataFile(file_name.c_str(), std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading file " << file_name);
    std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");
    // DICe text output file
    DEBUG_MSG("Importer_Projector::read_vector_field(): reading vector field from DICe text results file: " << file_name);
    int_t x_var_index = -1;
    int_t y_var_index = -1;
    int_t num_values_per_line = tokens.size();
    for(int_t i=0;i<num_values_per_line;++i){
      DEBUG_MSG("Importer_Projector::read_vector_field(): found field " << tokens[i] << " in results file");
      if(strcmp(tokens[i].c_str(),field_name_x.c_str())==0){
        x_var_index = i;
      }
      if(strcmp(tokens[i].c_str(),field_name_y.c_str())==0){
        y_var_index = i;
      }
    }
    DEBUG_MSG("Importer_Projector::read_vector_field(): using column: " << x_var_index << " as field " << field_name_x);
    DEBUG_MSG("Importer_Projector::read_vector_field(): using column: " << y_var_index << " as field " << field_name_y);
    TEUCHOS_TEST_FOR_EXCEPTION(x_var_index < 0,std::runtime_error,"Error could not find " << field_name_x);
    TEUCHOS_TEST_FOR_EXCEPTION(y_var_index < 0,std::runtime_error,"Error could not find " << field_name_y);
    while (!dataFile.eof())
    {
      std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");
      if(tokens.size()==0)continue;
      TEUCHOS_TEST_FOR_EXCEPTION((int_t)tokens.size()!=num_values_per_line,std::runtime_error,"Invalid line in file (inconsistent number of values per line)");
      for(int_t i=0;i<num_values_per_line;++i){
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[i]),std::runtime_error,"Invalid (non-numeric) line entry in file");
      }
      field_x.push_back(strtod(tokens[x_var_index].c_str(),NULL));
      field_y.push_back(strtod(tokens[y_var_index].c_str(),NULL));
    } // end file read
    dataFile.close();
  }
    // exodus file
  else if(file_name.find(exo_ext)){
    DEBUG_MSG("Importer_Projector::read_vector_field(): reading vector field from DICe exodus results file: " << file_name);
    field_x = read_exodus_field(file_name,field_name_x,step+1);
    field_y = read_exodus_field(file_name,field_name_y,step+1);
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized file format " << file_name);
  }
}

bool
Importer_Projector::is_valid_vector_source_field(const std::string & file_name,
  const std::string & field_name){

  // automatically append the components to the field name
  const std::string field_name_x = field_name + "_X";
  const std::string field_name_y = field_name + "_Y";

  DEBUG_MSG("Importer_Projector::is_valid_source_field: looking for vector fields " << field_name_x << " and " << field_name_y );

  // determine the file type
  const std::string text_ext(".txt");
  const std::string exo_ext(".e");

  // text file, could be a set of points or DICe output file
  if(file_name.find(text_ext)!=std::string::npos){
    // read the first line of the file
    std::fstream dataFile(file_name.c_str(), std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading file " << file_name);
    std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");
    int_t num_values_per_line = tokens.size();
    for(int_t i=0;i<num_values_per_line;++i){
      DEBUG_MSG("Importer_Projector::read_vector_field(): found field " << tokens[i] << " in results file");
      if(strcmp(tokens[i].c_str(),field_name_x.c_str())==0||strcmp(tokens[i].c_str(),field_name_y.c_str())==0){
        return true;
      }
    }
    dataFile.close();
  }
  // exodus file
  else if(file_name.find(exo_ext)){
    std::vector<std::string> field_names = DICe::mesh::read_exodus_field_names(file_name);
    for(size_t i=0;i<field_names.size();++i){
      if(field_names[i]==field_name_x||field_names[i]==field_name_y){
        return true;
      }
    }
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized file format " << file_name);
  }
  return false;
}


void
Importer_Projector::read_coordinates(const std::string & file_name,
  std::vector<scalar_t> & coords_x,
  std::vector<scalar_t> & coords_y){
  coords_x.clear();
  coords_y.clear();

  // determine the file type
  const std::string text_ext(".txt");
  const std::string csv_ext(".csv");
  const std::string exo_ext(".e");

  // text file, could be a set of points or DICe output file
  if(file_name.find(text_ext)!=std::string::npos||file_name.find(csv_ext)!=std::string::npos){
    // read the first line of the file
    std::fstream dataFile(file_name.c_str(), std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading file " << file_name);
    std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");

    // locations file
    if(file_name.find(csv_ext)!=std::string::npos||tokens.size()==2){
      DEBUG_MSG("Importer_Projector::read_coordinates(): reading points from file: " << file_name);
      // go back to the beginning of the file
      dataFile.clear();
      dataFile.seekg(0, std::ios::beg);
      // read each line of the file
      while (!dataFile.eof())
      {
        std::vector<std::string> tokens = tokenize_line(dataFile," \r\n,");
        if(tokens.size()==0)continue;
        TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=2,std::runtime_error,"Invalid line in file (should be two values per line)");
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[0])||!is_number(tokens[1]),std::runtime_error,"Invalid line in file (non-numeric entry)");
        coords_x.push_back(strtod(tokens[0].c_str(),NULL));
        coords_y.push_back(strtod(tokens[1].c_str(),NULL));
      } // end file read
      DEBUG_MSG("Importer_Projector::read_coordinates(): number of points: " << coords_x.size());
    }

    // DICe text output file
    else{
      DEBUG_MSG("Importer_Projector::read_coordinates(): reading points from DICe text results file: " << file_name);
      read_vector_field(file_name,"COORDINATE",coords_x,coords_y);
    }
    dataFile.close();
  }
    // exodus file
  else if(file_name.find(exo_ext)){
    DEBUG_MSG("Importer_Projector::read_coordinates(): reading target points from DICe exodus results file: " << file_name);
    read_vector_field(file_name,"INITIAL_COORDINATES",coords_x,coords_y);
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unrecognized file format " << file_name);
  }
}



} // mesh
} // DICe
