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
#include <DICe_Parser.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_Schema.h>
#include <DICe_PostProcessor.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_LAPACK.hpp>

#include <nanoflann.hpp>

#include <string>

#if DICE_MPI
#  include <mpi.h>
#endif

using namespace DICe;

void project_to_points(Teuchos::ArrayRCP<scalar_t> & points_x,
  Teuchos::ArrayRCP<scalar_t> & points_y,
  const int_t num_neigh,
  const std::vector<std::vector<int_t> > & neighbors,
  const std::vector<std::vector<scalar_t> > & neighbor_dist_x,
  const std::vector<std::vector<scalar_t> > & neighbor_dist_y,
  const std::vector<scalar_t> & disp_x,
  const std::vector<scalar_t> & disp_y,
  Teuchos::RCP<MultiField> & proj_disp_x,
  Teuchos::RCP<MultiField> & proj_disp_y){
  static int_t N = 3;
  static std::vector<int> IPIV(N+1,0);
  static int_t LWORK = N*N;
  static int_t INFO = 0;
  static std::vector<double> WORK(LWORK,0.0);
  Teuchos::LAPACK<int,double> lapack;

  static Teuchos::ArrayRCP<double> u_x(num_neigh,0.0);
  static Teuchos::ArrayRCP<double> u_y(num_neigh,0.0);
  static Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
  static Teuchos::ArrayRCP<double> X_t_u_y(N,0.0);
  static Teuchos::ArrayRCP<double> coeffs_x(N,0.0);
  static Teuchos::ArrayRCP<double> coeffs_y(N,0.0);
  static Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
  static Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

  const int_t num_points = points_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(num_points<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)points_y.size()!=num_points,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbors.size()!=num_points,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbor_dist_x.size()!=num_points,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbor_dist_y.size()!=num_points,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(proj_disp_x->get_map()->get_num_global_elements()!=num_points,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(proj_disp_y->get_map()->get_num_global_elements()!=num_points,std::runtime_error,"");

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
      for(int_t j=0;j<num_neigh;++j)
        X_t(i,j) = 0.0;
    }
    // gather the displacements of the neighbors
    for(int_t j=0;j<num_neigh;++j){
      neigh_id = neighbors[pt][j];
      u_x[j] = disp_x[neigh_id];
      u_y[j] = disp_y[neigh_id];
    }
    // set up the X^T matrix
    for(int_t j=0;j<num_neigh;++j){
      X_t(0,j) = 1.0;
      X_t(1,j) = neighbor_dist_x[pt][j];
      X_t(2,j) = neighbor_dist_y[pt][j];
    }
    // set up X^T*X
    for(int_t k=0;k<N;++k){
      for(int_t m=0;m<N;++m){
        for(int_t j=0;j<num_neigh;++j){
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
    // update the field values
    proj_disp_x->local_value(pt) = coeffs_x[0];
    proj_disp_y->local_value(pt) = coeffs_y[0];
  }// end point loop
}

// generic text reader
void read_text_results_file_coordinates(std::string & file_name,
  std::vector<scalar_t> & coords_x,
  std::vector<scalar_t> & coords_y){
  coords_x.clear();
  coords_y.clear();

  std::fstream dataFile(file_name.c_str(), std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading text results file " << file_name);
  // read each line of the file
  int_t x_coord_index = 0;
  int_t y_coord_index = 0;
  int_t num_values_per_line = 0;
  bool first_line = true;
  while (!dataFile.eof())
  {
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile," \r\n,");
    if(first_line){
      num_values_per_line = tokens.size();
      // expecting header info
      bool coords_x_found = false;
      bool coords_y_found = false;
      for(int_t i=0;i<tokens.size();++i){
        DEBUG_MSG("DICe_Strain(): found field " << tokens[i] << " in results file");
        if(strcmp(tokens[i].c_str(),"COORDINATE_X")==0){
          coords_x_found = true;
          x_coord_index = i;
        }
        if(strcmp(tokens[i].c_str(),"COORDINATE_Y")==0){
          coords_y_found = true;
          y_coord_index = i;
        }
      }
      DEBUG_MSG("DICe_Strain(): using column: " << x_coord_index << " as the coord x field");
      DEBUG_MSG("DICe_Strain(): using column: " << y_coord_index << " as the coord y field");
      TEUCHOS_TEST_FOR_EXCEPTION(!coords_x_found,std::runtime_error,"Error could not find x coords field");
      TEUCHOS_TEST_FOR_EXCEPTION(!coords_y_found,std::runtime_error,"Error could not find y coords field");
      first_line = false;
    }
    else{
      if(tokens.size()==0)continue;
      TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=num_values_per_line,std::runtime_error,"Invalid line in file");
      for(int_t i=0;i<num_values_per_line;++i){
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[i]),std::runtime_error,"Invalid (non-numeric) line entry in file");
      }
      coords_x.push_back(strtod(tokens[x_coord_index].c_str(),NULL));
      coords_y.push_back(strtod(tokens[y_coord_index].c_str(),NULL));
    }
  } // end text file read
}

// generic text reader
void read_text_results_file_displacements(std::string & file_name,
  std::vector<scalar_t> & disp_x,
  std::vector<scalar_t> & disp_y){

  disp_x.clear();
  disp_y.clear();

  std::fstream dataFile(file_name.c_str(), std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading text results file " << file_name);
  // read each line of the file
  int_t x_disp_index = 0;
  int_t y_disp_index = 0;
  int_t num_values_per_line = 0;
  bool first_line = true;
  while (!dataFile.eof())
  {
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile," \r\n,");
    if(first_line){
      num_values_per_line = tokens.size();
      // expecting header info
      bool disp_x_found = false;
      bool disp_y_found = false;
      for(int_t i=0;i<tokens.size();++i){
        DEBUG_MSG("DICe_Strain(): found field " << tokens[i] << " in results file");
        if(strcmp(tokens[i].c_str(),"SUBSET_DISPLACEMENT_X")==0){
          x_disp_index = i;
          disp_x_found = true;
        }
        if(strcmp(tokens[i].c_str(),"SUBSET_DISPLACEMENT_Y")==0){
          y_disp_index = i;
          disp_y_found = true;
        }
        if(strcmp(tokens[i].c_str(),"DISPLACEMENT_X")==0){
          x_disp_index = i;
          disp_x_found = true;
        }
        if(strcmp(tokens[i].c_str(),"DISPLACEMENT_Y")==0){
          y_disp_index = i;
          disp_y_found = true;
        }
      }
      DEBUG_MSG("DICe_Strain(): using column: " << x_disp_index << " as the displacement x field");
      DEBUG_MSG("DICe_Strain(): using column: " << y_disp_index << " as the displacement y field");
      TEUCHOS_TEST_FOR_EXCEPTION(!disp_x_found,std::runtime_error,"Error could not find x displacement field");
      TEUCHOS_TEST_FOR_EXCEPTION(!disp_y_found,std::runtime_error,"Error could not find y displacement field");
      first_line = false;
    }
    else{
      if(tokens.size()==0)continue;
      TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=num_values_per_line,std::runtime_error,"Invalid line in file");
      for(int_t i=0;i<num_values_per_line;++i){
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[i]),std::runtime_error,"Invalid (non-numeric) line entry in file");
      }
      disp_x.push_back(strtod(tokens[x_disp_index].c_str(),NULL));
      disp_y.push_back(strtod(tokens[y_disp_index].c_str(),NULL));
    }
  } // end text file read
}


int main(int argc, char *argv[]) {
  try{
    DICe::initialize(argc,argv);

    int_t proc_size = 1;
    int_t proc_rank = 0;
#if DICE_MPI
    MPI_Comm_size(MPI_COMM_WORLD,&proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

    Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);

    print_banner();

    // command line options
    bool has_locs_file = false;
    bool exodus_format = false;
    bool use_nonlocal = false;
    Teuchos::ArrayRCP<scalar_t> points_x;
    Teuchos::ArrayRCP<scalar_t> points_y;
    std::vector<scalar_t> imported_coords_x;
    std::vector<scalar_t> imported_coords_y;
    int_t num_imported_points;
    Teuchos::RCP<Point_Cloud<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud<scalar_t>());
    Teuchos::RCP<my_kd_tree_t> kd_tree;
    const int_t num_neigh = 5;
    std::vector<std::vector<int_t> > neighbors;
    std::vector<std::vector<scalar_t> > neighbor_dist_x;
    std::vector<std::vector<scalar_t> > neighbor_dist_y;
    std::string disp_x_name = "NOT_FOUND";
    std::string disp_y_name = "NOT_FOUND";
    int_t num_time_steps = 0;
    Teuchos::RCP<DICe::mesh::Mesh> mesh;
    std::string locations_file = "";
    std::string output_pre = "DICe_strain_solution";
    *outStream << "number of args:     " << argc << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(argc<3,std::runtime_error,
      "Invalid command line. Syntax: dice_strain results_file_1 ... results_file_n strain_window_size_in_pixels [locations_file] [ouput_prefix] [1: for use_nonlocal_formulation]");
    std::vector<std::string> results_files;
    for(int_t i=0;i<argc;++i){
      std::string res_file = argv[i+1];
      if(is_number(argv[i+1])) break;
      DEBUG_MSG("DICe_Strain(): using results file: " << res_file);
      results_files.push_back(res_file);
    }
    const int_t sg_size = std::atoi(argv[1 + results_files.size()]);
    *outStream << "strain gauge size:  " << sg_size << " pixels" << std::endl;
    std::string opt_arg_1 = "";
    std::string opt_arg_2 = "";
    std::string opt_arg_3 = "";
    if(argc>(int_t)results_files.size()+2){
      opt_arg_1 = argv[results_files.size()+2];
      if(argc>(int_t)results_files.size()+3){
        opt_arg_2 = argv[results_files.size()+3];
        if(argc>(int_t)results_files.size()+4){
          opt_arg_3 = argv[results_files.size()+4];
        }
      }
      if(is_number(opt_arg_3)){
        use_nonlocal = true;
      }
      if(is_number(opt_arg_1)){
        use_nonlocal = true;
      }else{
        const std::string text_ext(".txt");
        const std::string csv_ext(".csv");
        if(opt_arg_1.find(text_ext)!=std::string::npos||opt_arg_1.find(csv_ext)!=std::string::npos){
          TEUCHOS_TEST_FOR_EXCEPTION(opt_arg_2.find(text_ext)!=std::string::npos||opt_arg_2.find(csv_ext)!=std::string::npos,
            std::runtime_error,"Error can't have .txt in output_prefix");
          locations_file = opt_arg_1;
          *outStream << "using locs file:    " << locations_file << std::endl;
          has_locs_file = true;
          if(argc>(int_t)results_files.size()+3){
            if(is_number(opt_arg_2)){
              use_nonlocal = true;
            }
            else{
              output_pre = opt_arg_2;
            }
          }
        }
        else{ // its a output prefix
          output_pre = opt_arg_1;
        }
      }
    }
    else{
      *outStream << "using point locations in results file for default strain calculation locations" << std::endl;
      *outStream << "to compute strain at custom locations, specify a locations file as the last arg on the command line" << std::endl;
    }
    *outStream << "using prefix: " << output_pre << std::endl;
    const std::string exodus(".e");
    const std::string txt(".txt");
    if(results_files[0].find(txt)!=std::string::npos){
      DEBUG_MSG("DICe_Strain()::results file is text format");
      for(size_t i=0;i<results_files.size();++i){
        TEUCHOS_TEST_FOR_EXCEPTION(results_files[i].find(txt)==std::string::npos,std::runtime_error,
          "Error, all files must be text input if the first is text format: " << results_files[i]);
      }
    }
    else if(results_files[0].find(exodus)!=std::string::npos){
      exodus_format = true;
      TEUCHOS_TEST_FOR_EXCEPTION(results_files.size()!=1,std::runtime_error,"Error, only one results file may be specified for exodus format");
      DEBUG_MSG("DICe_Strain(): results file is exodus format");
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Unknown results file format: " << results_files[0]);
    }

    // read the results file and interpolate the results to the requested points:
    if(exodus_format){
      // read the exodus file:
      mesh = DICe::mesh::read_exodus_mesh(results_files[0],"dummy_output_name.e");
      num_time_steps = mesh->num_time_steps_imported();
      DEBUG_MSG("DICe_Strain(): number of time steps loaded from results file: " << num_time_steps);
      std::vector<std::string> imported_field_names = mesh->get_imported_fields();
      bool coords_x_found = false;
      bool coords_y_found = false;
      for(size_t i=0;i<imported_field_names.size();++i){
        DEBUG_MSG("DICe_Strain(): found field " << imported_field_names[i] << " in results file");
        if(strcmp(imported_field_names[i].c_str(),"INITIAL_COORDINATES_X")==0)
          coords_x_found = true;
        if(strcmp(imported_field_names[i].c_str(),"INITIAL_COORDINATES_Y")==0)
          coords_y_found = true;
        if(strcmp(imported_field_names[i].c_str(),"SUBSET_DISPLACEMENT_X")==0)
          disp_x_name=imported_field_names[i];
        if(strcmp(imported_field_names[i].c_str(),"SUBSET_DISPLACEMENT_Y")==0)
          disp_y_name=imported_field_names[i];
        if(strcmp(imported_field_names[i].c_str(),"DISPLACEMENT_X")==0)
          disp_x_name=imported_field_names[i];
        if(strcmp(imported_field_names[i].c_str(),"DISPLACEMENT_Y")==0)
          disp_y_name=imported_field_names[i];
      }
      DEBUG_MSG("DICe_Strain(): using field: " << disp_x_name << " as the displacement x field");
      DEBUG_MSG("DICe_Strain(): using field: " << disp_y_name << " as the displacement y field");
      TEUCHOS_TEST_FOR_EXCEPTION(!coords_x_found,std::runtime_error,"Error could not find x coords field");
      TEUCHOS_TEST_FOR_EXCEPTION(!coords_y_found,std::runtime_error,"Error could not find y coords field");
      TEUCHOS_TEST_FOR_EXCEPTION(disp_x_name=="NOT_FOUND",std::runtime_error,"Error could not find x disp field");
      TEUCHOS_TEST_FOR_EXCEPTION(disp_y_name=="NOT_FOUND",std::runtime_error,"Error could not find y disp field");

      // default: use the points from the results file to compute strains
      Teuchos::RCP<MultiField> icx = mesh->get_imported_field("INITIAL_COORDINATES_X",0);
      Teuchos::RCP<MultiField> icy = mesh->get_imported_field("INITIAL_COORDINATES_Y",0);
      num_imported_points = icx->get_map()->get_num_global_elements();
      imported_coords_x.resize(num_imported_points);
      imported_coords_y.resize(num_imported_points);
      for(int_t i=0;i<num_imported_points;++i){
        imported_coords_x[i] = icx->local_value(i);
        imported_coords_y[i] = icy->local_value(i);
      }
    }
    // use text format reader
    else{
      num_time_steps = results_files.size();
      read_text_results_file_coordinates(results_files[0],imported_coords_x,imported_coords_y);
      num_imported_points = imported_coords_x.size();
      DEBUG_MSG("DICe_Strain(): text file reader number of imported points: " << num_imported_points);
      TEUCHOS_TEST_FOR_EXCEPTION(imported_coords_x.size()<=0,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(imported_coords_x.size()!=imported_coords_y.size(),std::runtime_error,"");
    }

    // if custom points are defined in locations_file
    if(has_locs_file){
      DEBUG_MSG("DICe_Strain(): reading custom evaluation points from file: " << locations_file);
      assert(points_x.size()==0);
      assert(points_y.size()==0);
      std::vector<scalar_t> proj_coords_x;
      std::vector<scalar_t> proj_coords_y;
      // read the locations from the text file
      std::fstream dataFile(locations_file.c_str(), std::ios_base::in);
      TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(),std::runtime_error,"Error reading locations file " << locations_file);
      // read each line of the file
      while (!dataFile.eof())
      {
        Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile," \r\n,");
        if(tokens.size()==0)continue;
        TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=2,std::runtime_error,"Invalid line in file");
        TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[0])||!is_number(tokens[1]),std::runtime_error,"Invalid line in file");
        proj_coords_x.push_back(strtod(tokens[0].c_str(),NULL));
        proj_coords_y.push_back(strtod(tokens[1].c_str(),NULL));
      } // end file read
      const int_t num_proj_points = proj_coords_x.size();
      DEBUG_MSG("DICe_Strain(): number of projection points: " << num_proj_points);
      points_x.resize(num_proj_points);
      points_y.resize(num_proj_points);
      for(int_t i=0;i<num_proj_points;++i){
        points_x[i] = proj_coords_x[i];
        points_y[i] = proj_coords_y[i];
      }
      // set up a neighbor list of the closest results file points to the custom requested points
      TEUCHOS_TEST_FOR_EXCEPTION(imported_coords_x.size()<=0,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(imported_coords_y.size()<=0,std::runtime_error,"");
      // create neighborhood lists using nanoflann:
      DEBUG_MSG("DICe_Strain(): creating the point cloud using nanoflann");
      const int_t num_imported_coords = imported_coords_x.size();
      point_cloud->pts.resize(num_imported_coords);
      for(int_t i=0;i<num_imported_coords;++i){
        point_cloud->pts[i].x = imported_coords_x[i];
        point_cloud->pts[i].y = imported_coords_y[i];
        point_cloud->pts[i].z = 0.0;
      }
      DEBUG_MSG("DICe_Strain(): building the kd-tree");
      kd_tree = Teuchos::rcp(new my_kd_tree_t(3 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
      kd_tree->buildIndex();
      DEBUG_MSG("DICe_Strain(): kd-tree completed");
      std::vector<std::pair<size_t,scalar_t> > ret_matches;
      DEBUG_MSG("DICe_Strain(): executing neighbor search");
      neighbors.resize(num_proj_points);
      neighbor_dist_x.resize(num_proj_points);
      neighbor_dist_y.resize(num_proj_points);
      for(int_t i=0;i<num_proj_points;++i){
        neighbors[i].resize(num_neigh);
        neighbor_dist_x[i].resize(num_neigh);
        neighbor_dist_y[i].resize(num_neigh);
      }
      std::vector<size_t> ret_index(num_neigh);
      std::vector<scalar_t> out_dist_sqr(num_neigh);
      std::vector<scalar_t> query_pt(3,0.0);
      for(int_t i=0;i<num_proj_points;++i){
        query_pt[0] = points_x[i];
        query_pt[1] = points_y[i];
        query_pt[2] = 0.0;
        kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
        for(int_t neigh = 0;neigh<num_neigh;++neigh){
          neighbors[i][neigh] = ret_index[neigh];
          neighbor_dist_x[i][neigh] = imported_coords_x[ret_index[neigh]] - points_x[i];
          neighbor_dist_y[i][neigh] = imported_coords_y[ret_index[neigh]] - points_y[i];
        }
      }
      DEBUG_MSG("DICe_Strain(): custom points neighborhood has been initialized");
    } // end has_locs_file
    else{
      points_x.resize(num_imported_points);
      points_y.resize(num_imported_points);
      for(int_t i=0;i<num_imported_points;++i){
        points_x[i] = imported_coords_x[i];
        points_y[i] = imported_coords_y[i];
      }
    }

    DEBUG_MSG("DICe_Strain(): number of points to compute strain " << points_x.size());
    TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()<=0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(points_y.size()<=0,std::runtime_error,"");

    // create a schema to manage the results:
    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set(DICe::output_delimiter,",");
    params->set(DICe::omit_output_row_id,true);
    params->set(DICe::sort_txt_output,true);
    params->set(DICe::output_prefix,output_pre);
    Teuchos::RCP<Teuchos::ParameterList> output_sublist = Teuchos::rcp(new Teuchos::ParameterList());
    output_sublist->set("COORDINATE_X",true);
    output_sublist->set("COORDINATE_Y",true);
    output_sublist->set("DISPLACEMENT_X",true);
    output_sublist->set("DISPLACEMENT_Y",true);
    if(use_nonlocal){
      output_sublist->set("NLVC_STRAIN_XX",true);
      output_sublist->set("NLVC_STRAIN_YY",true);
      output_sublist->set("NLVC_STRAIN_XY",true);
    }
    else{
      output_sublist->set("VSG_STRAIN_XX",true);
      output_sublist->set("VSG_STRAIN_YY",true);
      output_sublist->set("VSG_STRAIN_XY",true);
    }
    output_sublist->set("SIGMA",true);
    output_sublist->set("MATCH",true);
    Teuchos::RCP<Teuchos::ParameterList> post_sublist = Teuchos::rcp(new Teuchos::ParameterList());
    if(use_nonlocal){
      post_sublist->set(DICe::horizon_diameter_in_pixels,sg_size);
      params->set(DICe::post_process_nlvc_strain,*post_sublist);
    }
    else{
      post_sublist->set(DICe::strain_window_size_in_pixels,sg_size);
      params->set(DICe::post_process_vsg_strain,*post_sublist);
    }
    params->set(DICe::output_spec,*output_sublist);

    Teuchos::RCP<Schema> schema = Teuchos::rcp(new Schema(100,100,0.0,params));
    // initialize the schema with the subset coordinates
    schema->set_first_frame_index(0);
    const int_t dummy_subset_size = 25;
    schema->initialize(points_x,points_y,dummy_subset_size);

    for(int_t time_step=0;time_step<num_time_steps;++time_step){
      schema->update_image_frame();

      // fill the displacement values of the schema:
      Teuchos::RCP<MultiField> disp_x = schema->mesh()->get_field(DICe::mesh::field_enums::SUBSET_DISPLACEMENT_X_FS);
      Teuchos::RCP<MultiField> disp_y = schema->mesh()->get_field(DICe::mesh::field_enums::SUBSET_DISPLACEMENT_Y_FS);

      std::vector<scalar_t> imported_disp_x(num_imported_points,0.0);
      std::vector<scalar_t> imported_disp_y(num_imported_points,0.0);

      if(exodus_format){
        Teuchos::RCP<MultiField> idx = mesh->get_imported_field(disp_x_name,time_step);
        Teuchos::RCP<MultiField> idy = mesh->get_imported_field(disp_y_name,time_step);
        if(!has_locs_file){
          DEBUG_MSG("DICe_Strain(): importing the displacement values from exodus results file as copy");
          disp_x->update(1.0,*idx,0.0);
          disp_y->update(1.0,*idy,0.0);
        }else{
          for(int_t i=0;i<num_imported_points;++i){
            imported_disp_x[i] = idx->local_value(i);
            imported_disp_y[i] = idy->local_value(i);
          }
        }
      }
      // text format
      else{
        read_text_results_file_displacements(results_files[time_step],imported_disp_x,imported_disp_y);
        TEUCHOS_TEST_FOR_EXCEPTION((int_t)imported_disp_x.size()!=num_imported_points,std::runtime_error,"");
        TEUCHOS_TEST_FOR_EXCEPTION((int_t)imported_disp_x.size()!=num_imported_points,std::runtime_error,"");
        if(!has_locs_file){
          DEBUG_MSG("DICe_Strain(): importing the displacement values from text results file as copy");
          for(int_t i=0;i<num_imported_points;++i){
            disp_x->local_value(i) = imported_disp_x[i];
            disp_y->local_value(i) = imported_disp_y[i];
          }
        }
      } // end text reader

      if(has_locs_file){
        DEBUG_MSG("DICe_Strain(): projecting the displacement values from the results file to the requested points");
        project_to_points(points_x,points_y,num_neigh,neighbors,
          neighbor_dist_x,neighbor_dist_y,
          imported_disp_x,imported_disp_y,disp_x,disp_y);
      }

      // FIXME: assumes post processor is of length 1 and the first entry is the VSG_STRAIN post processor
      if(time_step==0) (*schema->post_processors())[0]->pre_execution_tasks();
      (*schema->post_processors())[0]->execute();

      schema->write_output("./",output_pre,false,true);

      schema->mesh()->print_field_stats();
    }

    DICe::finalize();
  }
  catch(std::exception & e){
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
