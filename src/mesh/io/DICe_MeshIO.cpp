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

#include <boost/algorithm/string.hpp>
#include <exodusII.h>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

namespace DICe {
namespace mesh {

Teuchos::RCP<Mesh> read_exodus_mesh(const std::string & serial_input_filename,
  const std::string & serial_output_filename)
{
  std::stringstream in_file_base, out_file_base;
  // find the position of the file extension
  // only alow .g or .e
  size_t pos_in = serial_input_filename.find(".g");
  if(pos_in==std::string::npos){
    pos_in = serial_input_filename.find(".e");
  }
  if(pos_in==std::string::npos)
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, input mesh files must have the extension .g or .e");
  size_t pos_out = serial_output_filename.find(".g");
  if(pos_out==std::string::npos){
    pos_out = serial_output_filename.find(".e");
  }
  if(pos_out==std::string::npos)
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, output mesh files must have the extension .g or .e");

  in_file_base << serial_input_filename;
  out_file_base << serial_output_filename;

  int_t num_procs = 1;
  int_t my_rank = 0;
#ifdef HAVE_MPI
  int_t mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  TEUCHOS_TEST_FOR_EXCEPTION(!mpi_is_initialized,std::runtime_error,"Error: if MPI is enabled, MPI must be initialized at this point.");
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
#endif
  if(num_procs>1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, read_exodus_mesh has only been implemented for serial use.");

    std::stringstream temp_ss1, temp_ss2, file_name_post;
    temp_ss1 << num_procs;
    const size_t procs_num_digits = temp_ss1.str().length();
    temp_ss2 << my_rank;
    const size_t rank_num_digits = temp_ss2.str().length();
    const int_t num_fill_zeros = procs_num_digits - rank_num_digits;
    file_name_post << "." << num_procs << ".";
    for(int_t i=0;i<num_fill_zeros;++i)
      file_name_post << "0";
    file_name_post << my_rank;

    in_file_base << file_name_post.str();
    out_file_base << file_name_post.str();
  }

  std::string input_filename = in_file_base.str();
  std::string output_filename = out_file_base.str();

  DEBUG_MSG("Input file name: " << input_filename);
  DEBUG_MSG("Output file name: " << output_filename);

  Teuchos::RCP<DICe::mesh::Mesh> mesh = Teuchos::rcp(new DICe::mesh::Mesh(input_filename,output_filename));
  const int_t p_rank = mesh->get_comm()->get_rank();
  int error_int;
  float version;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;
  /*open exodus II file */
  std::vector<char> writable(input_filename.size() + 1);
  std::copy(input_filename.begin(), input_filename.end(), writable.begin());
  const int input_exoid = ex_open(&writable[0],EX_READ,&CPU_word_size,&IO_word_size,&version);
  if (input_exoid < 0){
    std::stringstream oss;
    oss << "[p=" << p_rank << "] Reading mesh failure: " << input_filename << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  else
    mesh->set_input_exoid(input_exoid);

  /* read database parameters */
  char title[MAX_LINE_LENGTH + 1];
  int_t num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets, num_side_sets;
  error_int  = ex_get_init(input_exoid, title, &num_dim, &num_nodes, &num_elem,
    &num_elem_blk, &num_node_sets, &num_side_sets);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_init(): Failure");
  mesh->set_spatial_dimension(num_dim);

  DEBUG_MSG("  ------------------ Analysis Model Definition (processsor " << p_rank <<") --------------------------");
  DEBUG_MSG("  Title:             " << title);
  DEBUG_MSG("  Input file:        " << input_filename);
  DEBUG_MSG("  Output file:       " << output_filename);
  DEBUG_MSG("  Spatial dimension: " << num_dim);
  DEBUG_MSG("  Nodes:             " << num_nodes);
  DEBUG_MSG("  Elements:          " << num_elem);
  DEBUG_MSG("  Element blocks:    " << num_elem_blk);
  DEBUG_MSG("  Node sets:         " << num_node_sets);
  DEBUG_MSG("  Side sets:         " << num_side_sets);
  DEBUG_MSG("  --------------------------------------------------------------------------------------");

  // Read in element map (index to elem_id)
  // the elements get inserted into the mesh when the connectivity is read
  int_t * elem_map = new int_t[num_elem];
  error_int = ex_get_elem_num_map(input_exoid, elem_map);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_elem_num_map(): Failure");

  // Read in node map (index to node_id)
  int_t * node_map = new int_t[num_nodes];
  error_int = ex_get_node_num_map(input_exoid, node_map);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_node_num_map(): Failure");

  // put all the nodes in the mesh
  for(int_t i =0;i<num_nodes;++i)
  {
    Teuchos::RCP<Node> node_rcp = Teuchos::rcp(new DICe::mesh::Node(node_map[i],i));
    mesh->get_node_set()->insert(std::pair<int_t,Teuchos::RCP<Node> >(node_rcp->global_id(),node_rcp));
  }

  char elem_type_str[MAX_STR_LENGTH + 1];

  /* read element block parameters */
  int_t * block_ids = new int_t[num_elem_blk];
  int_t * num_elem_in_block = new int_t[num_elem_blk];
  int_t * num_nodes_per_elem = new int_t[num_elem_blk];
  int_t * num_attr = new int_t[num_elem_blk];
  error_int = ex_get_elem_blk_ids(input_exoid, block_ids);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_elem_blk_ids(): Failure");

  for (int_t i = 0; i < num_elem_blk; i++)
  {
    // make sure that a material has been defined for each block:
    const int_t block_id = block_ids[i];
    std::stringstream block_ss;
    block_ss << "BLOCK_" << block_id;
    const std::string block_name = block_ss.str();
//    if(analysis_model->block_id_material_map()->find(block_id)==analysis_model->block_id_material_map()->end())
//      log_parallel() << "    WARNING: analysis model: " << analysis_model->name() << " block " << block_name << " does not have a material defined." << std::endl;
    error_int = ex_get_elem_block(input_exoid, block_ids[i], elem_type_str,
      &(num_elem_in_block[i]), &(num_nodes_per_elem[i]),
      &(num_attr[i]));
    TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_elem_block(): Failure");
    boost::to_upper(elem_type_str);
    const Base_Element_Type elem_type = string_to_base_element_type(elem_type_str);
    mesh->get_block_type_map()->insert(std::pair<int_t,Base_Element_Type>(block_ids[i],elem_type));
  }

  /* read element connectivity */
  int_t elem_local_id = 0;
  for (int_t i = 0; i < num_elem_blk; i++)
  {
    int_t elem_index_in_block = 0;
    int_t * connectivity = new int_t[num_nodes_per_elem[i] * num_elem_in_block[i]];
    error_int = ex_get_elem_conn(input_exoid, block_ids[i], connectivity);
    //if(error_int)
    //{
    //  error() << "ex_get_elem_conn(): Failure" << std::endl;
    //  exit(1);
    //}
    for(int_t j=0; j<num_elem_in_block[i]; ++j)
    {
      connectivity_vector conn;
      for(int_t k=0;k<num_nodes_per_elem[i];++k)
      {
        const int_t global_node_id = node_map[connectivity[j*num_nodes_per_elem[i] + k] - 1];
        // find the node in the set with the same global id
        bool found_node = false;
        if(mesh->get_node_set()->find(global_node_id)!=mesh->get_node_set()->end()){
          found_node = true;
          conn.push_back(mesh->get_node_set()->find(global_node_id)->second);
        }
        else{
          TEUCHOS_TEST_FOR_EXCEPTION(!found_node,std::logic_error,"Could not find node rcp in set");
        }
      }
      // create an element and put it in the mesh
      const int_t elem_global_id = elem_map[elem_local_id];
      Teuchos::RCP<DICe::mesh::Element> element_rcp = Teuchos::rcp(new DICe::mesh::Element(conn,elem_global_id,elem_local_id,block_ids[i],elem_index_in_block));
      mesh->get_element_set()->push_back(element_rcp);
      elem_local_id++;
      elem_index_in_block++;
    }
    delete[] connectivity;
  }

  // since we are going to read in some field data like coordinates or volumes, etc
  // we pause here to generate the Tpetra maps that will organize this stuff
  mesh->create_elem_node_field_maps();

  mesh->create_field(field_enums::BLOCK_ID_FS);
  MultiField & block_id_field = *mesh->get_field(field_enums::BLOCK_ID_FS);
  DICe::mesh::element_set::const_iterator elem_it = mesh->get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it){
    block_id_field.local_value(elem_it->get()->local_id()) = elem_it->get()->block_id();
  }

  // put the element blocks in a field so they are accessible in parallel from other processors
  block_type_map::iterator blk_map_it = mesh->get_block_type_map()->begin();
  block_type_map::iterator blk_map_end = mesh->get_block_type_map()->end();
  for(;blk_map_it!=blk_map_end;++blk_map_it)
  {
    // TODO: do the same for faces/edges
    Teuchos::RCP<element_set> elem_set = Teuchos::rcp(new element_set);
//    Teuchos::RCP<node_set> node_set_ptr = Teuchos::rcp(new node_set);
//    node_set & node_set_ref = *node_set_ptr;
    for(elem_it=mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
    {
      if(elem_it->get()->block_id()==blk_map_it->first)
      {
        elem_set->push_back(*elem_it);
//        const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
//        for(size_t node_it=0;node_it<connectivity.size();++node_it)
//        {
//          // check if the node is already there:
//          bool not_found = true;
//          for(size_t i=0;i<node_set_ptr->size();++i)
//          {
//            if(node_set_ref[i]==connectivity[node_it]) not_found = false;
//          }
//          if(not_found) node_set_ptr->push_back(connectivity[node_it]);
//        }
      }
    }
    mesh->get_element_sets_by_block()->insert(std::pair<int_t,Teuchos::RCP<element_set> >(blk_map_it->first,elem_set));
//    mesh->get_node_sets_by_block()->insert(std::pair<int_t,Teuchos::RCP<node_set> >(blk_map_it->first,node_set_ptr));
  }

//#ifdef DICE_DEBUG_MSG
//    std::cout << " THESE ARE THE SIZES (nodes):" << std::endl;
//    std::map<int_t,Teuchos::RCP<node_set> >::iterator map_it = mesh->get_node_sets_by_block()->begin();
//    std::map<int_t,Teuchos::RCP<node_set> >::iterator map_end = mesh->get_node_sets_by_block()->end();
//    for(;map_it!=map_end;++map_it)
//    {
//      std::cout << "Block : " << map_it->first << std::endl;
//      node_set & node_set = *map_it->second;
//      for(int_t i=0;i<map_it->second.get()->size();++i)
//        std::cout << node_set[i]->global_id() << std::endl;
//    }
//  std::cout << " THESE ARE THE SIZES (elems):" << std::endl;
//  std::map<int_t,Teuchos::RCP<element_set> >::iterator elem_map_it = mesh->get_element_sets_by_block()->begin();
//  std::map<int_t,Teuchos::RCP<element_set> >::iterator elem_map_end = mesh->get_element_sets_by_block()->end();
//  for(;elem_map_it!=elem_map_end;++elem_map_it)
//  {
//    std::cout << "Block : " << elem_map_it->first << std::endl;
//    element_set & elem_set = *elem_map_it->second;
//    for(int_t i=0;i<elem_map_it->second.get()->size();++i)
//      std::cout << elem_set[i]->global_id() << std::endl;
//  }
//#endif
  /* read element attributes */
  // FOR NOW WE BYPASS THIS SINCE WE WILL CALC THE RADIUS AND VOLUME FROM FEM MESH

  /* import node sets */
  //log_parallel() << "  Warning: Exodus distribution factors for side sets and node sets are ignored and not written to output " << std::endl;

  int_t * node_set_ids = new int_t[num_node_sets];
  int_t * num_nodes_in_node_set = new int_t[num_node_sets];
  int_t * num_df_in_node_set = new int_t[num_node_sets];
  error_int = ex_get_node_set_ids(input_exoid, node_set_ids);
  //if(error_int)
  //{
  //  error() << "ex_get_node_set_ids(): Failure" << std::endl;
  //  exit(1);
  //}

  for (int_t i = 0; i < num_node_sets; i++)
  {
    //cout << " READING NODE SET: " << i << std::endl;
    error_int = ex_get_node_set_param(input_exoid, node_set_ids[i],&(num_nodes_in_node_set[i]), &(num_df_in_node_set[i]));
    TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_node_set_params(): Failure");

    TEUCHOS_TEST_FOR_EXCEPTION(num_nodes_in_node_set[i]!=num_df_in_node_set[i]&&num_df_in_node_set[i]!=0,
      std::logic_error, "read_exodus_mesh(): number of nodes in nodeset doesn't match the number of dist factors");
    int_t * ns_node_list = new int_t[num_nodes_in_node_set[i]];
    float * dist_fact = new float[num_nodes_in_node_set[i]];
    error_int = ex_get_node_set(input_exoid, node_set_ids[i], ns_node_list);
    //if(error_int) //TODO: better checking here, there may not be a node_set and that would make ex_get return -1
    //{
    //  error() << "ex_get_node_set(): Failure" << std::endl;
    //  exit(1);
    //}

    if (num_df_in_node_set[i] > 0)
    {
      error_int = ex_get_node_set_dist_fact(input_exoid, node_set_ids[i],dist_fact);
      TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_node_set_dist_fact(): Failure");
    }
    std::vector<int_t> bc_def;
    for(int_t it=0;it<num_nodes_in_node_set[i];++it)
    {
      //cout << ns_node_list[it] << std::endl;
      //cout << node_map[ns_node_list[it]-1] << std::endl;
      bc_def.push_back((int_t)node_map[ns_node_list[it]-1]);
    }
    mesh->get_node_bc_sets()->insert(std::pair<int_t,std::vector<int_t> >(node_set_ids[i],bc_def));
    delete[] ns_node_list;
    delete[] dist_fact;
  }

  /*  import_side_sets */
  /* concatenated side set read */
  int_t num_ss, elem_list_len, df_list_len, *ids, *ss_side_list,
  *num_side_per_set, *num_df_per_set, *elem_ind, *df_ind, *ss_elem_list;
  char * cdum; cdum=0;
  float fdum, * dist_fact;

  error_int = ex_inquire (input_exoid, EX_INQ_SIDE_SETS, &num_ss, &fdum, cdum);
  if (num_ss > 0) {
    error_int = ex_inquire(input_exoid, EX_INQ_SS_ELEM_LEN, &elem_list_len, &fdum, cdum);
    error_int = ex_inquire(input_exoid, EX_INQ_SS_DF_LEN, &df_list_len, &fdum, cdum);
    TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_inquire(): Failure");

    ids = new int_t[num_side_sets];
    num_side_per_set = new int_t[num_side_sets];
    num_df_per_set = new int_t[num_side_sets];
    elem_ind = new int_t[num_side_sets];
    df_ind = new int_t[num_side_sets];
    ss_elem_list = new int_t[elem_list_len];
    ss_side_list = new int_t[elem_list_len];
    dist_fact = new float[df_list_len];  // has to be float instead of scalar_t due to exodus function call

    error_int = ex_get_concat_side_sets (input_exoid, ids, num_side_per_set, num_df_per_set, elem_ind, df_ind, ss_elem_list, ss_side_list, dist_fact);
    TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_concat_side_sets(): Failure");

    side_set_info & ss_info = *mesh->get_side_set_info();
    ss_info.ids.resize(num_side_sets);
    ss_info.df_ind.resize(num_side_sets);
    ss_info.dist_factors.resize(df_list_len);
    ss_info.elem_ind.resize(num_side_sets);
    ss_info.num_df_per_set.resize(num_side_sets);
    ss_info.num_side_per_set.resize(num_side_sets);
    ss_info.ss_local_elem_list.resize(elem_list_len);
    ss_info.ss_global_elem_list.resize(elem_list_len);
    ss_info.ss_side_list.resize(elem_list_len);
    ss_info.normal_x.resize(elem_list_len);
    ss_info.normal_y.resize(elem_list_len);
    ss_info.normal_z.resize(elem_list_len);
    ss_info.side_size.resize(elem_list_len);
    ss_info.normals_are_populated = false;

    for(int_t k=0;k<num_side_sets;++k)
    {
      ss_info.ids[k] = ids[k];
      //std::cout << "ids[" << k << "]: " << ids[k] << std::endl;
      ss_info.num_side_per_set[k] = num_side_per_set[k];
      //std::cout << "num_sides_per_set[" << k << "]: " << num_side_per_set[k] << std::endl;
      ss_info.num_df_per_set[k] = num_df_per_set[k];
      //std::cout << "num_df_per_set[" << k << "]: " << num_df_per_set[k] << std::endl;
      ss_info.elem_ind[k] = elem_ind[k];
      //std::cout << "elem_ind[" << k << "]: " << elem_ind[k] << std::endl;
      ss_info.df_ind[k] = df_ind[k];
      //std::cout << "df_ind[" << k << "]: " << df_ind[k] << std::endl;
    }
    for(int_t k=0;k<elem_list_len;++k)
    {
      ss_info.ss_local_elem_list[k] = ss_elem_list[k];
      ss_info.ss_global_elem_list[k] = elem_map[ss_elem_list[k] - 1];
      //std::cout << " ss_elem_list[" << k << "]: " << ss_elem_list[k] << std::endl;
      ss_info.ss_side_list[k] = ss_side_list[k];
      //std::cout << " ss_side_list[" << k << "]: " << ss_side_list[k] << std::endl;
    }
    for(int_t k=0;k<df_list_len;++k)
    {
      ss_info.dist_factors[k] = dist_fact[k];
      //std::cout << " dist_fact[" << k << "]: " << dist_fact[k] << std::endl;
    }
    delete[] ids;
    delete[] num_side_per_set;
    delete[] num_df_per_set;
    delete[] df_ind;
    delete[] elem_ind;
    delete[] ss_elem_list;
    delete[] ss_side_list;
    delete[] dist_fact;

  }
  delete[] node_map;
  delete[] elem_map;
  delete[] node_set_ids;
  delete[] num_nodes_in_node_set;
  delete[] num_df_in_node_set;

  error_int = ex_close(input_exoid);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_close(): Failure");
  mesh->set_initialized();
  // memory clean up
  // This is needed because the exodus calls don't work with smart pointers
  delete[] block_ids;
  delete[] num_elem_in_block;
  delete[] num_nodes_per_elem;
  delete[] num_attr;

  mesh->create_face_cell_field_maps();

  // read the coordinates from the mesh file:
  DICe::mesh::read_exodus_coordinates(mesh);

  return mesh;
}

int_t
read_exodus_num_steps(Teuchos::RCP<Mesh> mesh){
  return read_exodus_num_steps(mesh->get_input_filename());
}

int_t
read_exodus_num_steps(const std::string & file_name){
  int_t num_steps = 0;
  float version;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;
  /*open exodus II file */
  std::vector<char> writable(file_name.size() + 1);
  std::copy(file_name.begin(), file_name.end(), writable.begin());
  //const char * file_name = analysis_model->input_file_name().c_str();
  const int input_exoid = ex_open(&writable[0],EX_READ,&CPU_word_size,&IO_word_size,&version);
  if (input_exoid < 0)
  {
    std::stringstream oss;
    oss << "Reading mesh failure: " << file_name;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  float ret_float;
  char ret_char;
  ex_inquire(input_exoid,EX_INQ_TIME,&num_steps,&ret_float,&ret_char);
  return num_steps;
}


std::vector<scalar_t>
read_exodus_field(const std::string & file_name,
  const std::string & field_name,
  const int_t step){

  std::vector<std::string> field_names = DICe::mesh::read_exodus_field_names(file_name);
  int_t var_index = -1;
  for(size_t i=0;i<field_names.size();++i){
    if(field_names[i]==field_name){
      var_index = i+1;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(var_index<=0,std::runtime_error,"Error, field not found in mesh: " << field_name);
  return read_exodus_field(file_name,var_index,step);
}

std::vector<scalar_t>
read_exodus_field(Teuchos::RCP<Mesh> mesh,
  const int_t var_index,
  const int_t step){
  return read_exodus_field(mesh->get_input_filename(),var_index,step);
}

std::vector<scalar_t>
read_exodus_field(const std::string & file_name,
  const int_t var_index,
  const int_t step){
  TEUCHOS_TEST_FOR_EXCEPTION(step==0,std::runtime_error,"Invalid step (<=0): " << step);
  const int_t num_steps = read_exodus_num_steps(file_name);
  TEUCHOS_TEST_FOR_EXCEPTION(step>num_steps,std::runtime_error,"Invalid step (>num_steps): " << step);
  float version;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;
  /*open exodus II file */
  std::vector<char> writable(file_name.size() + 1);
  std::copy(file_name.begin(), file_name.end(), writable.begin());
  //const char * file_name = analysis_model->input_file_name().c_str();
  const int input_exoid = ex_open(&writable[0],EX_READ,&CPU_word_size,&IO_word_size,&version);
  if (input_exoid < 0)
  {
    std::stringstream oss;
    oss << "Reading mesh failure: " << file_name;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  // get the number of nodes
  float ret_float = 0.0;
  char ret_char;
  int num_nodes = 0;
  ex_inquire(input_exoid, EX_INQ_NODES, &num_nodes, &ret_float, &ret_char);
  DEBUG_MSG("read_exodus_field(): number of nodes " << num_nodes);

  float ex_result[num_nodes];
  std::vector<scalar_t> result(num_nodes,0.0);
  ex_get_nodal_var(input_exoid,step,var_index,result.size(),&ex_result[0]);
  for(int_t i=0;i<num_nodes;++i){
    result[i] = ex_result[i];
  }
  return result;
}

std::vector<std::string>
read_exodus_field_names(Teuchos::RCP<Mesh> mesh){
  return read_exodus_field_names(mesh->get_input_filename());
}

std::vector<std::string>
read_exodus_field_names(const std::string & file_name){
  float version;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;

  std::vector<std::string> field_names;

  std::vector<char> writable(file_name.size() + 1);
  std::copy(file_name.begin(), file_name.end(), writable.begin());

  //const char * file_name = analysis_model->input_file_name().c_str();
  const int input_exoid = ex_open(&writable[0],EX_READ,&CPU_word_size,&IO_word_size,&version);
  if (input_exoid < 0)
  {
    std::stringstream oss;
    oss << "Reading mesh failure: " << file_name;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  // TODO read element variables
  // nodal variables
  int_t num_node_vars = 0;
  ex_get_var_param(input_exoid,"n",&num_node_vars);
  DEBUG_MSG("read_exodus_field_names(): number of nodal fields: " << num_node_vars);
  for(int_t i=1;i<=num_node_vars;++i){ // exodus is one based
    char nodal_var_name[100];
    ex_get_var_name(input_exoid,"n",i,&nodal_var_name[0]);
    std::string var_name = nodal_var_name;
    field_names.push_back(var_name);
  }
  return field_names;
}

void
read_exodus_coordinates(Teuchos::RCP<Mesh> mesh){
  int error_int;
  float version;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;

  /*open exodus II file */
  std::string file_name_str = mesh->get_input_filename();
  const int_t spa_dim = mesh->spatial_dimension();
  std::vector<char> writable(file_name_str.size() + 1);
  std::copy(file_name_str.begin(), file_name_str.end(), writable.begin());

  //const char * file_name = analysis_model->input_file_name().c_str();
  const int input_exoid = ex_open(&writable[0],EX_READ,&CPU_word_size,&IO_word_size,&version);
  if (input_exoid < 0)
  {
    std::stringstream oss;
    oss << "Reading mesh failure: " << file_name_str;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }

  // initialize the fields needed for coordinates etc:
  mesh->create_field(field_enums::INITIAL_COORDINATES_FS);

  const int_t p_rank = mesh->get_comm()->get_rank();

  /* read database parameters */
  char title[MAX_LINE_LENGTH + 1];
  int_t num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets, num_side_sets;
  error_int  = ex_get_init(input_exoid, title, &num_dim, &num_nodes, &num_elem,
    &num_elem_blk, &num_node_sets, &num_side_sets);
  TEUCHOS_TEST_FOR_EXCEPTION(num_dim!=spa_dim,std::runtime_error,"");

  Teuchos::RCP<MultiField > initial_coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);

  /* read nodal coordinates values and names from database */
  float * temp_x = new float[num_nodes];
  float * temp_y = new float[num_nodes];
  float * temp_z = (num_dim >= 3) ? new float[num_nodes] : 0;
  error_int = ex_get_coord(input_exoid, temp_x, temp_y, temp_z);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_get_coord(): Failure");

  // put coords in a tpetra vector
  for(int_t i=0;i<num_nodes;++i)  // i represents the local_id
  {
    initial_coords->local_value(i*spa_dim+0) = temp_x[i];
    initial_coords->local_value(i*spa_dim+1) = temp_y[i];
    if (num_dim >= 3)
    {
      initial_coords->local_value(i*spa_dim+2) = temp_z[i];
    }
  }
  delete[] temp_x;
  delete[] temp_y;
  if(num_dim>2)
    delete[] temp_z;

  // export the coordinates back to the non-overlap field
  mesh->field_overlap_export(initial_coords, field_enums::INITIAL_COORDINATES_FS, INSERT);
  //std::cout << "INITIAL COORDS: " << std::endl;
  //initial_coords_ptr->vec()->describe();

  // CELL COORDINATES

  mesh->create_field(field_enums::INITIAL_CELL_COORDINATES_FS);

  Teuchos::RCP<MultiField > coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);
  MultiField & initial_cell_coords = *mesh->get_field(field_enums::INITIAL_CELL_COORDINATES_FS);

  //compute the centroid from the coordinates of the nodes;
  DICe::mesh::element_set::const_iterator elem_it = mesh->get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    scalar_t centroid[num_dim];
    for(int_t i=0;i<num_dim;++i) centroid[i]=0.0;
    for(size_t node_it=0;node_it<connectivity.size();++node_it)
    {
      const Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
      centroid[0] += coords->local_value(node.get()->overlap_local_id()*spa_dim+0);
      centroid[1] += coords->local_value(node.get()->overlap_local_id()*spa_dim+1);
    }
    for(int_t dim=0;dim<num_dim;++dim)
    {
      centroid[dim] /= connectivity.size();
      initial_cell_coords.local_value(elem_it->get()->local_id()*spa_dim+dim) = centroid[dim];
    }
  }
  //std::cout << " INITIAL CELL COORDS: " << std::endl;
  //initial_cell_coords.vec()->describe();

  // PROCESSOR ID

  // create the cell_coords, cell_radius and cell_size fields:
  mesh->create_field(field_enums::PROCESSOR_ID_FS);
  MultiField & proc_id = *mesh->get_field(field_enums::PROCESSOR_ID_FS);
  elem_it = mesh->get_element_set()->begin();
  for(;elem_it!=elem_end;++elem_it){
    proc_id.local_value(elem_it->get()->local_id()) = p_rank;
  }
}

void create_output_exodus_file(Teuchos::RCP<Mesh> mesh,
  const std::string & output_folder){

  std::stringstream out_file;
  out_file << output_folder << mesh->get_output_filename();
  const std::string filename = out_file.str();
  std::vector<char> writable(filename.size() + 1);
  std::copy(filename.begin(), filename.end(), writable.begin());
  int_t spa_dim = mesh->spatial_dimension();
  int_t init_spa_dim = spa_dim;
  int_t num_nodes = mesh->num_nodes();
  int_t num_elem = mesh->num_elem();
  int error_int;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;
  /* create EXODUS II file */
  const int output_exoid = ex_create (&writable[0],EX_CLOBBER,&CPU_word_size, &IO_word_size);
  mesh->set_output_exoid(output_exoid);

  // check if model coordinates is a valid field
  bool use_model_coordinates = false;
  Teuchos::RCP<MultiField> model_x;
  Teuchos::RCP<MultiField> model_y;
  Teuchos::RCP<MultiField> model_z;
  try{
    mesh->get_field(field_enums::MODEL_COORDINATES_X_FS);
    use_model_coordinates = true;
  }catch(std::exception & e){
    use_model_coordinates = false;
  }
  if(use_model_coordinates){
    model_x = mesh->get_overlap_field(field_enums::MODEL_COORDINATES_X_FS);
    model_y = mesh->get_overlap_field(field_enums::MODEL_COORDINATES_Y_FS);
    model_z = mesh->get_overlap_field(field_enums::MODEL_COORDINATES_Z_FS);
    // also check that the field has values
    use_model_coordinates = mesh->get_field(field_enums::MODEL_COORDINATES_X_FS)->norm() > 1.0E-8;
  }
  if(use_model_coordinates){
    init_spa_dim = 3;
  }
  DEBUG_MSG("output_exoid: " << output_exoid << " file: " << out_file.str() << " spatial_dimension: " << spa_dim << " num_nodes: " << num_nodes << " num_elem: " << num_elem
     << " num_blocks: " << mesh->num_blocks() << " node_sets: "<< mesh->num_node_sets() << " side_sets: "<< mesh->get_side_set_info()->ids.size());

  error_int = ex_put_init(output_exoid, &writable[0], init_spa_dim, num_nodes,
    num_elem, mesh->num_blocks(), mesh->num_node_sets(),  mesh->get_side_set_info()->ids.size());
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error, "ex_put_init(): Failure");

  //  write initial coordinates and node/element maps

  float x[num_nodes];
  float y[num_nodes];
  float z[num_nodes];

  Teuchos::RCP<MultiField > coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);

  int_t * elem_map = new int_t[num_elem];
  int_t * node_map = new int_t[num_nodes];

  DICe::mesh::node_set::const_iterator node_it = mesh->get_node_set()->begin();
  DICe::mesh::node_set::const_iterator node_end = mesh->get_node_set()->end();
  for(;node_it!=node_end;++node_it)
  {
    const int_t local_id = node_it->second->overlap_local_id();
    if(use_model_coordinates){
      x[local_id] = model_x->local_value(local_id);
      y[local_id] = model_y->local_value(local_id);
      z[local_id] = model_z->local_value(local_id);
    }
    else{
      x[local_id] = coords->local_value(local_id*spa_dim+0);
      y[local_id] = coords->local_value(local_id*spa_dim+1);
      if(spa_dim > 2)
        z[local_id] = coords->local_value(local_id*spa_dim+2);
      else
        z[local_id] = 0.0;
    }
    node_map[local_id]=node_it->first;
  }
  DICe::mesh::element_set::const_iterator elem_it = mesh->get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    const int_t local_id = elem_it->get()->local_id();
    elem_map[local_id]=elem_it->get()->global_id() + 1;
  }

  error_int = ex_put_coord(output_exoid, x, y, z);
  char * coord_names[3];
  if(use_model_coordinates){
    coord_names[0] = (char*) "MODEL_COORDINATES_X";
    coord_names[1] = (char*) "MODEL_COORDINATES_Y";
    coord_names[2] = (char*) "MODEL_COORDINATES_Z";
  }
  else{
    coord_names[0] = (char*) "INITIAL_COORDINATES_X";
    coord_names[1] = (char*) "INITIAL_COORDINATES_Y";
    coord_names[2] = (char*) "INITIAL_COORDINATES_Z";
  }
  error_int = ex_put_coord_names(output_exoid, coord_names);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_coord_names(): Failure");
  error_int = ex_put_elem_num_map(output_exoid, elem_map);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_elem_num_map(): Failure");
  error_int = ex_put_node_num_map(output_exoid, node_map);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_node_num_map(): Failure");
  delete[] elem_map;


  //  write elem blocks
  DICe::mesh::block_type_map::iterator block_map_it = mesh->get_block_type_map()->begin();
  DICe::mesh::block_type_map::iterator block_map_end = mesh->get_block_type_map()->end();
  for(;block_map_it!=block_map_end;++block_map_it)
  {
    const Base_Element_Type elem_type_enum = block_map_it->second;
    std::string elem_type_str = tostring(elem_type_enum);
    char * elem_type = const_cast<char *>(elem_type_str.c_str());
    int_t num_nodes_per_elem = 0;
    if(elem_type_enum == HEX8)
      num_nodes_per_elem = 8;
    else if(elem_type_enum == MESHLESS){
      elem_type_str = "SPHERE"; // use sphere type for paraview
      elem_type = const_cast<char *>(elem_type_str.c_str());
      num_nodes_per_elem = 1;
    }
    else if(elem_type_enum == TETRA4 || elem_type_enum == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
      num_nodes_per_elem = 4;
    else if(elem_type_enum == PYRAMID5)
      num_nodes_per_elem = 5;
    else if(elem_type_enum == QUAD4)
      num_nodes_per_elem = 4;
    else if(elem_type_enum == TRI3)
      num_nodes_per_elem = 3;
    else if(elem_type_enum == TRI6)
      num_nodes_per_elem = 6;
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "invalid element type");
    }
    const int_t num_elem_in_block = mesh->num_elem_in_block(block_map_it->first);
    error_int = ex_put_elem_block(output_exoid, block_map_it->first, elem_type,
      num_elem_in_block, num_nodes_per_elem, 0);
    TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error, "ex_put_elem_block(): Failure");
  }

  //  write elem connectivities
  //  Heads up: the connectivities will not write to file until ex_close is called

  for(block_map_it = mesh->get_block_type_map()->begin();block_map_it!=block_map_end;++block_map_it)
  {
    const Base_Element_Type elem_type_enum = block_map_it->second;
    int_t num_nodes_per_elem = 0;
    if(elem_type_enum == HEX8)
      num_nodes_per_elem = 8;
    else if(elem_type_enum == MESHLESS)
      num_nodes_per_elem = 1;
    else if(elem_type_enum == TETRA4 || elem_type_enum == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
      num_nodes_per_elem = 4;
    else if(elem_type_enum == PYRAMID5)
      num_nodes_per_elem = 5;
    else if(elem_type_enum == QUAD4)
      num_nodes_per_elem = 4;
    else if(elem_type_enum == TRI3)
      num_nodes_per_elem = 3;
    else if(elem_type_enum == TRI6)
      num_nodes_per_elem = 6;
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "invalid element type");
    }

    const int_t num_elem_in_block = mesh->num_elem_in_block(block_map_it->first);
    int_t conn_index = 0;
    int_t * block_connect = new int_t[num_elem_in_block * num_nodes_per_elem];
    for(elem_it = mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
    {
      // filter the elements that belong to this block
      if(elem_it->get()->block_id()==block_map_it->first)
      {
        const connectivity_vector & connectivity = *elem_it->get()->connectivity();
        for(size_t node_it = 0;node_it<connectivity.size();++node_it)
        {
          const int_t stride = conn_index * num_nodes_per_elem + node_it;
          const Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
          const int_t local_node_id = node.get()->overlap_local_id();
          block_connect[stride] = local_node_id + 1;  // connectivity is !ALWAYS! 1 based in exodus file
        }
        conn_index++;
      }
    }
  //  for(int_t i=0;i<num_elem_in_block*num_nodes_per_elem; ++i)
  //    cout << " block_connect["<< i << "]" << block_connect[i] << std::endl;
    error_int = ex_put_elem_conn(output_exoid, block_map_it->first, block_connect);
//    if(error_int)
//    {
//      error() << "ex_put_elem_conn(): Failure" << std::endl;
//      exit(1);
//    }
    delete[] block_connect;
  }

  //  write_elem_attributes()
  //  WE DON'T WRITE ATTRIBUTES SINCE WE COMPUTE (VOLUME, RADIUS, ETC) FROM THE COORDINATES

  //  write node sets

  bc_set * node_sets = mesh->get_node_bc_sets();
  bc_set::iterator node_set_it = node_sets->begin();
  bc_set::iterator node_set_end = node_sets->end();
  for(;node_set_it!=node_set_end;++node_set_it)
  {
    //cout << " OUTPUTTING SET NUMBER " << set_number++ << std::endl;
    int_t * node_list = new int_t[node_set_it->second.size()];
    int_t * dist_facts = new int_t[node_set_it->second.size()];
    for(size_t i=0;i<node_set_it->second.size();++i)
    {
      for(int_t mapit=0;mapit<num_nodes;++mapit)
      {
        if(node_map[mapit]==node_set_it->second[i])
          node_list[i] = mapit + 1;
      }
      //node_list[i] =
      //node_list[i] = mesh->node_global_to_local_id(node_set_it->second[i]) + 1; //ids are 1 based in exodus
      dist_facts[i] = 0.0; // no dist factors stored or written to output
    }
    //for(int_t i=0;i<node_set_it->second.size();++i)
    //  cout << " node list: " << node_list[i] << std::endl;
    error_int = ex_put_node_set_param(output_exoid, node_set_it->first,
      node_set_it->second.size(), node_set_it->second.size());
    error_int = ex_put_node_set(output_exoid, node_set_it->first,node_list);
    error_int = ex_put_node_set_dist_fact(output_exoid, node_set_it->first,dist_facts);

    delete[] node_list;
  }
  delete[] node_map;

  //  write side sets
  int_t *ids, *ss_side_list,
    *num_side_per_set, *num_df_per_set, *elem_ind, *df_ind, *ss_elem_list;
  float *dist_fact;

  side_set_info & ss_info = *mesh->get_side_set_info();
  int_t num_side_sets = ss_info.ids.size();
  int_t elem_list_len = ss_info.ss_local_elem_list.size();
  int_t df_list_len = ss_info.dist_factors.size();

  ids = new int_t[num_side_sets];
  num_side_per_set = new int_t[num_side_sets];
  num_df_per_set = new int_t[num_side_sets];
  elem_ind = new int_t[num_side_sets];
  df_ind = new int_t[num_side_sets];
  ss_elem_list = new int_t[elem_list_len];
  ss_side_list = new int_t[elem_list_len];
  dist_fact = new float[df_list_len];  // has to be float instead of scalar_t due to exodus function call

  for(int_t k=0;k<num_side_sets;++k)
  {
    ids[k] = ss_info.ids[k];
    //cout << "ids[" << k << "]: " << ids[k] << std::endl;
    num_side_per_set[k] = ss_info.num_side_per_set[k];
    //cout << "num_sides_per_set[" << k << "]: " << num_side_per_set[k] << std::endl;
    num_df_per_set[k] = ss_info.num_df_per_set[k];
    //cout << "num_df_per_set[" << k << "]: " << num_df_per_set[k] << std::endl;
    elem_ind[k] = ss_info.elem_ind[k];
    //cout << "elem_ind[" << k << "]: " << elem_ind[k] << std::endl;
    df_ind[k] = ss_info.df_ind[k];
    //cout << "df_ind[" << k << "]: " << df_ind[k] << std::endl;
  }
  for(int_t k=0;k<elem_list_len;++k)
  {
    ss_elem_list[k] = ss_info.ss_local_elem_list[k];
    //cout << " ss_elem_list[" << k << "]: " << ss_elem_list[k] << std::endl;
    ss_side_list[k] = ss_info.ss_side_list[k];
    //cout << " ss_side_list[" << k << "]: " << ss_side_list[k] << std::endl;
  }
  for(int_t k=0;k<df_list_len;++k)
  {
    dist_fact[k] = ss_info.dist_factors[k];
    //cout << " dist_fact[" << k << "]: " << dist_fact[k] << std::endl;
  }
// put the values in the mesh
  if(num_side_sets>0){
    error_int = ex_put_concat_side_sets (output_exoid,&ids[0],&num_side_per_set[0],&num_df_per_set[0],&elem_ind[0],&df_ind[0],&ss_elem_list[0],&ss_side_list[0],&dist_fact[0]);
    TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error, "ex_put_concat_side_sets(): Failure" << error_int);
  }

  delete[] ids;
  delete[] num_side_per_set;
  delete[] num_df_per_set;
  delete[] df_ind;
  delete[] elem_ind;
  delete[] ss_elem_list;
  delete[] ss_side_list;
  delete[] dist_fact;
}

void
exodus_output_dump(Teuchos::RCP<Mesh> mesh,
  const int_t & time_step_num,
  const float & time_value)
{
  int error_int = 0;
  error_int = ex_put_time(mesh->get_output_exoid(), time_step_num, &time_value);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_time(): Failure");

  // write fields
  DICe::mesh::field_registry::iterator field_it = mesh->get_field_registry()->begin();
  DICe::mesh::field_registry::iterator field_end = mesh->get_field_registry()->end();
  for(;field_it!=field_end;++field_it)
  {
    if(!field_it->first.is_printable()) continue;
    if(!field_it->first.get_rank()==field_enums::ELEMENT_RANK && !field_it->first.get_rank()==field_enums::NODE_RANK) continue;
    const int_t num_comps = (field_it->first.get_field_type()==field_enums::VECTOR_FIELD_TYPE) ? mesh->spatial_dimension(): 1;

    if(field_it->first.get_rank()==field_enums::ELEMENT_RANK)
    {
      MultiField & field = *mesh->get_field(field_it->first);
      //Teuchos::ArrayRCP<const scalar_t> field_values = field.vec()->get_1d_view();
      DICe::mesh::block_type_map::iterator block_type_map_end = mesh->get_block_type_map()->end();
      std::string components[3];
      components[0] = (field_it->first.get_field_type()==field_enums::VECTOR_FIELD_TYPE) ? "X" : "";
      components[1] = "Y";
      components[2] = "Z";
      for (int_t comp = 0; comp < num_comps; ++comp)
      {
        for(DICe::mesh::block_type_map::iterator block_type_map_it = mesh->get_block_type_map()->begin();
            block_type_map_it!=block_type_map_end;++block_type_map_it)
        {
          const int_t num_elements = mesh->num_elem_in_block(block_type_map_it->first);
          if(num_elements==0) continue;
          float values[num_elements];
          DICe::mesh::element_set::const_iterator elem_it = mesh->get_element_set()->begin();
          DICe::mesh::element_set::const_iterator elem_end = mesh->get_element_set()->end();
          for(;elem_it!=elem_end;++elem_it)
          {
            if(elem_it->get()->block_id()!=block_type_map_it->first)continue; // filter out the elements not from this block
            values[elem_it->get()->index_in_block()] = field.local_value(elem_it->get()->local_id()*num_comps+comp);  // this may not work since the elements are in a set rather than a vector, may have to re-order them
          }
          const int_t var_index = get_var_index(mesh, tostring(field_it->first.get_name()), components[comp], field_it->first.get_rank());
          error_int = ex_put_elem_var(mesh->get_output_exoid(), time_step_num, var_index, block_type_map_it->first, mesh->num_elem_in_block(block_type_map_it->first),values);
          TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"Failure ex_put_elem_var(): " + tostring(field_it->first.get_name()));
        }
      }
    }
    else if(field_it->first.get_rank()==field_enums::NODE_RANK)
    {
      Teuchos::RCP<MultiField > field = mesh->get_overlap_field(field_it->first);
      float values[mesh->num_nodes()];
      std::string components[3];
      components[0] = (field_it->first.get_field_type()==field_enums::VECTOR_FIELD_TYPE) ? "X" : "";
      components[1] = "Y";
      components[2] = "Z";
      for (int_t comp = 0; comp < num_comps; ++comp)
      {
        DICe::mesh::node_set::const_iterator node_it = mesh->get_node_set()->begin();
        DICe::mesh::node_set::const_iterator node_end = mesh->get_node_set()->end();
        for(;node_it!=node_end;++node_it)
        {
          values[node_it->second->overlap_local_id()] = field->local_value(node_it->second->overlap_local_id()*num_comps+comp);
        }
        const int_t var_index = get_var_index(mesh, tostring(field_it->first.get_name()), components[comp], field_it->first.get_rank());
        error_int = ex_put_nodal_var(mesh->get_output_exoid(), time_step_num, var_index,mesh->num_nodes(), values);
      }
    }
//    else
//    {
//      std::stringstream oss;
//      oss << "exodus_output_dump(): unrecognized field rank or output functions have not been implemented for: " << tostring(field_it->first.get_rank()) << std::endl;
//      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
//    }
  }
  error_int = ex_update(mesh->get_output_exoid());
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_update(): Failure");
}


void
exodus_face_edge_output_dump(Teuchos::RCP<Mesh> mesh,
  const int_t & time_step_num,
  const float & time_value)
{
  const int_t spa_dim = mesh->spatial_dimension();
  int error_int = 0;
  error_int = ex_put_time(mesh->get_face_edge_output_exoid(), time_step_num, &time_value);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error, "ex_put_time(): Failure");

  const int_t num_values = mesh->num_internal_faces(); // + mesh->num_boundary_faces();

  // write fields
  DICe::mesh::field_registry::iterator field_it = mesh->get_field_registry()->begin();
  DICe::mesh::field_registry::iterator field_end = mesh->get_field_registry()->end();
  for(;field_it!=field_end;++field_it)
  {
    if(field_it->first.get_rank()!=field_enums::INTERNAL_FACE_EDGE_RANK) continue;
    const int_t num_comps = (field_it->first.get_field_type()==field_enums::VECTOR_FIELD_TYPE) ? 3 : 1; // spheres are hard coded to have 3 dims
    const int_t comp_stride = (field_it->first.get_field_type()==field_enums::VECTOR_FIELD_TYPE) ? spa_dim : 1;
    MultiField & field = *mesh->get_field(field_it->first);
    Teuchos::ArrayRCP<const scalar_t> field_values = field.get_1d_view();
    float values[num_values];
    std::string components[3];
    components[0] = (field_it->first.get_field_type()==field_enums::VECTOR_FIELD_TYPE) ? "X" : "";
    components[1] = "Y";
    components[2] = "Z";
    for (int_t comp = 0; comp < num_comps; ++comp)
    {
      DICe::mesh::internal_face_edge_set::const_iterator face_it = mesh->get_internal_face_edge_set()->begin();
      DICe::mesh::internal_face_edge_set::const_iterator face_end = mesh->get_internal_face_edge_set()->end();
      for(;face_it!=face_end;++face_it){
        if(comp >= spa_dim)
          values[face_it->get()->local_id()] = 0.0; // dummy values for 3d output from a 2d mesh (spheres in exodus require 3d output)
        else
          values[face_it->get()->local_id()] = field_values[face_it->get()->local_id()*comp_stride+comp];
      }
//      face_it = mesh->get_boundary_face_edge_set()->begin();
//      face_end = mesh->get_boundary_face_edge_set()->end();
//      for(;face_it!=face_end;++face_it){
//        if(comp >= spa_dim)
//          values[face_it->get()->local_id()] = 0.0; // dummy values for 3d output from a 2d mesh (spheres in exodus require 3d output)
//        else
//          values[face_it->get()->local_id()] = field_values[face_it->get()->local_id()*comp_stride+comp];
//      }
      const int_t var_index = get_var_index(mesh, tostring(field_it->first.get_name()), components[comp], field_it->first.get_rank(),true,false);
      error_int = ex_put_nodal_var(mesh->get_face_edge_output_exoid(), time_step_num, var_index, num_values, values);
    }
  }
  error_int = ex_update(mesh->get_face_edge_output_exoid());
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_update(): Failure");
}

int_t
get_var_index(Teuchos::RCP<Mesh> mesh,
  const std::string & name,
  const std::string & component,
  const field_enums::Entity_Rank rank,
  const bool ignore_dimension,
  const bool only_printable)
{
  std::string comp = "";
  if (component != "") comp = "_" + component;
  boost::to_upper(comp);
  const int_t spatial_dimension = mesh->spatial_dimension();
  // this needs to be improved... right now the exodusII file does not differente between scalars and vectors for the variable names
  // so we have to concatinate the list like this:
  std::vector<std::string> scalar_fields = mesh->get_field_names(rank,field_enums::SCALAR_FIELD_TYPE,only_printable);
  std::vector<std::string> vector_fields = mesh->get_field_names(rank,field_enums::VECTOR_FIELD_TYPE,only_printable);
  int_t num_scalar_variables = scalar_fields.size();
  int_t num_vector_variables = vector_fields.size();
  std::vector<std::string> var_names;
  for (int_t i = 0; i < num_scalar_variables; ++i)
  {
    var_names.push_back(scalar_fields[i]);
  }
  for (int_t i = 0; i < num_vector_variables; ++i)
  {
    var_names.push_back(vector_fields[i] + "_X");
    var_names.push_back(vector_fields[i] + "_Y");
    if (spatial_dimension > 2 || ignore_dimension) var_names.push_back(vector_fields[i] + "_Z");
  }
  // rip through the list looking for the specified field which is broken down by component
  for (size_t i = 0; i < var_names.size(); ++i)
  {
    //cout << " This is a var name: " << var_names[i] << std::endl;
    if (var_names[i] == name + comp)
    return i + 1;
  }
  return -1;
}

void
close_exodus_output(Teuchos::RCP<Mesh> mesh){
  ex_close(mesh->get_output_exoid());
}

void
close_face_edge_exodus_output(Teuchos::RCP<Mesh> mesh){
  ex_close(mesh->get_face_edge_output_exoid());
}


void
create_face_edge_output_variable_names(Teuchos::RCP<Mesh> mesh)
{
  int error_int;
  const int_t spatial_dimension = mesh->spatial_dimension();
  const int output_exoid = mesh->get_face_edge_output_exoid();

  // ELEMENT fields
  const std::vector<std::string> element_scalar_fields = mesh->get_field_names(field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::SCALAR_FIELD_TYPE,false);
  const std::vector<std::string> element_vector_fields = mesh->get_field_names(field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::VECTOR_FIELD_TYPE,false);

  int_t num_element_scalar_variables = element_scalar_fields.size();
  int_t num_element_vector_variables = element_vector_fields.size();
  int_t num_element_variables = num_element_scalar_variables
      + num_element_vector_variables * spatial_dimension;
  char* ele_var_names[num_element_variables];
  std::vector<std::string> element_string_type_names;
  for (int_t i = 0; i < num_element_scalar_variables; ++i)
  {
    element_string_type_names.push_back(element_scalar_fields[i]);
  }
  for (int_t i = 0; i < num_element_vector_variables; ++i)
  {
    std::string x_comp = element_vector_fields[i] + "_X";
    std::string y_comp = element_vector_fields[i] + "_Y";
    std::string z_comp = element_vector_fields[i] + "_Z";
    element_string_type_names.push_back(x_comp);
    element_string_type_names.push_back(y_comp);
    element_string_type_names.push_back(z_comp);
  }
  for (int_t i = 0; i < num_element_variables; ++i)
  {
    ele_var_names[i] = (char*) (element_string_type_names[i].c_str());
  }

  error_int = ex_put_var_param(output_exoid, (char*) "n", num_element_variables); // we trick exodus into thinking all of the fields are nodal
  error_int = ex_put_var_names(output_exoid, (char*) "n", num_element_variables,  // even though internally they are element fields
    ele_var_names);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_var_names(): Failure");

}

void
create_face_edge_output_exodus_file(Teuchos::RCP<Mesh> mesh,
  const std::string & output_folder)
{

  std::stringstream out_file;
  out_file << output_folder << mesh->get_face_edge_output_filename();
  const std::string filename = out_file.str();
  std::vector<char> writable(filename.size() + 1);
  std::copy(filename.begin(), filename.end(), writable.begin());
  int_t spatial_dimension = mesh->spatial_dimension();
  const int_t num_internal = mesh->num_internal_faces();
  int_t num_elem = num_internal;// + mesh->num_boundary_faces();
  int error_int;
  int_t CPU_word_size = 0;
  int_t IO_word_size = 0;
  /* create EXODUS II file */
  const int output_exoid = ex_create (&writable[0],EX_CLOBBER,&CPU_word_size, &IO_word_size);
  mesh->set_face_edge_output_exoid(output_exoid);

  error_int = ex_put_init(output_exoid, &writable[0], spatial_dimension, num_elem,
    num_elem, 1, 0, 0); // all of the nodes are put into one block
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_init(): Failure");

  //  write initial coordinates and node/element maps
  float x[num_elem];
  float y[num_elem];
  float z[num_elem];

  Teuchos::RCP<MultiField > internal_coords = mesh->get_field(field_enums::INTERNAL_FACE_EDGE_COORDINATES_FS);
  Teuchos::ArrayRCP<const scalar_t> internal_coords_values = internal_coords->get_1d_view();
  //Teuchos::RCP<MultiField > external_coords = mesh->get_field(field_enums::EXTERNAL_FACE_EDGE_COORDINATES_FS);
  //Teuchos::ArrayRCP<const scalar_t> external_coords_values = external_coords->get_1d_view();
  int_t * elem_map = new int_t[num_elem];
  int_t * node_map = new int_t[num_elem];

  DICe::mesh::internal_face_edge_set::const_iterator face_it = mesh->get_internal_face_edge_set()->begin();
  DICe::mesh::internal_face_edge_set::const_iterator face_end = mesh->get_internal_face_edge_set()->end();
  for(;face_it!=face_end;++face_it)
  {
    const int_t local_id = face_it->get()->local_id();
    x[local_id] = internal_coords_values[local_id*spatial_dimension+0];
    y[local_id] = internal_coords_values[local_id*spatial_dimension+1];
    if(spatial_dimension > 2)
      z[local_id] = internal_coords_values[local_id*spatial_dimension+2];
    else
      z[local_id] = 0.0;
    node_map[local_id]=face_it->get()->local_id(); // FIXME : does a global id need to be created for this?
    elem_map[local_id]=face_it->get()->local_id();
    //cout << " Element " << local_id << " x " << x[local_id] << " y " << y[local_id] << " z " << z[local_id] << endl;
  }
//  face_it = mesh->get_boundary_face_edge_set()->begin();
//  face_end = mesh->get_boundary_face_edge_set()->end();
//  for(;face_it!=face_end;++face_it)
//  {
//    const int_t local_id = face_it->get()->local_id();
//    x[local_id+num_internal] = external_coords_values[local_id*spatial_dimension+0];
//    y[local_id+num_internal] = external_coords_values[local_id*spatial_dimension+1];
//    if(spatial_dimension > 2)
//      z[local_id+num_internal] = external_coords_values[local_id*spatial_dimension+2];
//    else
//      z[local_id+num_internal] = 0.0;
//    node_map[local_id+num_internal]=face_it->get()->local_id()+1+num_internal; // FIXME : does a global id need to be created for this?
//    elem_map[local_id+num_internal]=face_it->get()->local_id()+1+num_internal; // ids are 1 based in exodus
//    //cout << " Element " << local_id << " x " << x[local_id] << " y " << y[local_id] << " z " << z[local_id] << endl;
//  }

  error_int = ex_put_coord(output_exoid, x, y, z);
  char * coord_names[3];
  coord_names[0] = (char*) "INTERNAL_FACE_EDGE_COORDINATES_X";
  coord_names[1] = (char*) "INTERNAL_FACE_EDGE_COORDINATES_Y";
  coord_names[2] = (char*) "INTERNAL_FACE_EDGE_COORDINATES_Z";
  error_int = ex_put_coord_names(output_exoid, coord_names);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_coord_names(): Failure");
  error_int = ex_put_elem_num_map(output_exoid, elem_map);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_elem_num_map(): Failure");
  error_int = ex_put_node_num_map(output_exoid, node_map);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_node_num_map(): Failure");
  delete[] elem_map;
  delete[] node_map;

  //  write elem blocks
  int_t num_nodes_per_elem = 1;
  const Base_Element_Type elem_type_enum = SPHERE;
  const std::string elem_type_str = tostring(elem_type_enum);
  char * elem_type = const_cast<char *>(elem_type_str.c_str());
  const int_t num_elem_in_block = mesh->num_internal_faces(); //+mesh->num_boundary_faces();
  error_int = ex_put_elem_block(output_exoid, 1, elem_type,
    num_elem_in_block, num_nodes_per_elem, 0); // no attributes put in output file
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_elem_block(): Failure");

  //  write elem connectivities
  //  HEADS UP: the connectivities will not write to file until ex_close is called
  int_t conn_index = 0;
  int_t * block_connect = new int_t[num_elem_in_block * num_nodes_per_elem];
  face_it = mesh->get_internal_face_edge_set()->begin();
  face_end = mesh->get_internal_face_edge_set()->end();
  for(;face_it!=face_end;++face_it)
  {
    block_connect[conn_index] = face_it->get()->local_id() + 1;  // connectivity is always 1-based in exodus
    conn_index++;
  }
//  face_it = mesh->get_boundary_face_edge_set()->begin();
//  face_end = mesh->get_boundary_face_edge_set()->end();
//  for(;face_it!=face_end;++face_it)
//  {
//    block_connect[conn_index] = face_it->get()->local_id() + 1 + num_internal;  // local_ids are 1 based in exodus
//    conn_index++;
//  }
  error_int = ex_put_elem_conn(output_exoid, 1, block_connect);
  delete[] block_connect;
}


void
create_exodus_output_variable_names(Teuchos::RCP<Mesh> mesh)
{
  int error_int;
  const int_t spatial_dimension = mesh->spatial_dimension();
  const int output_exoid = mesh->get_output_exoid();

  // NODAL fields
  const std::vector<std::string> nodal_scalar_fields = mesh->get_field_names(field_enums::NODE_RANK,field_enums::SCALAR_FIELD_TYPE,true);
  const std::vector<std::string> nodal_vector_fields = mesh->get_field_names(field_enums::NODE_RANK,field_enums::VECTOR_FIELD_TYPE,true);
  int_t num_nodal_scalar_variables = nodal_scalar_fields.size();
  int_t num_nodal_vector_variables = nodal_vector_fields.size();
  int_t num_nodal_variables = num_nodal_scalar_variables
      + num_nodal_vector_variables * spatial_dimension;
  std::vector<std::string> nodal_string_type_names;
  char* nodal_var_names[num_nodal_variables];
  for (int_t i = 0; i < num_nodal_scalar_variables; ++i)
  {

    nodal_string_type_names.push_back(nodal_scalar_fields[i]);
  }
  for (int_t i = 0; i < num_nodal_vector_variables; ++i)
  {
    std::string x_comp = nodal_vector_fields[i] + "_X";
    std::string y_comp = nodal_vector_fields[i] + "_Y";
    std::string z_comp = nodal_vector_fields[i] + "_Z";
    nodal_string_type_names.push_back(x_comp);
    nodal_string_type_names.push_back(y_comp);
    if (spatial_dimension > 2) nodal_string_type_names.push_back(z_comp);
  }
  for (int_t i = 0; i < num_nodal_variables; ++i)
  {
    nodal_var_names[i] = (char*) (nodal_string_type_names[i].c_str());
  }

  error_int = ex_put_var_param(output_exoid, (char*) "n", num_nodal_variables);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_var_param(): Failure");
  error_int = ex_put_var_names(output_exoid, (char*) "n", num_nodal_variables,nodal_var_names);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::logic_error,"ex_put_var_names     (): Failure");

  // ELEMENT fields
  const std::vector<std::string> element_scalar_fields = mesh->get_field_names(field_enums::ELEMENT_RANK,field_enums::SCALAR_FIELD_TYPE,true);
  const std::vector<std::string> element_vector_fields = mesh->get_field_names(field_enums::ELEMENT_RANK,field_enums::VECTOR_FIELD_TYPE,true);
  int_t num_element_scalar_variables = element_scalar_fields.size();
  int_t num_element_vector_variables = element_vector_fields.size();
  int_t num_element_variables = num_element_scalar_variables
      + num_element_vector_variables * spatial_dimension;
  char* ele_var_names[num_element_variables];
  std::vector<std::string> element_string_type_names;
  for (int_t i = 0; i < num_element_scalar_variables; ++i)
  {
    element_string_type_names.push_back(element_scalar_fields[i]);
  }
  for (int_t i = 0; i < num_element_vector_variables; ++i)
  {
    std::string x_comp = element_vector_fields[i] + "_X";
    std::string y_comp = element_vector_fields[i] + "_Y";
    std::string z_comp = element_vector_fields[i] + "_Z";
    element_string_type_names.push_back(x_comp);
    element_string_type_names.push_back(y_comp);
    if (spatial_dimension > 2) element_string_type_names.push_back(z_comp);
  }
  for (int_t i = 0; i < num_element_variables; ++i)
  {
    ele_var_names[i] = (char*) (element_string_type_names[i].c_str());
  }
  error_int = ex_put_var_param(output_exoid, (char*) "e", num_element_variables);
  error_int = ex_put_var_names(output_exoid, (char*) "e", num_element_variables,
    ele_var_names);
}

void
initialize_control_volumes(Teuchos::RCP<Mesh> mesh){

  // The control volume mesh objects are oganized as follows:
  // Element -> Subelement (if tri3 or tet4 Subelement=Element) -> Internal_Cell and Internal_Face_Edge

  // return if already done by some other physics for this mesh;
  if(mesh->control_volumes_are_initialized()) return;
  bool print_warning_hex = true;
  bool print_warning_quad = true;

  const int_t p_rank = mesh->get_comm()->get_rank();
  const int_t spa_dim = mesh->spatial_dimension();

  Teuchos::RCP<DICe::mesh::subelement_set> subelem_set = mesh->get_subelement_set();
  static Teuchos::RCP<DICe::mesh::Subelement> subelem_rcp;
  static int_t subelem_local_id;
  static int_t subelem_global_id;

  // create subelements for each hex8 or quad4:
  DICe::mesh::element_set::iterator elem_it = mesh->get_element_set()->begin();
  DICe::mesh::element_set::iterator elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    Teuchos::RCP<DICe::mesh::Element> elem = *elem_it;
    const DICe::mesh::connectivity_vector & connectivity = *elem->connectivity();

    const Base_Element_Type elem_type = mesh->get_block_type_map()->find(elem_it->get()->block_id())->second;
    if(elem_type == HEX8)
    {
      if(print_warning_hex && p_rank==0)
      {
        std::cout << "    NOTICE: User requested CVFEM discretization on hexahedral elements. Each hexahedral element will be " << std::endl
            << "            divided into six four-node tetrahedrons. The resulting discretization will be equivalent to " << std::endl
            << "            a mesh composed of linear tetrahedons conforming to the vertices of the original hex mesh." << std::endl << std::endl;
        print_warning_hex = false;
      }
      // create 6 subelements based on the connectivity:
      const int_t num_subelem_per_elem = 6;
      const int_t num_nodes_per_subelem = 4;
      Teuchos::RCP<DICe::mesh::Node> node_rcps[num_subelem_per_elem][num_nodes_per_subelem];
      node_rcps[0][0] = connectivity[0]; node_rcps[0][1] = connectivity[2]; node_rcps[0][2] = connectivity[5]; node_rcps[0][3] = connectivity[1];
      node_rcps[1][0] = connectivity[0]; node_rcps[1][1] = connectivity[2]; node_rcps[1][2] = connectivity[3]; node_rcps[1][3] = connectivity[4];
      node_rcps[2][0] = connectivity[0]; node_rcps[2][1] = connectivity[2]; node_rcps[2][2] = connectivity[4]; node_rcps[2][3] = connectivity[5];
      node_rcps[3][0] = connectivity[2]; node_rcps[3][1] = connectivity[6]; node_rcps[3][2] = connectivity[4]; node_rcps[3][3] = connectivity[5];
      node_rcps[4][0] = connectivity[2]; node_rcps[4][1] = connectivity[3]; node_rcps[4][2] = connectivity[4]; node_rcps[4][3] = connectivity[6];
      node_rcps[5][0] = connectivity[3]; node_rcps[5][1] = connectivity[6]; node_rcps[5][2] = connectivity[7]; node_rcps[5][3] = connectivity[4];

      connectivity_vector subelem_con;
      for(int_t i=0;i<num_nodes_per_subelem;++i) // initialize the vector with the 0th node
        subelem_con.push_back(connectivity[0]);

      for(int_t subelem_it=0;subelem_it<num_subelem_per_elem;++subelem_it)
      {
        for(int_t node_it=0;node_it<num_nodes_per_subelem;++node_it)
        {
          subelem_con[node_it] = node_rcps[subelem_it][node_it];
        }
        subelem_local_id = subelem_set->size();
        subelem_global_id = mesh->get_scalar_subelem_dist_map()->get_global_element(subelem_local_id);
        subelem_rcp = Teuchos::rcp(new DICe::mesh::Subelement(subelem_con,*elem_it,subelem_local_id,subelem_global_id,elem_it->get()->block_id()));
        subelem_set->push_back(subelem_rcp);
      }
    }
    else if(elem_type == TETRA4 || elem_type == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
    {
      subelem_local_id = subelem_set->size();
      subelem_global_id = mesh->get_scalar_subelem_dist_map()->get_global_element(subelem_local_id);
      subelem_rcp = Teuchos::rcp(new DICe::mesh::Subelement(*elem_it,subelem_local_id,subelem_global_id));
      subelem_set->push_back(subelem_rcp);
    }
    else if(elem_type == PYRAMID5)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"CVFEM not implemented yet for PYRAMID5");
    }
    else if(elem_type == QUAD4)
    {
      if(print_warning_quad && p_rank==0)
      {
        std::cout << "    NOTICE: User requested CVFEM discretization on quad4 elements. Each quad4 element will be " << std::endl
            << "            divided into two three-node triangles. The resulting discretization will be equivalent to " << std::endl
            << "            a mesh composed of linear triangles conforming to the vertices of the original quad mesh." << std::endl << std::endl;
        print_warning_quad = false;
      }
      connectivity_vector subelem_con;
      subelem_con.push_back(connectivity[0]);
      subelem_con.push_back(connectivity[1]);
      subelem_con.push_back(connectivity[2]);
      subelem_local_id = subelem_set->size();
      subelem_global_id = mesh->get_scalar_subelem_dist_map()->get_global_element(subelem_local_id);
      subelem_rcp = Teuchos::rcp(new DICe::mesh::Subelement(subelem_con,*elem_it,subelem_local_id,subelem_global_id,elem_it->get()->block_id()));
      subelem_set->push_back(subelem_rcp);

      subelem_con[0] = connectivity[0];
      subelem_con[1] = connectivity[2];
      subelem_con[2] = connectivity[3];
      subelem_local_id = subelem_set->size();
      subelem_global_id = mesh->get_scalar_subelem_dist_map()->get_global_element(subelem_local_id);
      subelem_rcp = Teuchos::rcp(new DICe::mesh::Subelement(subelem_con,*elem_it,subelem_local_id,subelem_global_id,elem_it->get()->block_id()));
      subelem_set->push_back(subelem_rcp);
    }
    else if(elem_type == TRI3)
    {
      subelem_local_id = subelem_set->size();
      subelem_global_id = mesh->get_scalar_subelem_dist_map()->get_global_element(subelem_local_id);
      subelem_rcp = Teuchos::rcp(new DICe::mesh::Subelement(*elem_it,subelem_local_id,subelem_global_id));
      subelem_set->push_back(subelem_rcp);
    }
    else
    {
      std::stringstream oss;
      oss << "initialize_control_volumes(): unknown element type: " << tostring(elem_type) << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
  }

  // provide a mapping of the subelements with block access
  DICe::mesh::block_type_map::iterator blk_map_it = mesh->get_block_type_map()->begin();
  DICe::mesh::block_type_map::iterator blk_map_end = mesh->get_block_type_map()->end();
  for(;blk_map_it!=blk_map_end;++blk_map_it)
  {
    Teuchos::RCP<DICe::mesh::subelement_set> subele_set = Teuchos::rcp(new DICe::mesh::subelement_set);
    DICe::mesh::subelement_set::iterator subele_it = mesh->get_subelement_set()->begin();
    DICe::mesh::subelement_set::iterator subele_end = mesh->get_subelement_set()->end();
    for(;subele_it!=subele_end;++subele_it)
    {
      if(subele_it->get()->block_id()==blk_map_it->first)
        subele_set->push_back(*subele_it);
    }
    mesh->get_subelement_sets_by_block()->insert(std::pair<int_t,Teuchos::RCP<DICe::mesh::subelement_set> >(blk_map_it->first,subele_set));
  }

  // create control volume fields:
  mesh->create_field(field_enums::INTERNAL_FACE_EDGE_NORMAL_FS);
  mesh->create_field(field_enums::INTERNAL_FACE_EDGE_COORDINATES_FS);
  mesh->create_field(field_enums::INTERNAL_FACE_EDGE_SIZE_FS);
  //mesh->create_field(field_enums::EXTERNAL_FACE_EDGE_NORMAL_FS);
  //mesh->create_field(field_enums::EXTERNAL_FACE_EDGE_COORDINATES_FS);
  //mesh->create_field(field_enums::EXTERNAL_FACE_EDGE_SIZE_FS);
  mesh->create_field(field_enums::INTERNAL_CELL_COORDINATES_FS);
  mesh->create_field(field_enums::INTERNAL_CELL_SIZE_FS);

  mesh->create_field(field_enums::INITIAL_SUBELEMENT_SIZE_FS);

  Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  Teuchos::RCP<MultiField > coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);

  Teuchos::ArrayRCP<const scalar_t> coords_values = coords->get_1d_view();

  // iterate over subelements to create internal cells and faces or edges
  // and the internal cell and face or edge fields:
  DICe::mesh::subelement_set::iterator subelem_it = mesh->get_subelement_set()->begin();
  DICe::mesh::subelement_set::iterator subelem_end = mesh->get_subelement_set()->end();
  for(;subelem_it!=subelem_end;++subelem_it)
  {
    Teuchos::RCP<DICe::mesh::Subelement> subelem = *subelem_it;
    const DICe::mesh::connectivity_vector & connectivity = *subelem->connectivity();
    for(size_t node_it=0;node_it<connectivity.size();++node_it)
    {
      Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
      const Base_Element_Type elem_type = mesh->get_block_type_map()->find(subelem->parent_element()->block_id())->second;
      if(elem_type == HEX8 || elem_type == TETRA4 || elem_type == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
      {
        DICe::mesh::tetra4_sub_elem_edge_legths_and_normals(mesh,coords_values,node,subelem);
      }
      else if(elem_type==QUAD4 || elem_type == TRI3)
        DICe::mesh::tri3_sub_elem_edge_legths_and_normals(mesh,coords_values,node,subelem);
      else
      {
        std::stringstream oss;
        oss << "initialize_control_volumes(): not implemented for element type: " << tostring(elem_type) << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
      }
    }
  }

  // ADD BOUNDARY FACES

  MultiField & edge_centroid = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_COORDINATES_FS);
  MultiField & edge_normal = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_NORMAL_FS);
  MultiField & edge_size = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_SIZE_FS);
  //Teuchos::RCP<DICe::mesh::external_face_edge_set> boundary_face_edge_set = mesh->get_boundary_face_edge_set();
  Teuchos::RCP<DICe::mesh::internal_face_edge_set> internal_face_edge_set = mesh->get_internal_face_edge_set();
  DICe::mesh::side_set_info & ss_info = *mesh->get_side_set_info();
  const int_t num_sets = ss_info.ids.size();
  TEUCHOS_TEST_FOR_EXCEPTION(num_sets==0,std::runtime_error,
    "Error: No side sets have been specified in the mesh (all edges of the domain must be part of a side set, and only one side set)");
  for(int_t id_it=1;id_it<=num_sets;++id_it){
    const int_t set_index = ss_info.get_set_index(id_it);
    const int_t num_sides_this_set = ss_info.num_side_per_set[set_index];
    const int_t start_index = ss_info.elem_ind[set_index];
    for(int_t i=0;i<num_sides_this_set;++i){
      const int_t side_index = start_index+i;
      const int_t elem_gid = ss_info.ss_global_elem_list[side_index];
      const int_t side_id = ss_info.ss_side_list[side_index];
      // check if this elem is local to this process:
      const int_t elem_lid = mesh->get_scalar_elem_dist_map()->get_local_element(elem_gid);
      if(elem_lid<0) continue;
      //std::cout << " SETTING UP BOUNDARY FACES FOR ELEMENT GID: " << elem_gid << " LID " << elem_lid << std::endl;
      // find the element for this side set
      Teuchos::RCP<DICe::mesh::Subelement> subelem = (*mesh->get_subelement_set())[elem_lid];
      //std::cout << " the subelem has the following lid " << subelem->local_id() << " gid " << subelem->global_id() << std::endl;
      const DICe::mesh::connectivity_vector & connectivity = *subelem->connectivity();
      for(size_t node_it=0;node_it<connectivity.size();++node_it){
        Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
        //std::cout << " node " << node->global_id() << std::endl;
      }
      Teuchos::RCP<DICe::mesh::Node> target_node_1;
      Teuchos::RCP<DICe::mesh::Node> target_node_2;

      if(side_id==1){
        target_node_1 = connectivity[0];
        target_node_2 = connectivity[1];
      }
      else if(side_id==2){
        target_node_1 = connectivity[1];
        target_node_2 = connectivity[2];
      }
      else if(side_id==3){
        target_node_1 = connectivity[2];
        target_node_2 = connectivity[0];
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid side ID ");
      }
      //std::cout << " target_node_1 gid " << target_node_1->global_id() << " target_node_2 gid " << target_node_2->global_id() << std::endl;
      // create TWO faces, one for each node:
      const int_t face_edge_local_id_1 = internal_face_edge_set->size(); //boundary_face_edge_set->size(); // + internal_face_edge_set->size();
      const int_t face_edge_global_id_1 = mesh->get_scalar_face_dist_map()->get_global_element(face_edge_local_id_1);
      //std::cout << " edge local id 1 " << face_edge_local_id_1 << " global id " << face_edge_global_id_1 << std::endl;
      Teuchos::RCP< Internal_Face_Edge > boundary_face_edge_1 = Teuchos::rcp( new Internal_Face_Edge(face_edge_local_id_1, face_edge_global_id_1,target_node_1,Teuchos::null,subelem));
      internal_face_edge_set->push_back(boundary_face_edge_1);
      const int_t face_edge_local_id_2 = internal_face_edge_set->size(); //boundary_face_edge_set->size(); // + internal_face_edge_set->size();
      const int_t face_edge_global_id_2 = mesh->get_scalar_face_dist_map()->get_global_element(face_edge_local_id_2);
      //std::cout << " edge local id 2 " << face_edge_local_id_2 << " global id " << face_edge_global_id_2 << std::endl;
      Teuchos::RCP< Internal_Face_Edge > boundary_face_edge_2 = Teuchos::rcp( new Internal_Face_Edge(face_edge_local_id_2, face_edge_global_id_2,target_node_2,Teuchos::null,subelem));
      internal_face_edge_set->push_back(boundary_face_edge_2);

      // add the cell and edge to the parent subelement:
      subelem->add_deep_relation(boundary_face_edge_1);
      subelem->add_deep_relation(boundary_face_edge_2);

      // centroid (2 of them):
      const scalar_t lx = coords_values[target_node_1->overlap_local_id()*spa_dim+0];
      const scalar_t rx = coords_values[target_node_2->overlap_local_id()*spa_dim+0];
      const scalar_t ly = coords_values[target_node_1->overlap_local_id()*spa_dim+1];
      const scalar_t ry = coords_values[target_node_2->overlap_local_id()*spa_dim+1];
      //std::cout << " left node " << lx << " " << ly << " right node " << rx << " " << ry << std::endl;
      const scalar_t dx = rx - lx;
      const scalar_t dy = ly - ry;
      const scalar_t lcx =  0.25*dx + lx;
      const scalar_t lcy = -0.25*dy + ly;
      const scalar_t rcx =  0.75*dx + lx;
      const scalar_t rcy = -0.75*dy + ly;
      // length_total (divide by two for each side):
      const scalar_t edge_length = std::sqrt(dx*dx + dy*dy);
      //std::cout << " left centroid " << lcx << " " << lcy << " right centroid " << rcx << " " << rcy << " length " << edge_length << std::endl;
      // normal (same for both sides):
      TEUCHOS_TEST_FOR_EXCEPTION(edge_length<=0.0,std::runtime_error,"");
      const scalar_t nx = dy/edge_length;
      const scalar_t ny = dx/edge_length;
      //std::cout << " normal " << nx << " " << ny << std::endl;

      edge_size.local_value(face_edge_local_id_1) = 0.5*edge_length;
      edge_size.local_value(face_edge_local_id_2) = 0.5*edge_length;
      edge_centroid.local_value(face_edge_local_id_1*spa_dim + 0) = lcx;
      edge_centroid.local_value(face_edge_local_id_1*spa_dim + 1) = lcy;
      edge_centroid.local_value(face_edge_local_id_2*spa_dim + 0) = rcx;
      edge_centroid.local_value(face_edge_local_id_2*spa_dim + 1) = rcy;
      edge_normal.local_value(face_edge_local_id_1*spa_dim + 0) = nx;
      edge_normal.local_value(face_edge_local_id_1*spa_dim + 1) = ny;
      edge_normal.local_value(face_edge_local_id_2*spa_dim + 0) = nx;
      edge_normal.local_value(face_edge_local_id_2*spa_dim + 1) = ny;

    }
  }

  // provide a mapping of the internal faces or edges by block access
  for(blk_map_it = mesh->get_block_type_map()->begin();blk_map_it!=blk_map_end;++blk_map_it)
  {
    Teuchos::RCP<DICe::mesh::internal_face_edge_set> face_set = Teuchos::rcp(new DICe::mesh::internal_face_edge_set);
    DICe::mesh::internal_face_edge_set::iterator face_it = mesh->get_internal_face_edge_set()->begin();
    DICe::mesh::internal_face_edge_set::iterator face_end = mesh->get_internal_face_edge_set()->end();
    for(;face_it!=face_end;++face_it)
    {
      if(face_it->get()->parent_subelement()->parent_element()->block_id()==blk_map_it->first)
        face_set->push_back(*face_it);
    }
//    face_it = mesh->get_boundary_face_edge_set()->begin();
//    face_end = mesh->get_boundary_face_edge_set()->end();
//    for(;face_it!=face_end;++face_it)
//    {
//      if(face_it->get()->parent_subelement()->parent_element()->block_id()==blk_map_it->first)
//        face_set->push_back(*face_it);
//    }
    mesh->get_internal_face_sets_by_block()->insert(std::pair<int_t,Teuchos::RCP<DICe::mesh::internal_face_edge_set> >(blk_map_it->first,face_set));
  }

  // output relations information
  //const int_t p_rank = mesh->get_comm()->getRank();
  scalar_t average_relations_this_proc = 0;
  int_t max_num_relations_this_proc = 0;
  int_t min_num_relations = INT_MAX;
  std::stringstream oss;
  oss << std::endl << "  [p=" << p_rank << "] Interal Face Edge Element Relations Table: " << std::endl << std::endl;
  for(subelem_it=mesh->get_subelement_set()->begin();subelem_it!=subelem_end;++subelem_it)
  {
    oss << "  Element: " << subelem_it->get()->global_id() << " | ";
    const int_t rel_size = subelem_it->get()->num_deep_relations(field_enums::INTERNAL_FACE_EDGE_RANK);
    average_relations_this_proc += rel_size;
    if(rel_size < min_num_relations) min_num_relations = rel_size;
    if(rel_size > max_num_relations_this_proc) max_num_relations_this_proc = rel_size;
    const std::vector<Teuchos::RCP<Mesh_Object> > * relations = subelem_it->get()->deep_relations(field_enums::INTERNAL_FACE_EDGE_RANK);
    oss << rel_size << " face or edge relations |";
    for(std::vector<Teuchos::RCP<Mesh_Object> >::const_iterator i=relations->begin();i!=relations->end();++i)
    {
      oss << " " << i->get()->global_id();
    }
    oss << std::endl;
  }

  average_relations_this_proc = average_relations_this_proc / mesh->get_subelement_set()->size();
  oss << "  Average subelement face edge relations this processor: " << average_relations_this_proc << std::endl;
  oss << "  Min subelement face edge relations this processor: " << min_num_relations << std::endl;
  oss << "  Max subelement face edge relations this processor: " << max_num_relations_this_proc << std::endl;
  mesh->set_mean_num_subelem_face_relations(average_relations_this_proc);
  mesh->set_min_num_subelem_face_relations(min_num_relations);
  mesh->set_max_num_subelem_face_relations(max_num_relations_this_proc);
  //if(p_rank==0)
  //  DEBUG_MSG(oss.str());
  //oss.str("");
  //oss.clear();

  scalar_t average_node_relations_this_proc = 0;
  int_t max_node_num_relations_this_proc = 0;
  int_t min_node_num_relations = INT_MAX;

  oss << std::endl << "  [p=" << p_rank << "] Node Relations Table: " << std::endl << std::endl;
  DICe::mesh::node_set::iterator node_it = mesh->get_node_set()->begin();
  DICe::mesh::node_set::iterator node_end = mesh->get_node_set()->end();
  for(;node_it!=node_end;++node_it)
  {
    Teuchos::RCP<DICe::mesh::Node> node = node_it->second;
    oss << "  Node: " << node->global_id() << " | ";
    const int_t rel_size = node->num_deep_relations(field_enums::NODE_RANK);
    average_node_relations_this_proc += rel_size;
    if(rel_size < min_node_num_relations) min_node_num_relations = rel_size;
    if(rel_size > max_node_num_relations_this_proc) max_node_num_relations_this_proc = rel_size;

    const std::vector<Teuchos::RCP<Mesh_Object> > * relations = node->deep_relations(field_enums::NODE_RANK);
    oss << rel_size << " neighbor node relations |";
    for(std::vector<Teuchos::RCP<Mesh_Object> >::const_iterator i=relations->begin();i!=relations->end();++i)
    {
      oss << " " << i->get()->global_id();
    }
    oss << std::endl;
  }
  average_node_relations_this_proc = average_node_relations_this_proc / mesh->get_node_set()->size();
  oss << "  Average to node node relations this processor: " << average_node_relations_this_proc << std::endl;
  oss << "  Min node to node relations this processor: " << min_node_num_relations << std::endl;
  oss << "  Max node to node relations this processor: " << max_node_num_relations_this_proc << std::endl;
  mesh->set_mean_num_node_relations(average_node_relations_this_proc);
  mesh->set_min_num_node_relations(min_node_num_relations);
  mesh->set_max_num_node_relations(max_node_num_relations_this_proc);

  //if(p_rank==0)
  //  DEBUG_MSG(oss.str());

  mesh->set_control_volumes_are_initialzed();
}

// Create the internal faces and edges for control volume calculations
// Only the right hand boundary object is computed for each subelement
// since every boundary object is shared by two nodes.
void
tri3_sub_elem_edge_legths_and_normals(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> cv_node,
  Teuchos::RCP<DICe::mesh::Subelement> subelement)
{
  const int_t spa_dim = mesh->spatial_dimension();
  const DICe::mesh::connectivity_vector & connectivity = *subelement->connectivity();
  const int_t num_nodes = connectivity.size();
  TEUCHOS_TEST_FOR_EXCEPTION(num_nodes!=3,std::invalid_argument,"Connectivity does not have the right number of nodes.");

  // figure out the local node id of the cv_node
  int_t cv_local_node_index = -1;
  for(int_t i=0;i<3;++i)
    if(connectivity[i]->global_id()==cv_node->global_id()) cv_local_node_index = i;
  TEUCHOS_TEST_FOR_EXCEPTION(cv_local_node_index==-1,std::invalid_argument,"Node was not found in the element.");

  Teuchos::RCP<DICe::mesh::Node> node_I;
  Teuchos::RCP<DICe::mesh::Node> node_A;
  Teuchos::RCP<DICe::mesh::Node> node_B;
  if(cv_local_node_index==0)
  {
    node_I = connectivity[0];
    node_A = connectivity[1];
    node_B = connectivity[2];
  }
  else if(cv_local_node_index==1)
  {
    node_I = connectivity[1];
    node_A = connectivity[2];
    node_B = connectivity[0];
  }
  else if(cv_local_node_index==2)
  {
    node_I = connectivity[2];
    node_A = connectivity[0];
    node_B = connectivity[1];
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Invalid cv_node_local_index.");

   //cout << " Target node id " << node_I->global_id() << " neighbor id " << node_A->global_id() << std::endl;
   //cout << " The size of this element is: " << cell_size_values[element->local_id()] << std::endl;

  scalar_t cntrd[] = {0.0, 0.0};
  compute_centroid_of_tri(coords_values,node_A,node_B,node_I,&cntrd[0]);

  scalar_t edge_length;
  scalar_t normal[2];
  scalar_t subtri_centroid[2];
  scalar_t subtri_edge_centroid[2];
  compute_submesh_obj_from_tri3(coords_values,node_A,node_B,node_I,
    cntrd,edge_length,normal,subtri_centroid,subtri_edge_centroid);

   //cout << " COMPUTED EDGE LENGTH " << edge_length << " normal " << normal[0] << " " << normal[1] << std::endl;
   //cout << " COMPUTED CENTROID: " << subtri_centroid[0] << " " << subtri_centroid[1] << std::endl;
   //cout << " COMPUTED EDGE CENTROID: " << subtri_edge_centroid[0] << " " << subtri_edge_centroid[1] << std::endl;
   // create the mesh object and push it into the mg_mesh internal objects list
   Teuchos::RCP<DICe::mesh::internal_cell_set> internal_cell_set = mesh->get_internal_cell_set();
   const int_t cell_local_id = internal_cell_set->size();
   const int_t cell_global_id = mesh->get_scalar_internal_cell_dist_map()->get_global_element(cell_local_id);
   Teuchos::RCP< Internal_Cell > internal_cell = Teuchos::rcp( new Internal_Cell(cell_local_id,cell_global_id,node_I,subelement));
   internal_cell_set->push_back(internal_cell);
   Teuchos::RCP<DICe::mesh::internal_face_edge_set> internal_face_edge_set = mesh->get_internal_face_edge_set();
   const int_t face_edge_local_id = internal_face_edge_set->size();
   const int_t face_edge_global_id = mesh->get_scalar_face_dist_map()->get_global_element(face_edge_local_id);
   Teuchos::RCP< Internal_Face_Edge > internal_face_edge = Teuchos::rcp( new Internal_Face_Edge(face_edge_local_id, face_edge_global_id,node_I,node_A,subelement));
   internal_face_edge_set->push_back(internal_face_edge);

   // add the cell and edge to the parent subelement:
   subelement->add_deep_relation(internal_face_edge);
   subelement->add_deep_relation(internal_cell);
   // add the target and neighbor node to each other's relations
   node_I->add_deep_relation(node_A);
   node_A->add_deep_relation(node_I);

   MultiField & edge_centroid = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_COORDINATES_FS);
   MultiField & cell_centroid = *mesh->get_field(field_enums::INTERNAL_CELL_COORDINATES_FS);
   MultiField & edge_normal = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_NORMAL_FS);
   MultiField & edge_size = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_SIZE_FS);
   MultiField & cell_size = *mesh->get_field(field_enums::INTERNAL_CELL_SIZE_FS);
   MultiField & subelement_size = *mesh->get_field(field_enums::INITIAL_SUBELEMENT_SIZE_FS);

   edge_size.local_value(face_edge_local_id) = edge_length;
   const scalar_t subelem_size = tri3_area(coords_values,node_A,node_B,node_I);
   const scalar_t internal_cell_size = subelem_size/3.0;
   subelement_size.local_value(subelement->local_id()) = subelem_size;
   cell_size.local_value(cell_local_id) = internal_cell_size;
   for(int_t dim=0;dim<2;++dim)
   {
     edge_centroid.local_value(face_edge_local_id*spa_dim + dim) = subtri_edge_centroid[dim];
     cell_centroid.local_value(cell_local_id*spa_dim+dim) = subtri_centroid[dim];
     edge_normal.local_value(face_edge_local_id*spa_dim + dim) = normal[dim];
   }
}

void
compute_submesh_obj_from_tri3(Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  const scalar_t * parent_centroid,
  scalar_t & edge_length,
  scalar_t * normal,
  scalar_t * subtri_centroid,
  scalar_t * subtri_edge_centroid)
{
  const int_t stride_A = node_A.get()->overlap_local_id();
  const scalar_t Ax = coords_values[stride_A*2+0];
  const scalar_t Ay = coords_values[stride_A*2+1];
  const int_t stride_B = node_B.get()->overlap_local_id();
  const scalar_t Bx = coords_values[stride_B*2+0];
  const scalar_t By = coords_values[stride_B*2+1];
  const int_t stride_I = node_I.get()->overlap_local_id();
  const scalar_t Ix = coords_values[stride_I*2+0];
  const scalar_t Iy = coords_values[stride_I*2+1];

  // compute the mid points of the lines that connect node_I to A, B, and C
  const scalar_t mpIAx = (Ax + Ix) / 2.0;
  const scalar_t mpIAy = (Ay + Iy) / 2.0;

  const scalar_t mpIBx = (Bx + Ix) / 2.0;
  const scalar_t mpIBy = (By + Iy) / 2.0;

  const scalar_t dx = mpIAx - parent_centroid[0];
  const scalar_t dy = parent_centroid[1] - mpIAy;

  edge_length = std::sqrt(dx*dx + dy*dy);
  normal[0] = dy/edge_length;
  normal[1] = dx/edge_length;

  subtri_centroid[0] = (mpIAx + mpIBx + parent_centroid[0] + Ix) / 4.0;
  subtri_centroid[1] = (mpIAy + mpIBy + parent_centroid[1] + Iy) / 4.0;

  subtri_edge_centroid[0] = (mpIAx + parent_centroid[0]) / 2.0;
  subtri_edge_centroid[1] = (mpIAy + parent_centroid[1]) / 2.0;
}


void
compute_centroid_of_tri(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  scalar_t * cntrd)
{
  // get the coords of the nodes:
  const int_t stride_A = node_A.get()->overlap_local_id();
  const scalar_t Ax = coords_values[stride_A*2+0];
  const scalar_t Ay = coords_values[stride_A*2+1];
  const int_t stride_B = node_B.get()->overlap_local_id();
  const scalar_t Bx = coords_values[stride_B*2+0];
  const scalar_t By = coords_values[stride_B*2+1];
  const int_t stride_I = node_I.get()->overlap_local_id();
  const scalar_t Ix = coords_values[stride_I*2+0];
  const scalar_t Iy = coords_values[stride_I*2+1];

  cntrd[0] = (Ax + Bx + Ix)/3.0;
  cntrd[1] = (Ay + By + Iy)/3.0;
}

void
compute_centroid_of_tet(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  scalar_t * cntrd)
{
  // get the coords of the nodes:
  //FIXME: DO WE NEED THE .get() calls on each node?
  const int_t stride_A = node_A.get()->overlap_local_id();
  const scalar_t Ax = coords_values[stride_A*3+0];
  const scalar_t Ay = coords_values[stride_A*3+1];
  const scalar_t Az = coords_values[stride_A*3+2];
  const int_t stride_B = node_B.get()->overlap_local_id();
  const scalar_t Bx = coords_values[stride_B*3+0];
  const scalar_t By = coords_values[stride_B*3+1];
  const scalar_t Bz = coords_values[stride_B*3+2];
  const int_t stride_C = node_C.get()->overlap_local_id();
  const scalar_t Cx = coords_values[stride_C*3+0];
  const scalar_t Cy = coords_values[stride_C*3+1];
  const scalar_t Cz = coords_values[stride_C*3+2];
  const int_t stride_I = node_I.get()->overlap_local_id();
  const scalar_t Ix = coords_values[stride_I*3+0];
  const scalar_t Iy = coords_values[stride_I*3+1];
  const scalar_t Iz = coords_values[stride_I*3+2];

  cntrd[0] = (Ax + Bx + Cx + Ix)/4.0;
  cntrd[1] = (Ay + By + Cy + Iy)/4.0;
  cntrd[2] = (Az + Bz + Cz + Iz)/4.0;
}

void
tetra4_sub_elem_edge_legths_and_normals(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> cv_node,
  Teuchos::RCP<DICe::mesh::Subelement> subelement)
{
  const int_t spa_dim = mesh->spatial_dimension();
  MultiField & face_centroid = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_COORDINATES_FS);
  MultiField & cell_centroid = *mesh->get_field(field_enums::INTERNAL_CELL_COORDINATES_FS);
  MultiField & face_normal = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_NORMAL_FS);
  MultiField & face_size = *mesh->get_field(field_enums::INTERNAL_FACE_EDGE_SIZE_FS);
  MultiField & cell_size = *mesh->get_field(field_enums::INTERNAL_CELL_SIZE_FS);
  MultiField & subelement_size = *mesh->get_field(field_enums::INITIAL_SUBELEMENT_SIZE_FS);

  const DICe::mesh::connectivity_vector & connectivity = *subelement->connectivity();
  const int_t num_nodes = connectivity.size();
  TEUCHOS_TEST_FOR_EXCEPTION(num_nodes!=4,std::invalid_argument,"Connectivity does not have the right number of nodes.");

  // determine which local node the cv_node represents:
  int_t cv_local_node_index = -1;
  for(int_t i=0;i<4;++i)
    if(connectivity[i]->global_id()==cv_node->global_id()) cv_local_node_index = i;
  TEUCHOS_TEST_FOR_EXCEPTION(cv_local_node_index==-1,std::invalid_argument,"Node was not found in the element.");

  Teuchos::RCP<DICe::mesh::Node> node_A, node_B, node_C;
  const Teuchos::RCP<DICe::mesh::Node> node_I = connectivity[cv_local_node_index];
  int_t node_index_table[3][3] = {{-1,-1,-1},{-1,-1,-1},{-1,-1,-1}};

  // if its the 0th node, compute three faces, otherwise only compute the right face
  // This is the only way to get the 6 internal faces of a tet without repeating computations.
  if(cv_local_node_index==0)
  {
    node_index_table[0][0] = 1; node_index_table[0][1] = 2; node_index_table[0][2] = 3;
    node_index_table[1][0] = 2; node_index_table[1][1] = 3; node_index_table[1][2] = 1;
    node_index_table[2][0] = 3; node_index_table[2][1] = 1; node_index_table[2][2] = 2;
  }
  else if(cv_local_node_index==1)
  {
    node_index_table[0][0] = 0; node_index_table[0][1] = 3; node_index_table[0][2] = 2;
  }
  else if(cv_local_node_index==2)
  {
    node_index_table[0][0] = 0; node_index_table[0][1] = 1; node_index_table[0][2] = 3;
  }
  else if(cv_local_node_index==3)
  {
    node_index_table[0][0] = 0; node_index_table[0][1] = 2; node_index_table[0][2] = 1;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"Bad switch in if statement for figuring out the local node id.");

  for(int_t i=0;i<3;++i)
  {
    if(node_index_table[i][0]==-1) continue;
    node_A = connectivity[node_index_table[i][0]];
    node_B = connectivity[node_index_table[i][1]];
    node_C = connectivity[node_index_table[i][2]];
    //cout << " I: " << node_I->global_id() <<  " A: " << node_A->global_id() << " B: " << node_B->global_id() << " C: " << node_C->global_id() << std::endl;
    scalar_t face_area;
    scalar_t normal[3];
    scalar_t subtet_centroid[3];
    scalar_t subtet_face_centroid[3];

    scalar_t cntrd[] = {0.0, 0.0, 0.0};
    compute_centroid_of_tet(coords_values,node_A,node_B,node_C,node_I,&cntrd[0]);
    const scalar_t subelem_size = tetra4_volume(coords_values,node_A,node_B,node_C,node_I);//cell_size_values[element->local_id()]/4.0;
    const scalar_t internal_cell_size = subelem_size/4.0;
    subelement_size.local_value(subelement->local_id()) = subelem_size;

    compute_submesh_obj_from_tet4(coords_values,
      node_A,node_B,node_C,node_I,&cntrd[0],
      face_area,normal,subtet_centroid,subtet_face_centroid);
    //cout << " TARGET_ID: " << node_I->global_id() << " NEIGHBOR_ID: " << node_B->global_id() << std::endl;
    //cout << " COMPUTED SUBTET_COORDINATES: " << subtet_centroid[0] << " " << subtet_centroid[1] << " " << subtet_centroid[2] << std::endl;
    //cout << " COMPUTED FACE_COORDINATES: " << subtet_face_centroid[0] << " " << subtet_face_centroid[1] << " " << subtet_face_centroid[2] << std::endl;
    //cout << " COMPUTED FACE_AREA: " << face_area << std::endl;
    //cout << " COMPUTED FACE_NORMAL " << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;

    // create the mesh object and push it into the mg_mesh internal objects list
    if(i==0)
    {
      Teuchos::RCP<DICe::mesh::internal_cell_set> internal_cell_set = mesh->get_internal_cell_set();
      const int_t cell_local_id = internal_cell_set->size();
      const int_t cell_global_id = mesh->get_scalar_internal_cell_dist_map()->get_global_element(cell_local_id);
      Teuchos::RCP< Internal_Cell > internal_cell = Teuchos::rcp( new Internal_Cell(cell_local_id,cell_global_id,node_I,subelement));
      internal_cell_set->push_back(internal_cell);
      // add the cell to the parent subelement:
      subelement->add_deep_relation(internal_cell);

      cell_size.local_value(cell_local_id) = internal_cell_size;
      for(int_t dim=0;dim<3;++dim)
        cell_centroid.local_value(cell_local_id*spa_dim + dim) = subtet_centroid[dim];
    }
    Teuchos::RCP<DICe::mesh::internal_face_edge_set> internal_face_edge_set = mesh->get_internal_face_edge_set();
    const int_t face_edge_local_id = internal_face_edge_set->size();
    const int_t face_edge_global_id = mesh->get_scalar_face_dist_map()->get_global_element(face_edge_local_id);
    Teuchos::RCP< Internal_Face_Edge > internal_face_edge = Teuchos::rcp( new Internal_Face_Edge(face_edge_local_id,face_edge_global_id,node_I,node_B,subelement));
    internal_face_edge_set->push_back(internal_face_edge);
    // add the face to the subelement relations
    subelement->add_deep_relation(internal_face_edge);
    // add the target and neighbor node to each other's relations
    node_I->add_deep_relation(node_B);
    node_B->add_deep_relation(node_I);

    face_size.local_value(face_edge_local_id) = face_area;
    for(int_t dim=0;dim<3;++dim)
    {
      face_centroid.local_value(face_edge_local_id*spa_dim + dim) = subtet_face_centroid[dim];
      face_normal.local_value(face_edge_local_id*spa_dim + dim) = normal[dim];
    }
  }
}

void
compute_submesh_obj_from_tet4(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  const scalar_t * parent_centroid,
  scalar_t & face_area,
  scalar_t * normal,
  scalar_t * subtet_centroid,
  scalar_t * subtet_face_centroid)
{
  const int_t stride_A = node_A.get()->overlap_local_id();
  const scalar_t Ax = coords_values[stride_A*3+0];
  const scalar_t Ay = coords_values[stride_A*3+1];
  const scalar_t Az = coords_values[stride_A*3+2];
  const int_t stride_B = node_B.get()->overlap_local_id();
  const scalar_t Bx = coords_values[stride_B*3+0];
  const scalar_t By = coords_values[stride_B*3+1];
  const scalar_t Bz = coords_values[stride_B*3+2];
  const int_t stride_C = node_C.get()->overlap_local_id();
  const scalar_t Cx = coords_values[stride_C*3+0];
  const scalar_t Cy = coords_values[stride_C*3+1];
  const scalar_t Cz = coords_values[stride_C*3+2];
  const int_t stride_I = node_I.get()->overlap_local_id();
  const scalar_t Ix = coords_values[stride_I*3+0];
  const scalar_t Iy = coords_values[stride_I*3+1];
  const scalar_t Iz = coords_values[stride_I*3+2];

  // compute the mid points of the lines that connect node_I to A, B, and C
  const scalar_t mpIAx = (Ax + Ix) / 2.0;
  const scalar_t mpIAy = (Ay + Iy) / 2.0;
  const scalar_t mpIAz = (Az + Iz) / 2.0;

  const scalar_t mpIBx = (Bx + Ix) / 2.0;
  const scalar_t mpIBy = (By + Iy) / 2.0;
  const scalar_t mpIBz = (Bz + Iz) / 2.0;
  const scalar_t mpIB[] = {mpIBx, mpIBy, mpIBz};

  const scalar_t mpICx = (Cx + Ix) / 2.0;
  const scalar_t mpICy = (Cy + Iy) / 2.0;
  const scalar_t mpICz = (Cz + Iz) / 2.0;

  // compute the centroid of each face
  const scalar_t cIABx = (Ix+Ax+Bx) / 3.0;
  const scalar_t cIABy = (Iy+Ay+By) / 3.0;
  const scalar_t cIABz = (Iz+Az+Bz) / 3.0;
  const scalar_t cIAB[] = {cIABx, cIABy, cIABz};

  const scalar_t cIACx = (Ix+Ax+Cx) / 3.0;
  const scalar_t cIACy = (Iy+Ay+Cy) / 3.0;
  const scalar_t cIACz = (Iz+Az+Cz) / 3.0;

  const scalar_t cIBCx = (Ix+Cx+Bx) / 3.0;
  const scalar_t cIBCy = (Iy+Cy+By) / 3.0;
  const scalar_t cIBCz = (Iz+Cz+Bz) / 3.0;
  const scalar_t cIBC[] = {cIBCx, cIBCy, cIBCz};

  subtet_centroid[0] = (Ix + mpIAx + cIABx + mpIBx + parent_centroid[0] + cIBCx + cIACx + mpICx)/8.0;
  subtet_centroid[1] = (Iy + mpIAy + cIABy + mpIBy + parent_centroid[1] + cIBCy + cIACy + mpICy)/8.0;
  subtet_centroid[2] = (Iz + mpIAz + cIABz + mpIBz + parent_centroid[2] + cIBCz + cIACz + mpICz)/8.0;

  subtet_face_centroid[0] = (cIABx + parent_centroid[0] + cIBCx + mpIBx)/4.0;
  subtet_face_centroid[1] = (cIABy + parent_centroid[1] + cIBCy + mpIBy)/4.0;
  subtet_face_centroid[2] = (cIABz + parent_centroid[2] + cIBCz + mpIBz)/4.0;

  // compute the areas of the three four node resulting faces (made up of two three node triangles each)
  scalar_t test_normal[] = {0.0, 0.0, 0.0};
  face_area = 0.0;
  face_area += 0.5*cross3d_with_normal(&cIAB[0],&mpIB[0],&parent_centroid[0], &normal[0]);
  face_area += 0.5*cross3d_with_normal(&cIBC[0],&parent_centroid[0],&mpIB[0], &test_normal[0]);

  // The normals of both triangles that make up the square face on the domain should be the same
  //stringstream oss;
  //oss << normal[0] << " " << test_normal[0] << " " << normal[1] << " " << test_normal[1] << " " << normal[2] << " " << test_normal[2] << std::endl;
  // TODO turn off this checking once some testing has been done
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(normal[0]-test_normal[0])>1E-5,
    std::logic_error,"Normals that should be equal are not. ");// + oss.str());
}

scalar_t
tri3_area(Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C)
{
  scalar_t A[2], B[2], C[2];
  A[0] = coords_values[node_A.get()->overlap_local_id()*2+0];
  A[1] = coords_values[node_A.get()->overlap_local_id()*2+1];
  B[0] = coords_values[node_B.get()->overlap_local_id()*2+0];
  B[1] = coords_values[node_B.get()->overlap_local_id()*2+1];
  C[0] = coords_values[node_C.get()->overlap_local_id()*2+0];
  C[1] = coords_values[node_C.get()->overlap_local_id()*2+1];

  scalar_t cross_prod = cross(&A[0],&B[0],&C[0]);
  return 0.5 * std::abs(cross_prod);
}

scalar_t
tetra4_volume(Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C,
  Teuchos::RCP<DICe::mesh::Node> node_D)
{
  scalar_t a[4*4];
  a[0+0*4] = coords_values[node_A.get()->overlap_local_id()*3+0];
  a[0+1*4] = coords_values[node_B.get()->overlap_local_id()*3+1];
  a[0+2*4] = coords_values[node_C.get()->overlap_local_id()*3+2];
  a[0+3*4] = coords_values[node_D.get()->overlap_local_id()*3+0];
  a[1+0*4] = coords_values[node_A.get()->overlap_local_id()*3+1];
  a[1+1*4] = coords_values[node_B.get()->overlap_local_id()*3+2];
  a[1+2*4] = coords_values[node_C.get()->overlap_local_id()*3+0];
  a[1+3*4] = coords_values[node_D.get()->overlap_local_id()*3+1];
  a[2+0*4] = coords_values[node_A.get()->overlap_local_id()*3+2];
  a[2+1*4] = coords_values[node_B.get()->overlap_local_id()*3+0];
  a[2+2*4] = coords_values[node_C.get()->overlap_local_id()*3+1];
  a[2+3*4] = coords_values[node_D.get()->overlap_local_id()*3+2];
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  const scalar_t det = determinant_4x4(&a[0]);
  return std::abs(det) / 6.0;
}


void
create_cell_size_and_radius(Teuchos::RCP<Mesh> mesh)
{
  if(mesh->cell_sizes_are_initialized())return;
  // create the cell_coords, cell_radius and cell_size fields:
  mesh->create_field(field_enums::INITIAL_CELL_SIZE_FS);
  mesh->create_field(field_enums::INITIAL_CELL_RADIUS_FS);

  Teuchos::RCP<MultiField > coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);
  Teuchos::ArrayRCP<const scalar_t> coords_values = coords->get_1d_view();
  DICe::mesh::element_set::const_iterator elem_it = mesh->get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    // switch on element type
    const Teuchos::RCP<DICe::mesh::Element> elem = *elem_it;
    // switch on element type
    const Base_Element_Type elem_type = mesh->get_block_type_map()->find(elem.get()->block_id())->second;
    if(elem_type == HEX8)
      hex8_volume_radius(mesh,coords_values,elem);
    else if(elem_type == TETRA4 || elem_type == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
      tetra4_volume_radius(mesh,coords_values,elem);
    else if(elem_type == PYRAMID5)
      pyramid5_volume_radius(mesh,coords_values,elem);
    else if(elem_type == QUAD4)
      quad4_area_radius(mesh,coords_values,elem);
    else if(elem_type == TRI3)
      tri3_area_radius(mesh,coords_values,elem);
    else
    {
      std::stringstream oss;
      oss << "create_cell_size_and_radius(): unknown element type: " << tostring(elem_type) << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
  }
  mesh->set_cell_sizes_are_initialized();
}


void
hex8_volume_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element)
{
  MultiField & cell_size = *mesh->get_field(field_enums::INITIAL_CELL_SIZE_FS);
  MultiField & cell_radius = *mesh->get_field(field_enums::INITIAL_CELL_RADIUS_FS);
  const DICe::mesh::connectivity_vector & connectivity = *element.get()->connectivity();

  // this is translated from the sphgen3d fortran code
  const int_t num_nodes = connectivity.size();
  if(num_nodes!=8)
  {
    std::stringstream oss;
    oss << "hex8_volume_radius(): the connectivity does not have the right number of nodes: " << num_nodes << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }

  scalar_t x[num_nodes];
  for(int_t i=0;i<num_nodes;++i) x[i]=0.0;
  scalar_t y[num_nodes];
  for(int_t i=0;i<num_nodes;++i) y[i]=0.0;
  scalar_t z[num_nodes];
  for(int_t i=0;i<num_nodes;++i) z[i]=0.0;

  //std::stringstream oss;
  //oss << " ELEMENT: " << element->global_id() << std::endl;
  for(size_t node_it=0;node_it<connectivity.size();++node_it)
  {
    const Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
    const int_t stride = node.get()->overlap_local_id();
    x[node_it] = coords_values[stride*3+0];
    y[node_it] = coords_values[stride*3+1];
    z[node_it] = coords_values[stride*3+2];
    //oss << "   coords " << node_it << ": " << x[node_it] << " " << y[node_it] << " " << z[node_it] << std::endl;
  }
  //cout << oss.str();

  const scalar_t Z24 = z[1] - z[3];
  const scalar_t Z52 = z[4] - z[1];
  const scalar_t Z45 = z[3] - z[4];
  const scalar_t G1 = ( y[1]*(z[5]-z[2]-Z45) + y[2]*Z24 + y[3]*(z[2]-z[7]-Z52) + y[4]*(z[7]-z[5]-Z24) + y[5]*Z52 + y[7]*Z45 ) / 12.0;
  const scalar_t Z31 = z[2] - z[0];
  const scalar_t Z63 = z[5] - z[2];
  const scalar_t Z16 = z[0] - z[5];
  const scalar_t G2 = ( y[2]*(z[6]-z[3]-Z16) + y[3]*Z31 + y[0]*(z[3]-z[4]-Z63) + y[5]*(z[4]-z[6]-Z31) + y[6]*Z63 + y[4]*Z16 ) / 12.0;
  const scalar_t Z42 = z[3] - z[1];
  const scalar_t Z74 = z[6] - z[3];
  const scalar_t Z27 = z[1] - z[6];
  const scalar_t G3 = ( y[3]*(z[7]-z[0]-Z27) + y[0]*Z42 + y[1]*(z[0]-z[5]-Z74) + y[6]*(z[5]-z[7]-Z42) + y[7]*Z74 + y[5]*Z27 ) / 12.0;
  const scalar_t Z13 = z[0] - z[2];
  const scalar_t Z81 = z[7] - z[0];
  const scalar_t Z38 = z[2] - z[7];
  const scalar_t  G4 = ( y[0]*(z[4]-z[1]-Z38) + y[1]*Z13 + y[2]*(z[1]-z[6]-Z81) + y[7]*(z[6]-z[4]-Z13) + y[4]*Z81 + y[6]*Z38 ) / 12.0;
  const scalar_t Z86 = z[7] - z[5];
  const scalar_t Z18 = z[0] - z[7];
  const scalar_t Z61 = z[5] - z[0];
  const scalar_t G5 = ( y[7]*(z[3]-z[6]-Z61) + y[6]*Z86 + y[5]*(z[6]-z[1]-Z18) + y[0]*(z[1]-z[3]-Z86) + y[3]*Z18 + y[1]*Z61 ) / 12.0;
  const scalar_t Z57 = z[4] - z[6];
  const scalar_t Z25 = z[1] - z[4];
  const scalar_t Z72 = z[6] - z[1];
  const scalar_t G6 = ( y[4]*(z[0]-z[7]-Z72) + y[7]*Z57 + y[6]*(z[7]-z[2]-Z25) + y[1]*(z[2]-z[0]-Z57) + y[0]*Z25 + y[2]*Z72 ) / 12.0;
  const scalar_t Z68 = z[5] - z[7];
  const scalar_t Z36 = z[2] - z[5];
  const scalar_t Z83 = z[7] - z[2];
  const scalar_t G7 = ( y[5]*(z[1]-z[4]-Z83) + y[4]*Z68 + y[7]*(z[4]-z[3]-Z36) + y[2]*(z[3]-z[1]-Z68) + y[1]*Z36 + y[3]*Z83 ) / 12.0;
  const scalar_t Z75 = z[6] - z[4];
  const scalar_t Z47 = z[3] - z[6];
  const scalar_t Z54 = z[4] - z[3];
  const scalar_t G8 = ( y[6]*(z[2]-z[5]-Z54) + y[5]*Z75 + y[4]*(z[5]-z[0]-Z47) + y[3]*(z[0]-z[2]-Z75) + y[2]*Z47 + y[0]*Z54 ) / 12.0;
  const scalar_t volume = x[0] * G1 + x[1] * G2 + x[2] * G3 + x[3] * G4 + x[4] * G5 + x[5] * G6 + x[6] * G7 + x[7] * G8;
  const scalar_t radius = std::pow(volume,1.0/3.0) * 0.5;

  cell_size.local_value(element.get()->local_id()) = volume;
  cell_radius.local_value(element.get()->local_id()) = radius;
}

void
tetra4_volume_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element)
{
  MultiField & cell_size = *mesh->get_field(field_enums::INITIAL_CELL_SIZE_FS);
  MultiField & cell_radius = *mesh->get_field(field_enums::INITIAL_CELL_RADIUS_FS);
  const DICe::mesh::connectivity_vector & connectivity = *element.get()->connectivity();

  // this is translated from the sphgen3d fortran code
  const int_t num_nodes = connectivity.size();
  if(num_nodes!=4)
  {
    std::stringstream oss;
    oss << "tetra4_volume_radius(): the connectivity does not have the right number of nodes: " << num_nodes << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  const Teuchos::RCP<DICe::mesh::Node> node_A = connectivity[0];
  const Teuchos::RCP<DICe::mesh::Node> node_B = connectivity[1];
  const Teuchos::RCP<DICe::mesh::Node> node_C = connectivity[2];
  const Teuchos::RCP<DICe::mesh::Node> node_D = connectivity[3];
  const scalar_t volume = tetra4_volume(coords_values,node_A,node_B,node_C,node_D);
  scalar_t radius = std::pow(volume,1.0/3.0) * 0.5;

  cell_size.local_value(element.get()->local_id()) = volume;
  cell_radius.local_value(element.get()->local_id()) = radius;
}

void
pyramid5_volume_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element)
{
  MultiField & cell_size = *mesh->get_field(field_enums::INITIAL_CELL_SIZE_FS);
  MultiField & cell_radius = *mesh->get_field(field_enums::INITIAL_CELL_RADIUS_FS);
  const DICe::mesh::connectivity_vector & connectivity = *element.get()->connectivity();

  const int_t num_nodes = connectivity.size();
  if(num_nodes!=5)
  {
    std::stringstream oss;
    oss << "pyramid5_volume_radius(): the connectivity does not have the right number of nodes: " << num_nodes << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }

  scalar_t A[3], B[3], C[3], D[3], E[3];
  Teuchos::RCP<DICe::mesh::Node> node;

  node = connectivity[0];
  int_t stride = node.get()->overlap_local_id();
  A[0] = coords_values[stride*3+0];
  A[1] = coords_values[stride*3+1];
  A[2] = coords_values[stride*3+2];
  node = connectivity[1];
  stride = node.get()->overlap_local_id();
  B[0] = coords_values[stride*3+0];
  B[1] = coords_values[stride*3+1];
  B[2] = coords_values[stride*3+2];
  node = connectivity[2];
  stride = node.get()->overlap_local_id();
  C[0] = coords_values[stride*3+0];
  C[1] = coords_values[stride*3+1];
  C[2] = coords_values[stride*3+2];
  node = connectivity[3];
  stride = node.get()->overlap_local_id();
  D[0] = coords_values[stride*3+0];
  D[1] = coords_values[stride*3+1];
  D[2] = coords_values[stride*3+2];
  node = connectivity[4];
  stride = node.get()->overlap_local_id();
  E[0] = coords_values[stride*3+0];
  E[1] = coords_values[stride*3+1];
  E[2] = coords_values[stride*3+2];

  // split into two triangles and sum cross products to get the base area
  scalar_t area = 0.0;
  scalar_t cross_prod = cross3d(&A[0],&B[0],&D[0]);
  area += 0.5 * std::abs(cross_prod);
  cross_prod = cross3d(&C[0],&B[0],&D[0]);
  area += 0.5 * std::abs(cross_prod);


  // find the shortest distance from the top of the pyramid to the base
  scalar_t height = 0.0;

  // compute the coefficients of the base plane equation:
  scalar_t coeff_a = A[1] * (B[2]-C[2]) + B[1] * (C[2]-A[2]) + C[1] * (A[2]-B[2]);
  scalar_t coeff_b = A[2] * (B[0]-C[0]) + B[2] * (C[0]-A[0]) + C[2] * (A[0]-B[0]);
  scalar_t coeff_c = A[0] * (B[1]-C[1]) + B[0] * (C[1]-A[1]) + C[0] * (A[1]-B[1]);
  scalar_t coeff_d = -A[0] * (B[1]*C[2] - C[1]*B[2])
                        -B[0] * (C[1]*A[2] - A[1]*C[2])
                        -C[0] * (A[1]*B[2] - B[1]*A[2]);

  height = ( coeff_a * E[0] + coeff_b * E[1] + coeff_c * E[2]  + coeff_d ) / std::sqrt( coeff_a * coeff_a + coeff_b * coeff_b + coeff_c * coeff_c );
  height = std::abs(height);
  //cout << "Element: " << element.global_id() << " area " << area << " coeff_a " << coeff_a << " coeff_b " << coeff_b << " coeff_c " << coeff_c  << " coeff_d " << coeff_d << " hegiht " << height << std::endl;

  scalar_t volume = 1.0/3.0 * area * height;
  const scalar_t radius = std::pow(volume,1.0/3.0) * 0.5;

  cell_size.local_value(element.get()->local_id()) = volume;
  cell_radius.local_value(element.get()->local_id()) = radius;
}

void
quad4_area_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element)
{
  MultiField & cell_size = *mesh->get_field(field_enums::INITIAL_CELL_SIZE_FS);
  MultiField & cell_radius = *mesh->get_field(field_enums::INITIAL_CELL_RADIUS_FS);
  const DICe::mesh::connectivity_vector & connectivity = *element.get()->connectivity();

  const int_t num_nodes = connectivity.size();
  if(num_nodes!=4)
  {
    std::stringstream oss;
    oss << "quad4_area_radius(): the connectivity does not have the right number of nodes: " << num_nodes << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }

  scalar_t A[2], B[2], C[2], D[2];
  Teuchos::RCP<DICe::mesh::Node> node;

  node = connectivity[0];
  int_t stride = node.get()->overlap_local_id();
  A[0] = coords_values[stride*2+0];
  A[1] = coords_values[stride*2+1];
  node = connectivity[1];
  stride = node.get()->overlap_local_id();
  B[0] = coords_values[stride*2+0];
  B[1] = coords_values[stride*2+1];
  node = connectivity[2];
  stride = node.get()->overlap_local_id();
  C[0] = coords_values[stride*2+0];
  C[1] = coords_values[stride*2+1];
  node = connectivity[3];
  stride = node.get()->overlap_local_id();
  D[0] = coords_values[stride*2+0];
  D[1] = coords_values[stride*2+1];

  scalar_t area = 0.0;

  // split into two triangles and sum cross products
  scalar_t cross_prod = cross(&A[0],&B[0],&D[0]);
  area += 0.5 * std::abs(cross_prod);
  cross_prod = cross(&C[0],&B[0],&D[0]);
  area += 0.5 * std::abs(cross_prod);

  const scalar_t radius = std::sqrt(area/M_PI);

  cell_size.local_value(element.get()->local_id()) = area;
  cell_radius.local_value(element.get()->local_id()) = radius;
}

void
tri3_area_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element)
{
  MultiField & cell_size = *mesh->get_field(field_enums::INITIAL_CELL_SIZE_FS);
  MultiField & cell_radius = *mesh->get_field(field_enums::INITIAL_CELL_RADIUS_FS);
  const DICe::mesh::connectivity_vector & connectivity = *element.get()->connectivity();

  const int_t num_nodes = connectivity.size();
  if(num_nodes!=3)
  {
    std::stringstream oss;
    oss << "tri3_area_radius(): the connectivity does not have the right number of nodes: " << num_nodes << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  Teuchos::RCP<DICe::mesh::Node> node_A, node_B, node_C;
  node_A = connectivity[0];
  node_B = connectivity[1];
  node_C = connectivity[2];

  const scalar_t area = tri3_area(coords_values,node_A,node_B,node_C);
  const scalar_t radius = std::sqrt(area/M_PI);

  cell_size.local_value(element.get()->local_id()) = area;
  cell_radius.local_value(element.get()->local_id()) = radius;
}

} // mesh
} // DICe
