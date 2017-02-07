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

#include <DICe_Decomp.h>
#include <DICe_Parser.h>
#include <DICe_ParameterUtilities.h>
#include <DICe_Cine.h>
#include <DICe_PostProcessor.h>

#include <Teuchos_oblackholestream.hpp>

namespace DICe {

Decomp::Decomp(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params):
    num_global_subsets_(0),
    image_width_(-1),
    image_height_(-1){
  comm_ = Teuchos::rcp(new MultiField_Comm());

  std::vector<std::string> image_files;
  std::vector<std::string> stereo_image_files;
  DICe::decipher_image_file_names(input_params,image_files,stereo_image_files);

  // set up the positions of all the mesh points or subsets
  Teuchos::ArrayRCP<scalar_t> subset_centroids_x;
  Teuchos::ArrayRCP<scalar_t> subset_centroids_y;
  Teuchos::RCP<std::vector<int_t> > neighbor_ids;
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids;
  populate_global_coordinate_vector(subset_centroids_x,subset_centroids_y,neighbor_ids,obstructing_subset_ids,image_files[0],input_params,correlation_params);
  initialize(subset_centroids_x,subset_centroids_y,neighbor_ids,obstructing_subset_ids,correlation_params);
}

Decomp::Decomp(Teuchos::ArrayRCP<scalar_t> subset_centroids_x,
  Teuchos::ArrayRCP<scalar_t> subset_centroids_y,
  Teuchos::RCP<std::vector<int_t> > neighbor_ids,
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params):
    num_global_subsets_(0),
    image_width_(-1),
    image_height_(-1){
  TEUCHOS_TEST_FOR_EXCEPTION(subset_centroids_x.size()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(subset_centroids_x.size()!=subset_centroids_y.size(),std::runtime_error,"");
  comm_ = Teuchos::rcp(new MultiField_Comm());
  num_global_subsets_ = subset_centroids_x.size();
  initialize(subset_centroids_x,subset_centroids_y,neighbor_ids,obstructing_subset_ids,correlation_params);
}

void
Decomp::initialize(Teuchos::ArrayRCP<scalar_t> subset_centroids_x,
  Teuchos::ArrayRCP<scalar_t> subset_centroids_y,
  Teuchos::RCP<std::vector<int_t> > neighbor_ids,
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params){

  DEBUG_MSG("Decomp::initialize(): num global subsets: " << num_global_subsets_);

  // create an evenly split map to start:
  id_decomp_map_ = Teuchos::rcp(new MultiField_Map(num_global_subsets_,0,*comm_));

  this_proc_gid_order_ = std::vector<int_t>(id_decomp_map_->get_num_local_elements(),-1);
  for(int_t i=0;i<id_decomp_map_->get_num_local_elements();++i)
    this_proc_gid_order_[i] = id_decomp_map_->get_global_element(i);

  // if there are blocking subsets, they need to be on the same processor and put in order:
  create_obstruction_dist_map(obstructing_subset_ids);

  // if there are seeds involved, the decomp must respect these
  create_seed_dist_map(neighbor_ids,obstructing_subset_ids,correlation_params);

  // lastly, create the overlap vectors of points needed for computing neighbor values, etc.
  const bool is_parallel = comm_->get_size() > 1;
  // determine the max strain window size:
  // TODO find a way to make sure all strain window sizes are captured here
  //      as it is not, it only checks for NLVC and VSG sizes
  scalar_t max_strain_window_size = 0.0;
  scalar_t tmp_strain_window_size = 0.0;
  if(correlation_params!=Teuchos::null){
    if(correlation_params->isParameter(DICe::post_process_vsg_strain)){
      Teuchos::ParameterList vsg_sublist = correlation_params->sublist(DICe::post_process_vsg_strain);
      TEUCHOS_TEST_FOR_EXCEPTION(!vsg_sublist.isParameter(DICe::strain_window_size_in_pixels),std::runtime_error,"");
      scalar_t tmp_strain_window_size = vsg_sublist.get<int_t>(DICe::strain_window_size_in_pixels);
      if(tmp_strain_window_size > max_strain_window_size) max_strain_window_size = tmp_strain_window_size;
    }
    if(correlation_params->isParameter(DICe::post_process_nlvc_strain)){
      Teuchos::ParameterList nlvc_sublist = correlation_params->sublist(DICe::post_process_nlvc_strain);
      TEUCHOS_TEST_FOR_EXCEPTION(!nlvc_sublist.isParameter(DICe::horizon_diameter_in_pixels),std::runtime_error,"");
      tmp_strain_window_size = nlvc_sublist.get<int_t>(DICe::horizon_diameter_in_pixels);
      if(tmp_strain_window_size > max_strain_window_size) max_strain_window_size = tmp_strain_window_size;
    }
  }
  DEBUG_MSG("Decomp::Decomp(): max strain window size " << max_strain_window_size);

  // only do the neighbor searching if neighbors are needed:
  if(max_strain_window_size > 0.0 && is_parallel){
    // collect all the ids local to this processor already:
    std::set<int_t> id_list;
    Teuchos::ArrayView<const int_t> my_gids = id_decomp_map_->get_global_element_list();
    for(int_t i=0;i<my_gids.size();++i){
      id_list.insert(my_gids[i]);
    }
    // Do a neighborhood search for all possible neighbors for post-processors
    Teuchos::RCP<Decomp_Point_Cloud<scalar_t> > point_cloud = Teuchos::rcp(new Decomp_Point_Cloud<scalar_t>());
    point_cloud->pts.resize(num_global_subsets_);
    for(int_t i=0;i<num_global_subsets_;++i){
      point_cloud->pts[i].x = subset_centroids_x[i];
      point_cloud->pts[i].y = subset_centroids_y[i];
    }
    DEBUG_MSG("Decomp::Decomp(): building the kd-tree");
    Teuchos::RCP<decomp_kd_tree_t> kd_tree = Teuchos::rcp(new decomp_kd_tree_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
    kd_tree->buildIndex();
    DEBUG_MSG("Decomp::Decomp(): kd-tree completed");
    std::vector<std::pair<size_t,scalar_t> > ret_matches;
    nanoflann::SearchParams params;
    params.sorted = true; // sort by distance in ascending order
    const scalar_t tiny = 1.0E-5;
    scalar_t neigh_rad_2 = (scalar_t)max_strain_window_size/2.0;
    neigh_rad_2 *= neigh_rad_2;
    neigh_rad_2 += tiny;
    scalar_t query_pt[2];
    for(int_t i=0;i<my_gids.size();++i){
      const int_t gid = my_gids[i];
      assert(gid>=0&&gid<num_global_subsets_);
      query_pt[0] = subset_centroids_x[gid];
      query_pt[1] = subset_centroids_y[gid];
      kd_tree->radiusSearch(&query_pt[0],neigh_rad_2,ret_matches,params);
      for(size_t j=0;j<ret_matches.size();++j){
        id_list.insert(ret_matches[j].first);
      }
    }
    DEBUG_MSG("Decomp::Decomp(): neighbor list constructed");

    // at this point, the max neighbors should be included so create a reduced set of coordinates
    const int_t num_overlap_points = id_list.size();
    neighbor_ids_ = Teuchos::rcp(new std::vector<int_t>(num_overlap_points,-1));
    overlap_coords_x_.resize(num_overlap_points,0.0);
    overlap_coords_y_.resize(num_overlap_points,0.0);
    std::set<int_t>::iterator list_it = id_list.begin();
    std::set<int_t>::iterator list_end = id_list.end();
    std::vector<int_t> id_list_vec(id_list.size(),-1);
    int_t ii=0;
    for(;list_it!=list_end;++list_it){
      overlap_coords_x_[ii] = subset_centroids_x[*list_it];
      overlap_coords_y_[ii] = subset_centroids_y[*list_it];
      if(neighbor_ids!=Teuchos::null)
        (*neighbor_ids_)[ii] = (*neighbor_ids)[*list_it];
      id_list_vec[ii] = *list_it;
      ii++;
    }
    Teuchos::ArrayView<const int_t> id_list_array(&id_list_vec[0],id_list.size());
    id_decomp_overlap_map_ = Teuchos::rcp(new MultiField_Map(-1,id_list_array,0,*comm_));
    DEBUG_MSG("Decomp::Decomp(): coordinate list has been trimmed");
  } // end has strain windows && is parallel
  else{
    // set the neighbor ids
    neighbor_ids_ = Teuchos::rcp(new std::vector<int_t>(id_decomp_map_->get_num_local_elements(),-1));
    overlap_coords_x_.resize(id_decomp_map_->get_num_local_elements(),0.0);
    overlap_coords_y_.resize(id_decomp_map_->get_num_local_elements(),0.0);
    for(int_t i=0;i<id_decomp_map_->get_num_local_elements();++i){
      const int_t gid = id_decomp_map_->get_global_element(i);
      overlap_coords_x_[i] = subset_centroids_x[gid];
      overlap_coords_y_[i] = subset_centroids_y[gid];
      if(neighbor_ids!=Teuchos::null)
        (*neighbor_ids_)[i] = (*neighbor_ids)[gid];
    }
    id_decomp_overlap_map_ = id_decomp_map_;
  }
  DEBUG_MSG("[PROC "<< comm_->get_rank() <<"] num local subsets:    " << id_decomp_map_->get_num_local_elements());
  DEBUG_MSG("[PROC "<< comm_->get_rank() <<"] num overlap subsets:  " << id_decomp_overlap_map_->get_num_local_elements());
}

void
Decomp::create_obstruction_dist_map(Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids){
  if(obstructing_subset_ids==Teuchos::null) return;
  if(obstructing_subset_ids->size()==0) return;

  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  if(proc_id == 0) DEBUG_MSG("Subsets have obstruction dependencies.");
  // set up the groupings of subset ids that have to stay together:
  // Note: this assumes that the obstructions are only one relation deep
  // i.e. the blocking subset cannot itself have a subset that blocks it
  // TODO address this to make it more general
  std::set<int_t> eligible_ids;
  for(int_t i=0;i<num_global_subsets_;++i)
    eligible_ids.insert(i);
  std::vector<std::set<int_t> > obstruction_groups;
  std::map<int_t,int_t> earliest_id_can_appear;
  std::set<int_t> assigned_to_a_group;
  std::map<int_t,std::vector<int_t> >::iterator map_it = obstructing_subset_ids->begin();
  for(;map_it!=obstructing_subset_ids->end();++map_it){
    int_t greatest_subset_id_among_obst = 0;
    for(size_t j=0;j<map_it->second.size();++j)
      if(map_it->second[j] > greatest_subset_id_among_obst) greatest_subset_id_among_obst = map_it->second[j];
    earliest_id_can_appear.insert(std::pair<int_t,int_t>(map_it->first,greatest_subset_id_among_obst));

    if(assigned_to_a_group.find(map_it->first)!=assigned_to_a_group.end()) continue;
    std::set<int_t> dependencies;
    dependencies.insert(map_it->first);
    eligible_ids.erase(map_it->first);
    // gather for all the dependencies for this subset
    for(size_t j=0;j<map_it->second.size();++j){
      dependencies.insert(map_it->second[j]);
      eligible_ids.erase(map_it->second[j]);
    }
    // no search all the other obstruction sets for any ids currently in the dependency list
    std::set<int_t>::iterator dep_it = dependencies.begin();
    for(;dep_it!=dependencies.end();++dep_it){
      std::map<int_t,std::vector<int_t> >::iterator search_it = obstructing_subset_ids->begin();
      for(;search_it!=obstructing_subset_ids->end();++search_it){
        if(assigned_to_a_group.find(search_it->first)!=assigned_to_a_group.end()) continue;
        // if any of the ids are in the current dependency list, add the whole set:
        bool match_found = false;
        if(*dep_it==search_it->first) match_found = true;
        for(size_t k=0;k<search_it->second.size();++k){
          if(*dep_it==search_it->second[k]) match_found = true;
        }
        if(match_found){
          dependencies.insert(search_it->first);
          eligible_ids.erase(search_it->first);
          for(size_t k=0;k<search_it->second.size();++k){
            dependencies.insert(search_it->second[k]);
            eligible_ids.erase(search_it->second[k]);
          }
          // reset the dependency index because more items were added to the list
          dep_it = dependencies.begin();
          // remove this set of obstruction ids since they have already been added to a group
          assigned_to_a_group.insert(search_it->first);
        } // match found
      } // obstruction set
    } // dependency it
    obstruction_groups.push_back(dependencies);
  } // outer obstruction set it
  if(proc_id == 0) DEBUG_MSG("[PROC " << proc_id << "] There are " << obstruction_groups.size() << " obstruction groupings: ");
//  std::stringstream ss;
//  for(size_t i=0;i<obstruction_groups.size();++i){
//    ss << "[PROC " << proc_id << "] Group: " << i << std::endl;
//    std::set<int_t>::iterator j = obstruction_groups[i].begin();
//    for(;j!=obstruction_groups[i].end();++j){
//      ss << "[PROC " << proc_id << "]   id: " << *j << std::endl;
//    }
//  }
//  ss << "[PROC " << proc_id << "] Eligible ids: " << std::endl;
//  for(std::set<int_t>::iterator elig_it=eligible_ids.begin();elig_it!=eligible_ids.end();++elig_it){
//    ss << "[PROC " << proc_id << "]   " << *elig_it << std::endl;
//  }
//  if(proc_id == 0) DEBUG_MSG(ss.str());

  // divy up the obstruction groups among the processors:
  int_t obst_group_gid = 0;
  std::vector<std::set<int_t> > local_subset_gids(num_procs);
  while(obst_group_gid < (int_t)obstruction_groups.size()){
    for(int_t p_id=0;p_id<num_procs;++p_id){
      if(obst_group_gid < (int_t)obstruction_groups.size()){
        //if(p_id==proc_id){
        local_subset_gids[p_id].insert(obstruction_groups[obst_group_gid].begin(),obstruction_groups[obst_group_gid].end());
        //}
        obst_group_gid++;
      }
      else break;
    }
  }
  // assign the rest based on who has the least amount of subsets
  for(std::set<int_t>::iterator elig_it = eligible_ids.begin();elig_it!=eligible_ids.end();++elig_it){
    int_t proc_with_fewest_subsets = 0;
    int_t lowest_num_subsets = num_global_subsets_;
    for(int_t i=0;i<num_procs;++i){
      if((int_t)local_subset_gids[i].size() <= lowest_num_subsets){
        lowest_num_subsets = local_subset_gids[i].size();
        proc_with_fewest_subsets = i;
      }
    }
    local_subset_gids[proc_with_fewest_subsets].insert(*elig_it);
  }
  // order the subset ids so that they respect the dependencies:
  std::vector<int_t> local_gids;
  std::set<int_t>::iterator set_it = local_subset_gids[proc_id].begin();
  for(;set_it!=local_subset_gids[proc_id].end();++set_it){
    // not in the list of subsets with blockers
    if(obstructing_subset_ids->find(*set_it)==obstructing_subset_ids->end()){
      local_gids.push_back(*set_it);
    }
    // in the list of subsets with blockers, but has no blocking ids
    else if(obstructing_subset_ids->find(*set_it)->second.size()==0){
      local_gids.push_back(*set_it);
    }

  }
  set_it = local_subset_gids[proc_id].begin();
  for(;set_it!=local_subset_gids[proc_id].end();++set_it){
    if(obstructing_subset_ids->find(*set_it)!=obstructing_subset_ids->end()){
      if(obstructing_subset_ids->find(*set_it)->second.size()>0){
        TEUCHOS_TEST_FOR_EXCEPTION(earliest_id_can_appear.find(*set_it)==earliest_id_can_appear.end(),
          std::runtime_error,"");
        local_gids.push_back(*set_it);
      }
    }
  }

//  ss.str(std::string());
//  ss.clear();
//  ss << "[PROC " << proc_id << "] Has the following subset ids: " << std::endl;
//  for(size_t i=0;i<local_gids.size();++i){
//    ss << "[PROC " << proc_id << "] " << local_gids[i] <<  std::endl;
//  }
//  DEBUG_MSG(ss.str());

  // copy the ids in order
  this_proc_gid_order_ = std::vector<int_t>(local_gids.size(),-1);
  for(size_t i=0;i<local_gids.size();++i)
    this_proc_gid_order_[i] = local_gids[i];

  // sort the vector so the map is not in execution order
  std::sort(local_gids.begin(),local_gids.end());
  Teuchos::ArrayView<const int_t> lids_grouped_by_obstruction(&local_gids[0],local_gids.size());
  id_decomp_map_ = Teuchos::rcp(new MultiField_Map(num_global_subsets_,lids_grouped_by_obstruction,0,*comm_));
}


void
Decomp::create_seed_dist_map(Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params){
  Initialization_Method init_method = USE_FIELD_VALUES;
  if(correlation_params!=Teuchos::null){
    if(correlation_params->isParameter(DICe::initialization_method)){
      if(correlation_params->isType<std::string>(DICe::initialization_method)){
        std::string init_string = correlation_params->get<std::string>(DICe::initialization_method,"USE_FIELD_VALUES");
        init_method = DICe::string_to_initialization_method(init_string);
      }
      else{
        init_method = correlation_params->get<DICe::Initialization_Method>(DICe::initialization_method);
      }
    }
  }
  if(init_method!=USE_NEIGHBOR_VALUES&&init_method!=USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY) return;
  // distribution according to seeds map (one-to-one, not all procs have entries)
  // If the initialization method is USE_NEIGHBOR_VALUES or USE_NEIGHBOR_VALUES_FIRST_STEP_ONLY, the
  // first step has to have a special map that keeps all subsets that use a particular seed
  // on the same processor (the parallelism is limited to the number of seeds).
  const int_t proc_id = comm_->get_rank();
  const int_t num_procs = comm_->get_size();

  if(neighbor_ids!=Teuchos::null){
    // catch the case that this is an TRACKING_ROUTINE run, but seed values were specified for
    // the individual subsets. In that case, the seed map is not necessary because there are
    // no initializiation dependencies among subsets, but the seed map will still be used since it
    // will be activated when seeds are specified for a subset.
    if(obstructing_subset_ids!=Teuchos::null){
      if(obstructing_subset_ids->size()>0){
        bool print_warning = false;
        for(size_t i=0;i<neighbor_ids->size();++i){
          if((*neighbor_ids)[i]!=-1) print_warning = true;
        }
        if(print_warning && proc_id==0){
          std::cout << "*** Waring: Seed values were specified for an analysis with obstructing subsets. " << std::endl;
          std::cout << "            These values will be used to initialize subsets for which a seed has been specified, but the seed map " << std::endl;
          std::cout << "            will be set to the distributed map because grouping subsets by obstruction trumps seed ordering." << std::endl;
          std::cout << "            Seed dependencies between neighbors will not be enforced." << std::endl;
        }
        return;
      }
    }
    std::vector<int_t> local_subset_gids_grouped_by_roi;
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbor_ids->size()!=num_global_subsets_,std::runtime_error,"");
    std::vector<int_t> this_group_gids;
    std::vector<std::vector<int_t> > seed_groupings;
    std::vector<std::vector<int_t> > local_seed_groupings;
    for(int_t i=num_global_subsets_-1;i>=0;--i){
      this_group_gids.push_back(i);
      // if this subset is a seed, break this grouping and insert it in the set
      if((*neighbor_ids)[i]==-1){
        seed_groupings.push_back(this_group_gids);
        this_group_gids.clear();
      }
    }
    // TODO order the sets by their sizes and load balance better:
    // divy up the seed_groupings round-robin style:
    int_t group_gid = 0;
    int_t local_total_id_list_size = 0;
    while(group_gid < (int_t)seed_groupings.size()){
      // reverse the order so the subsets are computed from the seed out
      for(int_t p_id=0;p_id<num_procs;++p_id){
        if(group_gid < (int_t)seed_groupings.size()){
          if(p_id==proc_id){
            std::reverse(seed_groupings[group_gid].begin(), seed_groupings[group_gid].end());
            local_seed_groupings.push_back(seed_groupings[group_gid]);
            local_total_id_list_size += seed_groupings[group_gid].size();
          }
          group_gid++;
        }
        else break;
      }
    }
    DEBUG_MSG("[PROC " << proc_id << "] Has " << local_seed_groupings.size() << " local seed grouping(s)");
    for(size_t i=0;i<local_seed_groupings.size();++i){
      DEBUG_MSG("[PROC " << proc_id << "] local group id: " << i);
      for(size_t j=0;j<local_seed_groupings[i].size();++j){
        DEBUG_MSG("[PROC " << proc_id << "] gid: " << local_seed_groupings[i][j] );
      }
    }
    // concat local subset ids:
    local_subset_gids_grouped_by_roi.reserve(local_total_id_list_size);
    for(size_t i=0;i<local_seed_groupings.size();++i){
      local_subset_gids_grouped_by_roi.insert( local_subset_gids_grouped_by_roi.end(),
        local_seed_groupings[i].begin(),
        local_seed_groupings[i].end());
    }

    // copy the ids in order
    this_proc_gid_order_ = std::vector<int_t>(local_subset_gids_grouped_by_roi.size(),-1);
    for(size_t i=0;i<local_subset_gids_grouped_by_roi.size();++i)
      this_proc_gid_order_[i] = local_subset_gids_grouped_by_roi[i];

    // sort the vector so the map is not in execution order
    std::sort(local_subset_gids_grouped_by_roi.begin(),local_subset_gids_grouped_by_roi.end());
    Teuchos::ArrayView<const int_t> lids_grouped_by_roi(&local_subset_gids_grouped_by_roi[0],local_total_id_list_size);
    id_decomp_map_ = Teuchos::rcp(new MultiField_Map(num_global_subsets_,lids_grouped_by_roi,0,*comm_));
  }
}

void
Decomp::populate_global_coordinate_vector(Teuchos::ArrayRCP<scalar_t> & subset_centroids_x,
  Teuchos::ArrayRCP<scalar_t> & subset_centroids_y,
  Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids,
  const std::string & image_file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & input_params,
  const Teuchos::RCP<Teuchos::ParameterList> & correlation_params){;

  const int_t dim = 2;
  const int_t proc_rank = comm_->get_rank();
  Teuchos::RCP<Image> image;
  Teuchos::ArrayRCP<scalar_t> coords_x;
  Teuchos::ArrayRCP<scalar_t> coords_y;
  int_t num_global_subsets = 0;
  image_width_ = -1;
  image_height_ = -1;

  if(proc_rank==0){
    // only proc 0 reads the image:
    Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
    imgParams->set(DICe::compute_image_gradients,true);
    // determine if the image is from cine high speed video file:
//    if(image_file_name==DICe::cine_file){
//      Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader;
//      int_t cine_num_images = -1;
//      int_t cine_first_frame_index = -1;
//      int_t cine_ref_index = -1;
//      const std::string cine_file_name = input_params->get<std::string>(DICe::cine_file);
//      std::stringstream cine_name;
//      cine_name << input_params->get<std::string>(DICe::image_folder) << cine_file_name;
//      Teuchos::oblackholestream bhs; // outputs nothing
//      cine_reader = Teuchos::rcp(new DICe::cine::Cine_Reader(cine_name.str(),&bhs,false));
//      cine_num_images = cine_reader->num_frames();
//      cine_first_frame_index = cine_reader->first_image_number();
//      DEBUG_MSG("Decomp::Decomp(): cine first frame index: " << cine_first_frame_index);
//      cine_ref_index = input_params->get<int_t>(DICe::cine_ref_index,cine_first_frame_index);
//      TEUCHOS_TEST_FOR_EXCEPTION(cine_ref_index < cine_first_frame_index,std::invalid_argument,"");
//      TEUCHOS_TEST_FOR_EXCEPTION(cine_ref_index - cine_first_frame_index > cine_num_images,std::invalid_argument,"");
//      // convert the cine ref, start and end index to the DICe indexing, not cine indexing (begins with zero)
//      cine_ref_index = cine_ref_index - cine_first_frame_index;
//      DEBUG_MSG("Decomp::Decomp(): reading cine image from file: " << cine_file_name << " index " << cine_ref_index);
//      image = cine_reader->get_frame(cine_ref_index,true,false,imgParams);
//    }
//    else{
      DEBUG_MSG("Decomp::Decomp(): reading image from file: " << image_file_name);
      image = Teuchos::rcp( new Image(image_file_name.c_str(),imgParams));
//    }
    image_width_ = image->width();
    image_height_ = image->height();
  } // end proc 0 reads image
  assert((proc_rank==0&&image!=Teuchos::null)||(proc_rank>0&&image==Teuchos::null)); // only proc 0 should have an image
  // communicate the image width and height to all processes:
  // this is a dummy field that is used to communicate up to two values from proc 0 to all procs
  // this may be used for only one value with one dummy value passed along
  Teuchos::Array<int_t> zero_owned_ids;
  if(proc_rank==0){
    zero_owned_ids.push_back(0); // both entries of this field are owned by proc zero in this map
    zero_owned_ids.push_back(1);
  }
  Teuchos::Array<int_t> all_owned_ids;
  all_owned_ids.push_back(0);
  all_owned_ids.push_back(1);
  Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField_Map> all_map = Teuchos::rcp (new MultiField_Map(-1, all_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField> zero_data = Teuchos::rcp(new MultiField(zero_map,1,true));
  Teuchos::RCP<MultiField> all_data = Teuchos::rcp(new MultiField(all_map,1,true));
  if(proc_rank==0){
    zero_data->local_value(0) = image_width_;
    zero_data->local_value(1) = image_height_;
  }
  // now export the zero owned values to all
  MultiField_Exporter exporter(*all_map,*zero_data->get_map());
  all_data->do_import(zero_data,exporter,INSERT);
  if(proc_rank!=0){
    image_width_ = all_data->local_value(0);
    image_height_ = all_data->local_value(1);
  }
  DEBUG_MSG("[PROC "<<proc_rank <<"] image width:  " << image_width_);
  DEBUG_MSG("[PROC "<<proc_rank <<"] image height: " << image_height_);

  // now determine the list of coordinates:
  Teuchos::RCP<DICe::Subset_File_Info> subset_info;

  // if the subset locations are specified in an input file, read them in (else they will be defined later)
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs;
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > blocking_subset_ids;
  Teuchos::RCP<std::vector<scalar_t> > subset_centroids_on_0 = Teuchos::rcp(new std::vector<scalar_t>());
  Teuchos::RCP<std::set<int_t> > force_simplex;
  const bool has_subset_file = input_params->isParameter(DICe::subset_file);
  DICe::Subset_File_Info_Type subset_info_type = DICe::SUBSET_INFO;
  if(has_subset_file){
    std::string fileName = input_params->get<std::string>(DICe::subset_file);
    subset_info = DICe::read_subset_file(fileName,image_width_,image_height_);
    subset_info_type = subset_info->type;
  }
 if(!has_subset_file || subset_info_type==DICe::REGION_OF_INTEREST_INFO){
    TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::step_size),std::runtime_error,
      "Error, step size has not been specified");
    DEBUG_MSG("Correlation point centroids were not specified by the user. \nThey will be evenly distrubed in the region"
      " of interest with separation (step_size) of " << input_params->get<int_t>(DICe::step_size) << " pixels.");
    TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_size),std::runtime_error,
      "Error, the subset size has not been specified"); // required for all square subsets case
    //subset_size = input_params->get<int_t>(DICe::subset_size);
    // only processor 0 reads the image and constructs/checks the points for SSSIG and the results are communicated to the rest of the procs
    Teuchos::RCP<std::vector<int_t> > neighbor_ids_on_0 = Teuchos::rcp(new std::vector<int_t>());
    if(proc_rank==0){
      Optimization_Method optimization_method = GRADIENT_BASED;
      if(correlation_params!=Teuchos::null){
        if(correlation_params->isParameter(DICe::optimization_method)){
          if(correlation_params->isType<std::string>(DICe::optimization_method)){
            std::string opt_string = correlation_params->get<std::string>(DICe::optimization_method,"GRADIENT_BASED");
            optimization_method = DICe::string_to_optimization_method(opt_string);
          }
          else{
            optimization_method = correlation_params->get<DICe::Optimization_Method>(DICe::optimization_method);
          }
        }
      }
      const scalar_t default_threshold = optimization_method==GRADIENT_BASED || optimization_method==GRADIENT_BASED_THEN_SIMPLEX ? 50.0: -1.0;
      const scalar_t grad_threshold = correlation_params->get<double>(DICe::sssig_threshold,default_threshold);
      DICe::create_regular_grid_of_correlation_points(*subset_centroids_on_0,*neighbor_ids_on_0,input_params,image,subset_info,grad_threshold);
      // check all the subsets and eliminate ones with a gradient ratio too low
      num_global_subsets = subset_centroids_on_0->size()/dim; // divide by three because the stride is x y neighbor_id
      assert(neighbor_ids_on_0->size()==subset_centroids_on_0->size()/2);
    } // end proc 0

    // communicate the number of global subsets to all procs:
    if(proc_rank==0){
      zero_data->local_value(0) = num_global_subsets;
      zero_data->local_value(1) = -1;
    }
    // now export the zero owned values to all
    all_data->do_import(zero_data,exporter,INSERT);
    if(proc_rank!=0){
      num_global_subsets = all_data->local_value(0);
    }
    DEBUG_MSG("[PROC "<<proc_rank <<"] num_global_subsets:  " << num_global_subsets);

    // communicate a size three multifield to all procs with field 0 being x coords 1 being y coords and 2 being neigh ids
    Teuchos::Array<int_t> field_zero_owned_ids;
    if(proc_rank==0){
      field_zero_owned_ids = Teuchos::Array<int_t>(num_global_subsets);
    }
    for(int_t i=0;i<field_zero_owned_ids.size();++i){
      field_zero_owned_ids[i] = i;
    }
    Teuchos::Array<int_t> field_all_owned_ids(num_global_subsets);
    for(int_t i=0;i<field_all_owned_ids.size();++i){
      field_all_owned_ids[i] = i;
    }
    Teuchos::RCP<MultiField_Map> field_zero_map = Teuchos::rcp (new MultiField_Map(-1, field_zero_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField_Map> field_all_map = Teuchos::rcp (new MultiField_Map(-1, field_all_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField> field_zero_data = Teuchos::rcp(new MultiField(field_zero_map,3,true));
    Teuchos::RCP<MultiField> field_all_data = Teuchos::rcp(new MultiField(field_all_map,3,true));
    if(proc_rank==0){
      for(int_t i=0;i<num_global_subsets;++i){
        field_zero_data->local_value(i,0) = (*subset_centroids_on_0)[i*2+0];
        field_zero_data->local_value(i,1) = (*subset_centroids_on_0)[i*2+1];
        field_zero_data->local_value(i,2) = (*neighbor_ids_on_0)[i];
      }
    }
    // now export the zero owned values to all
    MultiField_Exporter field_exporter(*field_all_map,*field_zero_data->get_map());
    field_all_data->do_import(field_zero_data,field_exporter,INSERT);
    subset_centroids_x.resize(num_global_subsets,0.0);
    subset_centroids_y.resize(num_global_subsets,0.0);
    neighbor_ids = Teuchos::rcp(new std::vector<int_t>(num_global_subsets));
    for(int_t i=0;i<num_global_subsets;++i){
      subset_centroids_x[i] = field_all_data->local_value(i,0);
      subset_centroids_y[i] = field_all_data->local_value(i,1);
      (*neighbor_ids)[i] = field_all_data->local_value(i,2);
    }
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(subset_info==Teuchos::null,std::runtime_error,"");
    subset_centroids_on_0 = subset_info->coordinates_vector;
    num_global_subsets = subset_centroids_on_0->size()/dim;
    subset_centroids_x.resize(num_global_subsets,0.0);
    subset_centroids_y.resize(num_global_subsets,0.0);
    for(int_t i=0;i<num_global_subsets;++i){
      subset_centroids_x[i] = (*subset_centroids_on_0)[i*2+0];
      subset_centroids_y[i] = (*subset_centroids_on_0)[i*2+1];
    }
    neighbor_ids = subset_info->neighbor_vector;
    conformal_area_defs = subset_info->conformal_area_defs;
    obstructing_subset_ids = subset_info->id_sets_map;
    force_simplex = subset_info->force_simplex;
    if((int_t)subset_info->conformal_area_defs->size()<num_global_subsets){
      // Only require this if not all subsets are conformal:
      TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_size),std::runtime_error,
        "Error, the subset size has not been specified");
      //subset_size = input_params->get<int_t>(DICe::subset_size);
    }
  }
 TEUCHOS_TEST_FOR_EXCEPTION(subset_centroids_x.size()!=num_global_subsets,std::runtime_error,"");
 TEUCHOS_TEST_FOR_EXCEPTION(subset_centroids_y.size()!=num_global_subsets,std::runtime_error,"");
 TEUCHOS_TEST_FOR_EXCEPTION(num_global_subsets<=0,std::runtime_error,"");
 num_global_subsets_ = num_global_subsets;

  // at this point we should have the global list of valid points on all processors with only proc 0 having read the image

//
//  DEBUG_MSG("[PROC "<<proc_rank <<"] SUBSET_COORDS: ");
//  for(int_t i=0;i<subset_centroids->size()/2;++i){
//    DEBUG_MSG("[PROC "<<proc_rank <<"] "<< i << " x " << (*subset_centroids)[i*2+0] << " y " << (*subset_centroids)[i*2+1]);
//  }
//  DEBUG_MSG("[PROC "<<proc_rank <<"] NEIGH_IDS: ");
//  for(int_t i=0;i<subset_centroids->size()/2;++i){
//    DEBUG_MSG("[PROC "<<proc_rank <<"] "<< i << " " << (*neighbor_ids)[i]);
//  }
}

}// End DICe Namespace
