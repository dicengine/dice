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
#include <DICe_ImageIO.h>

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
  TEUCHOS_TEST_FOR_EXCEPTION(image_files.size()<=0,std::runtime_error,"");

  // set up the positions of all the mesh points or subsets
  Teuchos::ArrayRCP<scalar_t> subset_centroids_x;
  Teuchos::ArrayRCP<scalar_t> subset_centroids_y;
  Teuchos::RCP<std::vector<int_t> > neighbor_ids;
  Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids;
  populate_coordinate_vectors(image_files[0],input_params,correlation_params,
      subset_centroids_x,subset_centroids_y,neighbor_ids,obstructing_subset_ids);
  // only proc 0 should have the global coords at this point
  TEUCHOS_TEST_FOR_EXCEPTION(comm_->get_rank()==0&&subset_centroids_x.size()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(comm_->get_rank()==0&&subset_centroids_y.size()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(comm_->get_rank()!=0&&subset_centroids_x.size()!=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(comm_->get_rank()!=0&&subset_centroids_y.size()!=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(num_global_subsets_<=0,std::runtime_error,"");
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
Decomp::initialize(const Teuchos::ArrayRCP<scalar_t> subset_centroids_x,
  const Teuchos::ArrayRCP<scalar_t> subset_centroids_y,
  Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
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

  // at this point the id_decomp_map should be one-to-one in all circumstances
  TEUCHOS_TEST_FOR_EXCEPTION(!id_decomp_map_->is_one_to_one(),std::runtime_error,"");

  // lastly, create the overlap vectors of points needed for computing VSG strain, etc.
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
  if(is_parallel){
    Teuchos::Array<int_t> field_zero_owned_ids;
    // communicate the number of global subsets to all processors:
    // this is a dummy field that is used to communicate a value from proc 0 to all procs
    Teuchos::Array<int_t> zero_owned_ids;
    if(comm_->get_rank()==0){
      for(int_t i=0;i<comm_->get_size();++i)
        zero_owned_ids.push_back(i); // one begin index and end index for all processors
    }
    Teuchos::Array<int_t> all_owned_ids;
    all_owned_ids.push_back(comm_->get_rank());
    Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField_Map> all_map = Teuchos::rcp (new MultiField_Map(-1, all_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField> zero_data = Teuchos::rcp(new MultiField(zero_map,2,true));
    Teuchos::RCP<MultiField> all_data = Teuchos::rcp(new MultiField(all_map,2,true));
    int_t num_ids_to_get = comm_->get_rank()==0 ? num_global_subsets_ : 1;
    Teuchos::Array<int_t> remote_gids(num_ids_to_get);
    for(int_t i=0;i<num_ids_to_get;++i)
      remote_gids[i] = i;
    Teuchos::Array<int_t> remote_proc_ids(num_ids_to_get);
    id_decomp_map_->get_remote_index_list(remote_gids,remote_proc_ids);
//    for(int_t i=0;i<remote_proc_ids.size();++i)
//      std::cout << comm_->get_rank() << " gid " << remote_gids[i] << " pid " <<  remote_proc_ids[i] << std::endl;
    std::vector<std::set<int_t> > owned_ids_on_each_proc(comm_->get_size(),std::set<int_t>());
    if(comm_->get_rank()==0){
      // get the ids on each proc from the id_decomp_map
      for(int_t i=0;i<num_global_subsets_;++i){
        owned_ids_on_each_proc[remote_proc_ids[i]].insert(remote_gids[i]);
      }

      if(max_strain_window_size > 0.0){
        // processor 0 does a neighborhood search and scatters the results to all processors
        TEUCHOS_TEST_FOR_EXCEPTION(subset_centroids_x.size()!=num_global_subsets_||subset_centroids_y.size()!=num_global_subsets_,std::runtime_error,"");
        Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
        point_cloud->pts.resize(num_global_subsets_);
        for(int_t i=0;i<num_global_subsets_;++i){
          point_cloud->pts[i].x = subset_centroids_x[i];
          point_cloud->pts[i].y = subset_centroids_y[i];
        }
        DEBUG_MSG("Decomp::Decomp(): building the kd-tree");
        Teuchos::RCP<kd_tree_2d_t> kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
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
        for(int_t proc=0;proc<comm_->get_size();++proc){
          std::set<int_t> neighbors_to_add;
          // iterate all the ids local to this processor and find all the neighbors
          for(std::set<int_t>::const_iterator it=owned_ids_on_each_proc[proc].begin();it!=owned_ids_on_each_proc[proc].end();++it){
            assert(*it>=0&&*it<num_global_subsets_);
            query_pt[0] = subset_centroids_x[*it];
            query_pt[1] = subset_centroids_y[*it];
            kd_tree->radiusSearch(&query_pt[0],neigh_rad_2,ret_matches,params);
            for(size_t j=0;j<ret_matches.size();++j){
              neighbors_to_add.insert(ret_matches[j].first);
//              if(neighbor_ids!=Teuchos::null){
//                if(j==1&&(int_t)neighbor_ids->size()>*it&&(*neighbor_ids)[*it]>0){ // check for rebalance needed, and that this is not a seed subset
//                  (*neighbor_ids)[*it] = ret_matches[j].first;
//                }
//              }
            } // loop over all neighbors
          } // loop over all dist ids on that processor
          // add all the neighbors to the id list
          for(std::set<int_t>::const_iterator it = neighbors_to_add.begin(); it!=neighbors_to_add.end(); ++it)
            owned_ids_on_each_proc[proc].insert(*it);
        } // loop over each processor
        DEBUG_MSG("Decomp::Decomp(): neighbor list constructed");
      } // end window_size > 0
      // determine the size of the mega position vector with all the overlap coords
      int_t total_num_overlap_pts = 0;
      for(int_t proc=0;proc<comm_->get_size();++proc){
        total_num_overlap_pts+=owned_ids_on_each_proc[proc].size();
      }

      DEBUG_MSG("Decomp::Decomp(): total number of overlap points: " << total_num_overlap_pts);
      field_zero_owned_ids = Teuchos::Array<int_t>(total_num_overlap_pts);
      for(int_t i=0;i<field_zero_owned_ids.size();++i){
        field_zero_owned_ids[i] = i;
      }
      // scatter the bounds to all procs
      int_t current_index = 0;
      for(int_t proc=0;proc<comm_->get_size();++proc){
        zero_data->local_value(proc,0) = current_index; // start index
        zero_data->local_value(proc,1) = current_index+owned_ids_on_each_proc[proc].size()-1; // end index
        current_index += owned_ids_on_each_proc[proc].size();
      }
    } // end processor 0
    // now export the zero owned values to all
    MultiField_Exporter exporter(*all_map,*zero_data->get_map());
    all_data->do_import(zero_data,exporter,INSERT);
    const int_t start_index = all_data->local_value(0);
    const int_t end_index = all_data->local_value(1);
    const int_t num_overlap = end_index - start_index + 1;
    DEBUG_MSG("[PROC "<< comm_->get_rank() <<"] subset index range " << start_index << " to " << end_index);
    Teuchos::RCP<MultiField_Map> field_zero_map = Teuchos::rcp (new MultiField_Map(-1, field_zero_owned_ids,0,*comm_));
    Teuchos::Array<int_t> field_dist_owned_ids(num_overlap);
    for(int_t i=0;i<num_overlap;++i)
      field_dist_owned_ids[i] = start_index + i;
    Teuchos::RCP<MultiField_Map> field_dist_map =  Teuchos::rcp(new MultiField_Map(-1,field_dist_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField> field_zero_data = Teuchos::rcp(new MultiField(field_zero_map,4,true)); // the cols are x, y, gid, neigh_id
    Teuchos::RCP<MultiField> field_dist_data = Teuchos::rcp(new MultiField(field_dist_map,4,true));
    if(comm_->get_rank()==0){
      int_t current_index = 0;
      for(int_t proc=0;proc<comm_->get_size();++proc){
        for(std::set<int_t>::const_iterator it=owned_ids_on_each_proc[proc].begin();it!=owned_ids_on_each_proc[proc].end();++it){
          field_zero_data->local_value(current_index,0) = subset_centroids_x[*it];
          field_zero_data->local_value(current_index,1) = subset_centroids_y[*it];
          field_zero_data->local_value(current_index,2) = *it;
          if(neighbor_ids!=Teuchos::null){
            field_zero_data->local_value(current_index,3) = (int_t)neighbor_ids->size() > *it ? (*neighbor_ids)[*it]: -1;
          }
          else
            field_zero_data->local_value(current_index,3) = -1;
          current_index ++;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(current_index!=field_zero_owned_ids.size(),std::runtime_error,"");
    } // end proc 0
    MultiField_Exporter field_exporter(*field_dist_map,*field_zero_data->get_map());
    field_dist_data->do_import(field_zero_data,field_exporter,INSERT);
    Teuchos::Array<int_t> field_dist_owned_gids(num_overlap);
    for(int_t i=0;i<num_overlap;++i){
      field_dist_owned_gids[i] = field_dist_data->local_value(i,2);
      //  DEBUG_MSG("[PROC "<< comm_->get_rank() <<"] has a subset at " << field_dist_data->local_value(i,0) << " " << field_dist_data->local_value(i,1));
    }
    // now that the coords have been communicated, store them in the overlap coords vectors
    overlap_coords_x_.resize(num_overlap,0.0);
    overlap_coords_y_.resize(num_overlap,0.0);
    neighbor_ids_ = Teuchos::rcp(new std::vector<int_t>(num_overlap,-1));
    for(int_t i=0;i<num_overlap;++i){
      overlap_coords_x_[i] = field_dist_data->local_value(i,0);
      overlap_coords_y_[i] = field_dist_data->local_value(i,1);
      (*neighbor_ids_)[i] = field_dist_data->local_value(i,3);
    }
    id_decomp_overlap_map_ = Teuchos::rcp(new MultiField_Map(-1,field_dist_owned_gids,0,*comm_));
    DEBUG_MSG("Decomp::Decomp(): coordinate list has been trimmed");
  } // end is parallel
  else{
    // set the neighbor ids
    neighbor_ids_ = Teuchos::rcp(new std::vector<int_t>(id_decomp_map_->get_num_local_elements(),-1));
    overlap_coords_x_.resize(id_decomp_map_->get_num_local_elements(),0.0);
    overlap_coords_y_.resize(id_decomp_map_->get_num_local_elements(),0.0);
    for(int_t i=0;i<id_decomp_map_->get_num_local_elements();++i){
      const int_t gid = id_decomp_map_->get_global_element(i);
      assert(gid<subset_centroids_x.size());
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

  if(neighbor_ids==Teuchos::null) return;

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

  // process zero divies up the subsets:
  Teuchos::Array<int_t> field_zero_owned_ids;
  if(proc_id==0){
    field_zero_owned_ids = Teuchos::Array<int_t>(num_global_subsets_);
  }
  for(int_t i=0;i<field_zero_owned_ids.size();++i){
    field_zero_owned_ids[i] = i;
  }
  Teuchos::RCP<MultiField_Map> field_zero_map = Teuchos::rcp (new MultiField_Map(-1, field_zero_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField> field_zero_data = Teuchos::rcp(new MultiField(field_zero_map,1,true));

  // dummy field to communicate the extents to each processor
  Teuchos::Array<int_t> zero_owned_ids;
  if(proc_id==0){
    for(int_t i=0;i<num_procs;++i)
      zero_owned_ids.push_back(i);
  }
  Teuchos::Array<int_t> all_owned_ids;
  all_owned_ids.push_back(proc_id);
  Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField_Map> all_map = Teuchos::rcp (new MultiField_Map(-1, all_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField> zero_data = Teuchos::rcp(new MultiField(zero_map,2,true));
  Teuchos::RCP<MultiField> all_data = Teuchos::rcp(new MultiField(all_map,2,true));

  if(proc_id==0){
    std::vector<int_t> local_subset_gids_grouped_by_roi;
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)neighbor_ids->size()!=num_global_subsets_,std::runtime_error,"");
    std::vector<int_t> this_group_gids;
    std::vector<std::vector<int_t> > seed_groupings;
    for(int_t i=num_global_subsets_-1;i>=0;--i){
      this_group_gids.push_back(i);
      // if this subset is a seed, break this grouping and insert it in the set
      if((*neighbor_ids)[i]==-1){
        seed_groupings.push_back(this_group_gids);
        this_group_gids.clear();
      }
    }
    // reverse all the seed groupings:
    for(size_t i=0;i<seed_groupings.size();++i){
      std::reverse(seed_groupings[i].begin(), seed_groupings[i].end());
    }
    // split the groups among processors
    std::vector<int_t> send_to_proc(seed_groupings.size(),0);
    int_t p_id = 0;
    for(size_t i=0;i<seed_groupings.size();++i){
      send_to_proc[i] = p_id++;
      if(p_id==num_procs) p_id=0;
    }
    // restack the ids in processor order
    int_t id_count = 0;
    for(int_t proc=0;proc<num_procs;++proc){
      zero_data->local_value(proc,0) = id_count;
      for(size_t i=0;i<seed_groupings.size();++i){
        if(send_to_proc[i]==(int_t)proc){
          for(size_t j=0;j<seed_groupings[i].size();++j){
            field_zero_data->local_value(id_count++) = seed_groupings[i][j];
          }
        } // end proc id matches
      } // end loop over seed groupings
      zero_data->local_value(proc,1) = id_count - 1;
    } // end proc loop
    // now field zero list is ordered by groupings and processors
  }// end processor 0

  // communicate the extents of the ordered list to all procs:
  MultiField_Exporter exporter(*all_map,*zero_data->get_map());
  all_data->do_import(zero_data,exporter,INSERT);
  const int_t start_id = all_data->local_value(0,0);
  const int_t end_id = all_data->local_value(0,1);
  const int_t num_ids = end_id - start_id + 1;
  DEBUG_MSG("[PROC "<<proc_id <<"] Decomp::create_seed_dist_map(): owns neigbor id list elements " << start_id << " through " << end_id << " num ids " << num_ids);

  // create the dist map for the neigh ids:
  Teuchos::Array<int_t> field_all_owned_ids(num_ids);
  for(int_t i=0;i<field_all_owned_ids.size();++i){
    field_all_owned_ids[i] = start_id + i;
  }
  Teuchos::RCP<MultiField_Map> field_all_map = Teuchos::rcp (new MultiField_Map(-1, field_all_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField> field_all_data = Teuchos::rcp(new MultiField(field_all_map,1,true));
  // import the neigh ids local to this processor:
  MultiField_Exporter field_exporter(*field_all_map,*field_zero_data->get_map());
  field_all_data->do_import(field_zero_data,field_exporter,INSERT);
  const int_t num_local_subsets = field_all_data->get_map()->get_num_local_elements();
  for(int_t i=0;i<num_local_subsets;++i)
    DEBUG_MSG("[PROC " << proc_id << "] Decomp::create_seed_dist_map(): proc has subset id " << field_all_data->local_value(i));

  // copy the ids in order
  this_proc_gid_order_ = std::vector<int_t>(num_local_subsets,-1);
  for(int_t i=0;i<num_local_subsets;++i)
    this_proc_gid_order_[i] = field_all_data->local_value(i);

  std::vector<int_t> ordered_local_list(num_local_subsets);
  for(int_t i=0;i<num_local_subsets;++i)
    ordered_local_list[i] = field_all_data->local_value(i);

  // sort the vector so the map is not in execution order
  std::sort(ordered_local_list.begin(),ordered_local_list.end());
  Teuchos::ArrayView<const int_t> lids_grouped_by_roi(&ordered_local_list[0],num_local_subsets);
  id_decomp_map_ = Teuchos::rcp(new MultiField_Map(num_global_subsets_,lids_grouped_by_roi,0,*comm_));
}

void
Decomp::populate_coordinate_vectors(const std::string & image_file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & input_params,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params,
    Teuchos::ArrayRCP<scalar_t> & subset_centroids_x,
    Teuchos::ArrayRCP<scalar_t> & subset_centroids_y,
    Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
    Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids){

  subset_centroids_x.clear();
  subset_centroids_y.clear();
  neighbor_ids = Teuchos::rcp(new std::vector<int_t>()); // initialize to zero length

  //const int_t num_procs = comm_->get_size();
  const int_t proc_rank = comm_->get_rank();
  const int_t dim = 2;

  // read the image dimensions on all processors
  int_t img_w = -1;
  int_t img_h = -1;
  int_t subset_size = -1;
  if(input_params->isParameter(DICe::subset_size)){
    subset_size = input_params->get<int_t>(DICe::subset_size);
  }
  utils::read_image_dimensions(image_file_name.c_str(),img_w,img_h);
  DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): image width:  " << img_w);
  DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): image height: " << img_h);
  image_width_ = img_w;
  image_height_ = img_h;

  // processor 0 creates the list of correlation points and divys them up for checking the SSSIG if necessary...
  Teuchos::RCP<std::vector<scalar_t> > subset_centroids = Teuchos::rcp(new std::vector<scalar_t>());
  std::vector<int_t> neigh_ids_on_0;
  int_t num_global_subsets_pre_sssig = 0;
//  if(proc_rank==0){
    // now determine the list of coordinates:
    // if the subset locations are specified in an input file, read them in (else they will be defined later)
    Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs;
    //Teuchos::RCP<std::map<int_t,std::vector<int_t> > > blocking_subset_ids;
    const bool has_subset_file = input_params->isParameter(DICe::subset_file);
    DICe::Subset_File_Info_Type subset_info_type = DICe::SUBSET_INFO;
    if(has_subset_file){
      std::string fileName = input_params->get<std::string>(DICe::subset_file);
      subset_info_ = DICe::read_subset_file(fileName,img_w,img_h);
      subset_info_type = subset_info_->type;
    }
    if(!has_subset_file || subset_info_type==DICe::REGION_OF_INTEREST_INFO){
      TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::step_size),std::runtime_error,
        "Error, step size has not been specified");
      DEBUG_MSG("Correlation point centroids were not specified by the user. \nThey will be evenly distrubed in the region"
        " of interest with separation (step_size) of " << input_params->get<int_t>(DICe::step_size) << " pixels.");
      TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_size),std::runtime_error,
        "Error, the subset size has not been specified"); // required for all square subsets case
      DICe::create_regular_grid_of_correlation_points(*subset_centroids,neigh_ids_on_0,input_params,img_w,img_h,subset_info_);
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(subset_info_==Teuchos::null,std::runtime_error,"");
      subset_centroids = subset_info_->coordinates_vector;
      conformal_area_defs = subset_info_->conformal_area_defs;
      obstructing_subset_ids = subset_info_->id_sets_map;
      if(subset_info_->conformal_area_defs->size()<subset_centroids->size()/dim){
        // Only require this if not all subsets are conformal:
        TEUCHOS_TEST_FOR_EXCEPTION(!input_params->isParameter(DICe::subset_size),std::runtime_error,
          "Error, the subset size has not been specified");
      }
    }
    num_global_subsets_pre_sssig = subset_centroids->size()/dim; // divide by three because the stride is x y neighbor_id
    DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): num global subsets before SSSIG:  " << num_global_subsets_pre_sssig);
    //for(int_t i=0;i<num_global_subsets_pre_sssig;++i){
    //  std::cout << "subset " << i << " " << (*subset_centroids_on_0)[i*2+0] << " " << (*subset_centroids_on_0)[i*2+1] << std::endl;
    //}
//  } // end processor 0

  // check the SSSIG criteria if necessary:
  bool sssig_check_done = false;
  Teuchos::RCP<MultiField> field_zero_data;
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
  const scalar_t grad_threshold = correlation_params->get<double>(DICe::sssig_threshold,50.0);
  if((optimization_method==GRADIENT_BASED || optimization_method==GRADIENT_BASED_THEN_SIMPLEX)&&grad_threshold > 0.0&&subset_size>0){
    sssig_check_done = true;
    // split up the points across processors and check the SSSIG:

    // communicate the number of global subsets to all processors:
    // this is a dummy field that is used to communicate a value from proc 0 to all procs
    Teuchos::Array<int_t> zero_owned_ids;
    if(proc_rank==0){
      zero_owned_ids.push_back(0); // both entries of this field are owned by proc zero in this map
    }
    Teuchos::Array<int_t> all_owned_ids;
    all_owned_ids.push_back(0);
    Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField_Map> all_map = Teuchos::rcp (new MultiField_Map(-1, all_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField> zero_data = Teuchos::rcp(new MultiField(zero_map,1,true));
    Teuchos::RCP<MultiField> all_data = Teuchos::rcp(new MultiField(all_map,1,true));
    if(proc_rank==0){
      zero_data->local_value(0) = num_global_subsets_pre_sssig;
    }
    // now export the zero owned values to all
    MultiField_Exporter exporter(*all_map,*zero_data->get_map());
    all_data->do_import(zero_data,exporter,INSERT);
    if(proc_rank!=0){
      num_global_subsets_pre_sssig = all_data->local_value(0);
    }
    DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): num global subsets before SSSIG (post comm):  " << num_global_subsets_pre_sssig);

    Teuchos::Array<int_t> field_zero_owned_ids;
    if(proc_rank==0){
      field_zero_owned_ids = Teuchos::Array<int_t>(num_global_subsets_pre_sssig);
    }
    for(int_t i=0;i<field_zero_owned_ids.size();++i){
      field_zero_owned_ids[i] = i;
    }
    Teuchos::RCP<MultiField_Map> field_zero_map = Teuchos::rcp (new MultiField_Map(-1, field_zero_owned_ids,0,*comm_));
    Teuchos::RCP<MultiField_Map> field_dist_map =  Teuchos::rcp(new MultiField_Map(num_global_subsets_pre_sssig,0,*comm_));
    field_zero_data = Teuchos::rcp(new MultiField(field_zero_map,4,true));
    Teuchos::RCP<MultiField> field_dist_data = Teuchos::rcp(new MultiField(field_dist_map,4,true));
    // fields are 0: coord_x, 1: coord_y, 2: neigh_id 3: is_valid 1.0 for true (sssig)
    if(proc_rank==0){
      for(int_t i=0;i<num_global_subsets_pre_sssig;++i){
        field_zero_data->local_value(i,0) = (*subset_centroids)[i*2+0];
        field_zero_data->local_value(i,1) = (*subset_centroids)[i*2+1];
        field_zero_data->local_value(i,2) = (int_t)neigh_ids_on_0.size() > i ? neigh_ids_on_0[i]: -1;
        field_zero_data->local_value(i,3) = 1.0;
      }
    }
    // now export the zero owned values to all
    MultiField_Exporter field_exporter(*field_dist_map,*field_zero_data->get_map());
    field_dist_data->do_import(field_zero_data,field_exporter,INSERT);
    const int_t num_check_points = field_dist_data->get_map()->get_num_local_elements();
    DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): num points to check for sssig: " << num_check_points);
    // determine the image extents needed for this processor:
    int_t min_x = img_w;
    int_t min_y = img_h;
    int_t max_x = 0;
    int_t max_y = 0;
    for(int_t i=0;i<num_check_points;++i){
      if(field_dist_data->local_value(i,0)<min_x) min_x = field_dist_data->local_value(i,0);
      if(field_dist_data->local_value(i,0)>max_x) max_x = field_dist_data->local_value(i,0);
      if(field_dist_data->local_value(i,1)<min_y) min_y = field_dist_data->local_value(i,1);
      if(field_dist_data->local_value(i,1)>max_y) max_y = field_dist_data->local_value(i,1);
    }
    min_x = min_x - subset_size > 0 ? min_x - subset_size : 0;
    max_x = max_x + subset_size < img_w ? max_x + subset_size : img_w;
    min_y = min_y - subset_size >0 ? min_y - subset_size : 0;
    max_y = max_y + subset_size < img_h ? max_y + subset_size : img_h;
    DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): image extents " << min_x << " " << max_x << " " << min_y << " " << max_y);
    // load images for each processor
    Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
    imgParams->set(DICe::compute_image_gradients,true);
    Teuchos::RCP<Image> sssig_image = Teuchos::rcp( new Image(image_file_name.c_str(),min_x,min_y,max_x-min_x+1,max_y-min_y+1,imgParams));
    TEUCHOS_TEST_FOR_EXCEPTION(!sssig_image->has_gradients(),std::runtime_error,
      "Error, testing valid points for SSSIG tol, but image gradients have not been computed");
    for(int_t i=0;i<num_check_points;++i){
      const int_t cx = field_dist_data->local_value(i,0);
      const int_t cy = field_dist_data->local_value(i,1);
      //DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): checking ssig for point " << field_dist_data->local_value(i,0) << " " << field_dist_data->local_value(i,1));
      // check the gradient SSSIG threshold
      scalar_t SSSIG = 0.0;
      const int_t left_x = cx - subset_size/2;
      const int_t right_x = left_x + subset_size;
      const int_t top_y = cy - subset_size/2;
      const int_t bottom_y = top_y + subset_size;
      for(int_t y=top_y;y<bottom_y;++y){
        for(int_t x=left_x;x<right_x;++x){
          SSSIG += sssig_image->grad_x(x-min_x,y-min_y)*sssig_image->grad_x(x-min_x,y-min_y) +
              sssig_image->grad_y(x-min_x,y-min_y)*sssig_image->grad_y(x-min_x,y-min_y);
        }
      }
      SSSIG /= (subset_size*subset_size);
      if(SSSIG < grad_threshold) field_dist_data->local_value(i,3) = 0.0;
      //DEBUG_MSG("[PROC "<<proc_rank <<"] x " << cx << " y " << cy << " SSSIG: " << SSSIG << " threshold " << grad_threshold << " pass " << field_dist_data->local_value(i,2));
    } // end subset check loop

    // communicate the valid subsets back to process 0
    MultiField_Exporter field_exporter_rev(*field_zero_map,*field_dist_map);
    field_zero_data->do_import(field_dist_data,field_exporter_rev,INSERT);
  } // end sssig needs to be checked

  if(proc_rank==0){
    // collect only the valid subsets (that pass sssig if necessary)
    int_t num_valid_points = subset_centroids->size()/2;
    if(sssig_check_done){
      TEUCHOS_TEST_FOR_EXCEPTION(field_zero_data==Teuchos::null,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(field_zero_data->get_map()->get_num_local_elements()<=0,std::runtime_error,"");
      num_valid_points = 0;
      for(int_t i=0;i<field_zero_data->get_map()->get_num_local_elements();++i){
        if(field_zero_data->local_value(i,3)>0.0)
          num_valid_points++;
      }
    }
    DEBUG_MSG("[PROC "<<proc_rank <<"] Decomp::populate_coordinate_vectors(): num global subsets post sssig check " << num_valid_points);
    // check if the neighbor ids need to be rebalanced (some may have been invalidated by the sssig check)
    bool has_seeds = false;
    if(subset_info_!=Teuchos::null)
      if(subset_info_->size_map->size() > 0)
        has_seeds = true;
    if(num_valid_points!=num_global_subsets_pre_sssig && has_seeds){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, seeds and SSSIG thresholding cannot be used simultaneously.");
    }
    neighbor_ids = Teuchos::rcp(new std::vector<int_t>(num_valid_points));
    subset_centroids_x.resize(num_valid_points);
    subset_centroids_y.resize(num_valid_points);
    int_t valid_index=0;
    if(sssig_check_done){
      for(int_t i=0;i<field_zero_data->get_map()->get_num_local_elements();++i){
        if(field_zero_data->local_value(i,3)>0.0){
          subset_centroids_x[valid_index] = field_zero_data->local_value(i,0);
          subset_centroids_y[valid_index] = field_zero_data->local_value(i,1);
          (*neighbor_ids)[valid_index] = field_zero_data->local_value(i,2);
          valid_index++;
        }
      }
    } // end sssig_check_done
    else{
      for(int_t i=0;i<num_valid_points;++i){
        subset_centroids_x[i] = (*subset_centroids)[i*2+0];
        subset_centroids_y[i] = (*subset_centroids)[i*2+1];
        (*neighbor_ids)[i] = (int_t)neigh_ids_on_0.size() > i ? neigh_ids_on_0[i] : -1;
      }
    }
  } // end proc==0
  else{
    subset_centroids->clear();
  }

  // communicate the number of global subsets to all procs
  // this is a dummy field that is used to communicate a value from proc 0 to all procs
  Teuchos::Array<int_t> zero_owned_ids;
  if(proc_rank==0){
    zero_owned_ids.push_back(0); // both entries of this field are owned by proc zero in this map
  }
  Teuchos::Array<int_t> all_owned_ids;
  all_owned_ids.push_back(0);
  Teuchos::RCP<MultiField_Map> zero_map = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField_Map> all_map = Teuchos::rcp (new MultiField_Map(-1, all_owned_ids,0,*comm_));
  Teuchos::RCP<MultiField> zero_data = Teuchos::rcp(new MultiField(zero_map,1,true));
  Teuchos::RCP<MultiField> all_data = Teuchos::rcp(new MultiField(all_map,1,true));
  if(proc_rank==0){
    zero_data->local_value(0) = subset_centroids_x.size();
  }
  // now export the zero owned values to all
  MultiField_Exporter exporter(*all_map,*zero_data->get_map());
  all_data->do_import(zero_data,exporter,INSERT);
  num_global_subsets_ = all_data->local_value(0);
}

DICE_LIB_DLL_EXPORT
void
create_regular_grid_of_correlation_points(std::vector<scalar_t> & correlation_points,
  std::vector<int_t> & neighbor_ids,
  Teuchos::RCP<Teuchos::ParameterList> params,
  const int_t img_w,
  const int_t img_h,
  Teuchos::RCP<DICe::Subset_File_Info> subset_file_info){
  int proc_rank = 0;
#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif

  if(proc_rank==0) DEBUG_MSG("Creating a grid of regular correlation points");
  // note: assumes two dimensional
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::step_size),std::runtime_error,
    "Error, the step size was not specified");
  const int_t step_size = params->get<int_t>(DICe::step_size);
  // set up the control points
  TEUCHOS_TEST_FOR_EXCEPTION(step_size<=0,std::runtime_error,"Error: step size is <= 0");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::subset_size),std::runtime_error,
    "Error, the subset size was not specified");
  const int_t subset_size = params->get<int_t>(DICe::subset_size);
  correlation_points.clear();
  neighbor_ids.clear();
  bool seed_was_specified = false;
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > roi_defs;
  if(subset_file_info!=Teuchos::null){
    if(subset_file_info->conformal_area_defs!=Teuchos::null){
      if(proc_rank==0) DEBUG_MSG("Using ROIs from a subset file");
      roi_defs = subset_file_info->conformal_area_defs;
      if(proc_rank==0) DEBUG_MSG("create_regular_grid_of_correlation_points(): user requested " << roi_defs->size() <<  " ROI(s)");
      seed_was_specified = subset_file_info->size_map->size() > 0;
      if(seed_was_specified){
        TEUCHOS_TEST_FOR_EXCEPTION(subset_file_info->size_map->size()!=subset_file_info->displacement_map->size(),
          std::runtime_error,"Error the number of displacement guesses and seed locations must be the same");
      }
    }
    else{
      if(proc_rank==0) DEBUG_MSG("create_regular_grid_of_correlation_points(): subset file exists, but has no ROIs");
    }
  }
  if(roi_defs==Teuchos::null){ // wasn't populated above so create a dummy roi with the whole image:
    if(proc_rank==0) DEBUG_MSG("create_regular_grid_of_correlation_points(): creating dummy ROI of the entire image");
    Teuchos::RCP<DICe::Rectangle> rect = Teuchos::rcp(new DICe::Rectangle(img_w/2,img_h/2,img_w,img_h));
    DICe::multi_shape boundary_multi_shape;
    boundary_multi_shape.push_back(rect);
    DICe::Conformal_Area_Def conformal_area_def(boundary_multi_shape);
    roi_defs = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def> ());
    roi_defs->insert(std::pair<int_t,DICe::Conformal_Area_Def>(0,conformal_area_def));
  }

  // if no ROI is specified, the whole image is the ROI

  int_t current_subset_id = 0;

  std::map<int_t,DICe::Conformal_Area_Def>::iterator map_it=roi_defs->begin();
  for(;map_it!=roi_defs->end();++map_it){

    std::set<std::pair<int_t,int_t> > coords;
    std::set<std::pair<int_t,int_t> > excluded_coords;
    // collect the coords of all the boundary shapes
    for(size_t i=0;i<map_it->second.boundary()->size();++i){
      std::set<std::pair<int_t,int_t> > shapeCoords = (*map_it->second.boundary())[i]->get_owned_pixels();
      coords.insert(shapeCoords.begin(),shapeCoords.end());
    }
    // collect the coords of all the exclusions
    if(map_it->second.has_excluded_area()){
      for(size_t i=0;i<map_it->second.excluded_area()->size();++i){
        std::set<std::pair<int_t,int_t> > shapeCoords = (*map_it->second.excluded_area())[i]->get_owned_pixels();
        excluded_coords.insert(shapeCoords.begin(),shapeCoords.end());
      }
    }

    int_t num_rows = 0;
    int_t num_cols = 0;
    int_t seed_row = 0;
    int_t seed_col = 0;
    int_t x_coord = subset_size-1;
    int_t y_coord = subset_size-1;
    int_t seed_location_x = 0;
    int_t seed_location_y = 0;
    int_t seed_subset_id = -1;

    bool this_roi_has_seed = false;
    if(seed_was_specified){
      if(subset_file_info->size_map->find(map_it->first)!=subset_file_info->size_map->end()){
        seed_location_x = subset_file_info->size_map->find(map_it->first)->second.first;
        seed_location_y = subset_file_info->size_map->find(map_it->first)->second.second;
        if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " seed x: " << seed_location_x << " seed_y: " << seed_location_y);
        this_roi_has_seed = true;
      }
    }
    while(x_coord < img_w - subset_size) {
      if(x_coord + step_size/2 < seed_location_x) seed_col++;
      x_coord+=step_size;
      num_cols++;
    }
    while(y_coord < img_h - subset_size) {
      if(y_coord + step_size/2 < seed_location_y) seed_row++;
      y_coord+=step_size;
      num_rows++;
    }
    if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " has " << num_rows << " rows and " << num_cols << " cols, seed row " << seed_row << " seed col " << seed_col);
    //if(seed_was_specified&&this_roi_has_seed){
      x_coord = subset_size-1 + seed_col*step_size;
      y_coord = subset_size-1 + seed_row*step_size;
    if(valid_correlation_point(x_coord,y_coord,subset_size,img_w,img_h,coords,excluded_coords)){
      correlation_points.push_back((scalar_t)x_coord);
      correlation_points.push_back((scalar_t)y_coord);
      //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding seed correlation point " << x_coord << " " << y_coord);
      if(seed_was_specified&&this_roi_has_seed){
        seed_subset_id = current_subset_id;
        subset_file_info->seed_subset_ids->insert(std::pair<int_t,int_t>(seed_subset_id,map_it->first));
      }
      neighbor_ids.push_back(-1); // seed point cannot have a neighbor
      current_subset_id++;
    }
    else if(!(seed_row==0&&seed_col==0)){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, seed specified does not meet the SSSIG criteria for suffient image gradients");
    }

    // snake right from seed
    const int_t right_start_subset_id = current_subset_id;
    int_t direction = 1; // sign needs to be flipped if the seed row is the first row
    int_t row = seed_row;
    int_t col = seed_col;
    while(row>=0&&row<num_rows){
      if((direction==1&&row+1>=num_rows)||(direction==-1&&row-1<0)){
        direction *= -1;
        col++;
      }else{
        row += direction;
      }
      if(col>=num_cols)break;
      x_coord = subset_size - 1 + col*step_size;
      y_coord = subset_size - 1 + row*step_size;
      if(valid_correlation_point(x_coord,y_coord,subset_size,img_w,img_h,coords,excluded_coords)){
        correlation_points.push_back((scalar_t)x_coord);
        correlation_points.push_back((scalar_t)y_coord);
        //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding snake right correlation point " << x_coord << " " << y_coord);
        if(current_subset_id==right_start_subset_id)
          neighbor_ids.push_back(seed_subset_id);
        else
          neighbor_ids.push_back(current_subset_id - 1);
        current_subset_id++;
      }  // end valid point
    }  // end snake right
    // snake left from seed
    const int_t left_start_subset_id = current_subset_id;
    direction = -1;
    row = seed_row;
    col = seed_col;
    while(row>=0&&row<num_rows){
      if((direction==1&&row+1>=num_rows)||(direction==-1&&row-1<0)){
        direction *= -1;
        col--;
      }
      else{
        row += direction;
      }
      if(col<0)break;
      x_coord = subset_size - 1 + col*step_size;
      y_coord = subset_size - 1 + row*step_size;
      if(valid_correlation_point(x_coord,y_coord,subset_size,img_w,img_h,coords,excluded_coords)){
        correlation_points.push_back((scalar_t)x_coord);
        correlation_points.push_back((scalar_t)y_coord);
        //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding snake left correlation point " << x_coord << " " << y_coord);
        if(current_subset_id==left_start_subset_id)
          neighbor_ids.push_back(seed_subset_id);
        else
          neighbor_ids.push_back(current_subset_id-1);
        current_subset_id++;
      }  // valid point
    }  // end snake left
  }  // conformal area map
  TEUCHOS_TEST_FOR_EXCEPTION(neighbor_ids.size()!=correlation_points.size()/2,std::runtime_error,"");
  DEBUG_MSG("create_regular_grid_of_correlation_points(): complete, created " << correlation_points.size()/2 << " points");


// code t create simple grid of points, no snaking

//    int_t y_coord = subset_size-1;
//    while(y_coord < img_h - subset_size) {
//      int_t x_coord = subset_size-1;
//      while(x_coord < img_w - subset_size) {
//        if(valid_correlation_point(x_coord,y_coord,subset_size,img_w,img_h,coords,excluded_coords)){
//          correlation_points.push_back((scalar_t)x_coord);
//          correlation_points.push_back((scalar_t)y_coord);
//          //if(proc_rank==0) DEBUG_MSG("ROI " << map_it->first << " adding seed correlation point " << x_coord << " " << y_coord);
//        }
//        x_coord+=step_size;
//      } // end x loop
//      y_coord+=step_size;
//    } // end y loop

//  } // end roi loop
//  DEBUG_MSG("create_regular_grid_of_correlation_points(): complete, created " << correlation_points.size()/2 << " points");
}

DICE_LIB_DLL_EXPORT
bool valid_correlation_point(const int_t x_coord,
  const int_t y_coord,
  const int_t subset_size,
  const int_t img_w,
  const int_t img_h,
  std::set<std::pair<int_t,int_t> > & coords,
  std::set<std::pair<int_t,int_t> > & excluded_coords){
  // need to check if the point is interior to the image by at least one subset_size
  if(x_coord<subset_size-1) return false;
  if(x_coord>img_w-subset_size) return false;
  if(y_coord<subset_size-1) return false;
  if(y_coord>img_h-subset_size) return false;

  // only the centroid has to be inside the ROI
  if(coords.find(std::pair<int_t,int_t>(y_coord,x_coord))==coords.end()) return false;

  static std::vector<int_t> corners_x(4);
  static std::vector<int_t> corners_y(4);
  corners_x[0] = x_coord - subset_size/2;  corners_y[0] = y_coord - subset_size/2;
  corners_x[1] = corners_x[0]+subset_size; corners_y[1] = corners_y[0];
  corners_x[2] = corners_x[0]+subset_size; corners_y[2] = corners_y[0] + subset_size;
  corners_x[3] = corners_x[0];             corners_y[3] = corners_y[0] + subset_size;
  // check four points to see if any of the corners fall in an excluded region
  bool all_corners_in = true;
  for(int_t i=0;i<4;++i){
    if(excluded_coords.find(std::pair<int_t,int_t>(corners_y[i],corners_x[i]))!=excluded_coords.end())
      all_corners_in = false;
  }
  if(!all_corners_in) return false;

//  // check the gradient SSSIG threshold
//  if(grad_threshold > 0.0){
//    TEUCHOS_TEST_FOR_EXCEPTION(!image->has_gradients(),std::runtime_error,
//      "Error, testing valid points for SSSIG tol, but image gradients have not been computed");
//    scalar_t SSSIG = 0.0;
//    for(int_t y=corners_y[0];y<corners_y[2];++y){
//      for(int_t x=corners_x[0];x<corners_x[1];++x){
//        SSSIG += image->grad_x(x,y)*image->grad_x(x,y) + image->grad_x(x,y)*image->grad_x(x,y);
//      }
//    }
//    SSSIG /= (subset_size*subset_size);
//    //std::cout << "x " << x_coord << " y " << y_coord << " SSSIG: " << SSSIG << " threshold " << grad_threshold << std::endl;
//    if(SSSIG < grad_threshold) return false;
//  }
  return true;
}



}// End DICe Namespace
