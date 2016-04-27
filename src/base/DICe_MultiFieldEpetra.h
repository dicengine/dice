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

#ifndef DICE_MULTIFIELDEPETRA_H
#define DICE_MULTIFIELDEPETRA_H

#include <DICe.h>

#include <Epetra_ConfigDefs.h>
#if DICE_MPI
#  include <mpi.h>
#  include <Epetra_MpiComm.h>
#  include <Epetra_SerialComm.h>
#else
#  include <Epetra_SerialComm.h>
#endif

#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Vector.h>
#include <Epetra_Import.h>
#include <Epetra_Export.h>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace DICe {


/// \class DICe::MultiField_Comm
/// \brief MPI Communicator
class DICE_LIB_DLL_EXPORT
MultiField_Comm{
public:
  /// Default constructor with no arguments
  MultiField_Comm(){

#if DICE_MPI
  int mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  if(mpi_is_initialized)
    comm_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  else
    comm_ = Teuchos::rcp(new Epetra_SerialComm);
#else
  // otherwise use a serial communicator
  comm_ = Teuchos::rcp(new Epetra_SerialComm);
#endif
  }
  /// Returns the current processor id
  int get_rank()const{return comm_->MyPID();}

  /// Returns the number of processors
  int get_size()const{return comm_->NumProc();}

  /// Returns a pointer to the underlying communicator
  Teuchos::RCP<Epetra_Comm> get()const{return comm_;}
private:
  /// Pointer to the underlying communicator
  Teuchos::RCP<Epetra_Comm> comm_;
};

/// \class DICe::MultiField_Map
/// \brief MPI distribution map
class DICE_LIB_DLL_EXPORT
MultiField_Map{
public:
  /// \brief Constructor that splits the number of elements evenly across the processors
  /// \param num_elements the global number of elements
  /// \param index_base either 0 or 1
  /// \param comm the MPI communicator
  MultiField_Map(const int_t num_elements,
    const int_t index_base,
    MultiField_Comm & comm){
    map_ = Teuchos::rcp(new Epetra_Map(num_elements, index_base, *comm.get()));
  }

  /// \brief Constructor that takes a list of elements as arguments (this is not necessarily continguous or one-to-one)
  /// \param num_elements the global number of elements, if invalid is supplied, this number will be computed based on the element lists on each processor
  /// \param elements the array of global element ids
  /// \param index_base either 0 or 1
  /// \param comm MPI communicator
  MultiField_Map(const int_t num_elements,
    Teuchos::ArrayView<const int_t> elements,
    const int_t index_base, MultiField_Comm & comm){
    map_ = Teuchos::rcp(new Epetra_Map(num_elements, elements.size(), &elements[0], index_base, *comm.get()));
  }

  /// \brief Constructor that uses the local number of elements per processor and builds the global list
  /// \param num_elements the global number of elements (-1 forces tpetra to decide)
  /// \param num_local_elements the local number of elements
  /// \param index_base either 0 or 1
  /// \param comm the MPI communicator
  MultiField_Map(const int_t num_elements,
    const int_t num_local_elements,
    const int_t index_base,
    MultiField_Comm & comm){
    map_ = Teuchos::rcp(new Epetra_Map(num_elements, num_local_elements, index_base, *comm.get()));
  }

  /// destructor
  virtual ~MultiField_Map(){};

  /// Returns a pointer to the underlying map object
  Teuchos::RCP<const Epetra_Map> get()const{return map_;}

  /// \brief Return the local processor id for the given global id.
  /// Returns -1 if the global id is not on the processor
  /// \param global_id the global id of the element
  int_t get_local_element(const int_t global_id)const{
    return map_->LID(global_id);
  }

  /// \brief Return the global id for the given local id.
  /// \param local_id the local id of the element
  int get_global_element(const int_t local_id)const{
    return map_->GID(local_id);
  }

  /// Returns the total number of global elements
  int_t get_num_global_elements()const{
    return map_->NumGlobalElements();
  }

  /// Returns the number of local elements
  int_t get_num_local_elements()const{
    return map_->NumMyElements();
  }

  /// Returns the max of all the global ids
  int_t get_max_global_index()const{
    return map_->MaxAllGID();
  }

  /// returns a list of the remote indices
  /// \param GIDList the list of global IDs
  /// \param nodeIDList the list of node IDs
  void get_remote_index_list(Teuchos::Array<int_t> & GIDList, Teuchos::Array<int_t> & nodeIDList)const {
    const size_t num_entries = GIDList.size();
    Teuchos::Array<int_t> dummy(num_entries);
    map_->RemoteIDList(num_entries,&GIDList[0],&nodeIDList[0],&dummy[0]);
  }


  /// Reutrns an array view that lists the elements that are local to this process
  Teuchos::ArrayView<const int_t> get_local_element_list()const{
    return Teuchos::ArrayView<const int_t>(map_->MyGlobalElements(),map_->NumMyElements());
  }

  /// Returns true if the map is one to one
  bool is_one_to_one()const{
    return map_->IsOneToOne();
  }

  /// Print the map to the screen
  void describe()const{
    map_->Print(std::cout);
  }

private:
  /// Pointer to the underlying map object for this class
  Teuchos::RCP<const Epetra_Map> map_;
};

/// \class DICe::MultiField_Exporter
/// \brief An exporter directs the tranfer of information between distributed objects
class DICE_LIB_DLL_EXPORT
MultiField_Exporter{
public:
  /// \brief constructor that takes two maps as the arguments
  /// \param map_a the source map
  /// \param map_b the target map
  MultiField_Exporter(MultiField_Map & map_a,
    MultiField_Map & map_b){
    exporter_ = Teuchos::rcp(new Epetra_Export(*map_a.get(),*map_b.get()));
  }

  /// destructor
  virtual ~MultiField_Exporter(){};

  /// Returns a pointer to the underlying exporter object
  Teuchos::RCP<const Epetra_Export> get()const{
    return exporter_;
  }

private:
  /// Pointer to the underlying exporter object
  Teuchos::RCP<const Epetra_Export> exporter_;
};

/// \class DICe::MultiField_Importer
/// \brief An importer directs the tranfer of information between distributed objects
class DICE_LIB_DLL_EXPORT
MultiField_Importer{
public:
  /// \brief Default constructor with two maps as arguments (note the source and target have to be switched due to the Epetra syntax)
  /// \param map_a the source map
  /// \param map_b the target map
  MultiField_Importer(MultiField_Map & map_a,
    MultiField_Map & map_b){
    importer_ = Teuchos::rcp(new Epetra_Import(*map_b.get(),*map_a.get())); // note the target and the source switch for epetra importers
  }

  /// Destructor
  virtual ~MultiField_Importer(){};

  /// Returns a pointer to the underlying importer object
  Teuchos::RCP<const Epetra_Import> get()const{return importer_;}

private:
  /// Pointer to the underlying importer object
  Teuchos::RCP<const Epetra_Import> importer_;
};

/// \class DICe::MultiField
/// \brief A container class for the data structures used in DICe
/// This class was created to enable switching back and forth between data structures based on the
/// compute architecture
class DICE_LIB_DLL_EXPORT
MultiField {
public:
  /// \brief Default constructor
  /// \param map the map that organizes the data across processors
  /// \param num_fields the number of fields stored as columns
  /// \param zero_values set to true if the vectors should be initialized with zeros
  MultiField(Teuchos::RCP<MultiField_Map> & map,
    const int_t & num_fields,
    const bool & zero_values=false){
    map_ = map;
    epetra_mv_ = Teuchos::rcp(new Epetra_MultiVector(*map->get(),num_fields,true));
  }
  /// Destructor
  virtual ~MultiField(){};

  /// Returns a pointer to this field's map
  Teuchos::RCP<MultiField_Map> get_map()const{
    return map_;
  }

  /// Returns the number of fields
  int_t get_num_fields()const{
    return epetra_mv_->NumVectors();
  }

  /// Returns a pointer to the underlying vector data type
  Teuchos::RCP<Epetra_MultiVector> get()const{return epetra_mv_;}

  /// \brief value accessor
  /// \param global_id the global id of the intended element
  /// \param field_index the index of the field to access
  /// Warning: Epetra does not have a scalar type, its hard coded as double
  double & global_value(const int_t global_id,
    const int_t field_index=0){
    return (*epetra_mv_)[field_index][epetra_mv_->Map().LID(global_id)];
  }

  /// \brief value accessor
  /// \param local_id the local id of the intended element
  /// \param field_index the index of the field to access
  /// Warning: Epetra does not have a scalar type, its hard coded as double
  double & local_value(const int_t local_id,
    const int_t field_index=0){
    return (*epetra_mv_)[field_index][local_id];
  }

  /// \brief import the data from one distributed object to this one
  /// \param multifield the multifield to import
  /// \param importer the importer defines how the information will be transferred
  /// \param mode combine mode
  ///
  /// For more information about importing and exporting see the Trilinos docs
  void do_import(Teuchos::RCP<MultiField> multifield,
    MultiField_Importer & importer,
    const Combine_Mode mode=INSERT){
    if(mode==INSERT)
      epetra_mv_->Import(*multifield->get(),*importer.get(),Insert);
    else if(mode==ADD)
      epetra_mv_->Import(*multifield->get(),*importer.get(),Add);
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error, invalid combine mode.");
    }
  }

  /// \brief import the data from one distributed object to this one
  /// \param multifield the multifield to import
  /// \param importer the importer defines how the information will be transferred
  /// \param mode combine mode
  ///
  /// For more information about importing and exporting see the Trilinos docs
  void do_import(Teuchos::RCP<MultiField> multifield,
    MultiField_Exporter & exporter,
    const Combine_Mode mode=INSERT){
    if(mode==INSERT)
      epetra_mv_->Import(*multifield->get(),*exporter.get(),Insert);
    else if(mode==ADD)
      epetra_mv_->Import(*multifield->get(),*exporter.get(),Add);
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error, invalid combine mode.");
    }
  }

  /// \brief export the data from one distributed object to this one
  /// \param multifield the multifield to export
  /// \param exporter the exporter defines how the information will be transferred
  /// \param mode combine mode
  void do_export(Teuchos::RCP<MultiField> multifield,
    MultiField_Exporter & exporter,
    const Combine_Mode mode=INSERT){
    if(mode==INSERT)
      epetra_mv_->Export(*multifield->get(),*exporter.get(),Insert);
    else if(mode==ADD)
      epetra_mv_->Export(*multifield->get(),*exporter.get(),Add);
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error, invalid combine mode.");
    }
  }

  /// Return an array of values for the multifield (most only contain one vector so the first index is 0)
  Teuchos::ArrayRCP<const scalar_t> get_1d_view()const{
    // TODO find a way to avoid this copy. Doing it this way for now
    // because Epetra only has double type, not float so we have to copy/cast
    Teuchos::ArrayRCP<scalar_t> array(epetra_mv_->MyLength());
    for(int_t i=0;i<epetra_mv_->MyLength();++i){
      array[i] = (*epetra_mv_)[0][i];
    }
    return array;
  }


private:
  /// Pointer to the underlying data type
  Teuchos::RCP<Epetra_MultiVector> epetra_mv_;
  /// Pointer to the underlying map
  Teuchos::RCP<MultiField_Map> map_;
};

}// End DICe Namespace

#endif
