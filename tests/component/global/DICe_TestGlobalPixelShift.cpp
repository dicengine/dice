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
#include <DICe_GlobalUtils.h>
#include <DICe_Image.h>
#include <DICe_Schema.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;
using namespace DICe::global;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "generating the mms parameters" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> mms_params = Teuchos::rcp(new Teuchos::ParameterList());
  mms_params->set(DICe::problem_name,"div_curl_modulator");
  mms_params->set(DICe::phi_coeff,40.0);
  mms_params->set(DICe::b_coeff,2.0);
  mms_params->print(*outStream);

  *outStream << "generating the correlation and input params" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> input_params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<Teuchos::ParameterList> corr_params = Teuchos::rcp(new Teuchos::ParameterList());
  input_params->set(DICe::subset_file,"pixel_shift_roi.txt");
  input_params->set(DICe::output_folder,"");
  input_params->set(DICe::output_prefix,"pixel_shift");
  corr_params->set(DICe::use_global_dic,true);
  corr_params->set(DICe::global_solver,GMRES_SOLVER);
  corr_params->set(DICe::max_solver_iterations_fast,200);
//  corr_params->set(DICe::interpolation_method,BICUBIC);
//  corr_params->set(DICe::global_formulation,LEVENBERG_MARQUARDT);
//  corr_params->set(DICe::global_formulation,HORN_SCHUNCK);
  corr_params->set(DICe::global_formulation,UNREGULARIZED);
  corr_params->set(DICe::global_element_type,"TRI6");
//  corr_params->set(DICe::use_fixed_point_iterations,false);
  input_params->set(DICe::mesh_size,100.0);//4000.0);
  corr_params->set(DICe::num_image_integration_points,125);
//  corr_params->set(DICe::global_regularization_alpha,0.05);

  *outStream << "generating the mms problem" << std::endl;
  // create an MMS problem
  const int_t w = 500;
  const int_t h = 500;
  MMS_Problem_Factory mms_factory;
  Teuchos::RCP<MMS_Problem> prob = mms_factory.create(mms_params);

  // keep track of the errors
  std::vector<scalar_t> error_x;
  std::vector<scalar_t> error_y;

//  for(scalar_t shift = 0.0;shift<=1.0001;shift+=0.05){
  for(scalar_t shift = 0.0;shift<=1.0001;shift+=0.05){
    *outStream << "SHIFT " << shift << std::endl;
    *outStream << "creating the image set" << std::endl;

    Teuchos::ArrayRCP<intensity_t> ref_intens(w*h,0.0);
    Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
    for(int_t y=0;y<h;++y){
      for(int_t x=0;x<h;++x){
        scalar_t dx = x - shift;
        scalar_t dy = y - shift;
        scalar_t phi_0 = 0.0,phi=0.0;
        prob->phi(x,y,phi_0);
        prob->phi(dx,dy,phi);
        ref_intens[y*w+x] = 0.5 + phi_0*0.5;
        def_intens[y*w+x] = 0.5 + phi*0.5;
      }
    }
    Teuchos::RCP<Image> ref = Teuchos::rcp(new Image(w,h,ref_intens));
    std::stringstream ref_name;
    ref_name << "ref_pixel_shift_" << shift << ".tif";
    ref->write(ref_name.str());
    Teuchos::RCP<Image> def = Teuchos::rcp(new Image(w,h,def_intens));
    //std::stringstream def_name;
    //def_name << "def_pixel_shift_" << shift << ".tif";
    //def->write(def_name.str());

    *outStream << "creating the global roi file" << std::endl;

    std::ofstream roi_file;
    std::string roi_file_name = "pixel_shift_roi.txt";
    roi_file.open(roi_file_name);
    roi_file << "begin region_of_interest\n";
    roi_file << "  begin boundary\n";
    roi_file << "    begin polygon\n";
    roi_file << "      begin vertices\n";
    roi_file << "        20 20\n";
    roi_file << "        480 20\n";
    roi_file << "        480 480\n";
    roi_file << "        20 480\n";
    roi_file << "      end vertices\n";
    roi_file << "    end polygon\n";
    roi_file << "  end boundary\n";
    roi_file << "  use_regular_grid\n";
    roi_file << "  ic_value_x " << shift << "\n";
    roi_file << "  ic_value_y " << shift << "\n";
    roi_file << "  dirichlet_bc boundary 0 0 1 " << shift << " " << shift<<"\n";
    roi_file << "  dirichlet_bc boundary 0 1 2 " << shift << " " << shift<<"\n";
    roi_file << "  dirichlet_bc boundary 0 2 3 " << shift << " " << shift<<"\n";
    roi_file << "  dirichlet_bc boundary 0 3 0 " << shift << " " << shift<<"\n";
//    roi_file << "  neumann_bc boundary 0 0 1\n";
//    roi_file << "  neumann_bc boundary 0 1 2\n";
//    roi_file << "  neumann_bc boundary 0 2 3\n";
//    roi_file << "  neumann_bc boundary 0 3 0\n";
    roi_file << "end region_of_interest\n";
    roi_file.close();

    *outStream << "constructing a schema" << std::endl;

    DICe::Schema schema(ref,def,corr_params);
    schema.initialize(input_params);

    *outStream << "creating and populating the exact solution fields" << std::endl;
    //  schema->global_algorithm()->mesh()->create_field(DICe::mesh::field_enums::EXACT_SOL_VECTOR_FS);
    Teuchos::RCP<MultiField> coords = schema.global_algorithm()->mesh()->get_field(DICe::mesh::field_enums::INITIAL_COORDINATES_FS);
    Teuchos::RCP<MultiField> exact_sol = schema.global_algorithm()->mesh()->get_field(DICe::mesh::field_enums::EXACT_SOL_VECTOR_FS);
    for(int_t i=0;i<schema.global_algorithm()->mesh()->get_scalar_node_dist_map()->get_num_local_elements();++i){
      const int_t ix = i*2+0;
      const int_t iy = i*2+1;
      scalar_t bx=shift,by=shift;
      exact_sol->local_value(ix) = bx;
      exact_sol->local_value(iy) = by;
    }

    //output the grad images
    //  schema->ref_img()->write_grad_x("ref_pixel_shift_grad_x.tif");
    //  schema->ref_img()->write_grad_y("ref_pixel_shift_grad_y.tif");

    *outStream << "executing a correlation" << std::endl;
    schema.execute_correlation();

    // write the output etc.
    schema.post_execution_tasks();

    *outStream << "checking the output" << std::endl;

    scalar_t error_bx = 0.0;
    scalar_t error_by = 0.0;

    Teuchos::RCP<MultiField> disp = schema.global_algorithm()->mesh()->get_field(mesh::field_enums::DISPLACEMENT_FS);
    for(int_t i=0;i<schema.global_algorithm()->mesh()->get_scalar_node_dist_map()->get_num_local_elements();++i){
      const scalar_t b_x = disp->local_value(i*2+0);
      const scalar_t b_y = disp->local_value(i*2+1);
      const scalar_t b_exact_x = exact_sol->local_value(i*2+0);
      const scalar_t b_exact_y = exact_sol->local_value(i*2+1);
      error_bx += (b_exact_x - b_x)*(b_exact_x - b_x);
      error_by += (b_exact_y - b_y)*(b_exact_y - b_y);
    }
    error_bx = std::sqrt(error_bx);
    error_by = std::sqrt(error_by);

    error_x.push_back(error_bx);
    error_y.push_back(error_by);

    const scalar_t error_max = 1.0E-4;
    if(error_bx > error_max || error_by > error_max){
      *outStream << "Error, the solution error is too high" << std::endl;
      errorFlag++;
    }
    corr_params->remove(DICe::mesh_size);
    corr_params->remove(DICe::subset_file);
    corr_params->remove(DICe::output_folder);
    corr_params->remove(DICe::output_prefix);
  } // end shift loop

  *outStream << "-----------------------------------------------------------------------------------------------------------" << std::endl;
  *outStream << "Results Summary:" << std::endl;
  *outStream << "-----------------------------------------------------------------------------------------------------------" << std::endl;
  for(size_t i=0;i<error_x.size();++i){
    *outStream << "shift: " << std::setw(10) << i*0.05 << " error_x: " << std::setw(15)  << error_x[i] << " error_y: " << std::setw(15) << error_y[i] << std::endl;
  }
  *outStream << "-----------------------------------------------------------------------------------------------------------" << std::endl;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

