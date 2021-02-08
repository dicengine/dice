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

#include <DICe.h>
#include <DICe_GlobalUtils.h>
#include <DICe_Image.h>
#include <DICe_Schema.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>
#include <fstream>

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
  mms_params->set(DICe::phi_coeff,50.0);
  mms_params->set(DICe::b_coeff,2.0);
  mms_params->print(*outStream);

  *outStream << "generating the correlation and input params" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> input_params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<Teuchos::ParameterList> corr_params = Teuchos::rcp(new Teuchos::ParameterList());
  input_params->set(DICe::mesh_size,4000.0);
  input_params->set(DICe::subset_file,"image_terms_roi.txt");
  input_params->set(DICe::output_folder,"");
  input_params->set(DICe::output_prefix,"image_terms");
  input_params->set(DICe::image_folder,"");
  corr_params->set(DICe::use_global_dic,true);
  corr_params->set(DICe::global_formulation,UNREGULARIZED);
  corr_params->set(DICe::global_solver,GMRES_SOLVER);

  *outStream << "generating the mms problem" << std::endl;
  // create an MMS problem
  const int_t w = 500;
  const int_t h = 500;
  MMS_Problem_Factory mms_factory;
  Teuchos::RCP<MMS_Problem> prob = mms_factory.create(mms_params);

  *outStream << "creating the image set" << std::endl;

  Teuchos::ArrayRCP<storage_t> ref_intens(w*h,0.0);
  Teuchos::ArrayRCP<storage_t> def_intens(w*h,0.0);
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      // compute the velocity for this point
      work_t bx=0.0,by=0.0;
      prob->velocity(x,y,bx,by);
      //std::cout << " x " << x << " y " << y << " bx " << bx << " by " << by << std::endl;
      work_t dx = x - bx;
      work_t dy = y - by;
      work_t phi_0 = 0.0,phi=0.0;
      prob->phi(x,y,phi_0);
      prob->phi(dx,dy,phi);
      ref_intens[y*w+x] = static_cast<storage_t>(0.5*255.0 + phi_0*0.5*255.0);
      def_intens[y*w+x] = static_cast<storage_t>(0.5*255.0 + phi*0.5*255.0);
    }
  }

  Teuchos::RCP<Image> ref = Teuchos::rcp(new Image(w,h,ref_intens));
  ref->write("ref_image_terms.tif");
  Teuchos::RCP<Image> def = Teuchos::rcp(new Image(w,h,def_intens));
  def->write("def_image_terms.tif");
  input_params->set(DICe::reference_image,"ref_image_terms.tif");
  Teuchos::ParameterList def_img_params;
  def_img_params.set("def_image_terms.tif",true);
  input_params->set(DICe::deformed_images,def_img_params);


  *outStream << "creating the global roi file" << std::endl;

  std::ofstream roi_file;
  std::string roi_file_name = "image_terms_roi.txt";
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
  roi_file << "  dirichlet_bc boundary 0 0 1 2 use_subsets 27\n";
  roi_file << "  dirichlet_bc boundary 0 1 2 2 use_subsets 27\n";
  roi_file << "  dirichlet_bc boundary 0 2 3 2 use_subsets 27\n";
  roi_file << "  dirichlet_bc boundary 0 3 0 2 use_subsets 27\n";
  roi_file << "end region_of_interest\n";
  roi_file.close();

  *outStream << "constructing a schema" << std::endl;

  Teuchos::RCP<DICe::Schema> schema = Teuchos::rcp(new DICe::Schema(input_params,corr_params));
  schema->set_ref_image(ref);
  schema->set_def_image(def);

  *outStream << "creating and populating the exact solution fields" << std::endl;
//  schema->global_algorithm()->mesh()->create_field(DICe::field_enums::EXACT_SOL_VECTOR_FS);
  Teuchos::RCP<MultiField> coords = schema->global_algorithm()->mesh()->get_field(DICe::field_enums::INITIAL_COORDINATES_FS);
  Teuchos::RCP<MultiField> exact_sol = schema->global_algorithm()->mesh()->get_field(DICe::field_enums::EXACT_SOL_VECTOR_FS);
  for(int_t i=0;i<schema->global_algorithm()->mesh()->get_scalar_node_dist_map()->get_num_local_elements();++i){
    const int_t ix = i*2+0;
    const int_t iy = i*2+1;
    const work_t x = coords->local_value(ix);
    const work_t y = coords->local_value(iy);
    work_t bx=0.0,by=0.0;
    prob->velocity(x,y,bx,by);
    exact_sol->local_value(ix) = bx;
    exact_sol->local_value(iy) = by;
  }

  //output the grad images
  schema->ref_img()->write_grad_x("ref_image_terms_grad_x.tif");
  schema->ref_img()->write_grad_y("ref_image_terms_grad_y.tif");

  *outStream << "executing a correlation" << std::endl;
  schema->execute_correlation();

  // write the output etc.
  schema->post_execution_tasks();

  *outStream << "checking the output" << std::endl;

  work_t error_bx = 0.0;
  work_t error_by = 0.0;

  Teuchos::RCP<MultiField> disp = schema->global_algorithm()->mesh()->get_field(field_enums::DISPLACEMENT_FS);
  for(int_t i=0;i<schema->global_algorithm()->mesh()->get_scalar_node_dist_map()->get_num_local_elements();++i){
    int_t ix = i*2+0;
    int_t iy = i*2+1;
    const work_t b_x = disp->local_value(ix);
    const work_t b_y = disp->local_value(iy);
    const work_t b_exact_x = exact_sol->local_value(ix);
    const work_t b_exact_y = exact_sol->local_value(iy);
    error_bx += (b_exact_x - b_x)*(b_exact_x - b_x);
    error_by += (b_exact_y - b_y)*(b_exact_y - b_y);
  }
  error_bx = std::sqrt(error_bx);
  error_by = std::sqrt(error_by);

  *outStream << "ERROR_X: " << error_bx << " ERROR_Y: " << error_by << std::endl;

  const work_t error_max = 0.1;
  if(error_bx > error_max || error_by > error_max){
    *outStream << "Error, the solution error is too high" << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

