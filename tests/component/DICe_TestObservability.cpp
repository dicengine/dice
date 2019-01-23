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
#include <DICe_Schema.h>
#include <DICe_Image.h>
#include <DICe_FFT.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_LAPACK.hpp>

#include <iostream>
#include <random>
#include <fstream>

using namespace DICe;
using namespace DICe::field_enums;

intensity_t phi(const scalar_t & x, const scalar_t & y, const scalar_t & period, const scalar_t & L){
  assert(period!=0.0);
  const scalar_t freq = 1.0/period;
  const scalar_t gamma = 2.0*freq*DICE_PI;
  return 255.0*0.5 + 0.5*255.0*std::cos(gamma*x)*std::cos(gamma*y);
}

void compute_b(const scalar_t & x, const scalar_t & y, const scalar_t & period, const scalar_t & amplitude, const scalar_t & L,
  scalar_t & bx, scalar_t & by){
  bx = 0.0;
  by = 0.0;
  assert(period!=0.0);
  const scalar_t freq = 1.0/period;
  const scalar_t gamma = 2.0*freq*DICE_PI;
  bx = amplitude*0.5 + 0.5*amplitude*std::sin(gamma*x)*std::cos(gamma*y);
  by = amplitude*0.5 - 0.5*amplitude*std::cos(gamma*x)*std::sin(gamma*y);
}

struct computed_point
{
    scalar_t x_, y_, bx_, by_, sol_bx_, sol_by_,error_bx_,error_by_;
};

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  scalar_t errorTol = 10.0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "argc " << argc << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(argc!=1&&argc!=2&&argc!=7&&argc!=11&&argc!=12,std::runtime_error,"Error, invalid syntax");

  std::string var_name = "";
  scalar_t start_val = 0.0;
  scalar_t end_val = 0.0;
  scalar_t step_val = 0.0;
  int_t print_style = 0.0;
  std::string var_name2 = "";
  scalar_t start_val2 = 0.0;
  scalar_t end_val2 = 0.0;
  scalar_t step_val2 = 1.0;
  scalar_t noise_mag = 0.0; // counts
  scalar_t regularization_mag = 0.0; // counts

  // set up default for test case
  if(argc==1||argc==2){
    var_name = "subset_size";
    start_val = 10.0;
    end_val = 25.0;
    step_val = 5.0;
  }
  else{
    // parse the command line options
    TEUCHOS_TEST_FOR_EXCEPTION(argc!=7&&argc!=11&&argc!=12,std::runtime_error,"Error, invalid syntax");
    var_name = argv[1];
    start_val = std::strtod(argv[2],NULL);
    end_val = std::strtod(argv[3],NULL);
    step_val = std::strtod(argv[4],NULL);
    //noise_mag = std::strtod(argv[5],NULL); // TODO turn noise back on later!!
    print_style = std::stoi(argv[6]);
    if(argc > 7){
      TEUCHOS_TEST_FOR_EXCEPTION(argc!=11&&argc!=12,std::runtime_error,"");
      var_name2 = argv[7];
      start_val2 = std::strtod(argv[8],NULL);
      end_val2 = std::strtod(argv[9],NULL);
      step_val2 = std::strtod(argv[10],NULL);
    }
    if(argc > 11){
      TEUCHOS_TEST_FOR_EXCEPTION(argc!=12,std::runtime_error,"");
      regularization_mag = std::strtod(argv[11],NULL);
    }
  }

  std::default_random_engine generator;
  std::normal_distribution<intensity_t> distribution(0.0,0.5);

  *outStream << "variable name:      " << var_name << std::endl;
  *outStream << "start value:        " << start_val << std::endl;
  *outStream << "end value:          " << end_val << std::endl;
  *outStream << "step value:         " << step_val << std::endl;
  *outStream << "print style:        " << print_style << std::endl;
  *outStream << "variable name 2:    " << var_name2 << std::endl;
  *outStream << "start value 2:      " << start_val2 << std::endl;
  *outStream << "end value 2:        " << end_val2 << std::endl;
  *outStream << "step value 2:       " << step_val2 << std::endl;
  *outStream << "noise_mag:          " << noise_mag << std::endl;
  *outStream << "regularization_mag: " << regularization_mag << std::endl;

  // test for valid variable name
  if(var_name!="subset_size"&&
      var_name!="step_size"&&
      var_name!="phi_period"&&
      var_name!="b_period"&&
      var_name!="b_amp"){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid variable name " << var_name);
  }
  // test for valid variable name
  if(var_name2!="subset_size"&&
      var_name2!="step_size"&&
      var_name2!="phi_period"&&
      var_name2!="b_period"&&
      var_name2!="b_amp"&&
      var_name2!=""){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid variable 2 name " << var_name2);
  }

  std::stringstream results_stream;
  results_stream << std::setw(12) << var_name << std::setw(12) << var_name2 << std::setw(10) << "norm x" << std::setw(16) << "pk rel er_x(%)" << std::setw(15) << "std dev x (%)" <<
      std::setw(10) << "norm y" << std::setw(16) << "pk rel er_y(%)" << std::setw(15) << "std dev y(%)" <<std::endl;

  // Default values:

  const int_t L = 500;
  int_t subset_size = 25;
  int_t step_size = 5;
  scalar_t phi_period = 10; // pixels for each speckle (includes dark and light patch)
  scalar_t b_period = 100; // pixels for each wave in the disp profile
  scalar_t b_amp = 1.0;
  scalar_t param_value = 0.0;
  scalar_t param_value2 = 0.0;

  Teuchos::ArrayRCP<intensity_t> dummy_intensities(L*L,0.0);
  Teuchos::RCP<Image> dummy_img = Teuchos::rcp(new Image(L,L,dummy_intensities));
  Teuchos::RCP<Schema> schema;
  const bool inner_init_required = var_name=="subset_size"||var_name=="step_size"||var_name2=="subset_size"||var_name2=="step_size";

  if(!inner_init_required){
    schema = Teuchos::rcp(new DICe::Schema(L,L,step_size,step_size,subset_size));
    schema->set_ref_image(dummy_img);
    schema->set_def_image(dummy_img);
  }

  for(scalar_t val1=start_val;val1<=end_val;val1+=step_val){
    // parse the variables
    if(var_name=="subset_size"){
      TEUCHOS_TEST_FOR_EXCEPTION(val1-(int_t)val1!=0.0,std::runtime_error,"");
      subset_size = (int_t)val1;
      param_value = subset_size;
    }
    else if(var_name=="step_size"){
      TEUCHOS_TEST_FOR_EXCEPTION(val1-(int_t)val1!=0.0,std::runtime_error,"");
      step_size = (int_t)val1;
      param_value = step_size;
    }
    else if(var_name=="phi_period"){
      TEUCHOS_TEST_FOR_EXCEPTION(val1-(int_t)val1!=0.0,std::runtime_error,"");
      phi_period = (int_t)val1;
      param_value = phi_period;
    }
    else if(var_name=="b_period"){
      TEUCHOS_TEST_FOR_EXCEPTION(val1-(int_t)val1!=0.0,std::runtime_error,"");
      b_period = (int_t)val1;
      param_value = b_period;
    }
    else if(var_name=="b_amp"){
      b_amp = val1;
      param_value = b_amp;
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }

    for(scalar_t val2=start_val2;val2<=end_val2;val2+=step_val2){
      // parse the variables
      if(var_name2==""){
      }
      else if(var_name2=="subset_size"){
        TEUCHOS_TEST_FOR_EXCEPTION(val2-(int_t)val2!=0.0,std::runtime_error,"");
        subset_size = (int_t)val2;
        param_value2 = subset_size;
      }
      else if(var_name2=="step_size"){
        TEUCHOS_TEST_FOR_EXCEPTION(val2-(int_t)val2!=0.0,std::runtime_error,"");
        step_size = (int_t)val2;
        param_value2 = step_size;
      }
      else if(var_name2=="phi_period"){
        TEUCHOS_TEST_FOR_EXCEPTION(val2-(int_t)val2!=0.0,std::runtime_error,"");
        phi_period = (int_t)val2;
        param_value2 = phi_period;
      }
      else if(var_name2=="b_period"){
        TEUCHOS_TEST_FOR_EXCEPTION(val2-(int_t)val2!=0.0,std::runtime_error,"");
        b_period = (int_t)val2;
        param_value2 = b_period;
      }
      else if(var_name2=="b_amp"){
        b_amp = val2;
        param_value2 = b_amp;
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      *outStream << "subset " << subset_size << " step " << step_size << " phi_period " << phi_period << " b_period " << b_period << " b_amp " << b_amp << std::endl;

      if(inner_init_required){
        schema = Teuchos::rcp(new DICe::Schema(L,L,step_size,step_size,subset_size));
        schema->set_ref_image(dummy_img);
        schema->set_def_image(dummy_img);
      }
      const int_t fw = (L-2.0*subset_size)/step_size + 1;
      const int_t fh = (L-2.0*subset_size)/step_size + 1;
      assert(step_size!=0.0);
      const scalar_t sampling_b = 1.0/step_size;
      assert(b_period!=0.0);
      const scalar_t nyquist_b = 2.0/b_period;
      //*outStream << "Nyquist b: " << nyquist_b << " sampling b: " << sampling_b << " sampling rate must be greater than Nyquist" << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(nyquist_b >= sampling_b,std::runtime_error,"Error, sampling for b is below Nyquist freq.");
      Teuchos::ArrayRCP<intensity_t> intensities(L*L,0.0);
      for(int_t j=0;j<L;++j){
        for(int_t i=0;i<L;++i){
          intensities[j*L+i] = phi(i,j,phi_period,L);
        }
      }
      Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
      params->set(DICe::compute_image_gradients,true);
      Teuchos::RCP<Image> img = Teuchos::rcp(new Image(L,L,intensities,params));
      Teuchos::RCP<MultiField> bx_command = schema->mesh()->get_field(DICe::field_enums::FIELD_1_FS);
      Teuchos::RCP<MultiField> by_command = schema->mesh()->get_field(DICe::field_enums::FIELD_2_FS);
      Teuchos::RCP<MultiField> bx_computed = schema->mesh()->get_field(DICe::field_enums::FIELD_3_FS);
      Teuchos::RCP<MultiField> by_computed = schema->mesh()->get_field(DICe::field_enums::FIELD_4_FS);
      Teuchos::RCP<MultiField> error_x = schema->mesh()->get_field(DICe::field_enums::FIELD_5_FS);
      Teuchos::RCP<MultiField> error_y = schema->mesh()->get_field(DICe::field_enums::FIELD_6_FS);

      schema->update_frame_id();

      int N = 2;
      int *IPIV = new int[N+1];
      int LWORK = N*N;
      int INFO = 0;
      double *WORK = new double[LWORK];
      Teuchos::LAPACK<int_t,double> lapack;
      // Initialize storage:
      Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
      Teuchos::ArrayRCP<double> q(N,0.0);
      Teuchos::ArrayRCP<double> observed_b(N,0.0);

      // loop over the subsets
      scalar_t error_norm_x = 0.0;
      scalar_t error_norm_y = 0.0;
      for(int_t sub=0;sub<schema->local_num_subsets();++sub){
        H(0,0) = 0.0;
        H(1,0) = 0.0;
        H(0,1) = 0.0;
        H(1,1) = 0.0;
        q[0] = 0.0;
        q[1] = 0.0;
        observed_b[0] = 0.0;
        observed_b[1] = 0.0;
        scalar_t b_dot_grad_phi = 0.0;
        scalar_t bx = 0.0;
        scalar_t by = 0.0;
        const int_t xc = schema->local_field_value(sub,SUBSET_COORDINATES_X_FS);
        const int_t yc = schema->local_field_value(sub,SUBSET_COORDINATES_Y_FS);
        for(int_t j=yc-subset_size/2;j<yc+subset_size/2;++j){
          for(int_t i=xc-subset_size/2;i<xc+subset_size/2;++i){
            H(0,0) += img->grad_x(i,j)*img->grad_x(i,j);
            H(0,1) += img->grad_x(i,j)*img->grad_y(i,j);
            H(1,0) += img->grad_y(i,j)*img->grad_x(i,j);
            H(1,1) += img->grad_y(i,j)*img->grad_y(i,j);
            H(0,0) += regularization_mag;
            H(1,1) += regularization_mag;
            compute_b(i,j,b_period,b_amp,L,bx,by);
            if(noise_mag > 0.0){
                intensity_t pert = distribution(generator);
                //std::cout << "pert: " << pert << " noise_mag " << noise_mag << std::endl;
                bx+= pert*noise_mag;
                by+= pert*noise_mag;
            }
            b_dot_grad_phi = bx*img->grad_x(i,j) + by*img->grad_y(i,j);
            q[0] += b_dot_grad_phi * img->grad_x(i,j);
            q[1] += b_dot_grad_phi * img->grad_y(i,j);
          }
        }
        try
        {
          lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
        }
        catch(std::exception &e){
          std::cout << e.what() << '\n';
          return LINEAR_SOLVE_FAILED;
        }
        for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
        try
        {
          lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
        }
        catch(std::exception &e){
          std::cout << e.what() << '\n';
          return LINEAR_SOLVE_FAILED;
        }

        for(int_t i=0;i<N;++i)
          for(int_t j=0;j<N;++j)
            observed_b[i] += H(i,j)*q[j];

        compute_b(xc,yc,b_period,b_amp,L,bx,by);
        //std::cout << "observed x: " << observed_b[0] << " analytical x: " << bx << " observed y: " << observed_b[1] << " analytical_y: " << by << std::endl;

        bx_command->local_value(sub) = bx;
        by_command->local_value(sub) = by;
        bx_computed->local_value(sub) = observed_b[0];
        by_computed->local_value(sub) = observed_b[1];
        assert(b_amp!=0.0);
        error_x->local_value(sub) = (bx-observed_b[0])/b_amp*100.0;
        error_y->local_value(sub) = (by-observed_b[1])/b_amp*100.0;
        error_norm_x += (bx-observed_b[0])*(bx-observed_b[0]);
        error_norm_y += (by-observed_b[1])*(by-observed_b[1]);
      } // end subset loop
      error_norm_x = std::sqrt(error_norm_x);
      error_norm_y = std::sqrt(error_norm_y);

      // compute error measure based on finding the solution peaks and taking the average error at the peaks (Phil's method)

      std::vector<computed_point> approx_points(fw*fh);
      int_t subset_index = 0;
      for(int_t j=0;j<fh;++j){
        for(int_t i=0;i<fw;++i){
          const scalar_t x = schema->local_field_value(subset_index,SUBSET_COORDINATES_X_FS);
          const scalar_t y = schema->local_field_value(subset_index,SUBSET_COORDINATES_Y_FS);
          approx_points[subset_index].x_ = x;
          approx_points[subset_index].y_ = y;
          approx_points[subset_index].bx_ = bx_computed->local_value(subset_index);
          approx_points[subset_index].by_ = by_computed->local_value(subset_index);
          approx_points[subset_index].sol_bx_ = bx_command->local_value(subset_index);
          approx_points[subset_index].sol_by_ = by_command->local_value(subset_index);
          approx_points[subset_index].error_bx_ = error_x->local_value(subset_index);
          approx_points[subset_index].error_by_ = error_y->local_value(subset_index);
          subset_index++;
        }
      }
      // now sort the approximate points to pick out the peaks
      int_t num_peaks = ((L/b_period)*2-2)*(L/b_period-1);
      if(num_peaks > 100) num_peaks=100;
      *outStream << "num_peaks: " << num_peaks; // << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(num_peaks<=5,std::runtime_error,"");
      // bx-sort
      std::sort(std::begin(approx_points), std::end(approx_points), [](const computed_point& a, const computed_point& b)
        {
        // sort based on bx alone (assumes that by is a peak at the same location
        return a.bx_ > b.bx_;
        });
      // compute the average of the top num_peaks error
      scalar_t avg_error_x = 0.0;
      for(int_t i=0;i<num_peaks;++i){
        //*outStream << "peak: " << i << " command bx: " << approx_points[i].sol_bx_ << " comp bx: " << approx_points[i].bx_ <<  " rel error bx: " << approx_points[i].error_bx_ << "%" <<  std::endl;
        avg_error_x += std::abs(approx_points[i].error_bx_);
      }
      assert(num_peaks!=0);
      avg_error_x /= num_peaks;
      scalar_t std_dev_x = 0.0;
      for(int_t i=0;i<num_peaks;++i){
        std_dev_x += (approx_points[i].error_bx_-avg_error_x)*(approx_points[i].error_bx_-avg_error_x);
      }
      std_dev_x /= num_peaks;
      std_dev_x = std::sqrt(std_dev_x);
      // by-sort
      std::sort(std::begin(approx_points), std::end(approx_points), [](const computed_point& a, const computed_point& b)
        {
        // sort based on bx alone (assumes that by is a peak at the same location
        return a.by_ > b.by_;
        });
      // compute the average of the top num_peaks error
      scalar_t avg_error_y = 0.0;
      for(int_t i=0;i<num_peaks;++i){
        //*outStream << "peak: " << i << " command by: " << approx_points[i].sol_by_ << " rel error by: " << approx_points[i].error_by_ << std::endl;
        avg_error_y += std::abs(approx_points[i].error_by_);
      }
      avg_error_y /= num_peaks;
      scalar_t std_dev_y = 0.0;
      for(int_t i=0;i<num_peaks;++i){
        std_dev_y += (approx_points[i].error_by_-avg_error_y)*(approx_points[i].error_by_-avg_error_y);
      }
      std_dev_y /= num_peaks;
      std_dev_y = std::sqrt(std_dev_y);

//      scalar_t min_ex=0.0,max_ex=0.0,avg_ex=0.0,std_dev_x=0.0;
//      scalar_t min_ey=0.0,max_ey=0.0,avg_ey=0.0,std_dev_y=0.0;
//      schema->mesh()->field_stats(DICe::field_enums::FIELD_5_FS,min_ex,max_ex,avg_ex,std_dev_x,0);
//      schema->mesh()->field_stats(DICe::field_enums::FIELD_6_FS,min_ey,max_ey,avg_ey,std_dev_y,0);

      *outStream << " avg peak rel error x: " << avg_error_x << "% y: " << avg_error_y << "%" << std::endl;
      if(std::abs(avg_error_x) > errorTol){
        errorFlag++;
        *outStream << "Error, average error x too high." << std::endl;
      }
      if(std::abs(avg_error_y) > errorTol){
        errorFlag++;
        *outStream << "Error, average error y too high." << std::endl;
      }

      results_stream << std::setw(12) << param_value << std::setw(12) << param_value2 << std::setw(10) << error_norm_x << std::setw(16) << avg_error_x << std::setw(15) << std_dev_x <<
          std::setw(10) << error_norm_y << std::setw(16) << avg_error_y << std::setw(15) << std_dev_y << std::endl;

      delete [] IPIV;
      delete [] WORK;

      if(print_style==1)
        schema->write_output("./");

    } // end val2 iteration loop
  } // end val1 iteration loop

  *outStream << std::endl << "----------------------------------- RESULTS -----------------------------------" << std::endl << std::endl;
  *outStream << results_stream.str();

  // write the results to file:

  std::ofstream myfile ("observability_results.txt");
  if (myfile.is_open())
  {
    myfile << results_stream.str();
    myfile.close();
  }
  else *outStream << "Unable to write output file";

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

