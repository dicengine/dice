// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

#include <cassert>
#include <iostream>
#include <fstream>
#include <ctype.h>

#include <DICe_ImageIO.h>
#include <DICe_Rawi.h>

#if DICE_ENABLE_NETCDF
  #include <DICe_NetCDF.h>
#endif

namespace DICe{
namespace utils{

DICE_LIB_DLL_EXPORT
std::string netcdf_file_name(const char * decorated_netcdf_file){
  std::string netcdf_string(decorated_netcdf_file);
  // trim off the last underscore and the rest
  size_t found = netcdf_string.find("_frame_");
  if(found==std::string::npos){
    return netcdf_string;
  }
  std::string file_name = netcdf_string.substr(0,found);
  // add the .nc extension back
  file_name+=".nc";
  return file_name;
}

DICE_LIB_DLL_EXPORT
std::string video_file_name(const char * decorated_video_file){
  std::string video_string(decorated_video_file);
  // trim off the last underscore and the rest
  size_t found = video_string.find_last_of("_");
  std::string file_name;
  if(found==std::string::npos||
      (!std::isdigit(*(video_string.substr(found+1).begin()))&&*(video_string.substr(found+1).begin())!='-'&&video_string.substr(found+1,3)!="avg")){
    file_name = video_string;
  }else{
    file_name = video_string.substr(0,found);
  }
  // add the .video extension back
  const std::string ext = video_string.substr(video_string.find_last_of("."));
  file_name+=ext;
  return file_name;
}

DICE_LIB_DLL_EXPORT
int_t netcdf_index(const char * decorated_netcdf_file){
  std::string netcdf_string(decorated_netcdf_file);
  // trim off the last underscore and the rest
  size_t found = netcdf_string.find("_frame_");
  if(found==std::string::npos) return 0;
  size_t found_underscore = netcdf_string.find_last_of("_");
  size_t found_ext = netcdf_string.find(".nc");
  std::string first_num_str = netcdf_string.substr(found_underscore+1,found_ext - found_underscore - 1);
  DEBUG_MSG("netcdf_index(): " << first_num_str);
  return std::strtol(first_num_str.c_str(),NULL,0);
}

DICE_LIB_DLL_EXPORT
void video_index(const char * decorated_video_file,
  int_t & start_index,
  int_t & end_index,
  bool & is_avg){
  end_index = -1;
  is_avg = false;
  std::string video_string(decorated_video_file);
  // trim off the last underscore and the rest
  size_t found = video_string.find_last_of("_");
  std::string tail = video_string.substr(found);
  //DEBUG_MSG("video_index(): tail: " << tail);
  // determine if this is an average image
  size_t found_avg = tail.find("avg");
  size_t found_ext = tail.find_last_of(".");
  //DEBUG_MSG("video_index(): found average at position: " << found_avg);
  if(found_avg==std::string::npos){
    assert(found_ext!=std::string::npos);
    std::string index_str = tail.substr(1,found_ext-1);
    //DEBUG_MSG("video_index(): index_str: " << index_str);
    start_index = std::strtol(index_str.c_str(),NULL,0);
  }
  else{
    size_t found_dash = tail.find("to");
    std::string first_num_str = tail.substr(found_avg+3,found_dash-found_avg-3);
    std::string second_num_str = tail.substr(found_dash+2,found_ext-found_dash-2);
    //DEBUG_MSG("found_index(): first num " << first_num_str << " second num " << second_num_str);
    start_index = std::strtol(first_num_str.c_str(),NULL,0);
    end_index = std::strtol(second_num_str.c_str(),NULL,0);
    is_avg = true;
  }
}


DICE_LIB_DLL_EXPORT
Image_File_Type image_file_type(const char * file_name){
  Image_File_Type file_type = NO_SUCH_IMAGE_FILE_TYPE;
  // determine the file type based on the file_name
  const std::string file_str(file_name);
  const std::string rawi(".rawi");
  if(file_str.find(rawi)!=std::string::npos)
    return RAWI;
  const std::string cine(".cine");
  if(file_str.find(cine)!=std::string::npos)
    return CINE;
  const std::string tif("tif");
  const std::string tiff("tiff");
  const std::string tif_caps("TIF");
  const std::string tiff_caps("TIFF");
  if(file_str.find(tif)!=std::string::npos||file_str.find(tiff)!=std::string::npos||file_str.find(tif_caps)!=std::string::npos
      ||file_str.find(tiff_caps)!=std::string::npos)
    return TIFF;
  const std::string jpg("jpg");
  const std::string jpeg("jpeg");
  const std::string jpg_caps("JPG");
  const std::string jpeg_caps("JPEG");
  if(file_str.find(jpg)!=std::string::npos||file_str.find(jpeg)!=std::string::npos||file_str.find(jpg_caps)!=std::string::npos
      ||file_str.find(jpeg_caps)!=std::string::npos)
    return JPEG;
  const std::string bmp("bmp");
  const std::string bmp_caps("BMP");
  if(file_str.find(bmp)!=std::string::npos||file_str.find(bmp_caps)!=std::string::npos)
    return BMP;
  const std::string png("png");
  const std::string png_caps("PNG");
  if(file_str.find(png)!=std::string::npos||file_str.find(png_caps)!=std::string::npos)
    return PNG;
  const std::string nc(".nc");
  if(file_str.find(nc)!=std::string::npos)
    return NETCDF;
  const std::string avi(".avi");
  const std::string mp4(".mp4");
  if(file_str.find(avi)!=std::string::npos||file_str.find(mp4)!=std::string::npos)
    return VIDEO;
  return file_type;
}

DICE_LIB_DLL_EXPORT
void read_image_dimensions(const char * file_name,
  int_t & width,
  int_t & height){

  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);

  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  if(file_type==RAWI){
    read_rawi_image_dimensions(file_name,width,height);
  }
  else if(file_type==CINE||file_type==VIDEO){
    const std::string video_file = video_file_name(file_name);
    DEBUG_MSG("read_image_dimensions(): video file name: " << video_file);
    DICe::utils::Video_Singleton::instance().image_dimensions(video_file,width,height);
    TEUCHOS_TEST_FOR_EXCEPTION(width<=0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(height<=0,std::runtime_error,"");
  }
#if DICE_ENABLE_NETCDF
  else if(file_type==NETCDF){
    netcdf::NetCDF_Reader netcdf_reader;
    int_t num_time_steps = 0;
    const std::string netcdf_file = netcdf_file_name(file_name);
    DEBUG_MSG("read_image_dimensions(): netcdf file name: " << netcdf_file);
    netcdf_reader.get_image_dimensions(netcdf_file,width,height,num_time_steps);
  }
#endif
  else{
    DEBUG_MSG("read_image_dimensions(): (opencv) file name: " << file_name);
    cv::Mat image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
    if (image.empty()) {
      DEBUG_MSG("utils::read_image(): image is empty, it could have unicode characters in the name so trying to read with a buffer instead");
      std::ifstream f(file_name,std::iostream::binary);
      if(!f.good()){
        std::cout << "utils::read_image_dimensions(): image read failed." << std::endl;
        height = 0;
        width = 0;
        return;
      }
      std::filebuf* pbuf = f.rdbuf();
      size_t size = pbuf->pubseekoff(0, f.end, f.in);
      pbuf->pubseekpos(0, f.in);
      std::vector<uchar> buffer(size);
      pbuf->sgetn((char*)buffer.data(), size);
      image = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
        std::cout << "utils::read_image_dimensions(): image read failed." << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
    }
    height = image.rows;
    width = image.cols;
  }
}

template <typename S>
DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  Teuchos::ArrayRCP<S> & intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  int_t sub_w=0;
  int_t sub_h=0;
  int_t sub_offset_x=0;
  int_t sub_offset_y=0;
  storage_t cuttoff = 0;
  bool layout_right=true;
  bool is_subimage=false;
  bool filter_failed_pixels = true;
  bool convert_to_8_bit = true;
  bool reinit = false;
  bool use_threshold = false;
  bool buffer_persistence_guaranteed = false;
  if(params!=Teuchos::null){
    sub_w = params->isParameter(subimage_width) ? params->get<int_t>(subimage_width) : 0;
    sub_h = params->isParameter(subimage_height) ? params->get<int_t>(subimage_height) : 0;
    sub_offset_x = params->isParameter(subimage_offset_x) ? params->get<int_t>(subimage_offset_x) : 0;
    sub_offset_y = params->isParameter(subimage_offset_y) ? params->get<int_t>(subimage_offset_y) : 0;
    is_subimage=params->isParameter(subimage_width)||
        params->isParameter(subimage_height)||
        params->isParameter(subimage_offset_x)||
        params->isParameter(subimage_offset_y);
    layout_right = params->get<bool>(is_layout_right,true);
    reinit = params->get(reinitialize_cine_reader_threshold,reinit);
    use_threshold = params->get(use_threshold_for_failed_cine_pixels,use_threshold);
    filter_failed_pixels = params->get<bool>(filter_failed_cine_pixels,filter_failed_pixels);
    convert_to_8_bit = params->get<bool>(convert_cine_to_8_bit,convert_to_8_bit);
    buffer_persistence_guaranteed = params->get<bool>(DICe::buffer_persistence_guaranteed,buffer_persistence_guaranteed);
  }
  DEBUG_MSG("utils::read_image(): file name: " << file_name);
  DEBUG_MSG("utils::read_image(): sub_w: " << sub_w << " sub_h: " << sub_h << " offset_x: " << sub_offset_x << " offset_y: " << sub_offset_y);
  DEBUG_MSG("utils::read_image(): is_layout_right: " << layout_right);
  DEBUG_MSG("utils::read_image(): filter_failed_pixels: " << filter_failed_pixels);
  DEBUG_MSG("utils::read_image(): convert_to_8_bit: " << convert_to_8_bit);
  DEBUG_MSG("utils::read_image(): reinit: " << reinit);

  int_t width = 0;
  int_t height = 0;
  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);
  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  if(file_type==RAWI){
    if(is_subimage){
      std::cerr << "Error, reading only a portion of an image is not supported for rawi, file name: " << file_name << "\n";
      throw std::exception();
    }
    if(intensities.size()==0){
      read_rawi_image_dimensions(file_name,width,height);
      intensities = Teuchos::ArrayRCP<S>(width*height,0.0);
    }
    read_rawi_image(file_name,intensities.getRawPtr(),layout_right);
  }
  else if(file_type==VIDEO){
    const std::string video_file = video_file_name(file_name);
    DEBUG_MSG("utils::read_image(): intensity values deep copied from VideoCapture object");
    DEBUG_MSG("utils::read_image(): video file name: " << video_file);
    int_t end_index = -1;
    int_t start_index =-1;
    bool is_avg = false;
    video_index(file_name,start_index,end_index,is_avg);
    if(end_index<0) end_index = start_index;
    Teuchos::RCP<cv::VideoCapture> vc = DICe::utils::Video_Singleton::instance().video_capture(video_file);
    width = sub_w==0?(int_t)vc->get(cv::CAP_PROP_FRAME_WIDTH):sub_w;
    height = sub_h==0?(int_t)vc->get(cv::CAP_PROP_FRAME_HEIGHT):sub_h;
    if(intensities.size()==0)
      intensities = Teuchos::ArrayRCP<S>(width*height,0.0);
    for(int_t i=0;i<width*height;++i)
      intensities[i] = 0.0;
    if(is_avg){
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_width),std::runtime_error,"no sub images allowed for avg frame");
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_height),std::runtime_error,"no sub images allowed for avg frame");
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_offset_x),std::runtime_error,"no sub images allowed for avg frame");
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_offset_y),std::runtime_error,"no sub images allowed for avg frame");
    }
    const int_t num_frames = end_index - start_index + 1;
    DEBUG_MSG("utils::read_image(): video file num_frames to average: " << num_frames);
    for(int_t i=start_index;i<=end_index;++i){
      vc->set(cv::CAP_PROP_POS_FRAMES,i);
      cv::Mat frame;
      if(!vc->read(frame)||frame.empty()){
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"read image from video capture failed, frame " << i);
      }
      if(frame.channels()!=1)
        cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
      frame.convertTo(frame, CV_8UC1);
      for(int_t y=0;y<height;++y){
        for(int_t x=0;x<width;++x){
          intensities[y*width + x] += frame.at<uchar>(y,x);
        }
      }
    } // end avg loop
    if(num_frames>1){
      for(int_t y=0;y<height;++y){
        for(int_t x=0;x<width;++x){
          intensities[y*width + x] /= num_frames;
        }
      }
    }
    static storage_t threshold = 255; // frames read from video files are always converted to 8 bit UC so max value is always 255
    if(filter_failed_pixels){
      if(threshold==255){
        std::vector<storage_t> full_data(width*height,0.0);
        for(int_t i=0;i<width*height;++i) full_data[i] = intensities[i];
        std::sort(full_data.begin(),full_data.end());
        threshold = 0.98*full_data[full_data.size()-1];
      }
      cuttoff = threshold;
    }
  } // end video frame
  else if(file_type==CINE){
    const std::string cine_file = video_file_name(file_name);
    DEBUG_MSG("utils::read_image(): cine file name: " << cine_file);
    int_t end_index = -1;
    int_t start_index =-1;
    bool is_avg = false;
    video_index(file_name,start_index,end_index,is_avg);
    // get the image dimensions
    Teuchos::RCP<hypercine::HyperCine> hc;
    if(convert_to_8_bit)
      hc = DICe::utils::Video_Singleton::instance().hypercine(cine_file,hypercine::HyperCine::TO_8_BIT);
    else
      hc = DICe::utils::Video_Singleton::instance().hypercine(cine_file);
    width = sub_w==0?hc->width():sub_w;
    height = sub_h==0?hc->height():sub_h;
    if(is_avg){
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_width),std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_height),std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_offset_x),std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_offset_y),std::runtime_error,"");
      if(intensities.size()==0)
        intensities = Teuchos::ArrayRCP<S>(width*height,0.0);
      std::vector<storage_t> avg_data = hc->get_avg_frame(start_index,end_index);
      for(int_t i=0;i<width*height;++i)
          intensities[i] = avg_data[i];
    }else{
      static storage_t threshold = hc->max_possible_intensity();
      if(filter_failed_pixels){
        if(threshold==hc->max_possible_intensity()||reinit){
          std::vector<storage_t> full_data = hc->get_frame(start_index);
          std::sort(full_data.begin(),full_data.end());
          const scalar_t outlier = 0.98*full_data[full_data.size()-1];
          threshold = outlier;
          if(use_threshold){
            for(size_t i=0;i<full_data.size();++i){
              if((scalar_t)full_data[full_data.size()-i-1]<outlier){
                threshold = full_data[full_data.size()-i-1];
                break;
              }
            }
          }
        }
      } // end filter failed
      cuttoff = threshold;
      if(buffer_persistence_guaranteed){
        DEBUG_MSG("utils::read_image(): intensity values obtained via pointer to hypercine memory buffer");
        TEUCHOS_TEST_FOR_EXCEPTION(!hc->buffer_has_frame_window(start_index,sub_offset_x,width,sub_offset_y,height),
          std::runtime_error,"cine frame or window not in buffer, but should be");
        TEUCHOS_TEST_FOR_EXCEPTION(!(std::is_same<S,storage_t>::value),std::runtime_error,"error: cannot create a Scalar_Image (Image_<scalar_t>) "
            "from hypercine using buffer presistence unless hypercine is built with scalar_t as hypercine storage_t");
        // reinterpret cast needed to get the scalar image to compile since hc->data will always be storage_t pointer
        intensities = Teuchos::ArrayRCP<S>(reinterpret_cast<S*>(hc->data(start_index,sub_offset_x,width,sub_offset_y,height)),0,width*height,false);
      }else{
        DEBUG_MSG("utils::read_image(): intensity values deep copied from hypercine memory buffer");
        // manual copy of intensity values required
        if(intensities.size()==0)
          intensities = Teuchos::ArrayRCP<S>(width*height,0.0);
        // get pointer to the data in hypercine memory buffer
        storage_t * data = hc->data(start_index,sub_offset_x,width,sub_offset_y,height);
        // values need to be copied here since another call to hypercine read buffer for another image
        // would replace the existing intensity values that this image points to
        for(int_t i=0;i<intensities.size();++i)
          intensities[i] = data[i];
      }
    } // end not is_avg
  }
#ifdef DICE_ENABLE_NETCDF
  /// check if the file is a netcdf file
    else if(file_type==NETCDF){
      netcdf::NetCDF_Reader netcdf_reader;
      const std::string netcdf_file = netcdf_file_name(file_name);
      const int_t index = netcdf_index(file_name);
      if(sub_w>0||sub_h>0){
        width = sub_w;
        height = sub_h;
      }else{
        int_t num_time_steps = 0;
        netcdf_reader.get_image_dimensions(netcdf_file.c_str(),width,height,num_time_steps);
      }
      if(intensities.size()==0)
        intensities = Teuchos::ArrayRCP<S>(width*height,0);
      if(std::is_same<S,scalar_t>::value){
        scalar_t * intens_ptr = reinterpret_cast<scalar_t*>(intensities.getRawPtr()); // needed to get code to compile if scalar_t != storage_t (S)
        netcdf_reader.read_netcdf_image(netcdf_file.c_str(),index,intens_ptr,sub_w,sub_h,sub_offset_x,sub_offset_y,layout_right);
      }else{
        Teuchos::ArrayRCP<scalar_t> netcdf_intensities(width*height,0);
        netcdf_reader.read_netcdf_image(netcdf_file.c_str(),index,netcdf_intensities.getRawPtr(),sub_w,sub_h,sub_offset_x,sub_offset_y,layout_right);
        for(int_t i=0;i<netcdf_intensities.size();++i) // this conversion is needed since netcdf always stores values in float or double
          intensities[i] = static_cast<S>(netcdf_intensities[i]);
      }
    }
#endif
  else{
    // read the image using opencv:
    cv::Mat image;
    image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
    // make sure the filename didn't have any unicode characters causing problems
    if (image.empty()) {
      DEBUG_MSG("utils::read_image(): image is empty, it could have unicode characters in the name so trying to read with a buffer instead");
      std::ifstream f(file_name,std::iostream::binary);
      if(!f.good()){
        std::cout << "utils::read_image(): image read failed." << std::endl;
      }else{
        std::filebuf* pbuf = f.rdbuf();
        size_t size = pbuf->pubseekoff(0, f.end, f.in);
        pbuf->pubseekpos(0, f.in);
        std::vector<uchar> buffer(size);
        pbuf->sgetn((char*)buffer.data(), size);
        image = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
          std::cout << "utils::read_image(): image read failed." << std::endl;
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
        }
      }
    }
    width = sub_w==0?image.cols:sub_w;
    height = sub_h==0?image.rows:sub_h;
    assert(width+sub_offset_x <= image.cols);
    assert(height+sub_offset_y <= image.rows);
    if(intensities.size()==0)
      intensities = Teuchos::ArrayRCP<S>(width*height,0.0);
    for (int_t y=sub_offset_y; y<sub_offset_y+height; ++y) {
      uchar* p = image.ptr(y);
      p+=sub_offset_x;
      if(layout_right)
        for (int_t x=sub_offset_x; x<sub_offset_x+width;++x){
          intensities[(y-sub_offset_y)*width + x-sub_offset_x] = *p++;
        }
      else // otherwise assume layout left
        for (int_t x=sub_offset_x; x<sub_offset_x+width;++x){
          intensities[(x-sub_offset_x)*height+y-sub_offset_y] = *p++;
        }
    }
  }
  // apply any post processing of the images as requested
  if(file_type==CINE||file_type==VIDEO){
    if(filter_failed_pixels){
      if(use_threshold){ // the old way of doing things was to simply replace the failed pixel with a threshold value
        for(int_t i=0;i<intensities.size();++i)
          intensities[i] = std::min(intensities[i],static_cast<S>(cuttoff));
      }else{
        // ensure not on the boundary and
        // take an average of the surrounding pixels
        int_t px = 0, py = 0;
        for(int_t i=0;i<intensities.size();++i){
          if(intensities[i]>cuttoff){
            py = i/width;
            px = i - py*width;
            if(px>0&&px<width-1&&py>0&&py<height-1){
              intensities[i] = 0.125*(intensities[i-1]+intensities[i+1]+
                  intensities[i-width]+intensities[i-width+1]+intensities[i-width-1]+
                  intensities[i+width]+intensities[i+width+1]+intensities[i+width+1]);
            }else if(i>0){
              intensities[i] = intensities[i-1];
            }
          }
        }
      } // end not use_threshold
    } // end filter_failed
  } // end CINE or VIDEO

  if(params!=Teuchos::null){
    if(params->get<bool>(remove_outlier_pixels,false)){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"depricated function");
//      const intensity_t outlier_rep_value = params->get<double>(outlier_replacement_value,-1.0);
//      remove_outliers(width,height,intensities.getRawPtr(),outlier_rep_value);
    }
    if(params->get<bool>(spread_intensity_histogram,false)){
      spread_histogram(width,height,intensities.getRawPtr());
    }
    if(params->get<bool>(round_intensity_values,false)){
      round_intensities(width,height,intensities.getRawPtr());
    }
    if(params->get<bool>(floor_intensity_values,false)){
      floor_intensities(width,height,intensities.getRawPtr());
    }
    if(params->isParameter(undistort_images)){
      undistort_intensities(width,height,intensities.getRawPtr(),params);
    }
  }
}
#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void read_image(const char *,Teuchos::ArrayRCP<storage_t> &,const Teuchos::RCP<Teuchos::ParameterList> &);
#endif
template
DICE_LIB_DLL_EXPORT
void read_image(const char *,Teuchos::ArrayRCP<scalar_t> &,const Teuchos::RCP<Teuchos::ParameterList> &);


template <typename S>
DICE_LIB_DLL_EXPORT
void round_intensities(const int_t width,
  const int_t height,
  S * intensities){
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      intensities[y*width + x] = std::round(intensities[y*width + x]);
    }
  }
}

#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void round_intensities(const int_t,const int_t,storage_t *);
#endif
template
DICE_LIB_DLL_EXPORT
void round_intensities(const int_t,const int_t,scalar_t *);

//DICE_LIB_DLL_EXPORT
//void remove_outliers(const int_t width,
//  const int_t height,
//  S * intensities,
//  const S & rep_value){
//
//  std::vector<S> sorted_intensities(width*height,0.0);
//  for(int_t i=0;i<width*height;++i)
//    sorted_intensities[i] = intensities[i];
//  std::sort(sorted_intensities.begin(),sorted_intensities.end());
//  const S outlier_intens = 0.98 * sorted_intensities[width*height-1];
//  S replacement_intens = 0.0;
//  if(rep_value>=0.0){
//    replacement_intens = rep_value;
//  }else{
//    for(int_t i=0;i<width*height;++i){
//      if(sorted_intensities[width*height-i-1] < outlier_intens){
//        replacement_intens = sorted_intensities[width*height-i-1];
//        break;
//      }
//    }
//  }
//  DEBUG_MSG("utils::remove_outliers(): outlier intensity value: " << outlier_intens << " to be replaced with intensity: " << replacement_intens);
//  for(int_t i=0;i<width*height;++i){
//      if(intensities[i]>=outlier_intens)
//        intensities[i] = replacement_intens;
//  }
//}

template <typename S>
DICE_LIB_DLL_EXPORT
void floor_intensities(const int_t width,
  const int_t height,
  S * intensities){
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      intensities[y*width + x] = std::floor(intensities[y*width + x]);
    }
  }
}

#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void floor_intensities(const int_t,const int_t,storage_t *);
#endif
template
DICE_LIB_DLL_EXPORT
void floor_intensities(const int_t,const int_t,scalar_t *);

template <typename S>
DICE_LIB_DLL_EXPORT
void undistort_intensities(const int_t width,
  const int_t height,
  S * intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params){

  static bool first_run = true;
  static cv::Mat intrinsics = cv::Mat(3, 3, CV_32FC1);
  static cv::Mat dist_coeffs = cv::Mat(1,4,CV_32FC1);
  if(first_run){
    DEBUG_MSG("utils::undistort_intensities(): manually undistorting images");
    // This param must exist, otherwise this method would not be called
    TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,"");
    params->print(std::cout);
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(undistort_images),std::runtime_error,"");
    Teuchos::ParameterList cal_sublist = params->sublist(undistort_images);
    TEUCHOS_TEST_FOR_EXCEPTION(!cal_sublist.isParameter("fx"),std::runtime_error,"");
    const scalar_t fx = cal_sublist.get<double>("fx");
    TEUCHOS_TEST_FOR_EXCEPTION(!cal_sublist.isParameter("fy"),std::runtime_error,"");
    const scalar_t fy = cal_sublist.get<double>("fy");
    TEUCHOS_TEST_FOR_EXCEPTION(!cal_sublist.isParameter("cx"),std::runtime_error,"");
    const scalar_t cx = cal_sublist.get<double>("cx");
    TEUCHOS_TEST_FOR_EXCEPTION(!cal_sublist.isParameter("cy"),std::runtime_error,"");
    const scalar_t cy = cal_sublist.get<double>("cy");
    TEUCHOS_TEST_FOR_EXCEPTION(!cal_sublist.isParameter("k1"),std::runtime_error,"");
    const scalar_t k1 = cal_sublist.get<double>("k1");
    TEUCHOS_TEST_FOR_EXCEPTION(!cal_sublist.isParameter("k2"),std::runtime_error,"");
    const scalar_t k2 = cal_sublist.get<double>("k2");
    intrinsics.at<float>(0,0) = fx;
    intrinsics.at<float>(1,1) = fy;
    intrinsics.at<float>(0,2) = cx;
    intrinsics.at<float>(1,2) = cy;
    intrinsics.at<float>(2,2) = 1.0;
    dist_coeffs.at<float>(0,0) = k1;
    dist_coeffs.at<float>(0,1) = k2;
    dist_coeffs.at<float>(0,2) = 0.0;
    dist_coeffs.at<float>(0,3) = 0.0;
  }
  first_run = false;
  // convert intensity values to an opencv Mat
  cv::Mat img(height,width,CV_8UC1,cv::Scalar(0));
  cv::Mat out_img(height,width,CV_8UC1,cv::Scalar(0));
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      img.at<uchar>(y,x) = intensities[y*width + x];
    }
  }
  // undistort the mat and replace the intensity values
  cv::undistort(img,out_img,intrinsics,dist_coeffs);
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      intensities[y*width + x] = out_img.at<uchar>(y,x);
    }
  }
}

#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void undistort_intensities(const int_t,const int_t,storage_t *,const Teuchos::RCP<Teuchos::ParameterList> &);
#endif
template
DICE_LIB_DLL_EXPORT
void undistort_intensities(const int_t,const int_t,scalar_t *,const Teuchos::RCP<Teuchos::ParameterList> &);

template <typename S>
DICE_LIB_DLL_EXPORT
void spread_histogram(const int_t width,
  const int_t height,
  S * intensities){
  scalar_t max_intensity = std::numeric_limits<scalar_t>::min();
  scalar_t min_intensity = std::numeric_limits<scalar_t>::max();
  for(int_t i=0; i<width*height; ++i){
    if(intensities[i] > max_intensity) max_intensity = intensities[i];
    if(intensities[i] < min_intensity) min_intensity = intensities[i];
  }
  scalar_t range = 255.0;
  if(max_intensity > 255.0 && max_intensity <= 4096.0)
    range = 4096.0;
  else if(max_intensity > 4096.0)
    range = 65535.0;
  DEBUG_MSG("utils::spread_histogram(): range: " << range);
  DEBUG_MSG("utils::spread_histogram(): max intensity: " << max_intensity);
  DEBUG_MSG("utils::spread_histogram(): min intensity: " << min_intensity);
  if((max_intensity - min_intensity) == 0.0) return;
  const scalar_t fac = range / (max_intensity - min_intensity);
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x)
      intensities[y*width + x] = static_cast<S>((intensities[y*width + x]-min_intensity)*fac);
  }
}

#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void spread_histogram(const int_t,const int_t,storage_t *);
#endif
template
DICE_LIB_DLL_EXPORT
void spread_histogram(const int_t,const int_t,scalar_t *);

DICE_LIB_DLL_EXPORT
cv::Mat read_image(const char * file_name){
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  if(image_file_type(file_name)==CINE||image_file_type(file_name)==VIDEO){
    params->set(filter_failed_cine_pixels,true);
    params->set(convert_cine_to_8_bit,true);
    int_t width = 0;
    int_t height = 0;
    read_image_dimensions(file_name,width,height);
    // read the cine
    Teuchos::ArrayRCP<storage_t> intensities(width*height,0.0);
    read_image(file_name,intensities,params);
    cv::Mat img(height,width,CV_8UC1,cv::Scalar(0));
    for(int_t y=0;y<height;++y){
      for(int_t x=0;x<width;++x){
        img.at<uchar>(y,x) = intensities[y*width + x];
      }
    }
    return img;
  }else{
    cv::Mat image = cv::imread(file_name,cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      DEBUG_MSG("utils::read_image(): image is empty, it could have unicode characters in the name so trying to read with a buffer instead");
      std::ifstream f(file_name,std::iostream::binary);
      if(!f.good()){
        std::cout << "utils::read_image(): image read failed." << std::endl;
      }else{
        std::filebuf* pbuf = f.rdbuf();
        size_t size = pbuf->pubseekoff(0, f.end, f.in);
        pbuf->pubseekpos(0, f.in);
        std::vector<uchar> buffer(size);
        pbuf->sgetn((char*)buffer.data(), size);
        image = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
          std::cout << "utils::read_image(): image read failed." << std::endl;
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
        }
      }
    }
    return image;
  }
}

template <typename S>
DICE_LIB_DLL_EXPORT
void write_color_overlap_image(const char * file_name,
  const int_t width,
  const int_t height,
  S * bottom_intensities,
  S * top_intensities){

  S bot_max_intensity = std::numeric_limits<S>::min();
  S bot_min_intensity = std::numeric_limits<S>::max();
  S top_max_intensity = std::numeric_limits<S>::min();
  S top_min_intensity = std::numeric_limits<S>::max();
  for(int_t i=0; i<width*height; ++i){
    if(bottom_intensities[i] > bot_max_intensity) bot_max_intensity = bottom_intensities[i];
    if(bottom_intensities[i] < bot_min_intensity) bot_min_intensity = bottom_intensities[i];
    if(top_intensities[i] > top_max_intensity) top_max_intensity = top_intensities[i];
    if(top_intensities[i] < top_min_intensity) top_min_intensity = top_intensities[i];
  }
  S bot_fac = 1.0;
  S top_fac = 1.0;
  if((bot_max_intensity - bot_min_intensity) != 0.0)
    bot_fac = 0.5*255.0 / (bot_max_intensity - bot_min_intensity);
  if((top_max_intensity - top_min_intensity) != 0.0)
    top_fac = 0.5*255.0 / (top_max_intensity - top_min_intensity);

  cv::Mat out_img(height,width,CV_32FC3,cv::Scalar(0.0f,0.0f,0.0f));
  for (int_t y=0; y<height; ++y) {
      for (int_t x=0; x<width;++x){
        out_img.at<cv::Vec3b>(y,x)[1] = std::floor((bottom_intensities[y*width + x]-bot_min_intensity)*bot_fac);
        out_img.at<cv::Vec3b>(y,x)[2] = std::floor((top_intensities[y*width + x]-top_min_intensity)*top_fac);
      }
  }
  cv::imwrite(file_name,out_img);
}
#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void write_color_overlap_image(const char *,const int_t,const int_t,storage_t *,storage_t *);
#endif
template
DICE_LIB_DLL_EXPORT
void write_color_overlap_image(const char *,const int_t,const int_t,scalar_t *,scalar_t *);

template<typename S>
DICE_LIB_DLL_EXPORT
void write_image(const char * file_name,
  const int_t width,
  const int_t height,
  S * intensities,
  const bool is_layout_right){
  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);
  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  // rawi files are not scaled to the 8 bit range, because the file type holds double precision values
  if(file_type==RAWI){
    write_rawi_image(file_name,width,height,intensities,is_layout_right);
  }
  else{
    // check the range of the values and scale to 8 bit
    scalar_t max_value = 0.0;
    for(int_t i=0;i<width*height;++i)
      if(intensities[i]>max_value) max_value = intensities[i];
    scalar_t conversion_factor = 1.0;
    if(max_value > 255.0){
      conversion_factor = 255.0 / max_value;
    }
    DEBUG_MSG("write_image(): max intensity value " << max_value);
    DEBUG_MSG("write_image(): intensity value conversion factor " << conversion_factor);
    cv::Mat out_img(height,width,CV_8UC1,cv::Scalar(0));
    for (int_t y=0; y<height; ++y) {
      if(is_layout_right)
        for (int_t x=0; x<width;++x){
          out_img.at<uchar>(y,x) = std::round(intensities[y*width + x]*conversion_factor);
        }
      else
        for (int_t x=0; x<width;++x){
          out_img.at<uchar>(y,x) = std::round(intensities[x*height+y]*conversion_factor);
        }
    }
    cv::imwrite(file_name,out_img);
  }
}

#ifndef STORAGE_SCALAR_SAME_TYPE
template
DICE_LIB_DLL_EXPORT
void write_image(const char *,const int_t,const int_t,storage_t *,const bool);
#endif
template
DICE_LIB_DLL_EXPORT
void write_image(const char *,const int_t,const int_t,scalar_t *,const bool);

Teuchos::RCP<hypercine::HyperCine>
Video_Singleton::hypercine(const std::string & id,
  hypercine::HyperCine::Bit_Depth_Conversion_Type conversion_type){
  if(hypercine_map_.find(std::pair<std::string,hypercine::HyperCine::Bit_Depth_Conversion_Type>(id,conversion_type))==hypercine_map_.end()){
    DEBUG_MSG("Video_Singleton::hypercine(): insering a new HyperCine for file " << id << " conversion type " << conversion_type);
    Teuchos::RCP<hypercine::HyperCine> hypercine = Teuchos::rcp(new hypercine::HyperCine(id.c_str(),conversion_type));
    hypercine_map_.insert(std::pair<std::pair<std::string,hypercine::HyperCine::Bit_Depth_Conversion_Type>,Teuchos::RCP<hypercine::HyperCine> >
    (std::pair<std::string,hypercine::HyperCine::Bit_Depth_Conversion_Type>(id,conversion_type),hypercine));
    return hypercine;
  }else{
    DEBUG_MSG("Video_Singleton::hypercine(): reusing HyperCine for file " << id << " conversion type " << conversion_type);
    return hypercine_map_.find(std::pair<std::string,hypercine::HyperCine::Bit_Depth_Conversion_Type>(id,conversion_type))->second;
  }
}

Teuchos::RCP<cv::VideoCapture>
Video_Singleton::video_capture(const std::string & id){
  if(video_capture_map_.find(id)==video_capture_map_.end()){
    DEBUG_MSG("Video_Singleton::video_capture(): insering a new VideoCapture for file " << id );
    Teuchos::RCP<cv::VideoCapture> vc = Teuchos::rcp(new cv::VideoCapture(id));
    TEUCHOS_TEST_FOR_EXCEPTION(!vc->isOpened(),std::runtime_error,"video capture load failed for " << id);
    video_capture_map_.insert(std::pair<std::string,Teuchos::RCP<cv::VideoCapture> >(id,vc));
    return vc;
  }else{
    DEBUG_MSG("Video_Singleton::video_capture(): reusing VideoCapture for file " << id);
    return video_capture_map_.find(id)->second;
  }
}

DICE_LIB_DLL_EXPORT
bool is_cine_file(const std::string & file_name){
  // check if the file is cine or mp4 or throw an exception
  if(file_name.substr(file_name.find_last_of(".") + 1) == "cine" || file_name.substr(file_name.find_last_of(".") + 1) == "CINE")
    return true;
  else
    return false;
}

void
Video_Singleton::image_dimensions(const std::string & id, int_t & width, int_t & height)const{
  if(is_cine_file(id)){
    std::map<std::pair<std::string,hypercine::HyperCine::Bit_Depth_Conversion_Type>,Teuchos::RCP<hypercine::HyperCine> >::const_iterator it = hypercine_map_.begin();
    for(;it!=hypercine_map_.end();++it){
      if(it->first.first==id){
        DEBUG_MSG("Video_Singleton::image_dimensions(): using dims from existing hypercine " << id);
        width = it->second->width();
        height = it->second->height();
        return;
      }
    }
    // no matching (by name) hypercine objects found
    hypercine::HyperCine hc(id.c_str());
    width = hc.width();
    height = hc.height();
  }
  else{
    std::map<std::string,Teuchos::RCP<cv::VideoCapture> >::const_iterator it = video_capture_map_.begin();
    for(;it!=video_capture_map_.end();++it){
      if(it->first==id){
        DEBUG_MSG("Video_Singleton::image_dimensions(): using dims from existing VideoCapture " << id);
        TEUCHOS_TEST_FOR_EXCEPTION(!it->second->isOpened(),std::runtime_error,"Error: Video open failed " << id);
        width = (int_t)it->second->get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int_t)it->second->get(cv::CAP_PROP_FRAME_HEIGHT);
        return;
      }
    }
    // no matching (by name) video objects found
    cv::VideoCapture cap(id);
    TEUCHOS_TEST_FOR_EXCEPTION(!cap.isOpened(),std::runtime_error,"Error: Video open failed: " << id);
    width = (int_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  }
}

DICE_LIB_DLL_EXPORT
int_t video_file_frame_count(const std::string & video_name){
  if(is_cine_file(video_name))
    return DICe::utils::Video_Singleton::instance().hypercine(video_name)->file_frame_count();
  else{
    return (int_t)(DICe::utils::Video_Singleton::instance().video_capture(video_name)->get(cv::CAP_PROP_FRAME_COUNT));
  }
};

DICE_LIB_DLL_EXPORT
int_t video_file_first_frame_id(const std::string & cine_name){
  if(is_cine_file(cine_name))
    return DICe::utils::Video_Singleton::instance().hypercine(cine_name)->file_first_frame_id();
  else{
    return 0;
  }
};

DICE_LIB_DLL_EXPORT
void cine_file_read_buffer(const std::string & cine_name,
  const hypercine::HyperCine::Bit_Depth_Conversion_Type conversion_type,
  hypercine::HyperCine::HyperFrame & hf){
  DICe::utils::Video_Singleton::instance().hypercine(cine_name,conversion_type)->read_buffer(hf);
}

DICE_LIB_DLL_EXPORT
void cine_file_read_buffer(const std::string & cine_name,
  const hypercine::HyperCine::Bit_Depth_Conversion_Type conversion_type,
  const int_t frame,
  const int_t count){
  DICe::utils::Video_Singleton::instance().hypercine(cine_name,conversion_type)->hyperframe()->update_frames(frame,count);
  DICe::utils::Video_Singleton::instance().hypercine(cine_name,conversion_type)->read_buffer();
}



} // end namespace utils
} // end namespace DICe
