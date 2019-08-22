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

#include <cassert>
#include <iostream>

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
  // add the .cine extension back
  file_name+=".nc";
  return file_name;
}

DICE_LIB_DLL_EXPORT
std::string cine_file_name(const char * decorated_cine_file){
  std::string cine_string(decorated_cine_file);
  // trim off the last underscore and the rest
  size_t found = cine_string.find_last_of("_");
  std::string file_name = cine_string.substr(0,found);
  // add the .cine extension back
  file_name+=".cine";
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
void cine_index(const char * decorated_cine_file,
  int_t & start_index,
  int_t & end_index,
  bool & is_avg){
  end_index = -1;
  is_avg = false;
  std::string cine_string(decorated_cine_file);
  // trim off the last underscore and the rest
  size_t found = cine_string.find_last_of("_");
  std::string tail = cine_string.substr(found);
  //DEBUG_MSG("cine_index(): tail: " << tail);
  // determine if this is an average image
  size_t found_avg = tail.find("avg");
  size_t found_ext = tail.find(".cine");
  //DEBUG_MSG("cine_index(): found average at position: " << found_avg);
  if(found_avg==std::string::npos){
    assert(found_ext!=std::string::npos);
    std::string index_str = tail.substr(1,found_ext-1);
    //DEBUG_MSG("cine_index(): index_str: " << index_str);
    start_index = std::strtol(index_str.c_str(),NULL,0);
  }
  else{
    size_t found_dash = tail.find("to");
    std::string first_num_str = tail.substr(found_avg+3,found_dash-found_avg-3);
    std::string second_num_str = tail.substr(found_dash+2,found_ext-found_dash-2);
    //DEBUG_MSG("cine_index(): first num " << first_num_str << " second num " << second_num_str);
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
  if(file_str.find(tif)!=std::string::npos||file_str.find(tiff)!=std::string::npos)
    return TIFF;
  const std::string jpg("jpg");
  const std::string jpeg("jpeg");
  if(file_str.find(jpg)!=std::string::npos||file_str.find(jpeg)!=std::string::npos)
    return JPEG;
  const std::string bmp("bmp");
  if(file_str.find(bmp)!=std::string::npos)
    return BMP;
  const std::string png("png");
  if(file_str.find(png)!=std::string::npos)
    return PNG;
  const std::string nc(".nc");
  if(file_str.find(nc)!=std::string::npos)
    return NETCDF;
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
  else if(file_type==CINE){
    const std::string cine_file = cine_file_name(file_name);
    DEBUG_MSG("read_image_dimensions(): cine file name: " << cine_file);
    width = Image_Reader_Cache::instance().cine_reader(cine_file)->width();
    height = Image_Reader_Cache::instance().cine_reader(cine_file)->height();
    assert(width>0);
    assert(height>0);
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
    height = image.rows;
    width = image.cols;
  }
}

DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  intensity_t * intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  int_t sub_w=0;
  int_t sub_h=0;
  int_t sub_offset_x=0;
  int_t sub_offset_y=0;
  bool layout_right=true;
  bool is_subimage=false;
  bool filter_failed_pixels=true;
  bool convert_to_8_bit=true;
  if(params!=Teuchos::null){
    sub_w = params->get<int_t>(subimage_width,0);
    sub_h = params->get<int_t>(subimage_height,0);
    sub_offset_x = params->get<int_t>(subimage_offset_x,0);
    sub_offset_y = params->get<int_t>(subimage_offset_y,0);
    is_subimage=params->isParameter(subimage_width)||
        params->isParameter(subimage_height)||
        params->isParameter(subimage_offset_x)||
        params->isParameter(subimage_offset_y);
    layout_right = params->get<bool>(is_layout_right,true);
    filter_failed_pixels = params->get<bool>(filter_failed_cine_pixels,filter_failed_pixels);
    convert_to_8_bit = params->get<bool>(convert_cine_to_8_bit,convert_to_8_bit);
  }
  DEBUG_MSG("utils::read_image(): sub_w: " << sub_w << " sub_h: " << sub_h << " offset_x: " << sub_offset_x << " offset_y: " << sub_offset_y);
  DEBUG_MSG("utils::read_image(): is_layout_right: " << layout_right);
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
    read_rawi_image(file_name,intensities,layout_right);
  }
  else if(file_type==CINE){
    DEBUG_MSG("utils::read_image(): filter_failed_pixels: " << filter_failed_pixels);
    DEBUG_MSG("utils::read_image(): convert_to_8_bit: " << convert_to_8_bit);
    const std::string cine_file = cine_file_name(file_name);
    DEBUG_MSG("utils::read_image(): cine file name: " << cine_file);
    int_t end_index = -1;
    int_t start_index =-1;
    bool is_avg = false;
    cine_index(file_name,start_index,end_index,is_avg);
    // get the image dimensions
    Teuchos::RCP<DICe::cine::Cine_Reader> reader = Image_Reader_Cache::instance().cine_reader(cine_file);
    reader->initialize_filter(filter_failed_pixels,convert_to_8_bit);
    width = sub_w==0?reader->width():sub_w;
    height = sub_h==0?reader->height():sub_h;
    if(is_avg){
      reader->get_average_frame(start_index-reader->first_image_number(),end_index-reader->first_image_number(),
        sub_offset_x,sub_offset_y,width,height,intensities,layout_right);
    }else{
      reader->get_frame(sub_offset_x,sub_offset_y,width,height,intensities,layout_right,start_index-reader->first_image_number());
    }
  }
#ifdef DICE_ENABLE_NETCDF
  /// check if the file is a netcdf file
    else if(file_type==NETCDF){
      netcdf::NetCDF_Reader netcdf_reader;
      const std::string netcdf_file = netcdf_file_name(file_name);
      const int_t index = netcdf_index(file_name);
      netcdf_reader.read_netcdf_image(netcdf_file.c_str(),index,intensities,sub_w,sub_h,sub_offset_x,sub_offset_y,layout_right);
    }
#endif
  else{
    // read the image using opencv:
    cv::Mat image;
    image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
    width = sub_w==0?image.cols:sub_w;
    height = sub_h==0?image.rows:sub_h;
    assert(width+sub_offset_x <= image.cols);
    assert(height+sub_offset_y <= image.rows);
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
  if(params!=Teuchos::null){
    if(params->get<bool>(remove_outlier_pixels,false)){
      spread_histogram(width,height,intensities);
    }
    if(params->get<bool>(spread_intensity_histogram,false)){
      spread_histogram(width,height,intensities);
    }
    if(params->get<bool>(round_intensity_values,false)){
      round_intensities(width,height,intensities);
    }
    if(params->get<bool>(floor_intensity_values,false)){
      floor_intensities(width,height,intensities);
    }
  }
}

DICE_LIB_DLL_EXPORT
void round_intensities(const int_t width,
  const int_t height,
  intensity_t * intensities){
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      intensities[y*width + x] = std::round(intensities[y*width + x]);
    }
  }
}


DICE_LIB_DLL_EXPORT
void remove_outliers(const int_t width,
  const int_t height,
  intensity_t * intensities){

  std::vector<intensity_t> sorted_intensities(width*height,0.0);
  for(int_t i=0;i<width*height;++i)
    sorted_intensities[i] = intensities[i];
  std::sort(sorted_intensities.begin(),sorted_intensities.end());
  const intensity_t outlier_intens = 0.98 * sorted_intensities[width*height-1];
  intensity_t replacement_intens = 0.0;
  for(int_t i=0;i<width*height;++i){
    if(sorted_intensities[width*height-i-1] < outlier_intens){
      replacement_intens = sorted_intensities[width*height-i-1];
      break;
    }
  }
  DEBUG_MSG("utils::remove_outliers(): outlier intensity value: " << outlier_intens << " to be replaced with intensity: " << replacement_intens);
  for(int_t i=0;i<width*height;++i){
      if(intensities[i]>=outlier_intens)
        intensities[i] = replacement_intens;
  }
}

DICE_LIB_DLL_EXPORT
void floor_intensities(const int_t width,
  const int_t height,
  intensity_t * intensities){
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      intensities[y*width + x] = std::floor(intensities[y*width + x]);
    }
  }
}

DICE_LIB_DLL_EXPORT
void spread_histogram(const int_t width,
  const int_t height,
  intensity_t * intensities){
  intensity_t max_intensity = -1.0E10;
  intensity_t min_intensity = 1.0E10;
  for(int_t i=0; i<width*height; ++i){
    if(intensities[i] > max_intensity) max_intensity = intensities[i];
    if(intensities[i] < min_intensity) min_intensity = intensities[i];
  }
  intensity_t range = 255.0;
  if(max_intensity > 255.0 && max_intensity <= 4096.0)
    range = 4096.0;
  else if(max_intensity > 4096)
    range = 65535.0;
  DEBUG_MSG("utils::spread_histogram(): range: " << range);
  DEBUG_MSG("utils::spread_histogram(): max intensity: " << max_intensity);
  DEBUG_MSG("utils::spread_histogram(): min intensity: " << min_intensity);
  if((max_intensity - min_intensity) == 0.0) return;
  const intensity_t fac = range / (max_intensity - min_intensity);
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x)
      intensities[y*width + x] = (intensities[y*width + x]-min_intensity)*fac;
  }
}

DICE_LIB_DLL_EXPORT
cv::Mat read_image(const char * file_name){
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  if(image_file_type(file_name)==CINE){
    params->set(spread_intensity_histogram,true);
    params->set(filter_failed_cine_pixels,true);
    params->set(convert_cine_to_8_bit,true);
  }
  int_t width = 0;
  int_t height = 0;
  read_image_dimensions(file_name,width,height);
  // read the cine
  Teuchos::ArrayRCP<intensity_t> intensities(width*height,0.0);
  read_image(file_name,intensities.getRawPtr(),params);
  cv::Mat img(height,width,CV_8UC1,cv::Scalar(0));
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      img.at<uchar>(y,x) = intensities[y*width + x];
    }
  }
  return img;
}

DICE_LIB_DLL_EXPORT
void write_color_overlap_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * bottom_intensities,
  intensity_t * top_intensities){

  intensity_t bot_max_intensity = -1.0E10;
  intensity_t bot_min_intensity = 1.0E10;
  intensity_t top_max_intensity = -1.0E10;
  intensity_t top_min_intensity = 1.0E10;
  for(int_t i=0; i<width*height; ++i){
    if(bottom_intensities[i] > bot_max_intensity) bot_max_intensity = bottom_intensities[i];
    if(bottom_intensities[i] < bot_min_intensity) bot_min_intensity = bottom_intensities[i];
    if(top_intensities[i] > top_max_intensity) top_max_intensity = top_intensities[i];
    if(top_intensities[i] < top_min_intensity) top_min_intensity = top_intensities[i];
  }
  intensity_t bot_fac = 1.0;
  intensity_t top_fac = 1.0;
  if((bot_max_intensity - bot_min_intensity) != 0.0)
    bot_fac = 0.5*255.0 / (bot_max_intensity - bot_min_intensity);
  if((top_max_intensity - top_min_intensity) != 0.0)
    top_fac = 0.5*255.0 / (top_max_intensity - top_min_intensity);

  cv::Mat out_img(height,width,CV_8UC3,cv::Scalar(0,0,0));
  for (int_t y=0; y<height; ++y) {
      for (int_t x=0; x<width;++x){
        out_img.at<cv::Vec3b>(y,x)[1] = std::floor((bottom_intensities[y*width + x]-bot_min_intensity)*bot_fac);
        out_img.at<cv::Vec3b>(y,x)[2] = std::floor((top_intensities[y*width + x]-top_min_intensity)*top_fac);
      }
  }
  cv::imwrite(file_name,out_img);
}


DICE_LIB_DLL_EXPORT
void write_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * intensities,
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
    cv::Mat out_img(height,width,CV_8UC1,cv::Scalar(0));
    for (int_t y=0; y<height; ++y) {
      if(is_layout_right)
        for (int_t x=0; x<width;++x){
          out_img.at<uchar>(y,x) = std::round(intensities[y*width + x]);
        }
      else
        for (int_t x=0; x<width;++x){
          out_img.at<uchar>(y,x) = std::round(intensities[x*height+y]);
        }
    }
    cv::imwrite(file_name,out_img);
  }
}

Teuchos::RCP<DICe::cine::Cine_Reader>
Image_Reader_Cache::cine_reader(const std::string & id){
  if(cine_reader_map_.find(id)==cine_reader_map_.end()){
    Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader = Teuchos::rcp(new DICe::cine::Cine_Reader(id,NULL));
    cine_reader_map_.insert(std::pair<std::string,Teuchos::RCP<DICe::cine::Cine_Reader> >(id,cine_reader));
    return cine_reader;
  }
  else
    return cine_reader_map_.find(id)->second;
}

} // end namespace utils
} // end namespace DICe
