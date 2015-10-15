// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#include <DICe_Cine.h>

#include <fstream>
#include <stdint.h>
#include <time.h>
#include <string>
#include <sstream>

namespace DICe {
namespace cine {

// look up table for converting 10bit packed pixels to 12 bit (taken from Appendix 1 of the Vision Research Phantom Cine File Format Spec with permission)
const static uint16_t LinLUT[1024] =
{2, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 34, 35, 36, 37, 38, 39, 40,
 41, 42, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 79, 80,
 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
 121, 122, 123, 124, 125, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160,
 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 181, 182, 183, 184, 186, 187, 188, 189, 191, 192, 193, 194, 196, 197, 198, 200, 201, 202, 204, 205, 206, 208, 209, 210,
 212, 213, 215, 216, 217, 219, 220, 222, 223, 225, 226, 227, 229, 230, 232, 233, 235, 236, 238, 239, 241, 242, 244, 245, 247, 249, 250, 252, 253, 255, 257, 258, 260, 261, 263, 265, 266, 268, 270, 271, 273,
 275, 276, 278, 280, 281, 283, 285, 287, 288, 290, 292, 294, 295, 297, 299, 301, 302, 304, 306, 308, 310, 312, 313, 315, 317, 319, 321, 323, 325, 327, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348,
 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 377, 379, 381, 383, 385, 387, 389, 391, 394, 396, 398, 400, 402, 404, 407, 409, 411, 413, 416, 418, 420, 422, 425, 427, 429, 431, 434, 436,
 438, 441, 443, 445, 448, 450, 452, 455, 457, 459, 462, 464, 467, 469, 472, 474, 476, 479, 481, 484, 486, 489, 491, 494, 496, 499, 501, 504, 506, 509, 511, 514, 517, 519, 522, 524, 527, 529, 532, 535, 537,
 540, 543, 545, 548, 551, 553, 556, 559, 561, 564, 567, 570, 572, 575, 578, 581, 583, 586, 589, 592, 594, 597, 600, 603, 606, 609, 611, 614, 617, 620, 623, 626, 629, 632, 635, 637, 640, 643, 646, 649, 652,
 655, 658, 661, 664, 667, 670, 673, 676, 679, 682, 685, 688, 691, 694, 698, 701, 704, 707, 710, 713, 716, 719, 722, 726, 729, 732, 735, 738, 742, 745, 748, 751, 754, 758, 761, 764, 767, 771, 774, 777, 781,
 784, 787, 790, 794, 797, 800, 804, 807, 811, 814, 817, 821, 824, 828, 831, 834, 838, 841, 845, 848, 852, 855, 859, 862, 866, 869, 873, 876, 880, 883, 887, 890, 894, 898, 901, 905, 908, 912, 916, 919, 923,
 927, 930, 934, 938, 941, 945, 949, 952, 956, 960, 964, 967, 971, 975, 979, 982, 986, 990, 994, 998,1001,1005,1009,1013,1017,1021,1025,1028,1032,1036,1040,1044,1048,1052,1056,1060,1064,1068,1072,1076,1080,
 1084,1088,1092,1096,1100,1104,1108,1112,1116,1120, 1124,1128,1132,1137,1141,1145,1149,1153,1157,1162,1166,1170,1174,1178,1183,1187, 1191,1195,1200,1204,1208,1212,1217,1221,1225,1230,1234,1238,1243,1247,
 1251,1256, 1260,1264,1269,1273,1278,1282,1287,1291,1295,1300,1304,1309,1313,1318,1322,1327, 1331,1336,1340,1345,1350,1354,1359,1363,1368,1372,1377,1382,1386,1391,1396,1400, 1405,1410,1414,1419,1424,1428,
 1433,1438,1443,1447,1452,1457,1462,1466,1471,1476, 1481,1486,1490,1495,1500,1505,1510,1515,1520,1524,1529,1534,1539,1544,1549,1554, 1559,1564,1569,1574,1579,1584,1589,1594,1599,1604,1609,1614,1619,1624,
 1629,1634, 1639,1644,1649,1655,1660,1665,1670,1675,1680,1685,1691,1696,1701,1706,1711,1717, 1722,1727,1732,1738,1743,1748,1753,1759,1764,1769,1775,1780,1785,1791,1796,1801, 1807,1812,1818,1823,1828,1834,
 1839,1845,1850,1856,1861,1867,1872,1878,1883,1889, 1894,1900,1905,1911,1916,1922,1927,1933,1939,1944,1950,1956,1961,1967,1972,1978, 1984,1989,1995,2001,2007,2012,2018,2024,2030,2035,2041,2047,2053,2058,
 2064,2070, 2076,2082,2087,2093,2099,2105,2111,2117,2123,2129,2135,2140,2146,2152,2158,2164, 2170,2176,2182,2188,2194,2200,2206,2212,2218,2224,2231,2237,2243,2249,2255,2261, 2267,2273,2279,2286,2292,2298,
 2304,2310,2317,2323,2329,2335,2341,2348,2354,2360, 2366,2373,2379,2385,2392,2398,2404,2411,2417,2423,2430,2436,2443,2449,2455,2462, 2468,2475,2481,2488,2494,2501,2507,2514,2520,2527,2533,2540,2546,2553,
 2559,2566, 2572,2579,2586,2592,2599,2605,2612,2619,2625,2632,2639,2645,2652,2659,2666,2672, 2679,2686,2693,2699,2706,2713,2720,2726,2733,2740,2747,2754,2761,2767,2774,2781, 2788,2795,2802,2809,2816,2823,
 2830,2837,2844,2850,2857,2864,2871,2878,2885,2893, 2900,2907,2914,2921,2928,2935,2942,2949,2956,2963,2970,2978,2985,2992,2999,3006, 3013,3021,3028,3035,3042,3049,3057,3064,3071,3078,3086,3093,3100,3108,
 3115,3122, 3130,3137,3144,3152,3159,3166,3174,3181,3189,3196,3204,3211,3218,3226,3233,3241, 3248,3256,3263,3271,3278,3286,3294,3301,3309,3316,3324,3331,3339,3347,3354,3362, 3370,3377,3385,3393,3400,3408,
 3416,3423,3431,3439,3447,3454,3462,3470,3478,3486, 3493,3501,3509,3517,3525,3533,3540,3548,3556,3564,3572,3580,3588,3596,3604,3612, 3620,3628,3636,3644,3652,3660,3668,3676,3684,3692,3700,3708,3716,3724,
 3732,3740, 3749,3757,3765,3773,3781,3789,3798,3806,3814,3822,3830,3839,3847,3855,3863,3872, 3880,3888,3897,3905,3913,3922,3930,3938,3947,3955,3963,3972,3980,3989,3997,4006, 4014,4022,4031,4039,4048,4056,
 4064,4095,4095,4095,4095,4095,4095,4095,4095,4095 };

Cine_Reader::Cine_Reader(const std::string & file_name, std::ostream * out_stream):
  out_stream_(out_stream),
  bit_12_warning_(false)
{
  cine_header_ = read_cine_headers(file_name.c_str(),out_stream);
}

// If you're reading these comments, prepare yourself for the nonsense that is the .cine file format.
// Take a deep breath, read up on bit-shifting and get ready for hair pulling...
std::vector<Teuchos::RCP<Image> >
Cine_Reader::get_frames(const int_t frame_index_start, const int_t frame_index_end, const Teuchos::RCP<Teuchos::ParameterList> & params){

  int_t frame_start = frame_index_start;
  int_t frame_end = frame_index_end;

  int_t file_num_frames = cine_header_->header_.ImageCount;
  TEUCHOS_TEST_FOR_EXCEPTION(frame_start < 0,std::invalid_argument,"Error, index start < 0");
  TEUCHOS_TEST_FOR_EXCEPTION(frame_start >= file_num_frames,std::invalid_argument,"Error, index start > file_num_frames");
  TEUCHOS_TEST_FOR_EXCEPTION(frame_end < frame_start,std::invalid_argument,"Error, index end < index start");
  if(frame_end >= file_num_frames)
    frame_end = file_num_frames - 1;
  int_t img_width = cine_header_->bitmap_header_.biWidth;
  int_t img_height = cine_header_->bitmap_header_.biHeight;
  int_t num_pixels = img_width * img_height;

  // open the file
  std::ifstream cine_file (cine_header_->file_name_.c_str(), std::ios::in | std::ios::binary);
  if (cine_file.fail()){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Can't open the file: " + cine_header_->file_name_);
  }
  cine_file.seekg(0, std::ios::end);
  long long int file_size = cine_file.tellg(); // using long long here because the file may be huge
  int_t num_frames = frame_end - frame_start + 1;

  // factor of 8 to convert bytes to bits
  int_t bit_depth = (cine_header_->bitmap_header_.biSizeImage * 8) / (img_width * img_height);
  assert(bit_depth==8 || bit_depth==10 || bit_depth==16);
  int_t frame_size = cine_header_->bitmap_header_.biSizeImage;

  // size up the buffer and frame chunks
  const int64_t begin = cine_header_->image_offsets_[frame_start];
  const int64_t end = frame_end == file_num_frames - 1 ? file_size :
      cine_header_->image_offsets_[frame_end+1];
  assert(begin < file_size && end <= file_size);
  const long long int buffer_size = end - begin;
  const int_t frame_p_header_size = buffer_size / num_frames;
  assert(buffer_size % num_frames = 0);
  const int_t header_size = frame_p_header_size - frame_size; // could be different for each frame
  const int_t header_offset_8 = header_size / sizeof(uint8_t);
  assert(header_size%sizeof(uint8_t)==0);
  const int_t header_offset_16 = header_size / sizeof(uint16_t);
  assert(header_size%sizeof(uint16_t)==0);

  if(out_stream_){
    *out_stream_ << "Cine_Reader::get_frames(): cine file name:      " << cine_header_->file_name_ << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): frame start:         " << frame_start << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): frame end:           " << frame_end << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): file size:           " << file_size << " bytes" << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): internal range:      " << frame_start << " to " << frame_end << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): image dimensions:    " << img_width << " x " << img_height << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): num frames:          " << num_frames << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): bit depth:           " << bit_depth << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): frame size:          " << frame_size << " bytes" << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): frame header size:   " << header_size << " bytes" << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): frame + header size: " << frame_p_header_size << " bytes " << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): header offset 8:     " << header_offset_8 << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): header offset 16:    " << header_offset_16 << std::endl;
    *out_stream_ << "Cine_Reader::get_frames(): begin " << begin << " end " << end  << " buffer size: " << buffer_size << std::endl;
  }

  std::vector<Teuchos::RCP<Image> > frame_rcps(num_frames);

  // position to the first frame in this set:
  const int64_t begin_frame = cine_header_->image_offsets_[frame_start];
  cine_file.seekg(begin_frame);

  // read the buffer
  char * buffer = new char [buffer_size];
  intensity_t converted_intensity = 0.0;
  Teuchos::ArrayRCP<uint16_t> intensities_16;
  if(bit_depth==10)
    intensities_16 = Teuchos::ArrayRCP<uint16_t>(frame_size,0.0);

  cine_file.read(buffer,buffer_size);

  // cast the buffer to the approprate type (and size of each element)
  uint8_t * buff_ptr_8;
  uint16_t * buff_ptr_16;
  if(bit_depth==8 || bit_depth==10)
    buff_ptr_8 = reinterpret_cast<uint8_t*>(buffer);
  else
    buff_ptr_16 = reinterpret_cast<uint16_t*>(buffer);

  int_t max_chunk = cine_header_->bitmap_header_.biSizeImage / 5; // 5 bytes per chunk for 10 bit packed

  // serial version
  for(int_t frame=0;frame<num_frames;++frame){
    Teuchos::ArrayRCP<intensity_t> intensities  = Teuchos::ArrayRCP<intensity_t>(num_pixels,0.0);

    // read in the pixels
    if(bit_depth==8){
      // get to the start of the frame in the buffer
      long long int buff_start_index = frame*(frame_p_header_size) + header_offset_8;
      for(int_t y=0;y<img_height;++y){
        for(int_t x=0;x<img_width;++x){
          // the images are stored bottom up, not top down!
          intensities[(img_height - y - 1)*img_width + x] = buff_ptr_8[buff_start_index+y*img_width+x];
        }
      }
    }
    else if (bit_depth==16){
      long long int buff_start_index = frame*(frame_p_header_size) + header_offset_16;
      // the images are stored bottom up, not top down!
      uint16_t pixel_intensity;
      uint16_t max_intens = 0;
      for(int_t y=0;y<img_height;++y){
        for(int_t x=0;x<img_width;++x){
          pixel_intensity = buff_ptr_16[buff_start_index + y*img_width+x];
          if(pixel_intensity > max_intens) max_intens = pixel_intensity;
          // the range in the image array storage is always in terms of the 8bit value range (0-255)
          converted_intensity = static_cast<intensity_t>(pixel_intensity) * (255.0/65535.0);
          intensities[(img_height - y - 1)*img_width + x] = converted_intensity;
        }
      }
      // check to make sure the image is not 12bit stored as 16bit image:
      // if so, scale the numbers as if 12bit
      if(max_intens < 4096){
        if(out_stream_ && !bit_12_warning_){
          *out_stream_ << "*** Warning, .cine image: " << cine_header_->file_name_  << std::endl <<
              "             was detected to be 12bit depth, but stored and denoted in the header as 16bit." << std::endl <<
              "             The actual intensity value range is 0 to 4095, not 0 to 65535 as denoted in the header." << std::endl;
          bit_12_warning_ = true;
        }
        for(int_t i=0;i<img_height*img_width;++i){
          intensities[i] *= (65535.0/4095.0);
        }
      }
    }
    else if (bit_depth==10){
      long long int buff_start_index = frame*(frame_p_header_size) + header_offset_8;
      for(int_t i=0;i<frame_size;++i){
        intensities_16[i] = 0;
        intensities_16[i] = buff_ptr_8[buff_start_index + i];
      }
      // unpack the 10 bit image data from the array
      uint16_t two_byte = 0;
      int_t loc = 0;
      int_t pixel_index = 0;
      for (int_t chunk=0;chunk<max_chunk;++chunk){
        for (int_t i=0;i<4;++i){
          // create the single 16 bit combo
          two_byte = (intensities_16[loc+1] << 8) | (intensities_16[loc]); // endian swap the second byte then or it with the first
          endian_swap(two_byte);
          // shift the 10 bits to the right side of the 16 bit data type;
          two_byte = two_byte >> (6 - (i*2));
          // use a mask to zero out the left 6 bits
          two_byte = two_byte & 0x3FF; // 16 bits with only the right 10 active;
          //two_byte = two_byte & 0xFFC0; // 16 bits with only the left 10 active;
          // this next step is required because the original signal was companded from 12 bits to 10,
          // now we are expanding it back to 12:
          assert(two_byte>=0&&two_byte<1024);
          two_byte = LinLUT[two_byte];
          // save off the pixel
          converted_intensity = static_cast<intensity_t>(two_byte) * (255.0/4095.0);
          intensities[pixel_index] = converted_intensity; // packed bits are stored top down as usual
          pixel_index++;
          loc++;
        }
        loc++;
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "Error: invalid bit depth (or this bit-depth has not been implemented.");
    }
    frame_rcps[frame] = Teuchos::rcp(new Image(img_width,img_height,intensities,params));
  } // end frame
  delete[] buffer;
  cine_file.close();
  return frame_rcps;
}


Teuchos::RCP<Cine_Header>
read_cine_headers(const char *file, std::ostream * out_stream){

  std::ifstream cine_file (file, std::ios::in | std::ios::binary);
  if (cine_file.fail()){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"ERROR: Can't open the file: " + (std::string)file);
  }
  cine_file.seekg(0, std::ios::end);
  long long int file_size = cine_file.tellg();
  cine_file.seekg(0, std::ios::beg);

  // CINE HEADER

  if(out_stream) *out_stream << "\n** reading the cine header info:\n" << std::endl;

  cine_file_header header;
  cine_file.read(reinterpret_cast<char*>(&header.Type), sizeof(header.Type));
  if(out_stream) *out_stream << "file size:            " << file_size << std::endl;
  if(out_stream) *out_stream << "header type:          " << header.Type << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.Headersize), sizeof(header.Headersize));
  if(out_stream) *out_stream << "header size:          " << header.Headersize << std::endl;
  int test_size = 0;
  test_size += sizeof(header.Type);
  test_size += sizeof(header.Headersize);
  test_size += sizeof(header.Compression);
  test_size += sizeof(header.Version);
  test_size += sizeof(header.FirstMovieImage);
  test_size += sizeof(header.TotalImageCount);
  test_size += sizeof(header.FirstImageNo);
  test_size += sizeof(header.ImageCount);
  test_size += sizeof(header.OffImageHeader);
  test_size += sizeof(header.OffSetup);
  test_size += sizeof(header.OffImageOffsets);
  test_size += sizeof(header.TriggerTime);
  if(out_stream) *out_stream << "test size:            " << test_size << std::endl;
  assert(test_size == header.Headersize);
  cine_file.read(reinterpret_cast<char*>(&header.Compression), sizeof(header.Compression));
  if(out_stream) *out_stream << "header compression:   " << header.Compression << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(header.Compression!=0,std::runtime_error,
    "Error: compressed or color .cine files are not supported.");
  cine_file.read(reinterpret_cast<char*>(&header.Version), sizeof(header.Version));
  if(out_stream) *out_stream << "header version:       " << header.Version << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(header.Version!=1,std::runtime_error,
    "Error: only version 1 .cine files are not supported.");
  cine_file.read(reinterpret_cast<char*>(&header.FirstMovieImage), sizeof(header.FirstMovieImage));
  if(out_stream) *out_stream << "header first mov img: " << header.FirstMovieImage << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.TotalImageCount), sizeof(header.TotalImageCount));
  if(out_stream) *out_stream << "total image count:    " << header.TotalImageCount << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.FirstImageNo), sizeof(header.FirstImageNo));
  if(out_stream) *out_stream << "first image no:       " << header.FirstImageNo << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.ImageCount), sizeof(header.ImageCount));
  if(out_stream) *out_stream << "header image count:   " << header.ImageCount << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.OffImageHeader), sizeof(header.OffImageHeader));
  assert(header.OffImageHeader == test_size);
  if(out_stream) *out_stream << "offset image header:  " << header.OffImageHeader << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.OffSetup), sizeof(header.OffSetup));
  if(out_stream) *out_stream << "offset setup:         " << header.OffSetup << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.OffImageOffsets), sizeof(header.OffImageOffsets));
  if(out_stream) *out_stream << "offset image offsets: " << header.OffImageOffsets << std::endl;
  cine_file.read(reinterpret_cast<char*>(&header.TriggerTime), sizeof(header.TriggerTime));
  //if(out_stream) *out_stream << "trigger time:         " << header.TriggerTime << std::endl;

  // BITMAP HEADER

  if(out_stream) *out_stream << "\n** reading the cine bitmap header:\n" << std::endl;
  bitmap_info_header bitmap_header;

  // seek the file posision of the image header:
  cine_file.seekg(header.OffImageHeader);

  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biSize), sizeof(bitmap_header.biSize));
  if(out_stream) *out_stream << "bitmap header size:   " << bitmap_header.biSize << std::endl;
  int header_test_size = 0;
  header_test_size += sizeof(bitmap_header.biSize);
  header_test_size += sizeof(bitmap_header.biWidth);
  header_test_size += sizeof(bitmap_header.biHeight);
  header_test_size += sizeof(bitmap_header.biPlanes);
  header_test_size += sizeof(bitmap_header.biBitCount);
  header_test_size += sizeof(bitmap_header.biCompression);
  header_test_size += sizeof(bitmap_header.biSizeImage);
  header_test_size += sizeof(bitmap_header.biXPelsPerMeter);
  header_test_size += sizeof(bitmap_header.biYPelsPerMeter);
  header_test_size += sizeof(bitmap_header.biClrUsed);
  header_test_size += sizeof(bitmap_header.biClrImportant);
  //if(out_stream) *out_stream << "test header size:     " << header_test_size << std::endl;
  assert(header_test_size==bitmap_header.biSize);
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biWidth), sizeof(bitmap_header.biWidth));
  if(out_stream) *out_stream << "bitmap width:         " << bitmap_header.biWidth << std::endl;
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biHeight), sizeof(bitmap_header.biHeight));
  if(out_stream) *out_stream << "bitmap height:        " << bitmap_header.biHeight << std::endl;
  if(bitmap_header.biHeight < 0){
    std::cout <<"** Warning: the cine file has recorded the pixel array upside down" << std::endl;
    bitmap_header.biHeight *= -1;
  }
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biPlanes), sizeof(bitmap_header.biPlanes));
  if(out_stream) *out_stream << "bitmap num planes:    " << bitmap_header.biPlanes << std::endl;
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biBitCount), sizeof(bitmap_header.biBitCount));
  if(out_stream) *out_stream << "bitmap bit count:     " << bitmap_header.biBitCount << std::endl;
  assert((bitmap_header.biBitCount==8 || bitmap_header.biBitCount==16) && "Error: only 8 or 16 bits per pixel are supported");
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biCompression), sizeof(bitmap_header.biCompression));
  if(out_stream) *out_stream << "bitmap compression:   " << bitmap_header.biCompression << std::endl;
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biSizeImage), sizeof(bitmap_header.biSizeImage));
  if(out_stream) *out_stream << "bitmap image size:    " << bitmap_header.biSizeImage << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(bitmap_header.biSizeImage*header.ImageCount > file_size, std::runtime_error,
    "Error: file size is smaller than the number of reported frames would require.");
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biXPelsPerMeter), sizeof(bitmap_header.biXPelsPerMeter));
  if(out_stream) *out_stream << "bitmap x pels/meter:  " << bitmap_header.biXPelsPerMeter << std::endl;
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biYPelsPerMeter), sizeof(bitmap_header.biYPelsPerMeter));
  if(out_stream) *out_stream << "bitmap y pels/meter:  " << bitmap_header.biYPelsPerMeter << std::endl;
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biClrUsed), sizeof(bitmap_header.biClrUsed));
  if(out_stream) *out_stream << "bitmap colors used:   " << bitmap_header.biClrUsed << std::endl;
  assert(bitmap_header.biClrUsed == 0 && "Error: cine color files have not been implemented.");
  cine_file.read(reinterpret_cast<char*>(&bitmap_header.biClrImportant), sizeof(bitmap_header.biClrImportant));
  if(out_stream) *out_stream << "important colors:     " << bitmap_header.biClrImportant << std::endl;

  // create the return type
  std::stringstream fileName;
  fileName << file;
  Teuchos::RCP<Cine_Header> cine_header = Teuchos::rcp(new Cine_Header(fileName.str(),header, bitmap_header));

  // read the image offsets:
  cine_file.seekg(header.OffImageOffsets);
  for (int i=0;i<header.ImageCount;++i){
    int64_t offset;
    cine_file.read(reinterpret_cast<char*>(&offset), sizeof(offset));
    cine_header->image_offsets_[i] = offset;
  }

  // close the file:
  cine_file.close();

  return cine_header;
};

} // end cine namespace
} // end DICe Namespace
