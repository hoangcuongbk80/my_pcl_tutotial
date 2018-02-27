#include<vector>

//branch
#include <ros/ros.h>
#include <my_pcl_tutorial/segmentation_OpenCL.h>

// OpenCV specific includes
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>

using namespace cv;

#define fx 520.0
#define fy 520.0
#define cx 319.5
#define cy 239.5
#define cl_rows 480
#define cl_cols 640
#define cl_imgSize cl_rows*cl_cols
#define gausize 7
#define neighBor 3
#define normalThresh 1
#define distThresh 0.02
#define depthRE 0.001


cl_device_id    		        deviceID;
cl_context                  context = NULL;
cl_command_queue            command_queue = NULL;
cl_program                  program = NULL;
cl_kernel                   kernel = NULL;
cl_int                      ret;
cl_event                    kernelDone;
cl_mem                      gpu_depth, gpu_pdepth, gpu_normals, gpu_pnormals, gpu_gauKernel;
float                       *cpu_depth;
cl_float3                   *cpu_normals;
float                       *gauKernel;

ros::Publisher pub;
pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloudPtr (new pcl::PointCloud<pcl::PointXYZ>);
cv::Mat normals_image(cl_rows, cl_cols, CV_32FC1, Scalar(0));
cv::Mat depth_image(cl_rows, cl_cols, CV_32FC1, Scalar(0));
cv::Mat img(cl_rows, cl_cols, CV_32FC1, Scalar(0));
cv::Mat segmented_image(cl_rows, cl_cols, CV_32FC1, Scalar(0));
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ (new pcl::PointCloud<pcl::PointXYZRGB>);
int cluster_Index; int timePrint = 6; bool noData; int recurNum;

void getGaussian(int kernelHeight, int kernelWidth, double sigma)
{
    gauKernel = (float*)malloc(kernelWidth * kernelHeight * sizeof(float));
    float sum = 0.0000;

    for (int row = 0; row < kernelHeight ; row++) 
    {
        for (int col = 0 ; col < kernelWidth ; col++)
        {
            gauKernel[row * kernelWidth + col] = 1;
            //gauKernel[row * kernelWidth + col] = exp(-((row - kernelHeight / 2) * (row - kernelHeight /2) +
           // (col - kernelWidth / 2) * (col - kernelWidth / 2)) / (2*sigma*sigma)) / (2*M_PI*sigma*sigma);
            sum += gauKernel[row * kernelWidth + col];
        }
    }

    for (int row = 0; row < kernelHeight ; row++) 
    {
        for (int col = 0 ; col < kernelWidth ; col++)
        {
            gauKernel[row * kernelWidth + col] = gauKernel[row * kernelWidth + col] / sum;
        }
    }
}

void copygaussianToGPU(cl_context context, cl_command_queue command_queue)
{
    cl_int ret;
    gpu_gauKernel = clCreateBuffer(context, CL_MEM_READ_WRITE, gausize * gausize * sizeof(float), NULL, &ret);
    CheckCLError (ret, "Could not create clCreateBuffer gpu_gassianKernel.", "Created clCreateBuffer gpu_gassianKernel.");

    getGaussian(gausize, gausize, 1);

    ret = clEnqueueWriteBuffer(command_queue, gpu_gauKernel, CL_TRUE, 0, gausize * gausize * sizeof(float), gauKernel, 0, NULL, NULL);
    CheckCLError (ret, "Could not create clEnqueueWriteBuffer.", "Created clEnqueueWriteBuffer.");
    
    float total = 0;
    for (int row = 0 ; row < gausize ; row++) 
    {
      for (int col = 0 ; col < gausize ; col++) 
        {
            //std::cerr << gauKernel[row * gausize + col] << "  ";
            //if(col == gausize - 1) std::cerr << "\n";
            //total += gauKernel[row * gausize + col];
        }
    }
    //std::cerr << "total: " << total << "\n";
}

void GPUMemAlloc(cl_context context, cl_mem* cl_depth, cl_mem* cl_pdepth, cl_mem* cl_normals, cl_mem* cl_pnormals)
{
    cl_int         ret;
    cl_mem         depth, postdepth, normals, pnormals;

    depth = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_imgSize * sizeof(float), NULL, &ret);
    CheckCLError (ret, "Could not create clCreateBuffer depth.", "Created clCreateBuffer depth.");

    postdepth = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_imgSize * sizeof(float), NULL, &ret);
    CheckCLError (ret, "Could not create clCreateBuffer postdepth.", "Created clCreateBuffer postdepth.");

    normals = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_imgSize * sizeof(cl_float3), NULL, &ret);
    CheckCLError (ret, "Could not create clCreateBuffer normals.", "Created clCreateBuffer normals.");

    pnormals = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_imgSize * sizeof(cl_float3), NULL, &ret);
    CheckCLError (ret, "Could not create clCreateBuffer normals.", "Created clCreateBuffer normals.");

    cpu_depth = (float*)malloc(cl_imgSize * sizeof(float));
    cpu_normals = (cl_float3*)malloc(cl_imgSize * sizeof(cl_float3));

    *cl_depth = depth; *cl_pdepth = postdepth; *cl_normals = normals; *cl_pnormals = pnormals;
}

void copyDepthToCPU()
{
   int loop = 0;
   noData = true;
   for(int row=0; row < depth_image.rows; row++)
    {
       for(int col=0; col < depth_image.cols; col++)       
        {
          if(isnan(depth_image.at<float>(row, col)))
          {
            cpu_depth[loop] = 0; loop++; continue;
          }
          cpu_depth[loop] = depth_image.at<float>(row, col); loop++;
          noData = false;        
        }
    }
}

void copyDepthToGPU(cl_command_queue command_queue, cl_mem* gpu_depth)
{
    cl_int  ret;
    ret = clEnqueueWriteBuffer(command_queue, *gpu_depth, CL_TRUE, 0, cl_imgSize * sizeof(float), cpu_depth, 0, NULL, &kernelDone);
    ret = clWaitForEvents(1, &kernelDone);
    CheckCLError (ret, "Could not create clEnqueueWriteBuffer.", "Created clEnqueueWriteBuffer.");	
}

void OpenCL_start()
{
   InitializeOpenCL(&deviceID, &context, &command_queue, &program);
   GPUMemAlloc(context, &gpu_depth, &gpu_pdepth, &gpu_normals, &gpu_pnormals);
   copygaussianToGPU(context, command_queue);
}

void copyNormalsFromGPU(cl_command_queue command_queue, cl_mem gpu_normals)
{
   cv::Mat normals(depth_image.size(), CV_32FC3);
   normals_image = normals.clone();
   cl_int  ret;
   ret = clEnqueueReadBuffer(command_queue, gpu_normals, CL_TRUE, 0, cl_imgSize * sizeof(cl_float3), cpu_normals, 0, NULL, &kernelDone);
   ret = clWaitForEvents(1, &kernelDone);
   CheckCLError (ret, "Could not create clEnqueueReadBuffer.", "Created clEnqueueReadBuffer.");
}

void normalizeNormals()
{
   for(int i = 0; i < cl_imgSize; i++)
   {
     int row = i / cl_cols; int col = i % cl_cols;
     if(cpu_normals[i].x == 0 & cpu_normals[i].y == 0 & cpu_normals[i].z == 0) 
     {
       Vec3f d(0.0f, 0.0f, 0.0f); normals_image.at<Vec3f>(row, col) = d; continue;
     }
     float dzdx = cpu_normals[i].x;
     float dzdy = cpu_normals[i].y;
     Vec3f d(-dzdx, -dzdy, depthRE);
     Vec3f n; normalize(d, n, 1.0, 0.0, NORM_L2);
     normals_image.at<Vec3f>(row, col) = n;
   }
}

void OpenCL_process()
{
  clock_t start, end;
  double time_used;

  copyDepthToCPU();
  if(noData) return;

  start = clock();
  copyDepthToGPU(command_queue, &gpu_depth);
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  if(timePrint) std::cerr << "Time to execute copyDepthToGPU: " << time_used << "\n";
  
  //---------------------------0----------------------------
  // Pre-Processing
  kernel = clCreateKernel(program, "Pre_Processing", &ret);
  CheckCLError (ret, "Could not create clCreateKernel.", "Created clCreateKernel.");
  // Set OpenCL Kernel Parameters
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpu_gauKernel);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpu_depth); 
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&gpu_pdepth);

  // Execute OpenCL Kernel
  start = clock();
  size_t * global = (size_t*) malloc(sizeof(size_t)*2); global[0] = cl_rows; global[1] = cl_cols;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, &kernelDone);
  ret = clWaitForEvents(1, &kernelDone);
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  if(timePrint) std::cerr << "Time to execute Pre_Processing kernel : " << time_used << "\n";

  //---------------------------0----------------------------
  // Compute Normals
  // Create OpenCL Kernel
  kernel = clCreateKernel(program, "GPU_NormalCompute", &ret);
  CheckCLError (ret, "Could not create clCreateKernel.", "Created clCreateKernel.");
  // Set OpenCL Kernel Parameters
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpu_pdepth);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpu_normals);
  
  // Execute OpenCL Kernel
  start = clock();
  //size_t * global = (size_t*) malloc(sizeof(size_t)*2); global[0] = cl_rows; global[1] = cl_cols;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, &kernelDone);
  ret = clWaitForEvents(1, &kernelDone);
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  if(timePrint) std::cerr << "Time to execute GPU_NormalCompute kernel : " << time_used << "\n";
  //---------------------------0----------------------------

  start = clock();
  copyNormalsFromGPU(command_queue, gpu_normals);
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  if(timePrint) std::cerr << "Time to execute copyNormalsFromGPU: " << time_used << "\n";
}

int depthToClould()
{
   cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
   pcl::PointXYZRGB point;
   for(int row = 0; row < depth_image.rows; row++)
    {
       for(int col = 0; col < depth_image.cols; col++)       
        {
          if(isnan(depth_image.at<float>(row, col))) continue;
          double depth = depth_image.at<float>(row, col);
          point.x = (col-cx) * depth / fx;
          point.y = (row-cy) * depth / fy;
          point.z = depth;
          point.r = 255; point.b = 0; point.g =0;

          if (segmented_image.at<float>(row, col) == -1) continue; 
			    else if (segmented_image.at<float>(row, col) == 1) //Lime	#00FF00	(0,255,0)
			     {
				      point.r = 255;
				      point.g = 255;
				      point.b = 255;
			     }
			    else if (segmented_image.at<float>(row, col) == 2) // Blue	#0000FF	(0,0,255)
			     {
				      point.r = 255;
				      point.g = 0;
				      point.b = 0;
			     }
			    else if (segmented_image.at<float>(row, col) == 3) // Yellow	#FFFF00	(255,255,0)
			     {
				      point.r = 255;
				      point.g = 255;
				      point.b = 0;
			     }
			    else if (segmented_image.at<float>(row, col) == 4) //Cyan	#00FFFF	(0,255,255)
			     {
				      point.r = 0;
				      point.g = 255;
				      point.b = 255;
			     }
			    else if (segmented_image.at<float>(row, col) == 5) // Magenta	#FF00FF	(255,0,255)
			     {
				      point.r = 255;
				      point.g = 0;
				      point.b = 255;
			     }
			    else if (segmented_image.at<float>(row, col) == 6) // Olive	#808000	(128,128,0)
		     	 {
				      point.r = 128;
				      point.g = 128;
				      point.b = 0;
			     }
			    else if (segmented_image.at<float>(row, col) == 7) // Teal	#008080	(0,128,128)
			     {
				      point.r = 0;
				      point.g = 128;
				      point.b = 128;
			     }
			    else if (segmented_image.at<float>(row, col) == 8) // Purple	#800080	(128,0,128)
		     	 {
				      point.r = 128;
				      point.g = 0;
				      point.b = 128;
			     }
			    else
		   	     {
				      if ((int)segmented_image.at<float>(row, col) % 2 == 0)
				       {
					        point.r = 255 * segmented_image.at<float>(row, col) / cluster_Index;
					        point.g = 128;
					        point.b = 50;
				       }
				      else
				       {
					        point.r = 0;
					        point.g = 255 * segmented_image.at<float>(row, col) / cluster_Index;
					        point.b = 128;
				       }
            }

         cloud_->push_back(point);
        }
    }
    return 0;
}

void RecursiveSearch(int cluster_Index, float seedNormal, int row, int col)
{
  segmented_image.at<float>(row, col) = cluster_Index;
  recurNum++;
  if(recurNum > 100000) { recurNum--; return;} //To avoid core dumped by increasing number of recursive function

  for(int i = -neighBor; i < neighBor + 1; i++)
   for(int j = -neighBor; j < neighBor + 1; j++)
    {
       if(row + i > 0 & col + j > 0 & row + i < normals_image.rows & col + j < normals_image.cols)
       {
          if(!isnan(depth_image.at<float>(row + i, col + j)))
          if(segmented_image.at<float>(row + i, col + j) == 0)
          if(abs(depth_image.at<float>(row + i, col + j) - depth_image.at<float>(row, col)) < distThresh)
          {
              if(abs(seedNormal - normals_image.at<Vec3f>(row + i, col + j).val[0] - 
                normals_image.at<Vec3f>(row + i, col + j).val[1] - normals_image.at<Vec3f>(row + i, col + j).val[2]) < normalThresh)
                {
                   RecursiveSearch(cluster_Index, seedNormal, row + i, col + j);
                }
          }
       }
    }
  recurNum--;
}

void Clustering()
{
  cluster_Index = 0;
  cv::Mat segmented(depth_image.size(), CV_32FC1, Scalar(0));
  segmented_image = segmented.clone();
  int i = 0;
  for(int row = 0; row < depth_image.rows; row++)
   {
     for(int col = 0; col < depth_image.cols; col++)       
      {
         i++;
         if(isnan(depth_image.at<float>(row, col)) || row - neighBor < 0 || row + neighBor > depth_image.rows-1 || col - neighBor < 0 || col + neighBor > depth_image.cols-1)
         {
           segmented_image.at<float>(row, col) = -1;
           continue;
         }
         if(normals_image.at<Vec3f>(row, col).val[0] == 0.0f & normals_image.at<Vec3f>(row, col).val[1] == 0.0f & normals_image.at<Vec3f>(row, col).val[2] == 0.0f) 
          {
              segmented_image.at<float>(row, col) = -1; continue;
          }
         if(segmented_image.at<float>(row, col) == 0)
         {
           cluster_Index++;
           float seedNormal = normals_image.at<Vec3f>(row, col).val[0] + normals_image.at<Vec3f>(row, col).val[1] + normals_image.at<Vec3f>(row, col).val[2];
           recurNum = 0;
           RecursiveSearch(cluster_Index, seedNormal, row, col);
         }  
      }
   }
   std::cerr << "cluster_Index: " << cluster_Index << "\n";
}

void smoothing()
{
  GaussianBlur( depth_image, depth_image, Size( 5, 5), 0, 0 );
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  depthToClould();
  if(cloud_->size() == 0) return;
  sensor_msgs::PointCloud2 output;
  pcl::PCLPointCloud2 cloud_filtered;
  cloud_->header.frame_id = "camera_depth_optical_frame";
  pcl::toPCLPointCloud2(*cloud_, cloud_filtered);
  pcl_conversions::fromPCL(cloud_filtered, output);
  // Publish the data
  pub.publish (output);
}

void normal_cb (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;

  try
  {
    bridge = cv_bridge::toCvCopy(msg, "32FC1");
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform depth image.");
    return;
  }
  depth_image = bridge->image.clone();

  clock_t start, end;
  double time_used;

  start = clock();
  OpenCL_process();
  end = clock();
  if(noData) { std::cerr << "No Data!\n"; return; }
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  if(timePrint) std::cerr << "Total time for data transfering and kernel execution: " << time_used << "\n";

  normalizeNormals();
  start = clock();
  Clustering();
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  if(timePrint) std::cerr << "Time to execute Recursive function on CPU: " << time_used << "\n\n";
  
  cv::imshow("depth", depth_image);
  cv::waitKey(3);
  cv::imshow("Normals", normals_image);
  cv::waitKey(3);
  if(timePrint > 0) timePrint--;
}

bool test(int x)
{
   int y = 9;
   int z = 3;
   std::cerr << x << "\n";
   test(x+1);
}

int main (int argc, char** argv)
{
  OpenCL_start();
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;
  ros::NodeHandle n;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, cloud_cb);
  ros::Subscriber sub_depth = n.subscribe ("/camera/depth/image", 1, normal_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

  // Spin
  ros::spin ();

  //test(0);

      // Finalization 
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(gpu_depth);
    ret = clReleaseMemObject(gpu_pdepth);
    ret = clReleaseMemObject(gpu_normals);
    ret = clReleaseMemObject(gpu_pnormals);
    ret = clReleaseMemObject(gpu_gauKernel);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // Free host memory
    free(cpu_depth);
    free(cpu_normals);
    free(gauKernel);
}
