#include <string>
#include <cstring> 
#include <stdio.h>
#include <iostream>
#include <assert.h>

#ifdef __APPLE__		
#include <OpenCL/opencl.h>		
#else		
#include <CL/cl.h>	
#endif		

#define MAT_SIZE 100
#define MAT_NUMEL 10000
#define TOTAL_NUMEL 2560000
#define NUM_MATR 256
#define NUM_LOOP 200

// For error checking:
//#undef __OPENCL_NO_ERROR_CHECKING
#define __OPENCL_NO_ERROR_CHECKING

#ifdef __OPENCL_NO_ERROR_CHECKING
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)  \
	assert (CL_SUCCESS == __errNum__);
#else
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)  \
if (CL_SUCCESS != __errNum__)                               \
{															\
		char __msgBuf__[256];								\
		sprintf (__msgBuf__, "CL Error num %d: %s at line %d, file %s in function %s().\n", __errNum__, __failMsg__, __LINE__, __FILE__, __FUNCTION__);	\
		printf (__msgBuf__);								\
		getchar();											\
		printf("Failed on OpenCLError\n");					\
		assert (CL_SUCCESS != __errNum__);					\
		exit(0);											\
} else if (__passMsg__)										\
{															\
	printf("CL Success: %s\n", __passMsg__);				\
}				
#endif

void InitializeOpenCL(cl_device_id* pDeviceID, cl_context *pContext, cl_command_queue *pCommand_queue, cl_program *pProgram)
{
    cl_int                      errcode_ret;
    cl_platform_id		        platformID	= NULL;
    cl_platform_id		        *pPlatformIDs;
    cl_device_id		        deviceID;
    cl_context                  context = NULL;
    cl_command_queue            command_queue = NULL;
    cl_program                  program = NULL;
    cl_uint			            numPlatforms = 0;
    cl_uint                     num_devices;
    char 			            pPlatformVendor[256];
    char                        pDevVersion[256];

    // Load kernel source code	
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));
    dir_path.append("/segmentation_OpenCL.cl");

    FILE *fp;
    char *source_str;
    size_t source_size;
    fp = fopen(dir_path.c_str(), "r");

    fseek(fp, 0, SEEK_END);
    long fp_size = ftell(fp);
    std::cerr << "File length:" << fp_size << "\n";
    fseek(fp, 0, SEEK_SET);
    source_str = (char*)malloc((fp_size + 1)*sizeof(char));
    source_size = fread(source_str, 1, fp_size, fp);
    fclose(fp);

    // Get Platform and Device Info
    errcode_ret	= clGetPlatformIDs(0, NULL, &numPlatforms);
    CheckCLError (errcode_ret, "No platforms Found.", "OpenCL platforms found.");
    printf("numPlatforms is %d\n", numPlatforms);

    pPlatformIDs = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
    errcode_ret = clGetPlatformIDs(numPlatforms, pPlatformIDs, NULL); 
    CheckCLError (errcode_ret, "Could not get platform IDs.", "Got platform IDs.");
    for (int i = 0; i < numPlatforms; ++i) 
	 {
        errcode_ret = clGetPlatformInfo(pPlatformIDs[i],CL_PLATFORM_VENDOR,sizeof(pPlatformVendor),pPlatformVendor,NULL);
	    CheckCLError (errcode_ret, "Could not get platform info.", "Got platform info.");
        std::cerr << "Platform " << i << ": " << pPlatformVendor << "\n";
     }

    platformID = pPlatformIDs[1];

    clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &num_devices); //CL_DEVICE_TYPE_CPU
    errcode_ret = clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, sizeof(pDevVersion), pDevVersion, NULL);
    if (errcode_ret != CL_SUCCESS) 
     {
        printf("Error: couldn't get CL_DEVICE_VERSION!\n");
        exit(-1);
     }
    std::cerr << "num_devices: "  << num_devices << "\n";
    std::cerr << "pDevVersion: "  << pDevVersion << "\n";

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &errcode_ret);
    CheckCLError (errcode_ret, "Could not create CL context.", "Created CL context.");

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, deviceID, 0, &errcode_ret);
    CheckCLError (errcode_ret, "Could not create CL command queue.", "Created CL command queue.");
    
    // Create Kernel Program from the source
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode_ret);
    CheckCLError (errcode_ret, "Could not create clCreateProgramWithSource.", "Created clCreateProgramWithSource.");
   
   // Build Kernel Program
    errcode_ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    CheckCLError (errcode_ret, "Could not create clBuildProgram.", "Created clBuildProgram.");

    *pDeviceID = deviceID;
    *pContext = context;
    *pCommand_queue = command_queue;
    *pProgram = program;

    free(source_str);
}
