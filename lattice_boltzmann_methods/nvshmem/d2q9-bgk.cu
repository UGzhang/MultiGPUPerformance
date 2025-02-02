/*
** Modified by Youyi Zhang in 2025
**
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>


#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif



#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit( mpi_status );                                                       \
        }                                                                             \
    }

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }


#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

// #define DEBUG

/* struct to hold the parameter values */
typedef struct
{
    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */

    int    nyLocal;
    int    size;
    int    rank;

} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
    float speeds[NSPEEDS];
} t_speed;

MPI_Datatype MPI_T_SPEED;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int** obstacles_all_ptr);
int dataToDevices(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                  int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_h);

int dataToHost(t_param* params, t_speed* cells_d, t_speed* cells_h);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
void accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
void propagate_rebound_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);

void exchange_ghost_cells(const t_param* params, t_speed* cells);
inline void swap(t_speed** cells_d, t_speed** tmp_cells_d);

void scatter_obstacle(const t_param* params, int *obstacles_all, int *obstacles_local);
void gather_cell(const t_param* params, t_speed* cells, t_speed* cells_all);

void collectResult(const t_param params, t_speed* cells, float* av_vels, int* obstacles_all);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr,t_speed** cells_d, t_speed** tmp_cells_d,
             int** obstacles_ptr_d);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);


int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speed* cells     = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float*   av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
    int*     obstacles_all = NULL; 

    // pointer for devices
    t_speed* cells_d     = NULL;    
    t_speed* tmp_cells_d = NULL;    
    float* av_vels_d   = NULL;
    int*     obstacles_d = NULL;


    /* parse the command line */
    if (argc != 3) usage(argv[0]);
    else{
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    MPI_Init(&argc, &argv);
    double start_all = MPI_Wtime();

    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &obstacles_all);
    dataToDevices(&params, &cells_d, &tmp_cells_d, &obstacles_d, &av_vels_d, &cells, &obstacles);
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    PUSH_RANGE("Boltzmann kernel (nvshmem)", 0)
    for (int tt = 0; tt < params.maxIters; tt++)
    {
        accelerate_flow(params, cells_d, obstacles_d);
        // propagate+rebound+collision
        propagate_rebound_collision(params, cells_d, tmp_cells_d, obstacles_d);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        swap(&tmp_cells_d, &cells_d);
    }

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double stop = MPI_Wtime();
    POP_RANGE

    dataToHost(&params, cells_d, cells);
    if(params.nx <= 1024 && params.ny <= 1024) 
        collectResult(params, cells, av_vels, obstacles_all);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &cells_d, &tmp_cells_d,& obstacles_d );

    if (params.rank == 0)
    {
        double stop_all = MPI_Wtime();
        printf("%d, %d, %d, %d, %f, %f\n", params.nx, params.ny, params.size, params.maxIters, (stop - start), (stop_all-start_all));
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}


static int sizeOfRank(int rank, int size, int N)
{
    return N / size + ((N % size > rank) ? 1 : 0);
}

// convert NVSHMEM_SYMMETRIC_SIZE string to long long unsigned int
long long unsigned int parse_nvshmem_symmetric_size(char *value) {
    long long unsigned int units, size;

    assert(value != NULL);

    if (strchr(value, 'G') != NULL) {
        units=1e9;
    } else if (strchr(value, 'M') != NULL) {
        units=1e6;
    } else if (strchr(value, 'K') != NULL) {
        units=1e3;
    } else {
        units=1;
    }

    assert(atof(value) >= 0);
    size = (long long unsigned int) atof(value) * units;

    return size;
}

int init_nvshmem(t_param* params){
    
    MPI_Comm_rank(MPI_COMM_WORLD, &(params->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(params->size));

    int num_devices;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, params->rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }
    if ( 1 < num_devices && num_devices < local_size )
    {
        fprintf(stderr,"ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n", num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }
    if ( 1 == num_devices ) {
        // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        CUDA_RT_CALL(cudaSetDevice(0));
    } else {
        CUDA_RT_CALL(cudaSetDevice(local_rank));
    }
    CUDA_RT_CALL(cudaFree(0));

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;

    long long unsigned int mesh_size_per_rank = params->nx * (((params->ny - 2) + params->size - 1) / params->size + 2);
    long long unsigned int required_symmetric_heap_size = 2 * mesh_size_per_rank * sizeof(t_speed) * 1.1; 
    
    char * value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value) { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size) {
            fprintf(stderr, "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current NVSHMEM_SYMMETRIC_SIZE=%s\n", required_symmetric_heap_size, value);
            MPI_CALL(MPI_Finalize());
            return -1;
        }
    } else {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
        // if (!params->rank)
        //     printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu\n", required_symmetric_heap_size);
        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);


    MPI_Type_contiguous(NSPEEDS, MPI_FLOAT, &MPI_T_SPEED);
    MPI_Type_commit(&MPI_T_SPEED);

    return EXIT_SUCCESS;

}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int** obstacles_all_ptr)
{

    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->density));

    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->accel));

    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->omega));

    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    init_nvshmem(params);

    params->nyLocal = sizeOfRank(params->rank, params->size, params->ny);

    long long sizeLocal = (long long)(params->nyLocal + 2) * (long long)params->nx;

    CUDA_RT_CALL(cudaMallocHost((void**)cells_ptr, sizeof(t_speed) * sizeLocal));

    CUDA_RT_CALL(cudaMallocHost((void**)tmp_cells_ptr, sizeof(t_speed) * sizeLocal));


    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density      / 9.f;
    float w2 = params->density      / 36.f;

    for (long long jj = 0; jj < params->nyLocal+2; jj++)
    {
        for (long long ii = 0; ii < params->nx; ii++)
        {
            /* centre */
            (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
            (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
            (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
            (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
            (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
            (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
            (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
        }
    }


    CUDA_RT_CALL(cudaMallocHost((void**)obstacles_ptr, sizeof(int) * sizeLocal));
    memset(*obstacles_ptr, 0, sizeof(int) * sizeLocal);

    if(params->rank == 0){

        long long sizeAll = (long long)params->ny * params->nx;
        CUDA_RT_CALL(cudaMallocHost((void**)obstacles_all_ptr, sizeof(int) * sizeAll));

        /* first set all cells in obstacle(all) array to zero */
        for (long long jj = 0; jj < params->ny; jj++)
        {
            for (long long ii = 0; ii < params->nx; ii++)
            {
                (*obstacles_all_ptr)[ii + jj*params->nx] = 0;
            }
        }

        fp = fopen(obstaclefile, "r");

        if (fp == NULL)
        {
            sprintf(message, "could not open input obstacles file: %s", obstaclefile);
            die(message, __LINE__, __FILE__);
        }

        /* read-in the blocked cells list */
        while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
        {
            /* some checks */
            if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

            if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

            if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

            if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

            /* assign to array */
            long long index = xx + yy * (long long)params->nx;
            (*obstacles_all_ptr)[index] = blocked;
        }

        /* and close the file */
        fclose(fp);
    }

    scatter_obstacle(params,*obstacles_all_ptr,*obstacles_ptr);

    CUDA_RT_CALL(cudaMallocHost((void**)av_vels_ptr, sizeof(float) * params->maxIters));

    return EXIT_SUCCESS;
}

int dataToHost(t_param* params, t_speed* cells_d, t_speed* cells_h)
{

    cudaMemcpy(cells_h, cells_d, sizeof(t_speed) * (params->nyLocal+2) * params->nx, cudaMemcpyDeviceToHost);
    return EXIT_SUCCESS;
}


int dataToDevices(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                  int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_h){


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int maxNy = sizeOfRank(0, params->size, params->ny);
    int symmetricSize = (maxNy+2) * params->nx;

    *cells_d = (t_speed*)nvshmem_malloc(sizeof(t_speed) * symmetricSize);
    *tmp_cells_d = (t_speed*)nvshmem_malloc(sizeof(t_speed) * symmetricSize);
    *obstacles_d = (int*)nvshmem_malloc(sizeof(int) * symmetricSize);

    int size = (params->nyLocal+2) * params->nx;

    CUDA_RT_CALL(cudaMemcpyAsync(*cells_d, *cells_h, sizeof(t_speed) * size, cudaMemcpyHostToDevice, stream1));
    CUDA_RT_CALL(cudaMemcpyAsync(*obstacles_d, *obstacles_h, sizeof(int) * size, cudaMemcpyHostToDevice, stream2));
    CUDA_RT_CALL(cudaMemset(*tmp_cells_d, 0 ,sizeof(t_speed) * size));

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);


}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr,t_speed** cells_d, t_speed** tmp_cells_d,
             int** obstacles_ptr_d)
{
    /*
    ** free up allocated memory
    */
    cudaFreeHost(*cells_ptr);
    cudaFreeHost(*tmp_cells_ptr);
    cudaFreeHost(*obstacles_ptr);
    cudaFreeHost(*av_vels_ptr);

    nvshmem_free(*cells_d);
    nvshmem_free(*tmp_cells_d);
    nvshmem_free(*obstacles_ptr_d);
    nvshmem_finalize();

    // Clean up the custom MPI datatype
    MPI_Type_free(&MPI_T_SPEED);

    return EXIT_SUCCESS;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* an occupied cell */
            if (obstacles[ii + jj*params.nx])
            {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
                /* no obstacle */
            else
            {
                local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii + jj*params.nx].speeds[kk];
                }

                /* compute x velocity component */
                u_x = (cells[ii + jj*params.nx].speeds[1]
                       + cells[ii + jj*params.nx].speeds[5]
                       + cells[ii + jj*params.nx].speeds[8]
                       - (cells[ii + jj*params.nx].speeds[3]
                          + cells[ii + jj*params.nx].speeds[6]
                          + cells[ii + jj*params.nx].speeds[7]))
                      / local_density;
                /* compute y velocity component */
                u_y = (cells[ii + jj*params.nx].speeds[2]
                       + cells[ii + jj*params.nx].speeds[5]
                       + cells[ii + jj*params.nx].speeds[6]
                       - (cells[ii + jj*params.nx].speeds[4]
                          + cells[ii + jj*params.nx].speeds[7]
                          + cells[ii + jj*params.nx].speeds[8]))
                      / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            // if(ii == 1111 && jj ==11111) printf("%.12E\n", u);

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
        }
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}


void gather_cell(const t_param* params, t_speed* cells, t_speed* cells_all)
{
    int rank = params->rank;
    int size = params->size;
    int nx = params->nx;

    long long *rcvCounts = NULL;
    long long *displs = NULL;

    if(rank == 0){
        rcvCounts = (long long*) malloc(size * sizeof(long long));
        displs = (long long*) malloc(size * sizeof(long long));
    }


    long long local_count = (long long)nx * params->nyLocal;  // Number of elements per process
    MPI_Gather(&local_count, 1, MPI_LONG_LONG, rcvCounts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0]=0;
        // Rank 0 copies its own data
        memcpy(cells_all, cells + nx, local_count * sizeof(t_speed));
        // Rank 0 receives data from other processes
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + rcvCounts[i - 1];

            MPI_Recv(cells_all + displs[i], rcvCounts[i], MPI_T_SPEED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // Other processes send data to rank 0
        MPI_Send(cells + nx, local_count, MPI_T_SPEED, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        free(rcvCounts);
        free(displs);
    }

}



void collectResult(const t_param params, t_speed* cells, float* av_vels, int* obstacles_all){

    t_speed* cells_all = NULL;
    float* av_vels_all = NULL;
    int rank = params.rank;

    if(rank == 0){
        long long size = (long long)params.ny * params.nx * sizeof(t_speed);
        CUDA_RT_CALL(cudaMallocHost(&cells_all, size));
        CUDA_RT_CALL(cudaMallocHost(&av_vels_all, params.maxIters * sizeof(float)));
    }

    gather_cell(&params, cells, cells_all);

    if(rank == 0){
        write_values(params, cells_all, obstacles_all, av_vels_all);
        cudaFreeHost(cells_all);
        cudaFreeHost(av_vels_all);
        cudaFreeHost(obstacles_all);

    }
}


void scatter_obstacle(const t_param* params, int *obstacles_all, int *obstacles_local) {

    long long *rcvCounts = NULL;  // Array to store the number of elements each process will receive
    long long *displs = NULL;  // Array to store the displacement for each process
    int size = params->size;
    int rank = params->rank;

    // Allocate memory for receive counts and displacements on rank 0
    if (rank == 0) {
        rcvCounts = (long long*) malloc(size * sizeof(long long));
        displs = (long long*) malloc(size * sizeof(long long));
    }

    // Gather the number of elements each process will receive (nyLocal * nx)
    long long local_data_count = params->nyLocal * (long long)params->nx;  // Number of actual data elements for each process

    // Gather the receive counts on rank 0
    MPI_Gather(&local_data_count, 1, MPI_LONG_LONG, rcvCounts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + rcvCounts[i - 1];
        }
    }

    MPI_Request request;
    if (rank == 0) {   
        for (int i = 0; i < size; i++)
        { 
            MPI_CALL(MPI_Isend(obstacles_all+displs[i], rcvCounts[i], MPI_INT, i, 0, MPI_COMM_WORLD, &request));
        }
    }

    MPI_CALL(MPI_Irecv(obstacles_local+params->nx, local_data_count, MPI_INT, 0, 0, MPI_COMM_WORLD, &request));
    MPI_CALL(MPI_Wait(&request, MPI_STATUS_IGNORE));

    // Clean up rank 0 resources
    if (rank == 0) {
        free(rcvCounts);
        free(displs);
    }

}


inline void swap(t_speed** cells_d, t_speed** tmp_cells_d) {
    t_speed* swap = *tmp_cells_d;  // 暂存 tmp_cells_d
    *tmp_cells_d = *cells_d;      // 将 cells_d 赋值给 tmp_cells_d
    *cells_d = swap;               // 将暂存的值赋值给 cells_d
}


__global__ void exchange_kernel(t_speed* cells, int nx, int nyLocal, int down_pe, int up_pe, int recv_up){
    
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if(ii == 0)
    {
        int send_up = nyLocal * nx;      // send the last - 1
        int send_down = nx;              // send the 1 row
        int recv_down = 0;               // recv at 0 row

        // printf("rank=%d, cell=%p up=%d, down=%d\n" ,rank, cells, up_pe, down_pe);

        nvshmem_float_put((float *)(cells+recv_up), (float *)(cells+send_down), nx*9, down_pe);
        nvshmem_float_put((float *)(cells+recv_down), (float *)(cells+send_up), nx*9, up_pe);
        
    }
    
}



void exchange_ghost_cells(const t_param* params, t_speed* cells) {
    int rank = params->rank;
    int size = params->size;
    int nx = params->nx;
    int down = (rank == 0) ? (size - 1) : (rank - 1);
    int up = (rank + 1) % size;  

    int down_ny = sizeOfRank(down,size, params->ny);

    int recv_up = (down_ny + 1) * nx;
    // printf("rank=%d, down=%d, recv=%ld\n", rank, down, down_ny);    
    // printf("rank:%d, nyLocal:%d\n", rank, params->nyLocal);
    exchange_kernel<<<1,1>>>(cells, params->nx, params->nyLocal, down, up, recv_up);

}
