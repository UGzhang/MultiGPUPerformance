/*
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
#include <unistd.h> 
#include <math.h>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>


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
#define FINALSTATEFILE  "final_state_%d.dat"
#define AVVELSFILE      "av_vels.dat_%d.dat"

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
    int    yStart;
    int    yEnd;
    int    size;
    int    rank;

} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
    float speeds[NSPEEDS];
} t_speed;

// MPI_Datatype MPI_T_SPEED;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_all_ptr, float** av_vels_ptr );

int dataToDevices(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                  int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_all_h);

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

int computeY(t_param* params);
/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

double getTimeStamp()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}


static void print(t_param* params, t_speed* cell)
{
    int nx = params->nx;
printf("rank: %d\n", params->rank);

            for (int j = 0; j < params->nyLocal+2; j++) {
                printf("%02d:", j);
                for (int i = 0; i < nx; i++) {
                    // for(int k =0; k<NSPEEDS; k++)
                      printf("%12.6f", cell[j * nx + i].speeds[2]);

                }
                printf("\n");
            }
            fflush(stdout);

}


int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speed* cells     = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    float*   av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
    int*     obstacles_all = NULL; 

    // pointer for devices
    t_speed* cells_d     = NULL;    
    t_speed* tmp_cells_d = NULL;    
    float* av_vels_d   = NULL;
    int*     obstacles_d = NULL;


    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    double start_all = getTimeStamp();

    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles_all, &av_vels);
    dataToDevices(&params, &cells_d, &tmp_cells_d, &obstacles_d, &av_vels_d, &cells, &obstacles_all);
    

    double start = getTimeStamp();

    for (int tt = 0; tt < params.maxIters; tt++)
    {
        accelerate_flow(params, cells_d, obstacles_d);
        cudaDeviceSynchronize();
        nvshmem_barrier_all();
        
        // propagate+rebound+collision
        propagate_rebound_collision(params, cells_d, tmp_cells_d, obstacles_d);
                
        cudaDeviceSynchronize(); // waiting to delete
        nvshmem_barrier_all(); // waiting to delete
        
//         if ( tt ==2 && params.rank == 3 ) 
//         {
//         t_speed* temp = NULL;
//         cudaMallocHost((void**)&temp, sizeof(t_speed) * params.nx * (params.nyLocal+2));
//         cudaMemcpy(temp, cells_d, sizeof(t_speed) * params.nx * (params.nyLocal+2), cudaMemcpyDeviceToHost);
//         print(&params, temp);
//         cudaFreeHost(temp);
//         printf("------------exchange------------")
//         }
        swap(&tmp_cells_d, &cells_d);
        cudaDeviceSynchronize(); // waiting to delete
        nvshmem_barrier_all(); // waiting to delete

        exchange_ghost_cells(&params, cells_d);
        cudaDeviceSynchronize(); // waiting to delete
        nvshmem_barrier_all(); // waiting to delete



//         if ( tt ==1 && params.rank == 3 ) 
//         {
//          sleep(2);   
//         t_speed* temp = NULL;
//         cudaMallocHost((void**)&temp, sizeof(t_speed) * params.nx * (params.nyLocal+2));
//         cudaMemcpy(temp, cells_d, sizeof(t_speed) * params.nx * (params.nyLocal+2), cudaMemcpyDeviceToHost);
//         print(&params, temp);
//         cudaFreeHost(temp);
//         }

    }
    double stop = getTimeStamp();

    if (params.rank == 0)
    {
        printf("Runtime loop: %f s\n", stop-start);
    }

    dataToHost(&params, cells_d, cells);
    collectResult(params, cells, av_vels, obstacles_all);
    finalise(&params, &cells, &tmp_cells, &obstacles_all, &av_vels, &cells_d, &tmp_cells_d,& obstacles_d );

    if (params.rank == 0)
    {
        double stop_all = getTimeStamp();
        printf("Runtime all: %f s\n\n", stop_all-start_all);
    }

 
    return EXIT_SUCCESS;
}


static int sizeOfRank(int rank, int size, int N)
{
    return N / size + ((N % size > rank) ? 1 : 0);
}


int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_all_ptr, float** av_vels_ptr )
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


    nvshmem_init();

    params->rank = nvshmem_my_pe();
    params->size = nvshmem_n_pes();


    // MPI_Comm_rank(MPI_COMM_WORLD, &(params->rank));
    // MPI_Comm_size(MPI_COMM_WORLD, &(params->size));

    CUDA_RT_CALL(cudaSetDevice(params->rank));
    CUDA_RT_CALL(cudaFree(0));
    
    computeY(params);
    
//     printf("rank: %d, ny: %d, local ny: %d, start: %d, end: %d\n", params->rank, params->ny,params->nyLocal, params->yStart, params->yEnd);

    long long sizeLocal = (long long)(params->nyLocal + 2) * params->nx;
    
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
    


    CUDA_RT_CALL(cudaMallocHost((void**)av_vels_ptr, sizeof(float) * params->maxIters));
    
//         nvshmem_barrier_all();
//     exit(0);

    return EXIT_SUCCESS;
}

int computeY(t_param* params){
    

    
    int size = params->size;
    int rank = params->rank;
    int ny = params->ny;
  //local rows and columns for every local grid
  int nrows = ny / size;       /* integer division */
  int rem   = ny % size;
   int local_nrows=0, start=0, end=0;
  if (rem != 0)
  {  /* if there is a remainder */
    if (rank < rem)
      nrows += 1;  /* redistribute remainder to other ranks */
  }
  local_nrows = nrows;

  if (rem != 0 && rank >= rem)
  {
    start = (local_nrows + 1) * rem + local_nrows * (rank - rem);
    end   = (local_nrows + 1) * rem + local_nrows * (rank + 1 - rem);
  }
  else
  {
    start = local_nrows * rank;
    end   = local_nrows * (rank + 1);
  }
    
    params->nyLocal = local_nrows;
    params->yStart = start;
    params->yEnd = end;
    return EXIT_SUCCESS;
    
}

int dataToHost(t_param* params, t_speed* cells_d, t_speed* cells_h)
{
    cudaMemcpy(cells_h, cells_d, sizeof(t_speed) * (params->nyLocal+2) * params->nx, cudaMemcpyDeviceToHost);
    return EXIT_SUCCESS;
}


int dataToDevices(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                  int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_all_h)
{

    int nx = params->nx;
    int size = (params->nyLocal+2) * nx;
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    *cells_d = (t_speed*)nvshmem_malloc(sizeof(t_speed) * size);
    *tmp_cells_d = (t_speed*)nvshmem_malloc(sizeof(t_speed) * size);
    *obstacles_d = (int*)nvshmem_malloc(sizeof(int) * size);

    CUDA_RT_CALL(cudaMemcpyAsync(*cells_d, *cells_h, sizeof(t_speed) * size, cudaMemcpyHostToDevice, stream1));
    printf("rank %d yStart %d\n",params->rank, params->yStart );
    CUDA_RT_CALL(cudaMemcpyAsync(*obstacles_d+nx, *obstacles_all_h + params->yStart*nx, sizeof(int) * params->nyLocal*nx, cudaMemcpyHostToDevice, stream2));
    CUDA_RT_CALL(cudaMemset(*tmp_cells_d, 0 ,sizeof(t_speed) * size));

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return EXIT_SUCCESS;

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
//     MPI_Type_free(&MPI_T_SPEED);

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
    
    int yStart = params.yStart;

    char str[50];
    sprintf(str, FINALSTATEFILE, params.rank);
    
    fp = fopen(str, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 1; jj < params.nyLocal+1; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* an occupied cell */
            if (obstacles[ii + (jj+yStart-1)*params.nx])
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

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj+yStart-1, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
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

//     long long *rcvCounts = NULL;
//     long long *displs = NULL;

//     if(rank == 0){
//         rcvCounts = (long long*) malloc(size * sizeof(long long));
//         displs = (long long*) malloc(size * sizeof(long long));
//     }


//     long long local_count = (long long)nx * params->nyLocal;  // Number of elements per process
//     MPI_Gather(&local_count, 1, MPI_LONG_LONG, rcvCounts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         displs[0]=0;
//         // Rank 0 copies its own data
//         memcpy(cells_all, cells + nx, local_count * sizeof(t_speed));
//         // Rank 0 receives data from other processes
//         for (int i = 1; i < size; i++) {
//             displs[i] = displs[i - 1] + rcvCounts[i - 1];

//             MPI_Recv(cells_all + displs[i], rcvCounts[i], MPI_T_SPEED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }
//     } else {
//         // Other processes send data to rank 0
//         MPI_Send(cells + nx, local_count, MPI_T_SPEED, 0, 0, MPI_COMM_WORLD);
//     }

//     if(rank == 0){
//         free(rcvCounts);
//         free(displs);
//     }

}



void collectResult(const t_param params, t_speed* cells, float* av_vels, int* obstacles_all){


        write_values(params, cells, obstacles_all, av_vels);
}




inline void swap(t_speed** cells_d, t_speed** tmp_cells_d) {
    t_speed* swap = *tmp_cells_d;  // 暂存 tmp_cells_d
    *tmp_cells_d = *cells_d;      // 将 cells_d 赋值给 tmp_cells_d
    *cells_d = swap;               // 将暂存的值赋值给 cells_d
}


__global__ void exchange_kernel(t_speed* cells, int nx, int nyLocal, int down_pe, int up_pe, int rank){
    
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if(ii == 0){
        long long send_up = nyLocal * nx;      // send the last - 1
        int send_down = nx;              // send the 1 row
        int recv_down = 0;               // recv at 0 row
        long long recv_up = (nyLocal + 1) * nx; // recv at last row  
        
//         printf("rank: %d to rank %d -- send: %p, send: %p\n", rank, down_pe, cells+send_down, (float *)(cells+send_down));
//         printf("rank: %d to rank %d -- send: %p, send: %p\n", rank, up_pe, cells+send_up, (float *)(cells+send_up));
        nvshmem_float_put((float *)(cells+recv_up), (float *)(cells+send_down), nx*9, down_pe);
        nvshmem_float_put((float *)(cells+recv_down), (float *)(cells+send_up), nx*9, up_pe);
        
    }
    
}



void exchange_ghost_cells(const t_param* params, t_speed* cells) {
    int rank = params->rank;
    int size = params->size;
    int down = (rank == 0) ? (size - 1) : (rank - 1);
    int up = (rank + 1) % size;  // 

    
    exchange_kernel<<<1,1>>>(cells, params->nx, params->nyLocal,down,up,rank );

    nvshmem_barrier_all(); 

}


__global__ void accelerate_kernel(t_speed* cells, int* obstacles, int nyLocal, int nx, float w1, float w2)
{
 
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStrideX = gridDim.x * blockDim.x;
    
    int jdx = nyLocal - 1;

    for(int idx = ii; idx < nx; idx += gridStrideX){
          int index = idx + jdx * nx;
          /* if the cell is not occupied and
           ** we don't send a negative density */
            if (!obstacles[index] &&
                (cells[index].speeds[3] - w1) > 0.f &&
                (cells[index].speeds[6] - w2) > 0.f &&
                (cells[index].speeds[7] - w2) > 0.f) {

               /* increase 'east-side' densities */
                cells[index].speeds[1] += w1;
                cells[index].speeds[5] += w2;
                cells[index].speeds[8] += w2;

                /* decrease 'west-side' densities */
                cells[index].speeds[3] -= w1;
                cells[index].speeds[6] -= w2;
                cells[index].speeds[7] -= w2;
            }   
    }

}

void accelerate_flow(const t_param params, t_speed*  cells, int*  obstacles)
{
  if(params.rank == params.size - 1){

    dim3 threadsPerBlock(64); // 每个线程块 16x16 个线程
    dim3 blocksPerGrid(32);
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;
    accelerate_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, obstacles, params.nyLocal, params.nx, w1, w2);
    // cudaDeviceSynchronize();
  }

}



__global__ void propagate_rebound_collision_kernel
(t_speed* cells, t_speed* tmp_cells, int* obstacles, int nyLocal, int nx, float omega, bool rankIsLast, float w1_flow, float w2_flow)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */


  int idx = blockIdx.x * blockDim.x + threadIdx.x ;
  int jdx = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int gridStrideX = gridDim.x * blockDim.x;
  int gridStrideY = gridDim.y * blockDim.y;
  const int ySize = nyLocal+1;


  for(long long jj = jdx; jj < ySize; jj += gridStrideY)
  {
    for(long long ii = idx; ii < nx; ii += gridStrideX)
    {
      long long index = ii + jj * nx;

      int y_n = (jj + 1) % (nyLocal+2);
      int x_e = (ii + 1) % nx;
      int y_s = (jj == 0) ? (jj + nyLocal + 2 - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      float tmp[NSPEEDS];
      tmp[0] = cells[index].speeds[0]; /* central cell, no movement */
      tmp[1] = cells[x_w + jj*nx].speeds[1]; /* east */
      tmp[2] = cells[ii + y_s*nx].speeds[2]; /* north */
      tmp[3] = cells[x_e + jj*nx].speeds[3]; /* west */
      tmp[4] = cells[ii + y_n*nx].speeds[4]; /* south */
      tmp[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
      tmp[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
      tmp[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
      tmp[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */



      /* don't consider occupied cells */      
      if (!obstacles[index])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp[1]
                      + tmp[5]
                      + tmp[8]
                      - (tmp[3]
                         + tmp[6]
                         + tmp[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp[2]
                      + tmp[5]
                      + tmp[6]
                      - (tmp[4]
                         + tmp[7]
                         + tmp[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;


        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells[index].speeds[kk] = tmp[kk] + omega * (d_equ[kk] - tmp[kk]);
        }
      }
      else{
                /* called after propagate, so taking values from scratch space
                ** mirroring, and writing into main grid */
                tmp_cells[index].speeds[1] = tmp[3];
                tmp_cells[index].speeds[2] = tmp[4];
                tmp_cells[index].speeds[3] = tmp[1];
                tmp_cells[index].speeds[4] = tmp[2];
                tmp_cells[index].speeds[5] = tmp[7];
                tmp_cells[index].speeds[6] = tmp[8];
                tmp_cells[index].speeds[7] = tmp[5];
                tmp_cells[index].speeds[8] = tmp[6];
      }
    }

  }

}


void propagate_rebound_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles){

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid(32, 32);
    bool rankIsLast = (params.rank == params.size -1);
    float w1_flow = params.density * params.accel / 9.f;
    float w2_flow = params.density * params.accel / 36.f; 

    propagate_rebound_collision_kernel<<<blocksPerGrid, threadsPerBlock>>>
          (cells, tmp_cells, obstacles, params.nyLocal, params.nx, params.omega,rankIsLast,w1_flow,w2_flow);
    cudaDeviceSynchronize(); // waiting to delete


}