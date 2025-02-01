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
#include <string.h>
#include <math.h>
#include <sys/resource.h>
#include <mpi.h>
#include <cuda_runtime.h>

// #include "d2q9_kernels.cu"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"



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
  int    size;
  int    rank;
  int    nyLocal;
  int    yStart;
  int    yEnd;
} t_param;

MPI_Datatype MPI_T_SPEED;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed**  tmp_cells_ptr,
               int**  obstacles_ptr, float** av_vels_ptr);

void initialise_gpu(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_h);



/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
// __global__ void accelerate_kernel(t_speed* cells, int* obstacles, int ny, int nx, float w1, float w2);

void collision_cuda(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* h_tot_u);
float collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int tot_cells);
// __global__ void collision_kernel(t_speed* cells, t_speed* tmp_cells, int* obstacles, int nx, int nStart, int yEnd, int ny, float omega, float* d_tot_u) ;

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

void exchange_ghost_cells(const t_param* params, t_speed* cells);
void collectResult(const t_param *params, t_speed* cells, float* av_vels, int* obstacles);

void transDataToCPU(t_param* params, t_speed* cells_d, t_speed* cells_h);
/*
** main program:
** initialise, timestep loop, finalise
*/

static void print(t_param* params, t_speed* cell)
{
    int nx = params->nx;

    for (int i = 0; i < params->size; i++) {
        if (i == params->rank) {
            printf("### RANK %d "
                   "#######################################################\n",
                params->rank);
            for (int j = 0; j < params->ny; j++) {
                printf("%02d:", j);
                for (int i = 0; i < nx; i++) {
                    // for(int k =0; k<NSPEEDS; k++)
                      printf("%12.6f", cell[j * nx + i].speeds[2]);
                    
                      
                    printf("        ");
                }
                printf("\n");
            }
            fflush(stdout);
        }
        // MPI_Barrier(MPI_COMM_WORLD);
    }
}


// __global__ void print_kernel(t_speed* cells, int nx, int ny) {
//  int ii = blockIdx.x * blockDim.x + threadIdx.x;
//     int jj = blockIdx.y * blockDim.y + threadIdx.y;

//     int gridStrideX = gridDim.x * blockDim.x;
//     int gridStrideY = gridDim.y * blockDim.y;

//     for(int jdx = jj; jdx < ny; jdx += gridStrideY)
//       for(int idx = ii; idx < nx; idx += gridStrideX){
//       printf("%d, i=%d, j=%d\n", cells[idx+jdx*nx].speeds[2], idx, jdx);
//     }

// }

__global__ void print_ob(int* ob, int nx, int ny) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;

    int gridStrideX = gridDim.x * blockDim.x;
    int gridStrideY = gridDim.y * blockDim.y;

    for (int jdx = jj; jdx < ny; jdx += gridStrideY) {
        for (int idx = ii; idx < nx; idx += gridStrideX) {
            printf("%d, i=%d, j=%d\n", ob[idx + jdx * nx], idx, jdx);
        }
    }
}


int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  t_speed* cells_d     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells_d = NULL;    /* scratch space */
  t_speed* swap = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  int*     obstacles_d = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  float* av_vels_d   = NULL;     /* a record of the av. velocity computed for each timestep */

  float reynolds;

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
  MPI_Init(&argc, &argv);

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
 initialise_gpu (&params, &cells_d, &tmp_cells_d, &obstacles_d, &av_vels_d, &cells, &obstacles);

  /* calculate tot_cells value */
  int tot_cells=0;
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
      if (!obstacles[jj * params.nx + ii])
        tot_cells++;
  }



  //     dim3 threadsPerBlock(2, 2); // 每个线程块 16x16 个线程
  //   dim3 blocksPerGrid(1,1);
  //    print_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells_d, params.nx, params.ny);
  //   cudaDeviceSynchronize();
  //   cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   fprintf(stderr, "CUDA Error after kernel execution: %s\n", cudaGetErrorString(err));
  //   return -1;
  // }

  double start = MPI_Wtime();

  /* iterate for maxIters timesteps */
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow(params, cells_d, obstacles_d);
    av_vels[tt] = collision(params, cells_d, tmp_cells_d, obstacles_d, tot_cells);

    // if (tt == 1 &&params.rank == 2 ) {
    // t_speed* temp = NULL;
    // cudaMallocHost((void**)&temp, sizeof(t_speed) * params.nx * params.ny);
    // cudaMemcpy(temp, tmp_cells_d, sizeof(t_speed) * params.nx * params.ny, cudaMemcpyDeviceToHost);

    // print(&params, temp);
    // cudaFreeHost(temp);
    //     printf("------------------exchange----------------\n\n\n");
    // }

    swap = tmp_cells_d;
    tmp_cells_d = cells_d;
    cells_d = swap;

    
    exchange_ghost_cells(&params, cells_d);


    // if (params.rank == 0 && tt == 0 ) {
    // t_speed* temp = NULL;
    // cudaMallocHost((void**)&temp, sizeof(t_speed) * params.nx * params.ny);
    // cudaMemcpy(temp, tmp_cells_d, sizeof(t_speed) * params.nx * params.ny, cudaMemcpyDeviceToHost);

    // print(&params, temp);
    // cudaFreeHost(temp);
    // }

#ifdef DEBUG
    if (params.rank == 0)
    {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    }
#endif
  }

  double stop = MPI_Wtime();


  /* write final values and free memory */
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  reynolds =  av_vels[params.maxIters - 1] * params.reynolds_dim / viscosity;
  if (params.rank == 0)
  {
    printf("==done==\n");
    printf("Runtime: %f s\n", stop-start);
    
  }
  transDataToCPU(&params, cells_d, cells);

  collectResult(&params, cells, av_vels, obstacles);

  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  MPI_Finalize();
  return EXIT_SUCCESS;
}


float collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int tot_cells)
{
    // Host variable to hold the total velocity computed by CUDA
    float h_tot_u = 0.0f;
    float local_u = 0.0f;

    // Call the CUDA function to compute the total velocity
    collision_cuda(params, cells, tmp_cells, obstacles, &h_tot_u);

    MPI_Allreduce(&h_tot_u, &local_u, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Return the average velocity
    return local_u / (float)tot_cells;
}

static int sizeOfRank(int rank, int size, int N)
{
    return N / size + ((N % size > rank) ? 1 : 0);
}

void transDataToCPU(t_param* params, t_speed* cells_d, t_speed* cells_h){

    int size = params->ny * params->nx;
    cudaMemcpy(cells_h, cells_d, sizeof(t_speed) * size, cudaMemcpyDeviceToHost);

}



void initialise_gpu(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_h){
        
    int size = params->ny * params->nx;
    cudaMallocManaged((void**)cells_d, sizeof(t_speed) * size);
    cudaMallocManaged((void**)tmp_cells_d, sizeof(t_speed) * size);
    cudaMalloc((void**)obstacles_d, sizeof(int) * size);
    cudaMallocManaged((void**)av_vels_d, sizeof(float) * params->maxIters);

    // cudaMemset(*tmp_cells_d, 1,sizeof(t_speed)* size);
    cudaMemset(*av_vels_d, 0 ,sizeof(float)* size);
    cudaMemcpy(*cells_d, *cells_h, sizeof(t_speed) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(*obstacles_d, *obstacles_h, sizeof(int) * size, cudaMemcpyHostToDevice);
    // cudaMemset(*obstacles_d, 0 ,sizeof(int)* size);
    // if (tt == 0 &&params.rank == 0 ) {
    // int* temp = NULL;
    // cudaMallocHost((void**)&temp, sizeof(int) * params->nx * params->ny);
    // cudaMemcpy(temp, *obstacles_d, sizeof(int) * params->nx * params->ny, cudaMemcpyDeviceToHost);

    // // print(&params, temp);
    // // }

    // if(params->rank == 0){
    //   for(int i=0; i<params->ny; i++){
    // for(int j=0; j<params->nx; j++){
    //   printf("%d ", temp[j+i*params->nx]);
    // }
    // printf("\n");
    //   }
    // }
    // exit(0);

// cudaDeviceSynchronize();
// if(params->rank == 1){
//       dim3 threadsPerBlock(1, 1); // 每个线程块 16x16 个线程
//     dim3 blocksPerGrid(1,1);
//      print_ob<<<blocksPerGrid, threadsPerBlock>>>(*obstacles_d, params->nx, params->ny);
//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     fprintf(stderr, "CUDA Error after kernel execution: %s\n", cudaGetErrorString(err));
//     return ;
//   }
// }


}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed**  cells_ptr, t_speed**  tmp_cells_ptr,
               int**  obstacles_ptr, float** av_vels_ptr)
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


  //MPI initialize
  MPI_Comm_rank(MPI_COMM_WORLD, &(params->rank));
  MPI_Comm_size(MPI_COMM_WORLD, &(params->size));
  MPI_Type_contiguous(NSPEEDS, MPI_FLOAT, &MPI_T_SPEED);
  MPI_Type_commit(&MPI_T_SPEED);
  
  params->nyLocal = sizeOfRank(params->rank, params->size, params->ny); 

  int *rcvCounts = (int*) malloc(params->size * sizeof(int));  
  int *displs =  (int*) malloc(params->size * sizeof(int)); 
  int local_count = params->nyLocal;
  MPI_Allgather(&local_count, 1, MPI_INT, rcvCounts, 1, MPI_INT, MPI_COMM_WORLD);

  displs[0]=0;
  for (int i = 1; i < params->size; i++)
      displs[i] = displs[i - 1] + rcvCounts[i - 1];

  params->yStart = displs[params->rank];
  params->yEnd = (params->rank == params->size - 1) ? params->ny : displs[params->rank + 1];
  // printf("rank: %d -- %d, %d\n", params->rank, params->yStart, params->yEnd);

  free(displs);
  free(rcvCounts);


  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = (int*)malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
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

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
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
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}


void exchange_ghost_cells(const t_param* params, t_speed* cells){

  int nx = params->nx;
  int rank = params->rank;
  int size = params->size;
  int local_nrows = params->nyLocal;
    int start = params->yStart;

  int down = (rank == 0) ? (rank + size - 1) : (rank - 1);
  int up = (rank + 1) % size;
  
  int recv_down = (rank == 0) ? (params->ny - 1) * nx : (start - 1) * nx;
  int recv_up = (rank == size-1) ? 0 : (start + local_nrows) * nx;

  MPI_Request request[4];
    //halo exchange
    /* send to the down, receive from up */
    //memcpy(sendbuf, cells[rank * nrows * local_ncols].speeds, num * sizeof(float));

    MPI_Isend(cells + start * nx, nx, MPI_T_SPEED, down, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(cells + recv_up, nx, MPI_T_SPEED, up, 0, MPI_COMM_WORLD, &request[1]);
    // printf("rank: %d -- send: %d, rece: %d\n", rank, start, recv_up/nx);
    // t_speed* temp_send =NULL;
    // t_speed* temp_recv=NULL;


    // cudaMallocHost((void**)&temp_send, nx * sizeof(t_speed));
    // cudaMallocHost((void**)&temp_recv, nx * sizeof(t_speed));
    // cudaMemcpy(temp_send, (cells + start * nx), nx * sizeof(t_speed), cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp_recv,  (cells + recv_up), nx * sizeof(t_speed), cudaMemcpyDeviceToHost);


// Print the data from the sending buffer
// for (int x = 0; x < nx; x++) {
//     printf("%f ", temp_send[x].speeds[2]);
// }
// printf("\n");
// printf("rank: %d -- send: %d, rece: %d\n", rank, start, recv_up / nx);

// // Print the data from the receiving buffer
// for (int x = 0; x < nx; x++) {
//     printf("%f ", temp_recv[x].speeds[2]);
// }
// printf("\n");




    //MPI_Send(cells + start * nx, nx*9, MPI_FLOAT, down, 0, MPI_COMM_WORLD);
    //MPI_Recv(cells + recv_up, nx*9, MPI_FLOAT, up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // printf("rank: %d -- send: %d, rece: %d\n", rank, start, recv_up/nx);

    /* send to the up, receive from down */
    //memcpy(sendbuf, cells[rank * nrows * local_ncols + (local_nrows - 1) * local_ncols].speeds, num * sizeof(float));

    
    //MPI_Send(cells + start * nx + (local_nrows - 1) * nx, nx*9, MPI_FLOAT, up, 1, MPI_COMM_WORLD);
    //MPI_Recv(cells + recv_down, nx*9, MPI_FLOAT, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Isend(cells + start * nx + (local_nrows - 1) * nx, nx, MPI_T_SPEED, up, 1, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(cells + recv_down, nx, MPI_T_SPEED, down, 1, MPI_COMM_WORLD, &request[3]);

    // printf("rank: %d -- send: %d, rece: %d\n", rank, start+local_nrows - 1, recv_down/nx);

    MPI_Waitall(4, request, MPI_STATUS_IGNORE);


}




void collectResult(const t_param *params, t_speed* cells, float* av_vels, int* obstacles){
  
  int rank = params->rank;
  int size = params->size;
  int nx = params->nx;

  int *rcvCounts = NULL;  
  int *displs = NULL; 

  if(rank == 0){
    rcvCounts = (int*) malloc(size * sizeof(int));
    displs = (int*) malloc(size * sizeof(int));
  }
  int local_count = nx * params->nyLocal;  // Number of elements per process
  MPI_Gather(&local_count, 1, MPI_INT, rcvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(rank == 0){
    displs[0]=0;
    for (int i = 1; i < size; i++) 
        displs[i] = displs[i - 1] + rcvCounts[i - 1];
  }

  if(rank == 0){
    for (int i = 1; i < size; i++) {
            // printf("recv from %d, size = %d, pos = %d\n",  i, rcvCounts[i], displs[i]);
      MPI_Recv(cells + displs[i], rcvCounts[i], MPI_T_SPEED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
  }else{
    // printf("%d send to 0, size = %d, pos = %d\n", rank ,local_count, params->yStart*nx);
    MPI_Send(cells+params->yStart*nx, local_count, MPI_T_SPEED, 0, 0, MPI_COMM_WORLD);
  }

  if(rank == 0)   write_values(*params, cells, obstacles, av_vels);

}


int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  // Clean up the custom MPI datatype
  MPI_Type_free(&MPI_T_SPEED);

  return EXIT_SUCCESS;
}


float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
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
      int index = ii + jj*params.nx;
      /* an occupied cell */
      if (obstacles[index])
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
          local_density += cells[index].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[index].speeds[1]
               + cells[index].speeds[5]
               + cells[index].speeds[8]
               - (cells[index].speeds[3]
                  + cells[index].speeds[6]
                  + cells[index].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[index].speeds[2]
               + cells[index].speeds[5]
               + cells[index].speeds[6]
               - (cells[index].speeds[4]
                  + cells[index].speeds[7]
                  + cells[index].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
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
