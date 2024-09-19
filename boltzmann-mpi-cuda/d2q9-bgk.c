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
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include <cuda_runtime.h>

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
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr,t_speed** cells_d, t_speed** tmp_cells_d,
             int** obstacles_ptr_d);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

static void exchange_ghost_cells(const t_param* params, t_speed* cells);
void collectResult(const t_param params, t_speed* cells, float* av_vels, int* obstacles_all);
void scatter_obstacle(const t_param* params, int *obstacles_all, int *obstacles_local);
void gather_cell(const t_param* params, t_speed* cells, t_speed* cells_all);
void gather_vels(const t_param* params, float* av_vels, float* av_vels_all);

int rebound_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);


static void print(t_param* params, t_speed* cell)
{
    int nx = params->nx;

    for (int i = 0; i < params->size; i++) {
        if (i == params->rank) {
            printf("### RANK %d "
                   "#######################################################\n",
                params->rank);
            for (int j = 0; j < params->nyLocal + 2; j++) {
                printf("%02d:", j);
                for (int i = 0; i < nx; i++) {
                    // for(int k =0; k<NSPEEDS; k++)
                      printf("%12.6f ", cell[j * nx + i].speeds[2]);
                  
                }
                printf("\n");
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

static void print_ob(t_param* params, int* cell)
{
    int nx = params->nx;

    for (int i = 0; i < params->size; i++) {
        if (i == params->rank) {
            printf("### RANK %d "
                   "#######################################################\n",
                params->rank);
            for (int j = 0; j < params->nyLocal + 2; j++) {
                printf("%02d:", j);
                for (int i = 0; i < nx; i++) {
                    // for(int k =0; k<NSPEEDS; k++)
                      printf("%d ", cell[j * nx + i]);
                  
                }
                printf("\n");
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  t_speed* swap = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  int*     obstacles_all = NULL;
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  t_speed* cells_d     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells_d = NULL;    /* scratch space */
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
  

  MPI_Init(&argc, &argv);
  double start_all = MPI_Wtime();

  gettimeofday(&timstr, NULL); 
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;


  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &obstacles_all);
  dataToDevices(&params, &cells_d, &tmp_cells_d, &obstacles_d, &av_vels_d, &cells, &obstacles);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  double start = MPI_Wtime();


  for (int tt = 0; tt < params.maxIters; tt++)
  {

    // timestep(params, cells, tmp_cells, obstacles);
  // accelerate_flow(params, cells_d, obstacles_d);
  // propagate(params, cells_d, tmp_cells_d);
  // rebound(params, cells_d, tmp_cells_d, obstacles_d);
  // collision(params, cells_d, tmp_cells_d, obstacles_d);
  timestep(params, cells_d, tmp_cells_d, obstacles_d);

  // av_vels[tt] = av_velocity(params, cells, obstacles);


  swap = tmp_cells_d;
  tmp_cells_d = cells_d;
  cells_d = swap;




    exchange_ghost_cells(&params, cells_d);


    // if(tt==0)print(&params, cells);
    // exit(0);

  // print(&params, cells);

    // if(tt==0)print(&params, cells);

#ifdef DEBUG
    printf("rank: %d\n", params.rank);
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif



  } 
 double stop = MPI_Wtime();

 if (params.rank == 0)
  {
    printf("Runtime loop: %f s\n", stop-start);
  }

  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;


  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  dataToHost(&params, cells_d, cells);
  // Collate data from ranks here 
  collectResult(params, cells, av_vels, obstacles_all);


  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &cells_d, &tmp_cells_d,& obstacles_d );

  double stop_all = MPI_Wtime();
  if (params.rank == 0)
  {
    printf("==done==\n");
    printf("Runtime all: %f s\n", stop_all-start_all);
    
  }

// ss

  MPI_Finalize();
  return EXIT_SUCCESS;
}



float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  tot_u = 0.f;


  /* loop over all non-blocked cells */
  for (int jj = 1; jj < params.nyLocal+1; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;

      }
    }

  }

  return tot_u / (float)tot_cells;
}

static int sizeOfRank(int rank, int size, int N)
{
    return N / size + ((N % size > rank) ? 1 : 0);
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

  MPI_Comm_rank(MPI_COMM_WORLD, &(params->rank));
  MPI_Comm_size(MPI_COMM_WORLD, &(params->size));

  int num_devices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
  CUDA_RT_CALL(cudaSetDevice(params->rank%num_devices));
  CUDA_RT_CALL(cudaFree(0));

  MPI_Type_contiguous(NSPEEDS, MPI_FLOAT, &MPI_T_SPEED);
  MPI_Type_commit(&MPI_T_SPEED);


  params->nyLocal = sizeOfRank(params->rank, params->size, params->ny);

  // printf("rank: %d, nyLocal: %d \n", params->rank, params->nyLocal);

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * ((params->nyLocal+2) * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * ((params->nyLocal+2) * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->nyLocal+2; jj++)
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

// for (int jj = 0; jj < params->nyLocal + 2; jj++) {

//     for (int ii = 0; ii < params->nx; ii++) {
//         for (int k = 0; k < 9; k++) {

//           // if(jj == 0)  (*cells_ptr)[ii].speeds[k] = (float)(k*100);
//           // else if(jj == params->nyLocal + 1 )  (*cells_ptr)[jj * params->nx + ii].speeds[k] = (float)(k*100);
//             // 使用 ii, jj, k 的组合来初始化，确保每个网格点不同
//           // else 
//           (*cells_ptr)[jj * params->nx + ii].speeds[k] = (float)(ii + jj*params->nx + k*100+params->rank*1000+k);
//         }

//     }
// }
  


  /* the map of obstacles */
  *obstacles_ptr = (int *)malloc(sizeof(int) * ((params->nyLocal+2) * params->nx));
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
  memset(*obstacles_ptr, 0, sizeof(int) * ((params->nyLocal + 2) * params->nx));


  if(params->rank == 0){

    *obstacles_all_ptr = (int*)malloc(sizeof(int) * (params->ny * params->nx));

    if (*obstacles_all_ptr == NULL) die("cannot allocate column memory for obstacles(all)", __LINE__, __FILE__);

    /* first set all cells in obstacle(all) array to zero */
    for (int jj = 0; jj < params->ny; jj++)
    {
      for (int ii = 0; ii < params->nx; ii++)
      {
        (*obstacles_all_ptr)[ii + jj*params->nx] = 0;
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
      (*obstacles_all_ptr)[xx + yy*params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);
  }
      
  scatter_obstacle(params,*obstacles_all_ptr,*obstacles_ptr);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int dataToHost(t_param* params, t_speed* cells_d, t_speed* cells_h)
{

  cudaMemcpy(cells_h, cells_d, sizeof(t_speed) * (params->nyLocal+2) * params->nx, cudaMemcpyDeviceToHost);
  return EXIT_SUCCESS;
}


int dataToDevices(t_param* params, t_speed**  cells_d, t_speed**  tmp_cells_d,
                int**  obstacles_d, float** av_vels_d, t_speed**  cells_h, int** obstacles_h){

  int size = (params->nyLocal+2) * params->nx;
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  CUDA_RT_CALL(cudaMalloc((void**)cells_d, sizeof(t_speed) * size));

  CUDA_RT_CALL(cudaMalloc((void**)tmp_cells_d, sizeof(t_speed) * size));

  CUDA_RT_CALL(cudaMalloc((void**)obstacles_d, sizeof(int) * size));

  // cudaMalloc((void**)av_vels_d, sizeof(float) * sizeLocal);
  //   system("nvidia-smi");

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
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  cudaFree(*cells_d);
  cudaFree(*tmp_cells_d);
  cudaFree(*obstacles_ptr_d);


  // Clean up the custom MPI datatype
  MPI_Type_free(&MPI_T_SPEED);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 1; jj < params.nyLocal+1; jj++)
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

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
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


void gather_vels(const t_param* params, float* av_vels, float* av_vels_all){

    int iters = params->maxIters; 
  int size = params->size;
  
  // printf("rank: %d\n", params->rank);
  // for (int i = 0; i < iters; i++) {
  //     printf("av_vels[%d] = %f ", i, av_vels[i]);
  // }

  MPI_Reduce(av_vels, av_vels_all, params->maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (params->rank == 0) {
        // Calculate the global average velocity from the summed values
        // Note: You need to adjust this according to your actual data and how you interpret 'av_vels'
        for (int i = 0; i < iters; i++) {
            // Example: Normalize or compute average, depending on how av_vels is defined
            av_vels_all[i] /= size;  // Simple averaging example
        }

        // printf("Global average velocities:\n");
        // for (int i = 0; i < iters; i++) {
        //     printf("av_vels[%d] = %f\n", i, av_vels_all[i]);
        // }
    }
}

void gather_cell(const t_param* params, t_speed* cells, t_speed* cells_all)
{
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

    // Debugging output
    // print(&params, cells);
    // printf("rank %d, gather done\n", rank);
    // if(rank == 0){
    //    for (int j = 0; j < params->ny; j++) {
    //     for (int i = 0; i < nx; i++) {
    //       printf("%5.2f ", cells_all[j * nx + i].speeds[2]);
    //     }
    //     printf("\n");
    //    }
    // }

    // exit(0);
    
}



void collectResult(const t_param params, t_speed* cells, float* av_vels, int* obstacles_all){

  t_speed* cells_all = NULL;
  float* av_vels_all = NULL;

  int rank = params.rank;

  if(rank == 0){
  /* write final values and free memory */
    // printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    // printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    // printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    // printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    // printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);

    cells_all = (t_speed*)malloc(sizeof(t_speed) * params.ny * params.nx);
    if (cells_all == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    av_vels_all = (float*)malloc(params.maxIters * sizeof(float));
    if (av_vels_all == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  }


  gather_cell(&params, cells, cells_all);

  gather_vels(&params, av_vels, av_vels_all);

  if(rank == 0){

    write_values(params, cells_all, obstacles_all, av_vels_all);

    free(cells_all);
    cells_all = NULL;

    free(av_vels_all);
    av_vels_all = NULL;

    free(obstacles_all);
    obstacles_all = NULL;
  }


}


void scatter_obstacle(const t_param* params, int *obstacles_all, int *obstacles_local) {
  
    int *rcvCounts = NULL;  // Array to store the number of elements each process will receive
    int *displs = NULL;  // Array to store the displacement for each process
    int size = params->size;
    int rank = params->rank;

    // Allocate memory for receive counts and displacements on rank 0
    if (rank == 0) {
        rcvCounts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
    }

    // Gather the number of elements each process will receive (nyLocal * nx)
    int local_data_count = params->nyLocal * params->nx;  // Number of actual data elements for each process



    // Gather the receive counts on rank 0
    MPI_Gather(&local_data_count, 1, MPI_INT, rcvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 computes displacements
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + rcvCounts[i - 1];
        }
    }

    // Use MPI_Scatterv to send the relevant portions of cells_all to each process
    // The data is placed starting at the second row in cells_local (leaving first and last rows empty)
    MPI_Scatterv(obstacles_all, rcvCounts, displs, MPI_INT, 
                 &obstacles_local[params->nx], local_data_count, MPI_INT,  // Skip the first row for ghost cells
                 0, MPI_COMM_WORLD);

    // printf("Rank %d received data (with ghost cells):\n", rank);
    // for (int j = 1; j < params->nyLocal + 1; j++) {  // +2 includes the ghost cells
    //     for (int i = 0; i < params->nx; i++) {
    //         printf("%d ", obstacles_local[j * params->nx + i]);
    //     }
    //     printf("\n");
    // }
    // exit(0);

    // Clean up rank 0 resources
    if (rank == 0) {
        free(rcvCounts);
        free(displs);
    }
}

void exchange_ghost_cells(const t_param* params, t_speed* cells) {
    int rank = params->rank;
    int size = params->size;
    int nx = params->nx;
    int nyLocal = params->nyLocal;


int down = (rank == 0) ? (size - 1) : (rank - 1);  // 下方邻居
int up = (rank + 1) % size;  // 上方邻居

int send_up = nyLocal * nx;      // 发送本地的最后一行数据
int send_down = nx;              // 发送本地的第一行数据
int recv_down = 0;               // 接收放到下幽灵单元
int recv_up = (nyLocal + 1) * nx; // 接收放到上幽灵单元

MPI_Request request[4];

// 发送到下方 (down)，接收来自上方 (up)
MPI_CALL(MPI_Isend(cells + send_down, nx, MPI_T_SPEED, down, 0, MPI_COMM_WORLD, &request[0]));
MPI_CALL(MPI_Irecv(cells + recv_up, nx, MPI_T_SPEED, up, 0, MPI_COMM_WORLD, &request[1]));

// 发送到上方 (up)，接收来自下方 (down)
MPI_CALL(MPI_Isend(cells + send_up, nx, MPI_T_SPEED, up, 1, MPI_COMM_WORLD, &request[2]));
MPI_CALL(MPI_Irecv(cells + recv_down, nx, MPI_T_SPEED, down, 1, MPI_COMM_WORLD, &request[3]));

// 等待所有发送和接收完成
MPI_Waitall(4, request, MPI_STATUSES_IGNORE);


}