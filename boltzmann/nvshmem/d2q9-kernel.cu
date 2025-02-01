#include <nvshmem.h>
#include <nvshmemx.h>

#define NSPEEDS         9

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

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;


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

    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid(32);
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;
    accelerate_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, obstacles, params.nyLocal, params.nx, w1, w2);
    
  }

}



__global__ void propagate_rebound_collision_kernel
(t_speed* cells, t_speed* tmp_cells, int* obstacles, int nyLocal, int nx, float omega, bool rankIsLast, float w1_flow, float w2_flow, int down, int up, int recv_up)
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

      if (jj == 1){
        int send_down = nx + ii;
        nvshmem_float_put((float *)(tmp_cells+recv_up+ii), (float *)(tmp_cells+send_down), 9, down);

      }
      if(jj == nyLocal){
        int send_up = nyLocal * nx + ii;               
        int recv_down = ii;             
        nvshmem_float_put((float *)(tmp_cells+recv_down), (float *)(tmp_cells+send_up), 9, up);
      }
    }

  }

}

static int sizeOfRank(int rank, int size, int N)
{
    return N / size + ((N % size > rank) ? 1 : 0);
}


void propagate_rebound_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles){

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid(32, 32);
    bool rankIsLast = (params.rank == params.size -1);
    float w1_flow = params.density * params.accel / 9.f;
    float w2_flow = params.density * params.accel / 36.f; 


    int rank = params.rank;
    int size = params.size;
    int nx = params.nx;
    int down = (rank == 0) ? (size - 1) : (rank - 1);
    int up = (rank + 1) % size;  

    int down_ny = sizeOfRank(down,size, params.ny);
    int recv_up = (down_ny + 1) * nx;

    propagate_rebound_collision_kernel<<<blocksPerGrid, threadsPerBlock>>>
          (cells, tmp_cells, obstacles, params.nyLocal, params.nx, params.omega,rankIsLast,w1_flow,w2_flow, down, up, recv_up);



}