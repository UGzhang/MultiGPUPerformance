#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  int    size;
  int    rank;
  int    nyLocal;
  int    yStart;
  int    yEnd;
} t_param;

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;


__global__ void accelerate_kernel(t_speed* cells, int* obstacles, int ny, int nx, float w1, float w2) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStrideX = gridDim.x * blockDim.x;
    /* modify the 2nd row of the grid */
    int jdx = ny - 2;
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


int accelerate_flow(const t_param params, t_speed*  cells, int*  obstacles)
{
  dim3 threadsPerBlock(64); // 每个线程块 16x16 个线程
  dim3 blocksPerGrid(32);
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;
  accelerate_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, obstacles, params.ny, params.nx, w1, w2);

  return EXIT_SUCCESS;
}

//rebound, collision and av_velocity
__global__ void collision_kernel(t_speed* cells, t_speed* tmp_cells, int* obstacles, int nx, int ny, int yStart, int yEnd, float omega, float* d_tot_u, int rank) {
    
  //const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
       
  // const1 = 1 / c_sq, const2 = 1 / (2 * c_sq), const3 = 1 / (2 * c_sq * c_sq)
  const float const1 = 3.0f, const2 = 1.5f, const3 = 4.5f;
        
    float tmp[NSPEEDS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int jdx = blockIdx.y * blockDim.y + threadIdx.y;

    int gridStrideX = gridDim.x * blockDim.x;
    int gridStrideY = gridDim.y * blockDim.y;
    // if(rank == 1) printf("aa\n");

    for(int j = jdx; j < yEnd-yStart; j += gridStrideY)
    // for(int jj = jdx; jj<yEnd; jj += gridStrideY)
    {
      // if(jj >= yStart)
      // {
      for(int ii = idx; ii < nx; ii += gridStrideX)
      {
      int jj = j+yStart;
      int index = ii + jj*nx;
 
      int y_n = (jj + 1) % ny;
      int x_e = (ii + 1) % nx;
      int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

      tmp[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
      tmp[1] = cells[x_w + jj*nx].speeds[1]; /* east */
      tmp[2] = cells[ii + y_s*nx].speeds[2]; /* north */
      tmp[3] = cells[x_e + jj*nx].speeds[3]; /* west */
      tmp[4] = cells[ii + y_n*nx].speeds[4]; /* south */
      tmp[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
      tmp[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
      tmp[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
      tmp[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */

      // if(rank == 1) printf("%d, i=%d, j=%d\n", obstacles[idx + jdx * nx], idx, jdx);

      if (obstacles[index]) {
          
            tmp_cells[index].speeds[1] = tmp[3];
            tmp_cells[index].speeds[2] = tmp[4];
            tmp_cells[index].speeds[3] = tmp[1];
            tmp_cells[index].speeds[4] = tmp[2];
            tmp_cells[index].speeds[5] = tmp[7];
            tmp_cells[index].speeds[6] = tmp[8];
            tmp_cells[index].speeds[7] = tmp[5];
            tmp_cells[index].speeds[8] = tmp[6];
        } else {
            float local_density = 0.f;
            for (int kk = 0; kk < NSPEEDS; kk++) {
                local_density += tmp[kk];
                // if(rank == 0 && ii == 0 && jj ==1){
                //   printf("%f \n", tmp[kk]);
                // }
            }
            // if(rank == 0) printf("%f, i=%d, j=%d\n", local_density, idx, jdx);


            float u_x = (tmp[1] + tmp[5] + tmp[8] - (tmp[3] + tmp[6] + tmp[7])) / local_density;
            float u_y = (tmp[2] + tmp[5] + tmp[6] - (tmp[4] + tmp[7] + tmp[8])) / local_density;
            float u_sq = u_x * u_x + u_y * u_y;
            float u[NSPEEDS];
            u[1] =   u_x;        /* east */
            u[2] =         u_y;  /* north */
            u[3] = - u_x;        /* west */
            u[4] =       - u_y;  /* south */
            u[5] =   u_x + u_y;  /* north-east */
            u[6] = - u_x + u_y;  /* north-west */
            u[7] = - u_x - u_y;  /* south-west */
            u[8] =   u_x - u_y;  /* south-east */

            float d_equ[NSPEEDS];
            float w11=w1 * local_density, w22= w2 * local_density, u_sq1= 1.0f - u_sq*const2;
            /* zero velocity density: weight w0 */
            d_equ[0] = w0 * local_density
                   * (u_sq1);
            /* axis speeds: weight w1 */
            d_equ[1] = w11 * (u[1] * const1
                                         + (u[1] * u[1]) * const3
                                         + u_sq1);
            d_equ[2] = w11 * (u[2] * const1
                                         + (u[2] * u[2]) * const3
                                         + u_sq1);
            d_equ[3] = w11 * (u[3] * const1
                                         + (u[3] * u[3]) * const3
                                         + u_sq1);
            d_equ[4] = w11 * (u[4] * const1
                                         + (u[4] * u[4]) * const3
                                         + u_sq1);
            /* diagonal speeds: weight w2 */
            d_equ[5] = w22 * (u[5] * const1
                                         + (u[5] * u[5]) * const3
                                         + u_sq1);
            d_equ[6] = w22 * (u[6] * const1
                                         + (u[6] * u[6]) * const3
                                         + u_sq1);
            d_equ[7] = w22 * (u[7] * const1
                                         + (u[7] * u[7]) * const3
                                         + u_sq1);
            d_equ[8] = w22 * (u[8] * const1
                                         + (u[8] * u[8]) * const3
                                         + u_sq1);

            for (int kk = 0; kk < NSPEEDS; kk++) {
                tmp_cells[index].speeds[kk] = tmp[kk] + omega * (d_equ[kk] - tmp[kk]);
            }

            local_density = 0.f;
            for (int kk = 0; kk < NSPEEDS; kk++) {
                local_density += tmp_cells[index].speeds[kk];
            }

            u_x = (tmp_cells[index].speeds[1] + tmp_cells[index].speeds[5] + tmp_cells[index].speeds[8] - 
                   (tmp_cells[index].speeds[3] + tmp_cells[index].speeds[6] + tmp_cells[index].speeds[7])) / local_density;
            u_y = (tmp_cells[index].speeds[2] + tmp_cells[index].speeds[5] + tmp_cells[index].speeds[6] - 
                   (tmp_cells[index].speeds[4] + tmp_cells[index].speeds[7] + tmp_cells[index].speeds[8])) / local_density;

            float u_magnitude = sqrtf(u_x * u_x + u_y * u_y);

            atomicAdd(d_tot_u, u_magnitude);
        }
      
    }
  }
}


void collision_cuda(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* h_tot_u) {
    float* d_tot_u;
    cudaMalloc(&d_tot_u, sizeof(float));
    cudaMemset(d_tot_u, 0, sizeof(float));

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid(16, 16);

    // Launch the collision kernel
    collision_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, tmp_cells, obstacles, params.nx, params.ny, params.yStart, params.yEnd, params.omega, d_tot_u, params.rank);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_tot_u, d_tot_u, sizeof(float), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_tot_u);

}