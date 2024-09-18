
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

int accelerate_flow(const t_param params, t_speed*  cells, int*  obstacles)
{
  if(params.rank == params.size - 1){

    dim3 threadsPerBlock(64); // 每个线程块 16x16 个线程
    dim3 blocksPerGrid(32);
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;
    accelerate_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, obstacles, params.nyLocal, params.nx, w1, w2);
    // cudaDeviceSynchronize();

    return EXIT_SUCCESS;
  }

}


__global__ void propagate_kernel(t_speed* cells, t_speed* tmp_cells, int nyLocal, int nx )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int jdx = blockIdx.y * blockDim.y + threadIdx.y+1;

    int gridStrideX = gridDim.x * blockDim.x;
    int gridStrideY = gridDim.y * blockDim.y;
    const int ySize = nyLocal+1;

    for(int jj = jdx; jj < ySize; jj += gridStrideY){
        for(int ii = idx; ii < nx; ii += gridStrideX){
            int index = ii + jj * nx;
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */
            int y_n = (jj + 1) % (nyLocal+2);
            int x_e = (ii + 1) % nx;
            int y_s = (jj == 0) ? (jj + nyLocal + 2 - 1) : (jj - 1);
            int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            tmp_cells[index].speeds[0] = cells[index].speeds[0]; /* central cell, no movement */
            tmp_cells[index].speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
            tmp_cells[index].speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
            tmp_cells[index].speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
            tmp_cells[index].speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
            tmp_cells[index].speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
            tmp_cells[index].speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
            tmp_cells[index].speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
            tmp_cells[index].speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */
        }
    }
}


int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid(16, 16);

    propagate_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, tmp_cells, params.nyLocal, params.nx);
    // cudaDeviceSynchronize();


  return EXIT_SUCCESS;
}



__global__ void rebound_kernel(t_speed* cells, t_speed* tmp_cells, int* obstacles, int nyLocal, int nx )
{
  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int jdx = blockIdx.y * blockDim.y + threadIdx.y+ 1;

    int gridStrideX = gridDim.x * blockDim.x;
    int gridStrideY = gridDim.y * blockDim.y;

    const int ySize = nyLocal+1;

    for(int jj = jdx; jj < ySize; jj += gridStrideY){
        for(int ii = idx; ii < nx; ii += gridStrideX){
            int index = ii + jj * nx;
            if (obstacles[index])
            {
                /* called after propagate, so taking values from scratch space
                ** mirroring, and writing into main grid */
                cells[index].speeds[1] = tmp_cells[index].speeds[3];
                cells[index].speeds[2] = tmp_cells[index].speeds[4];
                cells[index].speeds[3] = tmp_cells[index].speeds[1];
                cells[index].speeds[4] = tmp_cells[index].speeds[2];
                cells[index].speeds[5] = tmp_cells[index].speeds[7];
                cells[index].speeds[6] = tmp_cells[index].speeds[8];
                cells[index].speeds[7] = tmp_cells[index].speeds[5];
                cells[index].speeds[8] = tmp_cells[index].speeds[6];
            }
            

        }
    }

}


int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid(16, 16);

    rebound_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, tmp_cells, obstacles, params.nyLocal, params.nx);
    // cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}


__global__ void collision_kernel(t_speed* cells, t_speed* tmp_cells, int* obstacles, int nyLocal, int nx, float omega)
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

  for(int jj = jdx; jj < ySize; jj += gridStrideY)
  {
    for(int ii = idx; ii < nx; ii += gridStrideX)
    {
      int index = ii + jj * nx;
      /* don't consider occupied cells */
      if (!obstacles[index])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[index].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[index].speeds[1]
                      + tmp_cells[index].speeds[5]
                      + tmp_cells[index].speeds[8]
                      - (tmp_cells[index].speeds[3]
                         + tmp_cells[index].speeds[6]
                         + tmp_cells[index].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[index].speeds[2]
                      + tmp_cells[index].speeds[5]
                      + tmp_cells[index].speeds[6]
                      - (tmp_cells[index].speeds[4]
                         + tmp_cells[index].speeds[7]
                         + tmp_cells[index].speeds[8]))
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
          cells[index].speeds[kk] = tmp_cells[index].speeds[kk]
                                                  + omega
                                                  * (d_equ[kk] - tmp_cells[index].speeds[kk]);
        }
      }
    }

  }

}


int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles){

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid(16, 16);

    collision_kernel<<<blocksPerGrid, threadsPerBlock>>>(cells, tmp_cells, obstacles, params.nyLocal, params.nx, params.omega);
    cudaDeviceSynchronize();


}