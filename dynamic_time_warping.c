
#include <math.h>
#include <stdio.h>

float search_optimal_path(float* costs, float* cumulative_costs, int num_rows, int num_cols)
{
    for(int row=0; row<num_rows; ++row)
    {
        for(int col=0; col<num_cols; ++col)
        {
            if(col==0 && row==0)
            {
                cumulative_costs[row*num_cols+col] = costs[row*num_cols+col];
            }
            else if(row==0)
            {
                cumulative_costs[row*num_cols+col] = (cumulative_costs[row*num_cols + col-1] + costs[row*num_cols + col]);
            }
            else if(col==0)
            {
                cumulative_costs[row*num_cols+col] = (cumulative_costs[(row-1)*num_cols + col] + costs[row*num_cols + col]);
            }
            else
            {
                float horizontal = cumulative_costs[row*num_cols + col-1] + costs[row*num_cols + col];
                float vertical = cumulative_costs[(row-1)*num_cols + col] + costs[row*num_cols + col];
                float diagonal = cumulative_costs[(row-1)*num_cols + col-1] + 1.41*costs[row*num_cols + col];
                cumulative_costs[row*num_cols + col] = fmin(fmin(horizontal, vertical), diagonal);
            }
        }
    }
    return cumulative_costs[(num_rows-1)*num_cols + (num_cols-1)];
}


void distance_matrix(float* path1, int num_rows, float* path2, int num_cols, int num_features, float* distances)
{
    for(int row=0; row<num_rows; ++row)
    {
        for(int col=0; col<num_cols; ++col)
        {
            float sum_diff = 0;
            for(int feat=0; feat<num_features; ++feat)
            {
                float diff = path1[row*num_features+feat] - path2[col*num_features+feat];
                diff *= diff;
                sum_diff += diff;
            }
            distances[row*num_cols+col] = sqrt(sum_diff);
        }
    }
}
