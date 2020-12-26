//
//  vectors_add.metal
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 26.11.2020.
//

#include <metal_stdlib>
using namespace metal;


kernel void vectors_add(device const float *vector_a,
                        device const float *vector_b,
                        device float *result_vector,
                        uint index [[thread_position_in_grid]]) {
    result_vector[index] = vector_a[index] + vector_b[index];
}


kernel void vectors_sub(device const float *vector_a,
                        device const float *vector_b,
                        device float *result_vector,
                        uint index [[thread_position_in_grid]]) {
    result_vector[index] = vector_a[index] - vector_b[index];
}


kernel void vectors_mul(device const float *vector_a,
                        device const float *vector_b,
                        device float *result_vector,
                        uint index [[thread_position_in_grid]]) {
    result_vector[index] = vector_a[index] * vector_b[index];
}


kernel void vectors_div(device const float *vector_a,
                        device const float *vector_b,
                        device float *result_vector,
                        uint index [[thread_position_in_grid]]) {
    result_vector[index] = vector_a[index] / vector_b[index];
}

