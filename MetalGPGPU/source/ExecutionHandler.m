//
//  ExecutionHandler.m
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//



#import "ExecutionHandler.h"

@implementation ExecutionHandler
{
    Context *gpuContext;
}

- (instancetype) init {
    self = [super init];
    if (self) {
        gpuContext = [Context createGPUDefautContext];
    }
    return self;
}


- (void) calculateMatrisesOnGPU {
    double time;
    printf("\n______________ Matrix Testig ____________\n");
    for (int i = 0; i < MATRIX_ITERATIONS; ++i) {
        shape matrixShape;
        matrixShape.width = FIRST_MATRIX_ROW_NUM * (unsigned long)pow(2, i + 1);
        matrixShape.height = FIRST_MATRIX_COLUMN_NUM * (unsigned long)pow(2, i + 1);
        [GPUMatrixOperations createAndMultRandomMatriciesWithShape:matrixShape andShape:matrixShape inContext:gpuContext withTimeVariable:&time];
        printf("GPU Dot product for matrices %lux%lu complited in %f seconds\n", (unsigned long)matrixShape.width, matrixShape.height, time);
    }
    printf("\n__________________________________________\n");
}

- (void) calculateVectorsOnGPU {
    double time;
    printf("\n______________Vectors Testig______________\n");
    for (int i = 0; i < VECTOR_ITERATIONS; ++i) {
        unsigned long vector_len = FIRST_MATRIX_ROW_NUM  * (unsigned long)pow(2, i + 1);
        [GPUCalculationFunction mulRandomDataWithLength: vector_len inContext:gpuContext withTimeVariable:&time];
        printf("GPU Dot product for vectors with lengths %lu complited in %f seconds\n", vector_len, time);
    }
    printf("\n__________________________________________\n");
}

- (void) calculateSIMDVectors {
    double time;
    printf("\n______________SIMD Vectors Testig______________\n");
    for (int i = 0; i < VECTOR_ITERATIONS; ++i) {
        unsigned long vector_len = FIRST_MATRIX_ROW_NUM  * (unsigned long)pow(2, i + 1);
        [SIMDComputations multTwoVectorsWithSize:vector_len andTimePointer:&time];
        printf("Product for SIMD vectors with lengths %lu complited in %f seconds\n", vector_len, time);
    }
    printf("\n_______________________________________________\n");
}

- (void) calculateMatrisesWithSimd {
    double time;
    printf("\n______________ Matrix Testig ____________\n");
    for (int i = 0; i < MATRIX_ITERATIONS; ++i) {
        shape matrixShape;
        matrixShape.width = FIRST_MATRIX_ROW_NUM * (unsigned long)pow(2, i + 1);
        matrixShape.height = FIRST_MATRIX_COLUMN_NUM * (unsigned long)pow(2, i + 1);
        [SIMDComputations dotProductOfTwoMatrixWithShape:matrixShape andShape:matrixShape withTimePointer:&time];
        printf("SIMD dot product for matrices %lux%lu complited in %f seconds\n", (unsigned long)matrixShape.width, matrixShape.height, time);
    }
    printf("\n__________________________________________\n");
}


- (void) calculateMatrisesGridWithSimd {
    double time;
    printf("\n______________ Matrix Testig ____________\n");
    for (int i = 0; i < MATRIX_ITERATIONS; ++i) {
        shape matrixShape;
        matrixShape.width = FIRST_MATRIX_ROW_NUM * (unsigned long)pow(2, i + 1);
        matrixShape.height = FIRST_MATRIX_COLUMN_NUM * (unsigned long)pow(2, i + 1);
        [SIMDComputations dotProductOfTwoMatrixAsGridWithShape:matrixShape andShape:matrixShape withTimePointer:&time];
        printf("SIMD dot product for matrices %lux%lu complited in %f seconds\n", (unsigned long)matrixShape.width, matrixShape.height, time);
    }
    printf("\n__________________________________________\n");
}


+ (void) startTesting {
    ExecutionHandler* handler = [[ExecutionHandler alloc] init];
    [handler calculateMatrisesOnGPU];
    [handler calculateVectorsOnGPU];
    [handler calculateSIMDVectors];
//    [handler calculateMatrisesWithSimd];
//    [handler calculateMatrisesGridWithSimd];
}

@end
