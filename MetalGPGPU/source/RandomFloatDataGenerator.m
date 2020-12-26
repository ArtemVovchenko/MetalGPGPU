//
//  RandomFloatDataGenerator.m
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#import "RandomFloatDataGenerator.h"

@implementation RandomFloatDataGenerator

+ (void) fillMTLBufferWithRandomData: (id <MTLBuffer>) buffer withLength: (NSUInteger) length {
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < length; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}


+ (float*) generateAlignedFloatDataWithLength: (unsigned long) length {
    float *dataPtr = (float*)_mm_malloc(sizeof(float) * length, 32);
    for (unsigned long index = 0; index < length; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
    return dataPtr;
}


+ (void) deallocateAlignedFloatDataFromPointer:(float*)dataPrt {
    _mm_free(dataPrt);
}

+ (float**) generateMatrixDataWithShape: (shape) matrixShape {
    float **outputMatrixPointer = (float**) malloc(sizeof(float*) * matrixShape.width);
    
    for (unsigned long i = 0; i < matrixShape.width; ++i) {
        outputMatrixPointer[i] = _mm_malloc(sizeof(float) * matrixShape.height, 32);
    }
    
    return outputMatrixPointer;
}

+ (void) deallocateMatrixFromPointer:(float **)matrixPointer withShape:(shape)matrixShape {
//    for (unsigned long i = 0; i < matrixShape.width; ++i) {
//        free(matrixPointer[i]);
//    }
    free(matrixPointer);
}

@end
