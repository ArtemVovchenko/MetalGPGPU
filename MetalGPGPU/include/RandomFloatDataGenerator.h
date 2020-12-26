//
//  RandomFloatDataGenerator.h
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#ifndef RandomFloatDataGenerator_h
#define RandomFloatDataGenerator_h

#include <immintrin.h>
@import Foundation;
@import MetalPerformanceShaders;

typedef struct {
    NSUInteger width;
    NSUInteger height;
} shape;


@interface RandomFloatDataGenerator : NSObject

+ (void) fillMTLBufferWithRandomData: (id <MTLBuffer>) buffer withLength: (NSUInteger) length;
 
+ (float*) generateAlignedFloatDataWithLength: (unsigned long) length;

+ (void) deallocateAlignedFloatDataFromPointer: (float*) dataPrt;

+ (float**) generateMatrixDataWithShape: (shape) matrixShape;

+ (void) deallocateMatrixFromPointer: (float**) matrixPointer withShape: (shape) matrixShape;

@end

#endif /* RandomFloatDataGenerator_h */
