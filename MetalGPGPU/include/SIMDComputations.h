//
//  SIMDConputations.h
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#ifndef SIMDComputations_h
#define SIMDComputations_h
#include <immintrin.h>
#import "GPUMatrixOperations.h"
#import "RandomFloatDataGenerator.h"

@interface SIMDComputations : NSObject

+ (void) multTwoVectorsWithSize: (unsigned long) length andTimePointer: (double*) time;

+ (void) dotProductOfTwoMatrixWithShape: (shape) aShape andShape: (shape) bShape withTimePointer: (double*) time;

+ (void) dotProductOfTwoMatrixAsGridWithShape: (shape) aShape andShape: (shape) bShape withTimePointer: (double*) time;

@end

#endif /* SIMDComputations_h */
