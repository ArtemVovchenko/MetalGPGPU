//
//  ExecutionHandler.h
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#ifndef ExecutionHandler_h
#define ExecutionHandler_h

#import "Context.h"
#import "GPUCalculationFunction.h"
#import "GPUMatrixOperations.h"
#import "SIMDComputations.h"

#define FIRST_MATRIX_ROW_NUM 64
#define FIRST_MATRIX_COLUMN_NUM 64
#define MATRIX_ITERATIONS 7

#define FIRST_VECTOR_LENGTH 64
#define VECTOR_ITERATIONS 22

@import Foundation;

@interface ExecutionHandler : NSObject

+ (void) startTesting;

@end

#endif /* ExecutionHandler_h */
