//
//  GPUMatrixOperations.h
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#ifndef GPUMatrixOperations_h
#define GPUMatrixOperations_h
#import "Context.h"
#import "RandomFloatDataGenerator.h"

@import Foundation;
@import MetalPerformanceShaders;
@import QuartzCore;


@interface GPUMatrixOperations : NSObject
    
 @property (readonly) shape MatrixShape;
 @property (readonly) Context* context;

 @property (readonly) MPSMatrixMultiplication* ExecutionKernel;
 @property (readonly) id <MTLCommandBuffer> CommandBuffer;
 
 @property (readonly) id <MTLBuffer> AMatrixBuffer;
 @property (readonly) id <MTLBuffer> BMatrixBuffer;
 @property (readonly) id <MTLBuffer> CMatrixBuffer;

+ (void) createAndMultRandomMatriciesWithShape: (shape) shapeA andShape: (shape) shapeB
                                     inContext: (Context*) context withTimeVariable: (double*) time;

@end



#endif /* GPUMatrixOperations_h */
