//
//  GPUCalculationFunction.h
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#ifndef GPUCalculationFunction_h
#define GPUCalculationFunction_h

#define MAX_FLOATS_PER_GPU_BUFFER 402653184

#import "RandomFloatDataGenerator.h"
#import "Context.h"

@import Foundation;
@import MetalPerformanceShaders;
@import QuartzCore;

typedef NS_ENUM (NSInteger, gpuFunction) {
    gpuFunctionVectorsAdd = 0,
    gpuFunctionVectorsSub = 1,
    gpuFunctionVectorsMul = 2,
    gpuFunctionVectorsDiv = 3
};


@interface GPUCalculationFunction : NSObject

 @property (readonly) Context* FunctionContext;

 @property (readonly) id <MTLBuffer> GPUDataFirstArgBuffer;
 @property (readonly) id <MTLBuffer> GPUDataSecondArgBuffer;
 @property (readonly) id <MTLBuffer> GPUDataResultBuffer;

 @property (readonly) id <MTLCommandBuffer> GPUCommandBuffer;
 @property (readonly) id <MTLComputeCommandEncoder> GPUCommandEncoder;
 @property (readonly) id <MTLComputePipelineState> GPUFunctionPSO;


+ (void) addRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time;
+ (void) subRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time;
+ (void) mulRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time;
+ (void) divRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time;

@end

#endif /* GPUCalculationFunction_h */
