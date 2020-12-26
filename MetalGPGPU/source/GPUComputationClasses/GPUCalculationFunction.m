//
//  GPUCalculationFunction.m
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#import "GPUCalculationFunction.h"

@implementation GPUCalculationFunction

- (instancetype) initWithFunction: (NSInteger) function inContext: (Context*) context {
    self = [super init];
    if (self) {
        
        switch (function) {
            case gpuFunctionVectorsAdd:
                _GPUFunctionPSO = [context createFunctionPiplineStateObjectWithName:@"vectors_add"];
                break;
                
            case gpuFunctionVectorsSub:
                _GPUFunctionPSO = [context createFunctionPiplineStateObjectWithName:@"vectors_sub"];
                break;
                
            case gpuFunctionVectorsMul:
                _GPUFunctionPSO = [context createFunctionPiplineStateObjectWithName:@"vectors_mul"];
                break;
                
            case gpuFunctionVectorsDiv:
                _GPUFunctionPSO = [context createFunctionPiplineStateObjectWithName:@"vectors_div"];
                break;
                
            default:
                break;
        }
        if (!_GPUFunctionPSO) {
            NSString *exceptionName = [NSString stringWithFormat:@"%@\n", @"An error occured while creating pipeline object for function"];
            NSString *exceptionReason = [NSString stringWithFormat:@"%@\n", @"Error: Incorrect Function Code"];
            @throw [NSException exceptionWithName:exceptionName reason:exceptionReason userInfo:nil];
        }
        
        _GPUCommandBuffer = [context createNewCommandBuffer];
        _GPUCommandEncoder = [_GPUCommandBuffer computeCommandEncoder];
        _FunctionContext = context;
    }
    return self;
}


- (void) sendRandomDataToFunctionWithLength:(NSUInteger)length {
    if (length > MAX_FLOATS_PER_GPU_BUFFER) {
        NSString *exceptionName = [NSString stringWithFormat:@"%@\n", @"An error occured while loading data to GPU buffers"];
        NSString *exceptionReason = [NSString stringWithFormat:@"%@\n", @"Error: Volume of the data must be less than 402653184 floats"];
        @throw [NSException exceptionWithName:exceptionName reason:exceptionReason userInfo:nil];
    }
    
    _GPUDataFirstArgBuffer = [_FunctionContext createDataBufferWithLength: sizeof(float) * length ];
    _GPUDataSecondArgBuffer = [_FunctionContext createDataBufferWithLength: sizeof(float) * length ];
    _GPUDataResultBuffer = [_FunctionContext createDataBufferWithLength: sizeof(float) * length ];
    
    [RandomFloatDataGenerator fillMTLBufferWithRandomData: _GPUDataFirstArgBuffer withLength: length];
    [RandomFloatDataGenerator fillMTLBufferWithRandomData: _GPUDataSecondArgBuffer withLength: length];
}

- (void) executeAndWaitWithTimeVariable: (double*) timeVariablePointer {
    [self encodeCommand: _GPUCommandEncoder];
    
    [_GPUCommandEncoder endEncoding];
    
    CFTimeInterval start, end;
    
    start = CACurrentMediaTime();
    [_GPUCommandBuffer commit];
    [_GPUCommandBuffer waitUntilCompleted];
    end = CACurrentMediaTime();
    
    if (!timeVariablePointer) {
        NSString *exceprionName = [NSString stringWithFormat:@"An exception occured while time measurment"];
        NSString *exceprionInfo = [NSString stringWithFormat:@"Time pointer = %p", timeVariablePointer];
        @throw [NSException exceptionWithName:exceprionName reason:exceprionInfo userInfo:nil];
    }
    
    *timeVariablePointer = end - start;
}


- (void) encodeCommand: (id<MTLComputeCommandEncoder>) computeEncoder {
    [computeEncoder setComputePipelineState:_GPUFunctionPSO];
    [computeEncoder setBuffer:_GPUDataFirstArgBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:_GPUDataSecondArgBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_GPUDataResultBuffer offset:0 atIndex:2];
    
    
    unsigned long buffersLength = (unsigned int)(_GPUDataFirstArgBuffer.length / sizeof(float));
    MTLSize gridSize = MTLSizeMake(buffersLength, 1, 1);
    NSUInteger threadGroupSize = _GPUFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > buffersLength) {
        threadGroupSize = buffersLength;
    }
    
    MTLSize threadgroupsize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupsize];
}


+ (GPUCalculationFunction *)createFunctionWithName:(NSInteger)name inContext:(Context *)context {
    return [[GPUCalculationFunction alloc] initWithFunction:name inContext:context];
}


- (void) generateRandomFloatData: (id<MTLBuffer>) buffer withLength: (NSUInteger) dataLength {
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < dataLength; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}


+ (void) addRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time {
    GPUCalculationFunction* addFunction = [GPUCalculationFunction createFunctionWithName:gpuFunctionVectorsAdd inContext:context];
    [addFunction sendRandomDataToFunctionWithLength:length];
    [addFunction executeAndWaitWithTimeVariable: time];
}


+ (void) subRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time {
    GPUCalculationFunction* subFunction = [GPUCalculationFunction createFunctionWithName:gpuFunctionVectorsSub inContext:context];
    [subFunction sendRandomDataToFunctionWithLength:length];
    [subFunction executeAndWaitWithTimeVariable: time];
}


+ (void) mulRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time {
    GPUCalculationFunction* mulFunction = [GPUCalculationFunction createFunctionWithName:gpuFunctionVectorsMul inContext:context];
    [mulFunction sendRandomDataToFunctionWithLength:length];
    [mulFunction executeAndWaitWithTimeVariable: time];
}


+ (void) divRandomDataWithLength: (NSUInteger) length inContext: (Context*) context withTimeVariable: (double*) time {
    GPUCalculationFunction* divFunction = [GPUCalculationFunction createFunctionWithName:gpuFunctionVectorsDiv inContext:context];
    [divFunction sendRandomDataToFunctionWithLength:length];
    [divFunction executeAndWaitWithTimeVariable: time];
}

@end
