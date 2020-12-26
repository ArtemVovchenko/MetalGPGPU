//
//  GPUMatrixOperations.m
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#import "GPUMatrixOperations.h"


const float alpha = 1.0;
const float beta = 0.0;

@implementation GPUMatrixOperations
{
    NSUInteger M;
    NSUInteger N;
    NSUInteger K;
}

- (instancetype) initRandomWithShapes: (shape) shapeA andShape: (shape) shapeB inContext: (Context*) context {
    self = [super init];
    if (self) {
        M = shapeA.width;
        N = shapeB.height;
        K = shapeA.height;
        
        _context = context;
        
        size_t ARowBytes = [MPSMatrixDescriptor rowBytesFromColumns:K dataType:MPSDataTypeFloat32];
        size_t BRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:N dataType:MPSDataTypeFloat32];
        size_t CRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:N dataType:MPSDataTypeFloat32];
        
        _AMatrixBuffer = [_context.GPUDevice newBufferWithLength: M * ARowBytes options:MTLResourceStorageModeShared ];
        _BMatrixBuffer = [_context.GPUDevice newBufferWithLength: K * BRowBytes options:MTLResourceStorageModeShared ];
        _CMatrixBuffer = [_context.GPUDevice newBufferWithLength: M * CRowBytes options:MTLResourceStorageModeShared ];
        
        
        [RandomFloatDataGenerator fillMTLBufferWithRandomData: _AMatrixBuffer withLength: M * K];
        [RandomFloatDataGenerator fillMTLBufferWithRandomData: _BMatrixBuffer withLength: N * K];
        
        
        MPSMatrixDescriptor* matrixDescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M columns:N rowBytes:ARowBytes dataType:MPSDataTypeFloat32];
        
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:_AMatrixBuffer descriptor:matrixDescriptor];
        matrixDescriptor.rows = K;
        matrixDescriptor.columns = N;
        matrixDescriptor.rowBytes = BRowBytes;
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:_BMatrixBuffer descriptor:matrixDescriptor];
        matrixDescriptor.rows = M;
        matrixDescriptor.rowBytes = CRowBytes;
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:_CMatrixBuffer descriptor:matrixDescriptor];
        
        _ExecutionKernel = [[MPSMatrixMultiplication alloc] initWithDevice:context.GPUDevice transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:beta];
        
        _CommandBuffer = [context.GPUDeviceQueue commandBuffer];
        
        [_ExecutionKernel encodeToCommandBuffer:_CommandBuffer leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    }
    return self;
}


- (void) executeAndWaitWithTimeMeasurement: (double*) timeVariablePointer {
    CFTimeInterval start, end;
    
    start = CACurrentMediaTime();
    [_CommandBuffer commit];
    [_CommandBuffer waitUntilCompleted];
    end = CACurrentMediaTime();
    
    if (!timeVariablePointer) {
        NSString *exceprionName = [NSString stringWithFormat:@"An exception occured while time measurment"];
        NSString *exceprionInfo = [NSString stringWithFormat:@"Time pointer = %p", timeVariablePointer];
        @throw [NSException exceptionWithName:exceprionName reason:exceprionInfo userInfo:nil];
    }
    *timeVariablePointer = end - start;
}



+ (void) createAndMultRandomMatriciesWithShape: (shape) shapeA andShape: (shape) shapeB inContext: (Context*) context withTimeVariable: (double*) time {
    GPUMatrixOperations* operation = [[GPUMatrixOperations alloc] initRandomWithShapes:shapeA andShape:shapeB inContext:context];
    [operation executeAndWaitWithTimeMeasurement:time];
}

@end
