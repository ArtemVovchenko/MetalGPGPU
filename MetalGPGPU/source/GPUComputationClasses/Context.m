//
//  Context.m
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#import "Context.h"

@implementation Context

- (instancetype) init {
    self = [super init];
    if (self) {
        _GPUDevice = MTLCreateSystemDefaultDevice();
        _GPUDeviceQueue = [_GPUDevice newCommandQueue];
        _GPUDeviceLibrary = [_GPUDevice newDefaultLibrary];
    }
    return self;
}

- (id <MTLCommandBuffer>) createNewCommandBuffer {
    return [_GPUDeviceQueue commandBuffer];
}

- (id <MTLComputePipelineState>) createFunctionPiplineStateObjectWithName:(NSString *)functionName {
    NSError *gpuComputePipeLineStateError = nil;
    
    id <MTLFunction> requestedFunction = [_GPUDeviceLibrary newFunctionWithName:functionName];
    id <MTLComputePipelineState> functionPiplineState = [_GPUDevice newComputePipelineStateWithFunction:requestedFunction
                error:&gpuComputePipeLineStateError];
    
    if (gpuComputePipeLineStateError != nil) {
        NSString *exceptionName = [NSString stringWithFormat:@"%@ %@", @"An error occured while creating pipeline object for function", functionName];
        NSString *exceptionReason = [NSString stringWithFormat:@"%@: %@", @"Error", gpuComputePipeLineStateError.localizedDescription];
        @throw [NSException exceptionWithName:exceptionName reason:exceptionReason userInfo:nil];
    }
    
    return functionPiplineState;
}

- (id <MTLBuffer>) createDataBufferWithLength: (NSUInteger) length {
    return [_GPUDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
}

+ (Context*) createGPUDefautContext {
    return [[Context alloc] init];
}

@end
