//
//  Context.h
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#ifndef Context_h
#define Context_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


@interface Context : NSObject

 @property (readonly) id <MTLDevice> GPUDevice;
 @property (readonly) id <MTLCommandQueue> GPUDeviceQueue;
 @property (readonly) id <MTLLibrary> GPUDeviceLibrary;
 
 - (id <MTLComputePipelineState>) createFunctionPiplineStateObjectWithName: (NSString*) functionName;
 
 - (id <MTLCommandBuffer>) createNewCommandBuffer;

 - (id <MTLBuffer>) createDataBufferWithLength: (NSUInteger) length;
 
 + (Context*) createGPUDefautContext;

@end


#endif /* Context_h */
