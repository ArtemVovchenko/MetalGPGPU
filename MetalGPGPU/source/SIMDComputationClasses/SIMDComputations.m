//
//  SIMDConputations.m
//  MetalGPGPU
//
//  Created by Artem Vovchenko on 27.11.2020.
//

#import "SIMDComputations.h"
@implementation SIMDComputations 

+ (void) multTwoVectorsWithSize: (unsigned long) length andTimePointer: (double*) time {
    float *aVector = [RandomFloatDataGenerator generateAlignedFloatDataWithLength:length];
    float *bVector = [RandomFloatDataGenerator generateAlignedFloatDataWithLength:length];
    float *res = _mm_malloc(sizeof(float) * length, 32);
    CFTimeInterval start, end;
    
    __m256 *aAsSimd = (__m256*) aVector;
    __m256 *bAsSimd = (__m256*) bVector;
    __m256 *resAsSimd = (__m256*) res;
    
    start = CACurrentMediaTime();
    for (unsigned long i = 0; i < length/8; ++i) {
        resAsSimd[i] = _mm256_mul_ps(aAsSimd[i], bAsSimd[i]);
    }
    end = CACurrentMediaTime();
    
    *time = end - start;
    
    [RandomFloatDataGenerator deallocateAlignedFloatDataFromPointer:aVector];
    [RandomFloatDataGenerator deallocateAlignedFloatDataFromPointer:bVector];
    _mm_free(res);
}


+ (void) dotProductOfTwoMatrixWithShape: (shape) aShape andShape: (shape) bShape withTimePointer: (double*) time {
    if (aShape.width != bShape.height) {
        
        NSString *exceptionName = [NSString stringWithFormat:
                                   @"Unable to multiply matrices with shpes %lux%lu and %lux%lu",
                                   (unsigned long)aShape.width, (unsigned long)aShape.height,
                                   (unsigned long)bShape.width, (unsigned long)bShape.height];
        @throw [NSException exceptionWithName:exceptionName reason:nil userInfo:nil];
    }
    unsigned long M = aShape.height;
    unsigned long N = bShape.width;
    unsigned long K = aShape.width;
    
    float *A = [RandomFloatDataGenerator generateAlignedFloatDataWithLength: aShape.height * aShape.width];
    float *B = [RandomFloatDataGenerator generateAlignedFloatDataWithLength: bShape.height * bShape.width];
    float *C = _mm_malloc(sizeof(float) * M * N, 32);
    CFTimeInterval start, end;
    
    start = CACurrentMediaTime();
    for (int i = 0; i < M; ++i) {
        float * c = C + i * N;
        
        for (int j = 0; j < N; j += 8) {
            _mm256_storeu_ps(c + j + 0, _mm256_setzero_ps());
        }
                
        for (int k = 0; k < K; ++k) {
            const float * b = B + k * N;
            __m256 a = _mm256_set1_ps(A[i*K + k]);
            for (int j = 0; j < N; j += 16) {
                _mm256_storeu_ps(c + j + 0, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 0), _mm256_loadu_ps(c + j + 0)));
                _mm256_storeu_ps(c + j + 8, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 8), _mm256_loadu_ps(c + j + 8)));
            }
        }
    }
    end = CACurrentMediaTime();
    
    *time = end - start;
    
    [RandomFloatDataGenerator deallocateAlignedFloatDataFromPointer:A];
    [RandomFloatDataGenerator deallocateAlignedFloatDataFromPointer:B];
    _mm_free(C);
    
}

+ (void) dotProductOfTwoMatrixAsGridWithShape: (shape) aShape andShape: (shape) bShape withTimePointer: (double*) time {
    if (aShape.width != bShape.height) {
        
        NSString *exceptionName = [NSString stringWithFormat:
                                   @"Unable to multiply matrices with shpes %lux%lu and %lux%lu",
                                   (unsigned long)aShape.width, (unsigned long)aShape.height,
                                   (unsigned long)bShape.width, (unsigned long)bShape.height];
        @throw [NSException exceptionWithName:exceptionName reason:nil userInfo:nil];
    }
    
    float **A = [RandomFloatDataGenerator generateMatrixDataWithShape:aShape];
    float **B = [RandomFloatDataGenerator generateMatrixDataWithShape:bShape];
    float **C = (float**)malloc(sizeof(float*) * aShape.width);
    for (unsigned long i = 0; i < aShape.width; ++i) {
        C[i] = _mm_malloc(sizeof(float) * bShape.height, 32);
    }
    
    unsigned long i, j, k;
    unsigned long N = aShape.width;
    
    CFTimeInterval start, end;
    
    start = CACurrentMediaTime();

    __m256i *c256 = (__m256i*) C[0];
    __m256i *b256 = (__m256i*) B[0];
    for (i = 0; i < N * N / 8; i++) {
        c256[i] = _mm256_setzero_si256();
    }

    for (i = 0; i < N; i++) {
    __m256i* c256i = c256 + i * N / 8;
        for (j = 0; j < N; j++) {
            __m256i r256 = _mm256_set1_epi32(A[i][j]);
            __m256i* b256i = b256 + j * N / 8;
            for (k = 0; k < N / 8; k++) {
                c256i[k] = _mm256_add_epi32(c256i[k], _mm256_mullo_epi32(r256, b256i[k]));
            }
        }
    }
    end = CACurrentMediaTime();
    
    *time = end - start;
    
    [RandomFloatDataGenerator deallocateMatrixFromPointer:A withShape:aShape];
    [RandomFloatDataGenerator deallocateMatrixFromPointer:B withShape:bShape];
//    for (unsigned long i = 0; i < aShape.width; ++i) {
//        _mm_free(C[i]);
//    }
    free(C);
}

@end
