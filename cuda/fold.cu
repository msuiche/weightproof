#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::uint64_t P = 0xffffffff00000001ULL;
constexpr std::uint64_t EPS = 0xffffffffULL;

struct Ext {
    std::uint64_t c[3];
};

__host__ __device__ inline bool operator==(const Ext& a, const Ext& b) {
    return a.c[0] == b.c[0] && a.c[1] == b.c[1] && a.c[2] == b.c[2];
}

__host__ __device__ inline std::uint64_t canonical(std::uint64_t value) {
    return value >= P ? value - P : value;
}

__host__ __device__ inline std::uint64_t reduce128(std::uint64_t hi, std::uint64_t lo) {
    const std::uint64_t high = hi >> 32;
    const std::uint64_t low = hi & EPS;
    std::uint64_t t = lo - high;
    t -= EPS & static_cast<std::uint64_t>(lo < high ? ~0ULL : 0ULL);
    const std::uint64_t m = (low << 32) - low;
    std::uint64_t r = t + m;
    r += EPS & static_cast<std::uint64_t>(r < m ? ~0ULL : 0ULL);
    return canonical(r);
}

__host__ __device__ inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t not_b = P - b;
    const std::uint64_t difference = a - not_b;
    return a < not_b ? difference - EPS : difference;
}

__host__ __device__ inline std::uint64_t sub(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t difference = a - b;
    return a < b ? difference - EPS : difference;
}

__host__ __device__ inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
#if defined(__CUDA_ARCH__)
    const std::uint64_t hi = __umul64hi(a, b);
#else
    const std::uint64_t hi = static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(a) * b) >> 64);
#endif
    return reduce128(hi, a * b);
}

__host__ __device__ inline Ext fold_value(
    const Ext& zero,
    const Ext& one,
    const Ext& challenge) {
    const std::uint64_t two = 2;
    const std::uint64_t d0 = sub(one.c[0], zero.c[0]);
    const std::uint64_t d1 = sub(one.c[1], zero.c[1]);
    const std::uint64_t d2 = sub(one.c[2], zero.c[2]);
    Ext out;
    const std::uint64_t product0 = add(
        mul(d0, challenge.c[0]),
        mul(two, add(mul(d1, challenge.c[2]), mul(d2, challenge.c[1]))));
    const std::uint64_t product1 = add(
        add(mul(d0, challenge.c[1]), mul(d1, challenge.c[0])),
        mul(two, mul(d2, challenge.c[2])));
    const std::uint64_t product2 = add(
        mul(d0, challenge.c[2]),
        add(mul(d1, challenge.c[1]), mul(d2, challenge.c[0])));
    out.c[0] = add(zero.c[0], product0);
    out.c[1] = add(zero.c[1], product1);
    out.c[2] = add(zero.c[2], product2);
    return out;
}

inline std::uint64_t reference_add(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) + b) % P);
}

inline std::uint64_t reference_sub(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) + P - b) % P);
}

inline std::uint64_t reference_mul(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % P);
}

Ext reference_fold(const Ext& zero, const Ext& one, const Ext& challenge) {
    const std::uint64_t d0 = reference_sub(one.c[0], zero.c[0]);
    const std::uint64_t d1 = reference_sub(one.c[1], zero.c[1]);
    const std::uint64_t d2 = reference_sub(one.c[2], zero.c[2]);
    const std::uint64_t product0 = reference_add(
        reference_mul(d0, challenge.c[0]),
        reference_mul(2, reference_add(
            reference_mul(d1, challenge.c[2]),
            reference_mul(d2, challenge.c[1]))));
    const std::uint64_t product1 = reference_add(
        reference_add(
            reference_mul(d0, challenge.c[1]),
            reference_mul(d1, challenge.c[0])),
        reference_mul(2, reference_mul(d2, challenge.c[2])));
    const std::uint64_t product2 = reference_add(
        reference_mul(d0, challenge.c[2]),
        reference_add(
            reference_mul(d1, challenge.c[1]),
            reference_mul(d2, challenge.c[0])));
    Ext out{{reference_add(zero.c[0], product0),
        reference_add(zero.c[1], product1),
        reference_add(zero.c[2], product2)}};
    return out;
}

__global__ void fold_kernel(
    const Ext* input,
    Ext* output,
    std::size_t output_count,
    std::size_t low_half,
    Ext challenge) {
    if (low_half == 0) {
        return;
    }
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) {
        return;
    }
    const std::size_t block = index / low_half;
    const std::size_t offset = index % low_half;
    const std::size_t zero = offset + 2 * low_half * block;
    const std::size_t one = zero + low_half;
    output[index] = fold_value(input[zero], input[one], challenge);
}

} // namespace

struct FoldContext {
    Ext* device_input = nullptr;
    Ext* device_output = nullptr;
    std::size_t input_capacity = 0;
    std::size_t output_capacity = 0;
    cudaStream_t stream = nullptr;

    ~FoldContext() {
        cudaFree(device_input);
        cudaFree(device_output);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

thread_local FoldContext fold_context;

cudaError_t reserve(Ext*& buffer, std::size_t& capacity, std::size_t count) {
    if (count <= capacity) {
        return cudaSuccess;
    }
    Ext* replacement = nullptr;
    const cudaError_t status = cudaMalloc(&replacement, count * sizeof(Ext));
    if (status == cudaSuccess) {
        cudaFree(buffer);
        buffer = replacement;
        capacity = count;
    }
    return status;
}

extern "C" int wp_fold_cuda(
    const Ext* input,
    std::size_t input_count,
    Ext* output,
    std::size_t output_count,
    std::size_t low_half,
    Ext challenge) {
    if (input == nullptr
        || output == nullptr
        || input_count == 0
        || input_count % 2 != 0
        || output_count != input_count / 2
        || low_half == 0
        || (low_half & (low_half - 1)) != 0
        || low_half > output_count
        || input_count % (2 * low_half) != 0) {
        return -1;
    }
    FoldContext& context = fold_context;
    if (context.stream == nullptr) {
        const cudaError_t status = cudaStreamCreate(&context.stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    cudaError_t status = reserve(context.device_input, context.input_capacity, input_count);
    if (status != cudaSuccess) {
        return status;
    }
    status = reserve(context.device_output, context.output_capacity, output_count);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMemcpyAsync(
        context.device_input,
        input,
        input_count * sizeof(Ext),
        cudaMemcpyHostToDevice,
        context.stream);
    if (status == cudaSuccess) {
        const int threads = 256;
        const int blocks = static_cast<int>((output_count + threads - 1) / threads);
        fold_kernel<<<blocks, threads, 0, context.stream>>>(
            context.device_input,
            context.device_output,
            output_count,
            low_half,
            challenge);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            output,
            context.device_output,
            output_count * sizeof(Ext),
            cudaMemcpyDeviceToHost,
            context.stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context.stream);
    }
    return status;
}

#ifndef WP_FOLD_LIBRARY

namespace {

int run_stride(
    const std::vector<Ext>& input,
    const Ext& challenge,
    std::size_t low_half) {
    if (input.empty()
        || input.size() % 2 != 0
        || low_half == 0
        || (low_half & (low_half - 1)) != 0
        || low_half > input.size() / 2
        || input.size() % (2 * low_half) != 0) {
        std::fprintf(stderr, "invalid fold input or stride\n");
        return 1;
    }
    const std::size_t output_count = input.size() / 2;
    std::vector<Ext> expected(output_count);
    for (std::size_t block = 0; block < input.size() / (2 * low_half); ++block) {
        for (std::size_t offset = 0; offset < low_half; ++offset) {
            const std::size_t zero = offset + 2 * low_half * block;
            const std::size_t one = zero + low_half;
            expected[offset + low_half * block] = reference_fold(input[zero], input[one], challenge);
        }
    }

    std::vector<Ext> actual(output_count);
    const int status = wp_fold_cuda(
        input.data(), input.size(), actual.data(), actual.size(), low_half, challenge);
    if (status != 0) {
        std::fprintf(stderr, "persistent CUDA fold failed at stride %zu: %d\n", low_half, status);
        return 1;
    }
    if (actual != expected) {
        std::fprintf(stderr, "mismatch at stride %zu\n", low_half);
        return 1;
    }
    bool nonzero = false;
    for (const Ext& value : actual) {
        nonzero |= value.c[0] != 0 || value.c[1] != 0 || value.c[2] != 0;
    }
    if (!nonzero) {
        std::fprintf(stderr, "vacuous output at stride %zu\n", low_half);
        return 1;
    }
    std::printf("stride %zu: %zu outputs match\n", low_half, output_count);
    return 0;
}

} // namespace

int main() {
    std::vector<Ext> input(1024);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i].c[0] = canonical(i * 0x9e3779b97f4a7c15ULL);
        input[i].c[1] = canonical(i * 0xd1b54a32d192ed03ULL);
        input[i].c[2] = canonical(i * 0x94d049bb133111ebULL);
    }
    const Ext challenge{{7, 11, 13}};
    for (std::size_t stride = 1; stride <= 512; stride *= 2) {
        const int result = run_stride(input, challenge, stride);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

#endif // WP_FOLD_LIBRARY
