#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <new>
#include <utility>

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

__host__ __device__ inline Ext ext_add(const Ext& a, const Ext& b) {
    return Ext{{add(a.c[0], b.c[0]), add(a.c[1], b.c[1]), add(a.c[2], b.c[2])}};
}

__host__ __device__ inline Ext ext_sub(const Ext& a, const Ext& b) {
    return Ext{{sub(a.c[0], b.c[0]), sub(a.c[1], b.c[1]), sub(a.c[2], b.c[2])}};
}

__host__ __device__ inline Ext ext_mul(const Ext& a, const Ext& b) {
    const std::uint64_t two = 2;
    return Ext{{add(mul(a.c[0], b.c[0]),
                   mul(two, add(mul(a.c[1], b.c[2]), mul(a.c[2], b.c[1])))),
        add(add(mul(a.c[0], b.c[1]), mul(a.c[1], b.c[0])),
            mul(two, mul(a.c[2], b.c[2]))),
        add(mul(a.c[0], b.c[2]),
            add(mul(a.c[1], b.c[1]), mul(a.c[2], b.c[0])))}};
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

Ext reference_ext_mul(const Ext& a, const Ext& b) {
    return Ext{{reference_add(reference_mul(a.c[0], b.c[0]),
                   reference_mul(2, reference_add(
                       reference_mul(a.c[1], b.c[2]), reference_mul(a.c[2], b.c[1])))),
        reference_add(
            reference_add(reference_mul(a.c[0], b.c[1]), reference_mul(a.c[1], b.c[0])),
            reference_mul(2, reference_mul(a.c[2], b.c[2]))),
        reference_add(reference_mul(a.c[0], b.c[2]),
            reference_add(reference_mul(a.c[1], b.c[1]), reference_mul(a.c[2], b.c[0])))}};
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

__global__ void dot_round_kernel(
    const Ext* a,
    const Ext* b,
    Ext* next_a,
    Ext* next_b,
    std::size_t count,
    Ext challenge,
    Ext* coefficients,
    bool advance) {
    __shared__ Ext zero_sums[256];
    __shared__ Ext one_sums[256];
    __shared__ Ext quadratic_sums[256];
    if (advance) {
        for (std::size_t index = 2 * threadIdx.x; index < count; index += 2 * blockDim.x) {
            next_a[index / 2] = fold_value(a[index], a[index + 1], challenge);
            next_b[index / 2] = fold_value(b[index], b[index + 1], challenge);
        }
        __syncthreads();
    }
    const Ext* values_a = advance ? next_a : a;
    const Ext* values_b = advance ? next_b : b;
    const std::size_t coefficient_count = advance ? count / 2 : count;
    const std::size_t thread = threadIdx.x;
    Ext zero{{0, 0, 0}};
    Ext one{{0, 0, 0}};
    Ext quadratic{{0, 0, 0}};
    for (std::size_t index = 2 * thread; index < coefficient_count; index += 2 * blockDim.x) {
        const Ext a0 = values_a[index];
        const Ext a1 = values_a[index + 1];
        const Ext b0 = values_b[index];
        const Ext b1 = values_b[index + 1];
        const Ext da = ext_sub(a1, a0);
        const Ext db = ext_sub(b1, b0);
        zero = ext_add(zero, ext_mul(a0, b0));
        one = ext_add(one, ext_mul(a1, b1));
        quadratic = ext_add(quadratic, ext_mul(da, db));
    }
    zero_sums[thread] = zero;
    one_sums[thread] = one;
    quadratic_sums[thread] = quadratic;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            zero_sums[thread] = ext_add(zero_sums[thread], zero_sums[thread + stride]);
            one_sums[thread] = ext_add(one_sums[thread], one_sums[thread + stride]);
            quadratic_sums[thread] = ext_add(
                quadratic_sums[thread], quadratic_sums[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        coefficients[0] = zero_sums[0];
        coefficients[1] = one_sums[0];
        coefficients[2] = quadratic_sums[0];
    }
}

} // namespace

struct FoldContext {
    Ext* device_input = nullptr;
    Ext* device_output = nullptr;
    Ext* host_input = nullptr;
    Ext* host_output = nullptr;
    std::size_t input_capacity = 0;
    std::size_t output_capacity = 0;
    std::size_t host_input_capacity = 0;
    std::size_t host_output_capacity = 0;
    cudaStream_t stream = nullptr;
    bool mapped = false;

    ~FoldContext() {
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFreeHost(host_input);
        cudaFreeHost(host_output);
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

cudaError_t reserve_pinned(Ext*& buffer, std::size_t& capacity, std::size_t count) {
    if (count <= capacity) {
        return cudaSuccess;
    }
    Ext* replacement = nullptr;
    const cudaError_t status = cudaHostAlloc(
        reinterpret_cast<void**>(&replacement), count * sizeof(Ext), cudaHostAllocMapped);
    if (status == cudaSuccess) {
        cudaFreeHost(buffer);
        buffer = replacement;
        capacity = count;
    }
    return status;
}

struct DotContext {
    Ext* a[2] = {nullptr, nullptr};
    Ext* b[2] = {nullptr, nullptr};
    Ext* coefficients = nullptr;
    std::size_t a_capacity[2] = {0, 0};
    std::size_t b_capacity[2] = {0, 0};
    std::size_t count = 0;
    std::size_t current = 0;
    cudaStream_t stream = nullptr;

    ~DotContext() {
        cudaFree(a[0]);
        cudaFree(a[1]);
        cudaFree(b[0]);
        cudaFree(b[1]);
        cudaFree(coefficients);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

extern "C" void* wp_dot_cuda_create(const Ext* a, const Ext* b, std::size_t count) {
    if (a == nullptr || b == nullptr || count == 0 || count % 2 != 0) {
        return nullptr;
    }
    DotContext* context = new (std::nothrow) DotContext();
    if (context == nullptr) {
        return nullptr;
    }
    context->count = count;
    cudaError_t status = cudaStreamCreate(&context->stream);
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&context->coefficients), 3 * sizeof(Ext));
    }
    if (status == cudaSuccess) {
        status = reserve(context->a[0], context->a_capacity[0], count);
    }
    if (status == cudaSuccess) {
        status = reserve(context->a[1], context->a_capacity[1], count / 2);
    }
    if (status == cudaSuccess) {
        status = reserve(context->b[0], context->b_capacity[0], count);
    }
    if (status == cudaSuccess) {
        status = reserve(context->b[1], context->b_capacity[1], count / 2);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            context->a[0], a, count * sizeof(Ext), cudaMemcpyHostToDevice, context->stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            context->b[0], b, count * sizeof(Ext), cudaMemcpyHostToDevice, context->stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context->stream);
    }
    if (status != cudaSuccess) {
        delete context;
        return nullptr;
    }
    return context;
}

extern "C" int wp_dot_cuda_coefficients(void* opaque, Ext* coefficients) {
    if (opaque == nullptr || coefficients == nullptr) {
        return -1;
    }
    DotContext& context = *static_cast<DotContext*>(opaque);
    const std::size_t next = 1 - context.current;
    dot_round_kernel<<<1, 256, 0, context.stream>>>(
        context.a[context.current],
        context.b[context.current],
        context.a[next],
        context.b[next],
        context.count,
        Ext{{0, 0, 0}},
        context.coefficients,
        false);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            coefficients,
            context.coefficients,
            3 * sizeof(Ext),
            cudaMemcpyDeviceToHost,
            context.stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context.stream);
    }
    return status;
}

extern "C" int wp_dot_cuda_round(void* opaque, Ext challenge, Ext* coefficients) {
    if (opaque == nullptr || coefficients == nullptr) {
        return -1;
    }
    DotContext& context = *static_cast<DotContext*>(opaque);
    const std::size_t next = 1 - context.current;
    dot_round_kernel<<<1, 256, 0, context.stream>>>(
        context.a[context.current],
        context.b[context.current],
        context.a[next],
        context.b[next],
        context.count,
        challenge,
        context.coefficients,
        true);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            coefficients,
            context.coefficients,
            3 * sizeof(Ext),
            cudaMemcpyDeviceToHost,
            context.stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context.stream);
    }
    if (status == cudaSuccess) {
        context.current = next;
        context.count /= 2;
    }
    return status;
}

extern "C" int wp_dot_cuda_download(void* opaque, Ext* a, Ext* b) {
    if (opaque == nullptr || a == nullptr || b == nullptr) {
        return -1;
    }
    DotContext& context = *static_cast<DotContext*>(opaque);
    cudaError_t status = cudaMemcpyAsync(
        a,
        context.a[context.current],
        context.count * sizeof(Ext),
        cudaMemcpyDeviceToHost,
        context.stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            b,
            context.b[context.current],
            context.count * sizeof(Ext),
            cudaMemcpyDeviceToHost,
            context.stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context.stream);
    }
    return status;
}

extern "C" void wp_dot_cuda_destroy(void* opaque) {
    if (opaque != nullptr) {
        DotContext* context = static_cast<DotContext*>(opaque);
        cudaStreamSynchronize(context->stream);
        delete context;
    }
}

extern "C" void wp_fold_cuda_set_mapped(bool mapped) {
    fold_context.mapped = mapped;
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
    cudaError_t status = cudaSuccess;
    if (context.mapped) {
        status = reserve_pinned(context.host_input, context.host_input_capacity, input_count);
        if (status != cudaSuccess) {
            return status;
        }
        status = reserve_pinned(context.host_output, context.host_output_capacity, output_count);
        if (status != cudaSuccess) {
            return status;
        }
        memcpy(context.host_input, input, input_count * sizeof(Ext));
    } else {
        status = reserve(context.device_input, context.input_capacity, input_count);
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
    }
    const int threads = 256;
    const int blocks = static_cast<int>((output_count + threads - 1) / threads);
    if (status == cudaSuccess) {
        if (context.mapped) {
            fold_kernel<<<blocks, threads, 0, context.stream>>>(
                context.host_input,
                context.host_output,
                output_count,
                low_half,
                challenge);
        } else {
            fold_kernel<<<blocks, threads, 0, context.stream>>>(
                context.device_input,
                context.device_output,
                output_count,
                low_half,
                challenge);
        }
        status = cudaGetLastError();
    }
    if (status == cudaSuccess && !context.mapped) {
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
    if (status == cudaSuccess && context.mapped) {
        memcpy(output, context.host_output, output_count * sizeof(Ext));
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

int run_dot_resident() {
    std::vector<Ext> a(1024);
    std::vector<Ext> b(1024);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i].c[0] = canonical(i * 0x9e3779b97f4a7c15ULL);
        a[i].c[1] = canonical(i * 0xd1b54a32d192ed03ULL);
        a[i].c[2] = canonical(i * 0x94d049bb133111ebULL);
        b[i].c[0] = canonical((i + 1) * 0xd1b54a32d192ed03ULL);
        b[i].c[1] = canonical((i + 1) * 0x94d049bb133111ebULL);
        b[i].c[2] = canonical((i + 1) * 0x9e3779b97f4a7c15ULL);
    }
    void* opaque = wp_dot_cuda_create(a.data(), b.data(), a.size());
    if (opaque == nullptr) {
        std::fprintf(stderr, "resident dot initialization failed\n");
        return 1;
    }
    const Ext challenges[2] = {{{0, 0, 0}}, {{17, 19, 23}}};
    {
        std::vector<Ext> uploaded_a(a.size());
        std::vector<Ext> uploaded_b(b.size());
        const int upload_status = wp_dot_cuda_download(opaque, uploaded_a.data(), uploaded_b.data());
        if (upload_status != 0 || uploaded_a != a || uploaded_b != b) {
            std::fprintf(stderr, "resident initial upload mismatch\n");
            wp_dot_cuda_destroy(opaque);
            return 1;
        }
    }
    for (const Ext& challenge : challenges) {
        std::vector<Ext> next_a(a.size() / 2);
        std::vector<Ext> next_b(b.size() / 2);
        for (std::size_t i = 0; i < a.size(); i += 2) {
            next_a[i / 2] = reference_fold(a[i], a[i + 1], challenge);
            next_b[i / 2] = reference_fold(b[i], b[i + 1], challenge);
        }
        Ext zero{{0, 0, 0}};
        Ext one{{0, 0, 0}};
        Ext quadratic{{0, 0, 0}};
        for (std::size_t i = 0; i < next_a.size(); i += 2) {
            const auto add_values = [](const Ext& x, const Ext& y) {
                return Ext{{reference_add(x.c[0], y.c[0]),
                    reference_add(x.c[1], y.c[1]), reference_add(x.c[2], y.c[2])}};
            };
            const Ext da{{reference_sub(next_a[i + 1].c[0], next_a[i].c[0]),
                reference_sub(next_a[i + 1].c[1], next_a[i].c[1]),
                reference_sub(next_a[i + 1].c[2], next_a[i].c[2])}};
            const Ext db{{reference_sub(next_b[i + 1].c[0], next_b[i].c[0]),
                reference_sub(next_b[i + 1].c[1], next_b[i].c[1]),
                reference_sub(next_b[i + 1].c[2], next_b[i].c[2])}};
            zero = add_values(zero, reference_ext_mul(next_a[i], next_b[i]));
            one = add_values(one, reference_ext_mul(next_a[i + 1], next_b[i + 1]));
            quadratic = add_values(quadratic, reference_ext_mul(da, db));
        }
        Ext actual[3];
        const int status = wp_dot_cuda_round(opaque, challenge, actual);
        std::vector<Ext> folded_a(next_a.size());
        std::vector<Ext> folded_b(next_b.size());
        const int download_status = wp_dot_cuda_download(opaque, folded_a.data(), folded_b.data());
        if (download_status != 0 || folded_a != next_a || folded_b != next_b) {
            std::fprintf(stderr, "resident fold mismatch\n");
            wp_dot_cuda_destroy(opaque);
            return 1;
        }
        if (status != 0 || !(actual[0] == zero) || !(actual[1] == one) || !(actual[2] == quadratic)) {
            std::fprintf(stderr,
                "resident coefficient mismatch (status %d): actual %llu %llu %llu expected %llu %llu %llu\n",
                status,
                static_cast<unsigned long long>(actual[0].c[0]),
                static_cast<unsigned long long>(actual[1].c[0]),
                static_cast<unsigned long long>(actual[2].c[0]),
                static_cast<unsigned long long>(zero.c[0]),
                static_cast<unsigned long long>(one.c[0]),
                static_cast<unsigned long long>(quadratic.c[0]));
            wp_dot_cuda_destroy(opaque);
            return 1;
        }
        a = std::move(next_a);
        b = std::move(next_b);
    }
    std::vector<Ext> actual_a(a.size());
    std::vector<Ext> actual_b(b.size());
    const int status = wp_dot_cuda_download(opaque, actual_a.data(), actual_b.data());
    wp_dot_cuda_destroy(opaque);
    if (status != 0 || actual_a != a || actual_b != b) {
        std::fprintf(stderr, "resident download mismatch\n");
        return 1;
    }
    std::printf("resident dot: two rounds and %zu final pairs match\n", a.size());
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
    for (int mapped = 0; mapped <= 1; ++mapped) {
        wp_fold_cuda_set_mapped(mapped != 0);
        std::printf("transfer mode: %s\n", mapped == 0 ? "copies" : "mapped");
        for (std::size_t stride = 1; stride <= 512; stride *= 2) {
            const int result = run_stride(input, challenge, stride);
            if (result != 0) {
                return result;
            }
        }
    }
    return run_dot_resident();
}

#endif // WP_FOLD_LIBRARY
