#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace asio = boost::asio;
using asio::ip::tcp;

constexpr int CHALLENGE_TIMEOUT = 60;
constexpr size_t MAX_POST_SIZE = 4 * 1024;
constexpr size_t PREFIX_RAW_LEN = 32;
constexpr size_t PREFIX_LEN = 44;
constexpr size_t CONSTANT_BUFFER_SIZE = MAX_POST_SIZE + PREFIX_LEN + 64;
constexpr int ATTEMPTS_PER_THREAD = 128;

bool ENABLE_LOGGING = false;
bool ENABLE_PROFILING_LOGGING = false;
#define LOG                                                                                                            \
    if (ENABLE_LOGGING)                                                                                                \
    std::cerr

__constant__ uint32_t k_const[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__constant__ char B64_CHARS_DEVICE[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
__constant__ uint8_t FIXED_DATA_CONST[CONSTANT_BUFFER_SIZE];

// Global start time for the program
std::chrono::steady_clock::time_point global_start;

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return __funnelshift_r(x, x, n); }

__device__ void sha256_transform(const uint8_t *data, uint32_t *s) {
    uint32_t w[64];
#pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = (data[4 * i] << 24) | (data[4 * i + 1] << 16) | (data[4 * i + 2] << 8) | data[4 * i + 3];
#pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t a = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t b = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + a + w[i - 7] + b;
    }
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t tmp1 = h + S1 + ch + k_const[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + tmp1;
        d = c;
        c = b;
        b = a;
        a = tmp1 + tmp2;
    }
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
}

__device__ void base64_encode_device(const uint8_t *in, uint8_t *out) {
    int j = 0;
#pragma unroll
    for (int i = 0; i < 30; i += 3) {
        uint32_t triple = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out[j++] = B64_CHARS_DEVICE[(triple >> 18) & 0x3F];
        out[j++] = B64_CHARS_DEVICE[(triple >> 12) & 0x3F];
        out[j++] = B64_CHARS_DEVICE[(triple >> 6) & 0x3F];
        out[j++] = B64_CHARS_DEVICE[triple & 0x3F];
    }
    out[j++] = B64_CHARS_DEVICE[in[30] >> 2];
    out[j++] = B64_CHARS_DEVICE[((in[30] & 3) << 4) | (in[31] >> 4)];
    out[j++] = B64_CHARS_DEVICE[(in[31] & 0xF) << 2];
    out[j++] = '=';
}

__device__ uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

__global__ void pow_kernel(int zeros, volatile int *flag, uint8_t *out_prefix, uint64_t seed_base, size_t fixed_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t state = seed_base * 6364136223846793005ULL + idx;

    for (int iter = 0; iter < ATTEMPTS_PER_THREAD; ++iter) {
        if (*flag)
            return;
        uint64_t r[4];
        for (int i = 0; i < 4; i++)
            r[i] = xorshift64star(&state);
        uint8_t raw[32];
        memcpy(raw, r, 32);
        uint8_t prefix[PREFIX_LEN];
        base64_encode_device(raw, prefix);
        uint32_t s[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        size_t total = PREFIX_LEN + fixed_len;
        size_t pad_len = (total + 9 + 63) / 64 * 64;
        uint64_t bit_len = total * 8;
        uint8_t block[64];
        for (size_t off = 0; off < pad_len; off += 64) {
#pragma unroll
            for (int j = 0; j < 64; ++j) {
                size_t p = off + j;
                if (p < PREFIX_LEN)
                    block[j] = prefix[p];
                else if (p < total)
                    block[j] = FIXED_DATA_CONST[p - PREFIX_LEN];
                else if (p == total)
                    block[j] = 0x80;
                else if (p >= pad_len - 8)
                    block[j] = (bit_len >> ((pad_len - 1 - p) * 8)) & 0xFF;
                else
                    block[j] = 0;
            }
            sha256_transform(block, s);
        }
        uint8_t hash[32];
        for (int i = 0; i < 8; i++) {
            hash[4 * i] = (s[i] >> 24) & 0xFF;
            hash[4 * i + 1] = (s[i] >> 16) & 0xFF;
            hash[4 * i + 2] = (s[i] >> 8) & 0xFF;
            hash[4 * i + 3] = s[i] & 0xFF;
        }
        int lz = 0;
        for (int i = 0; i < 32; i++) {
            uint8_t hi = hash[i] >> 4, lo = hash[i] & 0xF;
            if (hi == 0) {
                lz++;
                if (lo == 0)
                    lz++;
                else
                    break;
            } else
                break;
        }
        if (lz >= zeros) {
            if (atomicCAS((int *) flag, 0, 1) == 0) {
                memcpy(out_prefix, prefix, PREFIX_LEN);
                return;
            }
        }
    }
}

class CudaMem {
public:
    CudaMem() = default;

    explicit CudaMem(size_t size) { cudaMalloc(&ptr_, size); }
    ~CudaMem() {
        if (ptr_)
            cudaFree(ptr_);
    }
    void *get() const { return ptr_; }

    CudaMem(const CudaMem &) = delete;

    CudaMem &operator=(const CudaMem &) = delete;

    CudaMem(CudaMem &&o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }

    CudaMem &operator=(CudaMem &&o) noexcept {
        if (ptr_)
            cudaFree(ptr_);
        ptr_ = o.ptr_;
        o.ptr_ = nullptr;
        return *this;
    }

private:
    void *ptr_ = nullptr;
};

#define CUDA_CHECK(fn)                                                                                                 \
    do {                                                                                                               \
        cudaError_t err = (fn);                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) +       \
                                     " '" + cudaGetErrorName(err) + "' (" + cudaGetErrorString(err) + ")");            \
        }                                                                                                              \
    } while (0)

std::string perform_pow_cuda(const std::string &fixed_data, int zeros) {
    static bool initialized = false;
    if (!initialized) {
        auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpyToSymbol(B64_CHARS_DEVICE,
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", 65));
        if (ENABLE_PROFILING_LOGGING) {
            auto t1 = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - global_start).count();
            std::cerr << "[" << elapsed << "ms] Initialized GPU constant data (base64 table)\n";
        }
        initialized = true;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(FIXED_DATA_CONST, fixed_data.data(), fixed_data.size()));
    if (ENABLE_PROFILING_LOGGING) {
        auto t_copy1 = std::chrono::steady_clock::now();
        auto elapsed_copy = std::chrono::duration_cast<std::chrono::milliseconds>(t_copy1 - global_start).count();
        std::cerr << "[" << elapsed_copy << "ms] Copied fixed data to GPU (" << fixed_data.size() << " bytes)\n";
    }
    CudaMem d_flag(sizeof(int));
    CudaMem d_out(PREFIX_LEN);
    int flag_zero = 0;
    CUDA_CHECK(cudaMemcpy(d_flag.get(), &flag_zero, sizeof(int), cudaMemcpyHostToDevice));

    std::mt19937_64 gen(std::random_device{}());
    int thr = 512;
    int blk = 224;
    uint64_t total_hashes = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (true) {
        uint64_t seed_base = gen();
        pow_kernel<<<blk, thr>>>(zeros, reinterpret_cast<int *>(d_flag.get()), reinterpret_cast<uint8_t *>(d_out.get()),
                                 seed_base, fixed_data.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        total_hashes += static_cast<uint64_t>(blk) * thr * ATTEMPTS_PER_THREAD;

        int found;
        CUDA_CHECK(cudaMemcpy(&found, d_flag.get(), sizeof(int), cudaMemcpyDeviceToHost));
        if (found)
            break;

        CUDA_CHECK(cudaMemset(d_flag.get(), 0, sizeof(int)));
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - t0).count() >= CHALLENGE_TIMEOUT)
            throw std::runtime_error("POW timeout");
    }

    auto t1 = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(t1 - t0).count();
    double hashrate = total_hashes / time_sec;
    LOG << "Total hashes: " << total_hashes << "\n";
    LOG << "Time: " << time_sec * 1000 << "ms\n";
    LOG << "Hashrate: " << hashrate / 1e6 << " MH/s\n";

    uint8_t prefix[PREFIX_LEN];
    CUDA_CHECK(cudaMemcpy(prefix, d_out.get(), PREFIX_LEN, cudaMemcpyDeviceToHost));
    return std::string(reinterpret_cast<char *>(prefix), PREFIX_LEN);
}

class BeamClient {
public:
    BeamClient(asio::io_context &io_context, const std::string &host, int port) :
        resolver_(io_context), socket_(io_context) {
        auto endpoints = resolver_.resolve(host, std::to_string(port));
        asio::connect(socket_, endpoints);
        if (ENABLE_PROFILING_LOGGING) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - global_start).count();
            std::cerr << "[" << elapsed << "ms] Connection established\n";
        }
    }

    std::string read_line() {
        asio::read_until(socket_, buffer_, "\r\n");
        std::istream is(&buffer_);
        std::string line;
        std::getline(is, line);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        LOG << "RECV: " << line << "\n";
        return line;
    }

    void write_line(const std::string &data) {
        std::string line = data + "\r\n";
        LOG << "SEND: " << data << "\n";
        asio::write(socket_, asio::buffer(line));
    }

    void post_file(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open file");
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        if (size > MAX_POST_SIZE)
            throw std::runtime_error("File too large");
        file.seekg(0);
        std::string content(size, '\0');
        file.read(content.data(), size);

        std::string sent;
        if (read_line() != "HI")
            throw std::runtime_error("Bad greeting");
        write_line("POST");
        if (read_line() != "OK")
            throw std::runtime_error("POST rejected");
        std::istringstream ss(content);
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(ss, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            sent += line + "\r\n";
            lines.push_back(line);
        }
        for (const auto &l: lines) {
            write_line(l);
        }
        for (size_t i = 0; i < lines.size(); i++) {
            if (read_line() != "OK") [[unlikely]]
                throw std::runtime_error("POST rejected");
        }
        if (ENABLE_PROFILING_LOGGING) {
            auto t_lines_sent = std::chrono::steady_clock::now();
            auto elapsed_lines =
                    std::chrono::duration_cast<std::chrono::milliseconds>(t_lines_sent - global_start).count();
            std::cerr << "[" << elapsed_lines << "ms] All lines sent\n";
        }
        write_line("SUBMIT");
        auto challenge = read_line();
        if (ENABLE_PROFILING_LOGGING) {
            auto t_challenge = std::chrono::steady_clock::now();
            auto elapsed_challenge =
                    std::chrono::duration_cast<std::chrono::milliseconds>(t_challenge - global_start).count();
            std::cerr << "[" << elapsed_challenge << "ms] Challenge received\n";
        }
        std::vector<std::string> parts;
        boost::split(parts, challenge, boost::is_any_of(" "));
        if (parts.size() < 4 || parts[0] != "CHALLENGE" || parts[1] != "sha256")
            throw std::runtime_error("Invalid challenge");
        int zeros = std::stoi(parts[2]);
        std::string token = parts[3];
        std::string fixed_data = "\r\n" + sent + token + "\r\n";
        LOG << "Fixed size=" << fixed_data.size() << " bytes\n";
        auto prefix = perform_pow_cuda(fixed_data, zeros);
        if (ENABLE_PROFILING_LOGGING) {
            auto t_solution = std::chrono::steady_clock::now();
            auto elapsed_solution =
                    std::chrono::duration_cast<std::chrono::milliseconds>(t_solution - global_start).count();
            std::cerr << "[" << elapsed_solution << "ms] Solution found: " << prefix << "\n";
        }
        write_line("ACCEPTED " + prefix);
        auto result = read_line();
        if (ENABLE_PROFILING_LOGGING) {
            auto t_result = std::chrono::steady_clock::now();
            auto elapsed_result =
                    std::chrono::duration_cast<std::chrono::milliseconds>(t_result - global_start).count();
            std::cerr << "[" << elapsed_result << "ms] Result: " << result << "\n";
        }
        if (result.rfind("SENT", 0) == 0)
            std::cout << "Posted " << result.substr(5) << "\n";
        else
            throw std::runtime_error("Post failed");
    }

    void get_file(const std::string &id) {
        if (read_line() != "HI")
            throw std::runtime_error("Bad greeting");
        write_line("GET " + id);

        boost::system::error_code ec;
        std::array<char, 4096> buf;
        while (true) {
            std::size_t len = socket_.read_some(asio::buffer(buf), ec);
            if (ec == asio::error::eof) {
                break;
            }
            if (ec) {
                throw boost::system::system_error(ec);
            }
            std::cout.write(buf.data(), len);
        }
    }

private:
    tcp::resolver resolver_;
    tcp::socket socket_;
    asio::streambuf buffer_;
};

int main(int argc, char *argv[]) {
    global_start = std::chrono::steady_clock::now();
    if (ENABLE_PROFILING_LOGGING) {
        auto t_init = std::chrono::steady_clock::now();
        auto elapsed_init = std::chrono::duration_cast<std::chrono::milliseconds>(t_init - global_start).count();
        std::cerr << "[" << elapsed_init << "ms] Program started\n";
    }
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <post|get> <file_or_id> [--log] [--profile]\n";
        return 1;
    }

    bool log = false;
    std::string mode;
    std::string target;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--log") {
            log = true;
        } else if (arg == "--profile") {
            ENABLE_PROFILING_LOGGING = true;
        } else if (mode.empty()) {
            mode = arg;
        } else if (target.empty()) {
            target = arg;
        }
    }

    ENABLE_LOGGING = log;

    if (mode != "post" && mode != "get") {
        std::cerr << "Invalid mode '" << mode << "', expected 'post' or 'get'\n";
        return 1;
    }
    try {
        asio::io_context ctx;
        BeamClient client(ctx, "45.146.253.5", 6969);
        if (mode == "post") {
            client.post_file(target);
        } else {
            client.get_file(target);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
