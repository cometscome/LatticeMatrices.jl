// Full-lattice HISQ Dslash benchmark against an unmodified SIMULATeQCD tree.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "simulateqcd.h"
#include "modules/dslash/dslash.h"
#include "modules/hisq/hisqSmearing.h"

// SIMULATeQCD declares this virtual base entry point but its derived HISQ
// implementation is the only one used.  Define it in this external driver so
// the explicit template vtables emitted by dslash.cpp have a complete base.
template <typename SpinorLHS, typename SpinorRHS>
void DSlash<SpinorLHS, SpinorRHS>::Dslash(
    SpinorLHS&, const SpinorRHS&, bool)
{
    throw std::runtime_error("abstract DSlash base entry point called");
}

namespace {

constexpr size_t halo_depth_gauge = 2;
// Three is sufficient for the farthest (Naik) hop and matches nw=3 in
// LatticeMatrices.  SIMULATeQCD requires a decomposed local extent > 2*halo.
constexpr size_t halo_depth_spinor = 3;
#ifdef HISQ_BENCH_SINGLE
using BenchmarkReal = float;
constexpr const char* benchmark_precision = "Float32";
#else
using BenchmarkReal = double;
constexpr const char* benchmark_precision = "Float64";
#endif
constexpr BenchmarkReal naik_epsilon = static_cast<BenchmarkReal>(-0.083);
using Dimensions = std::array<int, 4>;

Dimensions parse_dimensions(const char* name, const Dimensions& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    Dimensions result{};
    std::stringstream stream(value);
    std::string component;
    for (int direction = 0; direction < 4; ++direction) {
        if (!std::getline(stream, component, ',')) {
            throw std::runtime_error(std::string(name)
                + " must contain four comma-separated integers");
        }
        result[direction] = std::stoi(component);
    }
    if (std::getline(stream, component, ',')) {
        throw std::runtime_error(std::string(name)
            + " must contain four comma-separated integers");
    }
    return result;
}

int environment_integer(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::stoi(value);
}

template <size_t HaloDepth>
void fill_thin_links(Gaugefield<BenchmarkReal, true, HaloDepth, R18>& gauge,
                     CommunicationBase& comm)
{
    using GInd = GIndexer<All, HaloDepth>;
    Gaugefield<BenchmarkReal, false, HaloDepth, R18> host_gauge(comm);
    auto accessor = host_gauge.getAccessor();
    auto lattice = GInd::getLatData();

    for (int t = 0; t < static_cast<int>(lattice.lt); ++t) {
        for (int z = 0; z < static_cast<int>(lattice.lz); ++z) {
            for (int y = 0; y < static_cast<int>(lattice.ly); ++y) {
                for (int x = 0; x < static_cast<int>(lattice.lx); ++x) {
                    const LatticeDimensions local(x, y, z, t);
                    const LatticeDimensions global = lattice.globalPos(local);
                    const int coordinate = (global[0] + 1)
                        + 3 * (global[1] + 1)
                        + 5 * (global[2] + 1)
                        + 7 * (global[3] + 1);
                    const gSite site = GInd::getSite(x, y, z, t);
                    for (int mu = 0; mu < 4; ++mu) {
                        SU3<BenchmarkReal> link;
                        for (int row = 0; row < 3; ++row) {
                            for (int column = 0; column < 3; ++column) {
                                const BenchmarkReal deterministic_re =
                                    static_cast<BenchmarkReal>(0.013) * (
                                    2 * (row + 1) - (column + 1)
                                    + coordinate + 3 * (mu + 1));
                                const BenchmarkReal deterministic_im =
                                    static_cast<BenchmarkReal>(0.017) * (
                                    (row + 1) + 2 * (column + 1)
                                    - coordinate + (mu + 1));
                                link(row, column) = COMPLEX(BenchmarkReal)(
                                    static_cast<BenchmarkReal>(0.05)
                                        * deterministic_re
                                        + (row == column
                                            ? static_cast<BenchmarkReal>(1)
                                            : static_cast<BenchmarkReal>(0)),
                                    static_cast<BenchmarkReal>(0.05)
                                        * deterministic_im);
                            }
                        }
                        accessor.setLink(GInd::getSiteMu(site, mu), link);
                    }
                }
            }
        }
    }
    gauge = host_gauge;
    gauge.updateAll();
}

std::string dimensions_text(const Dimensions& dimensions, char separator)
{
    std::ostringstream output;
    output << dimensions[0] << separator << dimensions[1] << separator
           << dimensions[2] << separator << dimensions[3];
    return output.str();
}

}  // namespace

int main(int argc, char** argv)
{
    stdLogger.setVerbosity(WARN);
    CommunicationBase comm(&argc, &argv);
    const Dimensions global = parse_dimensions(
        "HISQ_BENCH_GLOBAL", {16, 16, 16, 16});
    const Dimensions grid = parse_dimensions(
        "HISQ_BENCH_GRID", {1, 1, 1, 1});
    const int repetitions = environment_integer("HISQ_BENCH_ITERS", 20);
    const int samples = environment_integer("HISQ_BENCH_SAMPLES", 7);
    const int warmups = environment_integer("HISQ_BENCH_WARMUPS", 4);
    if (repetitions <= 0 || samples <= 0 || warmups < 0) {
        throw std::runtime_error("invalid benchmark repetition count");
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (std::accumulate(grid.begin(), grid.end(), 1,
            std::multiplies<int>()) != world_size) {
        throw std::runtime_error("HISQ_BENCH_GRID does not match MPI ranks");
    }
    for (int direction = 0; direction < 4; ++direction) {
        if (global[direction] % grid[direction] != 0
            || global[direction] / grid[direction] < 4) {
            throw std::runtime_error(
                "global lattice is incompatible with process grid");
        }
    }

    LatticeParameters parameters;
    parameters.set(
        LatticeDimensions(global[0], global[1], global[2], global[3]),
        LatticeDimensions(grid[0], grid[1], grid[2], grid[3]));
    comm.init(parameters.nodeDim());
    initIndexer(halo_depth_gauge, parameters, comm);
    initIndexer(halo_depth_spinor, parameters, comm);

    using ThinGauge =
        Gaugefield<BenchmarkReal, true, halo_depth_gauge, R18>;
    using FatGauge =
        Gaugefield<BenchmarkReal, true, halo_depth_gauge, R18>;
    using NaikGauge =
        Gaugefield<BenchmarkReal, true, halo_depth_gauge, U3R14>;
    ThinGauge thin(comm);
    FatGauge fat(comm);
    NaikGauge naik(comm);
    fill_thin_links(thin, comm);
    HisqSmearing<BenchmarkReal, true, halo_depth_gauge,
        R18, R18, R18, U3R14> smearing(
            thin, fat, naik, static_cast<double>(naik_epsilon));
    // SIMULATeQCD stores staggered and antiperiodic boundary phases in links.
    smearing.SmearAll(0.0, true);

    using Spinor =
        SpinorfieldAll<BenchmarkReal, true, halo_depth_spinor, 1>;
    Spinor first(comm, "hisq_dirac_benchmark_first");
    Spinor second(comm, "hisq_dirac_benchmark_second");
    first.one();
    first.updateAll();
    Spinor* input = &first;
    Spinor* output = &second;

    HisqDSlash<BenchmarkReal, true, Even,
        halo_depth_gauge, halo_depth_spinor, 1> even_to_odd(
            fat, naik, 0.0, static_cast<double>(naik_epsilon));
    HisqDSlash<BenchmarkReal, true, Odd,
        halo_depth_gauge, halo_depth_spinor, 1> odd_to_even(
            fat, naik, 0.0, static_cast<double>(naik_epsilon));

    auto apply_full_lattice = [&]() {
        even_to_odd.Dslash(output->odd, input->even, true);
        odd_to_even.Dslash(output->even, input->odd, true);
        std::swap(input, output);
    };

    for (int warmup = 0; warmup < warmups; ++warmup) {
        apply_full_lattice();
    }
    gpuDeviceSynchronize();

    std::vector<double> timings;
    timings.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        const auto start = std::chrono::steady_clock::now();
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            apply_full_lattice();
        }
        gpuDeviceSynchronize();
        const auto stop = std::chrono::steady_clock::now();
        const double local_ms = std::chrono::duration<double, std::milli>(
            stop - start).count() / repetitions;
        double maximum_ms = 0.0;
        MPI_Allreduce(&local_ms, &maximum_ms, 1, MPI_DOUBLE, MPI_MAX,
            MPI_COMM_WORLD);
        timings.push_back(maximum_ms);
    }

    if (comm.IamRoot()) {
        std::vector<double> sorted = timings;
        std::sort(sorted.begin(), sorted.end());
        int device = -1;
        cudaGetDevice(&device);
        gpuDeviceProp properties{};
        gpuGetDeviceProperties(&properties, device);
        std::cout << std::setprecision(12)
                  << "RESULT operation=HISQDirac code=SIMULATeQCD"
                  << " backend=cuda ranks=" << world_size
                  << " precision=" << benchmark_precision
                  << " global=" << dimensions_text(global, 'x')
                  << " grid=" << dimensions_text(grid, 'x')
                  << " iterations=" << repetitions << " samples=" << samples
                  << " device=\"" << properties.name << "\""
                  << " min_ms=" << sorted.front()
                  << " median_ms=" << sorted[sorted.size() / 2]
                  << " max_ms=" << sorted.back() << " all_ms=[";
        for (size_t index = 0; index < timings.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << timings[index];
        }
        std::cout << "]\n";
    }
    return 0;
}

// Compile the upstream implementation in this translation unit so the base
// template definition above is visible to all explicit HISQ instantiations.
#include "modules/dslash/dslash.cpp"
