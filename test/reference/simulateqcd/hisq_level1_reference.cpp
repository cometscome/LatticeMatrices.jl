// External numerical oracle for test/hisq_smearing.jl.
//
// This program is compiled against an unmodified SIMULATeQCD tree. It fills
// the same deterministic 4x4x4x4 thin links as the Julia test, constructs the
// unprojected level-1 Fat7 field through HisqSmearing::SmearLvl1, and prints
// layout-independent fingerprints for each direction.

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "simulateqcd.h"
#include "modules/hisq/hisqSmearing.h"

namespace {

constexpr int nc = 3;
constexpr int nx = 4;
constexpr int ny = 4;
constexpr int nz = 4;
constexpr int nt = 4;

template <size_t HaloDepth>
void fill_thin_links(Gaugefield<double, true, HaloDepth, R18>& gauge,
                     CommunicationBase& comm)
{
    using GInd = GIndexer<All, HaloDepth>;
    Gaugefield<double, false, HaloDepth, R18> host_gauge(comm);
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
                        SU3<double> link;
                        for (int row = 0; row < nc; ++row) {
                            for (int column = 0; column < nc; ++column) {
                                const double re = 0.013 * (
                                    2 * (row + 1) - (column + 1)
                                    + coordinate + 3 * (mu + 1));
                                const double im = 0.017 * (
                                    (row + 1) + 2 * (column + 1)
                                    - coordinate + (mu + 1));
                                link(row, column) = COMPLEX(double)(re, im);
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

template <size_t HaloDepth>
std::array<double, 5> fingerprint_direction(
    const Gaugefield<double, false, HaloDepth, R18>& gauge, int mu)
{
    using GInd = GIndexer<All, HaloDepth>;
    const auto accessor = gauge.getAccessor();
    auto lattice = GInd::getLatData();
    std::array<double, 5> result{0.0, 0.0, 0.0, 0.0, 0.0};

    for (int t = 0; t < static_cast<int>(lattice.lt); ++t) {
        for (int z = 0; z < static_cast<int>(lattice.lz); ++z) {
            for (int y = 0; y < static_cast<int>(lattice.ly); ++y) {
                for (int x = 0; x < static_cast<int>(lattice.lx); ++x) {
                    const LatticeDimensions local(x, y, z, t);
                    const LatticeDimensions global = lattice.globalPos(local);
                    const gSite site = GInd::getSite(x, y, z, t);
                    const SU3<double> link =
                        accessor.getLink(GInd::getSiteMu(site, mu));

                    for (int column = 0; column < nc; ++column) {
                        for (int row = 0; row < nc; ++row) {
                            const auto value = link(row, column);
                            const double re = real(value);
                            const double im = imag(value);
                            const double weight = 1.0 + row + nc * (
                                column + nc * (
                                    global[0] + nx * (
                                        global[1] + ny * (
                                            global[2] + nz * global[3]))));
                            result[0] += re;
                            result[1] += im;
                            result[2] += weight * re;
                            result[3] += weight * im;
                            result[4] += re * re + im * im;
                        }
                    }
                }
            }
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    constexpr size_t halo_depth = 0;
    using Gauge = Gaugefield<double, true, halo_depth, R18>;

    stdLogger.setVerbosity(WARN);
    CommunicationBase comm(&argc, &argv);
    LatticeParameters parameters;
    parameters.set(
        LatticeDimensions(nx, ny, nz, nt),
        LatticeDimensions(1, 1, 1, 1));
    comm.init(parameters.nodeDim());
    initIndexer(halo_depth, parameters, comm);

    Gauge thin_links(comm);
    Gauge level1(comm);
    Gauge level2_unused(comm);
    Gauge naik_unused(comm);
    fill_thin_links(thin_links, comm);

    HisqSmearing<double, true, halo_depth, R18, R18, R18, R18>
        smearing(thin_links, level2_unused, naik_unused);
    smearing.SmearLvl1(level1);

    Gaugefield<double, false, halo_depth, R18> level1_host(comm);
    level1_host = level1;

    if (comm.IamRoot()) {
        std::cout << std::setprecision(17);
        for (int mu = 0; mu < 4; ++mu) {
            const auto fp = fingerprint_direction(level1_host, mu);
            std::cout << "level1_mu" << (mu + 1) << "=("
                      << fp[0] << ", " << fp[1] << ", "
                      << fp[2] << ", " << fp[3] << ", "
                      << fp[4] << ")\n";
        }
    }
    return 0;
}
