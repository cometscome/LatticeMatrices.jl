// External numerical oracle for test/hisq_full_smearing.jl.
//
// This program is compiled against an unmodified SIMULATeQCD tree.  It uses
// the same well-conditioned deterministic 4^4 thin links as the Julia test,
// runs the complete unphased HISQ smearing, and prints fingerprints for the
// corrected fat and forward-anchored Naik links.

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
constexpr double naik_epsilon = -0.083;

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
                                const double deterministic_re = 0.013 * (
                                    2 * (row + 1) - (column + 1)
                                    + coordinate + 3 * (mu + 1));
                                const double deterministic_im = 0.017 * (
                                    (row + 1) + 2 * (column + 1)
                                    - coordinate + (mu + 1));
                                const double re = 0.05 * deterministic_re
                                    + (row == column ? 1.0 : 0.0);
                                const double im = 0.05 * deterministic_im;
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
    const Gaugefield<double, false, HaloDepth, R18>& gauge, int mu,
    bool shift_naik_to_forward_anchor)
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
                    const gSite origin = GInd::getSite(x, y, z, t);
                    const gSite stored_site = shift_naik_to_forward_anchor
                        ? GInd::site_up(origin, mu) : origin;
                    const SU3<double> link = accessor.getLink(
                        GInd::getSiteMu(stored_site, mu));

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

void print_fingerprint(
    const char* label, int mu, const std::array<double, 5>& fingerprint)
{
    std::cout << label << "_mu" << (mu + 1) << "=("
              << fingerprint[0] << ", " << fingerprint[1] << ", "
              << fingerprint[2] << ", " << fingerprint[3] << ", "
              << fingerprint[4] << ")\n";
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
    Gauge level2(comm);
    Gauge naik(comm);
    fill_thin_links(thin_links, comm);

    HisqSmearing<double, true, halo_depth, R18, R18, R18, R18>
        smearing(thin_links, level2, naik, naik_epsilon);
    smearing.SmearAll(0.0, false);

    Gaugefield<double, false, halo_depth, R18> level2_host(comm);
    Gaugefield<double, false, halo_depth, R18> naik_host(comm);
    level2_host = level2;
    naik_host = naik;

    if (comm.IamRoot()) {
        std::cout << std::setprecision(17);
        for (int mu = 0; mu < 4; ++mu) {
            print_fingerprint(
                "level2", mu,
                fingerprint_direction(level2_host, mu, false));
            // SIMULATeQCD stores the centered product at x+mu.  Read that
            // site so the reported value is anchored at Julia's x.
            print_fingerprint(
                "naik_forward", mu,
                fingerprint_direction(naik_host, mu, true));
        }
    }
    return 0;
}
