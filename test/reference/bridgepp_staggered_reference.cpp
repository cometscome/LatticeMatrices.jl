// Numerical oracle for test/staggered_dirac.jl.
//
// Compile this file against an unmodified Bridge++ 2.1.x build.  It uses the
// same deterministic fields as the Julia test and prints compact fingerprints
// of Fopr_Staggered::D and Fopr_Staggered::Ddag.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "bridge_setup.h"
#include "Field/field_F_1spinor.h"
#include "Field/field_G.h"
#include "Field/index_lex.h"
#include "Fopr/fopr_Staggered.h"

namespace {

void fill_gauge(Field_G& gauge)
{
  const int nc = CommonParameters::Nc();
  Index_lex index;
  for (int t = 0; t < CommonParameters::Nt(); ++t) {
    for (int z = 0; z < CommonParameters::Nz(); ++z) {
      for (int y = 0; y < CommonParameters::Ny(); ++y) {
        for (int x = 0; x < CommonParameters::Nx(); ++x) {
          const int site = index.site(x, y, z, t);
          const int coordinate = (x + 1) + 3 * (y + 1)
                               + 5 * (z + 1) + 7 * (t + 1);
          for (int mu = 0; mu < 4; ++mu) {
            for (int row = 0; row < nc; ++row) {
              for (int col = 0; col < nc; ++col) {
                const double re = 0.013 * (2 * (row + 1) - (col + 1)
                                         + coordinate + 3 * (mu + 1));
                const double im = 0.017 * ((row + 1) + 2 * (col + 1)
                                         - coordinate + (mu + 1));
                gauge.set_ri(row * nc + col, site, mu, re, im);
              }
            }
          }
        }
      }
    }
  }
}

void fill_fermion(Field_F_1spinor& psi)
{
  const int nc = CommonParameters::Nc();
  Index_lex index;
  for (int t = 0; t < CommonParameters::Nt(); ++t) {
    for (int z = 0; z < CommonParameters::Nz(); ++z) {
      for (int y = 0; y < CommonParameters::Ny(); ++y) {
        for (int x = 0; x < CommonParameters::Nx(); ++x) {
          const int site = index.site(x, y, z, t);
          const int coordinate = (x + 1) + 2 * (y + 1)
                               + 4 * (z + 1) + 8 * (t + 1);
          for (int color = 0; color < nc; ++color) {
            const double re = 0.019 * ((color + 1) + coordinate);
            const double im = 0.023 * (2 * (color + 1) - coordinate);
            psi.set_ri(color, site, 0, re, im);
          }
        }
      }
    }
  }
}

void print_fingerprint(const std::string& label,
                       const Field_F_1spinor& field)
{
  const int nc = CommonParameters::Nc();
  double sum_re = 0.0;
  double sum_im = 0.0;
  double weighted_re = 0.0;
  double weighted_im = 0.0;
  double norm2 = 0.0;
  for (int site = 0; site < CommonParameters::Nvol(); ++site) {
    for (int color = 0; color < nc; ++color) {
      const double re = field.cmp_r(color, site);
      const double im = field.cmp_i(color, site);
      const double weight = 1.0 + color + nc * site;
      sum_re += re;
      sum_im += im;
      weighted_re += weight * re;
      weighted_im += weight * im;
      norm2 += re * re + im * im;
    }
  }

  std::cout << std::setprecision(17)
            << label << "_sum_re=" << sum_re << '\n'
            << label << "_sum_im=" << sum_im << '\n'
            << label << "_weighted_re=" << weighted_re << '\n'
            << label << "_weighted_im=" << weighted_im << '\n'
            << label << "_norm2=" << norm2 << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
  const std::vector<int> lattice_size{4, 2, 2, 2};
  const std::vector<int> grid_size{1, 1, 1, 1};
  bridge_initialize(&argc, &argv);
  bridge_setup(lattice_size, grid_size, 1, 3, "stdout", "Crucial");

  Field_G gauge(CommonParameters::Nvol(), 4);
  Field_F_1spinor psi;
  Field_F_1spinor result;
  fill_gauge(gauge);
  fill_fermion(psi);

  Parameters staggered_parameters;
  staggered_parameters.set_double("quark_mass", 0.17);
  staggered_parameters.set_int_vector(
      "boundary_condition", std::vector<int>{1, 1, 1, -1});
  staggered_parameters.set_string("verbose_level", "Crucial");
  Fopr_Staggered staggered(staggered_parameters);
  staggered.set_config(&gauge);

  staggered.set_mode("D");
  staggered.mult(result, psi);
  print_fingerprint("D", result);

  staggered.set_mode("Ddag");
  staggered.mult(result, psi);
  print_fingerprint("Ddag", result);

  bridge_finalize();
  return 0;
}
