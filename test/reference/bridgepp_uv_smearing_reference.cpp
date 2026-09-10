#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "bridge_setup.h"
#include "Field/field_G.h"
#include "Field/index_lex.h"
#include "Smear/projection_Maximum_SU_N.h"
#include "Smear/projection_Stout_SU3.h"
#include "Smear/smear_APE.h"
#include "Smear/smear_HYP.h"

namespace {

void fill_raw(Field_G& gauge)
{
  const int nc = CommonParameters::Nc();
  Index_lex index;
  for (int t = 0; t < CommonParameters::Nt(); ++t) {
    for (int z = 0; z < CommonParameters::Nz(); ++z) {
      for (int y = 0; y < CommonParameters::Ny(); ++y) {
        for (int x = 0; x < CommonParameters::Nx(); ++x) {
          const int site = index.site(x, y, z, t);
          const int coordinate = x + 3 * y + 5 * z + 7 * t;
          for (int mu = 0; mu < 4; ++mu) {
            for (int row = 0; row < nc; ++row) {
              for (int col = 0; col < nc; ++col) {
                const double re = (row == col ? 1.0 : 0.0)
                  + 0.013 * (2 * (row + 1) - (col + 1)
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

void write_stage(const std::string& stage, const Field_G& gauge)
{
  const int nc = CommonParameters::Nc();
  Index_lex index;
  for (int t = 0; t < CommonParameters::Nt(); ++t) {
    for (int z = 0; z < CommonParameters::Nz(); ++z) {
      for (int y = 0; y < CommonParameters::Ny(); ++y) {
        for (int x = 0; x < CommonParameters::Nx(); ++x) {
          const int site = index.site(x, y, z, t);
          for (int mu = 0; mu < 4; ++mu) {
            for (int row = 0; row < nc; ++row) {
              for (int col = 0; col < nc; ++col) {
                const int component = row * nc + col;
                std::cout << stage << '\t' << mu + 1 << '\t'
                          << x << '\t' << y << '\t' << z << '\t' << t << '\t'
                          << row + 1 << '\t' << col + 1 << '\t'
                          << gauge.cmp_r(component, site, mu) << '\t'
                          << gauge.cmp_i(component, site, mu) << '\n';
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  const std::vector<int> lattice_size{4, 4, 4, 4};
  const std::vector<int> grid_size{1, 1, 1, 1};
  bridge_initialize(&argc, &argv);
  bridge_setup(lattice_size, grid_size, 1, 3, "stderr", "Crucial");

  Parameters projection_parameters;
  projection_parameters.set_int("maximum_number_of_iteration", 1000);
  projection_parameters.set_double("convergence_criterion", 1.0e-14);
  projection_parameters.set_string("verbose_level", "Crucial");
  Projection_Maximum_SU_N projection(projection_parameters);

  Parameters stout_projection_parameters;
  stout_projection_parameters.set_string("verbose_level", "Crucial");
  Projection_Stout_SU3 stout_projection(stout_projection_parameters);

  Parameters ape_parameters;
  std::vector<double> rho(16, 0.6 / 6.0);
  for (int mu = 0; mu < 4; ++mu) rho[mu + 4 * mu] = 0.6;
  ape_parameters.set_double("rho_uniform", 0.1);
  ape_parameters.set_string("verbose_level", "Crucial");
  Smear_APE ape(&projection, ape_parameters);
  ape.set_parameters(rho);

  Parameters hyp_parameters;
  hyp_parameters.set_double("alpha1", 0.75);
  hyp_parameters.set_double("alpha2", 0.6);
  hyp_parameters.set_double("alpha3", 0.3);
  hyp_parameters.set_string("verbose_level", "Crucial");
  Smear_HYP hyp(&projection, hyp_parameters);

  // Bridge++ implements HEX by using the HYP geometry with the analytic
  // Projection_Stout_SU3 at every level. Smear_HYP supplies the geometric
  // outer/middle/inner normalization factors 1/6, 1/4, and 1/2.
  Parameters hex_parameters;
  hex_parameters.set_double("alpha1", 0.125);
  hex_parameters.set_double("alpha2", 0.15);
  hex_parameters.set_double("alpha3", 0.15);
  hex_parameters.set_string("verbose_level", "Crucial");
  Smear_HYP hex(&stout_projection, hex_parameters);

  Field_G raw(CommonParameters::Nvol(), 4);
  Field_G zero(CommonParameters::Nvol(), 4);
  Field_G thin(CommonParameters::Nvol(), 4);
  Field_G ape_output(CommonParameters::Nvol(), 4);
  Field_G hyp_output(CommonParameters::Nvol(), 4);
  Field_G hex_output(CommonParameters::Nvol(), 4);
  fill_raw(raw);
  zero.set(0.0);

#pragma omp parallel
  {
    projection.project(thin, 0.0, zero, raw);
    ape.smear(ape_output, thin);
    hyp.smear(hyp_output, thin);
    hex.smear(hex_output, thin);
  }

  if (Communicator::is_primary()) {
    std::cout << std::setprecision(17);
    std::cout << "stage\tmu\tx0\tx1\tx2\tx3\trow\tcol\tre\tim\n";
    write_stage("thin", thin);
    write_stage("ape", ape_output);
    write_stage("hyp", hyp_output);
    write_stage("hex", hex_output);
  }

  bridge_finalize();
  return 0;
}
