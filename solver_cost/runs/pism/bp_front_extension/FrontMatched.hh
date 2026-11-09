// Benchmark-specific boundary adapter for released PISM 2.3.1.
// GPL-3.0-or-later; native PISM assembly and solver remain unchanged.
#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include "pism/stressbalance/blatter/Blatter.hh"
#include "pism/util/Grid.hh"

class FrontMatchedBlatter : public pism::stressbalance::Blatter {
public:
  FrontMatchedBlatter(std::shared_ptr<const pism::Grid> grid, int Mz,
                       int coarsening_factor, const std::vector<int>& mask, const std::vector<int>& fronts)
    : pism::stressbalance::Blatter(grid, Mz, coarsening_factor),
      m_mask(mask), m_fronts(fronts), m_mx(grid->Mx()), m_my(grid->My()) {
    if (!m_fronts.empty() && m_fronts.size()!=static_cast<size_t>(4*(m_mx-1)*(m_my-1)))
      throw std::runtime_error("Front mask size differs from horizontal elements.");
    if (m_mask.size() != static_cast<size_t>(m_mx*m_my))
      throw std::runtime_error("Boundary mask size differs from horizontal grid.");
  }
protected:
  bool dirichlet_node(const DMDALocalInfo &,
                      const pism::fem::Element3::GlobalIndex& I) override {
    // All sigma grids share the horizontal grid. This constrains the entire
    // ice column at masked nodes on every multigrid level.
    const int i=(I.i % m_mx + m_mx) % m_mx;
    const int j=(I.j % m_my + m_my) % m_my;
    return m_mask[j*m_mx+i] != 0;
  }

  void nodal_parameter_values(const pism::fem::Q1Element3 &element,
                             Parameters **P, int i, int j,
                             int *node_type, double *bottom, double *thickness,
                             double *surface, double *sea_level) const override {
    m_i=i; m_j=j;
    pism::stressbalance::Blatter::nodal_parameter_values(element,P,i,j,node_type,
                                                        bottom,thickness,surface,sea_level);
  }
  bool marine_boundary(int face, const int *node_type, const double *bottom,
                       const double *sea_level) override {
    if (m_fronts.empty())
      return pism::stressbalance::Blatter::marine_boundary(face,node_type,bottom,sea_level);
    const int i=(m_i % (m_mx-1) + m_mx-1) % (m_mx-1);
    const int j=(m_j % (m_my-1) + m_my-1) % (m_my-1);
    return m_fronts[4*(j*(m_mx-1)+i)+face] != 0;
  }
  // The base u_bc returns zero. Volume/basal/front kernels are unchanged.

private:
  std::vector<int> m_mask, m_fronts;
  mutable int m_i=0, m_j=0;
  int m_mx, m_my;
};
