%module bp_front_boundary
%include std_shared_ptr.i
%include std_vector.i
%template(BandIntVector) std::vector<int>;
%shared_ptr(pism::Grid)
%shared_ptr(pism::stressbalance::ShallowStressBalance)
%shared_ptr(pism::stressbalance::Blatter)
%import(module="PISM.cpp") "pism/util/Grid.hh"
%import(module="PISM.cpp") "pism/stressbalance/ShallowStressBalance.hh"
%import(module="PISM.cpp") "pism/stressbalance/blatter/Blatter.hh"
%shared_ptr(FrontMatchedBlatter)
%{
#include "FrontMatched.hh"
%}
%include "FrontMatched.hh"
