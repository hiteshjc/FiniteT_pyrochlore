#ifndef SEMICLASSICAL_HEADER
#define SEMICLASSICAL_HEADER

#include"classical_spin_hole_model.h"
using namespace std;

void semiclassical_mc_zeroT(Classical_Spin_Hole_Model &h, 
			    int iterations, 
		            int num_mc_samples, 
			    double start_delta);

void lanczos_real_semiclassical(Classical_Spin_Hole_Model &h, 
		      int iterations,
                      Spin_Config &sc, 
		      std::vector<double> &eigs, int how_many_eigenvecs, 
		      std::vector< std::vector<double> > &eigenvecs,
                      bool store_ham, bool ipr, std::vector<double> &best_vector);

void diag_t_matrix(Classical_Spin_Hole_Model &h, Spin_Config &sc_proposed, double &ground_state_energy);               

#endif
