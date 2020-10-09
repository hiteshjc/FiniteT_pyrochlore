#ifndef ED_HEADER
#define ED_HEADER

#include"hamiltonian.h"
#include"global.h"
#include"matrix_functions.h"
#include"number_functions.h"
#include"printing_functions.h"
#include"bethe_lapack_interface.h"
#include"math_utilities.h"
#include"hamiltonian_spin_functions.h"
#include"tmp_info.h"

void lanczos_save_ham_sym_hints(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);

void exact(Ham &h, std::vector<double> &eigs, Matrix &eigenvecs);
void load_wf_get_corrs(int nsites, 
		       string wffile, 
		       Simulation_Params &smp,
	      	       string type);

void load_wf_get_1rdms(string wffile, 
		      Simulation_Params &smp,
	      	      int nsites,
	      	      int nup_holes,
	      	      int ndn_holes);

void load_wf_get_2rdms(string wffile, 
		      Simulation_Params &smp,
	      	      int nsites,
	      	      int nup_holes,
	      	      int ndn_holes, string type);

void ed_with_hints_given(std::vector< std::vector<int> >                  const &map,
		      	 std::vector< std::vector< complex<double> > >    const &hints,
                         std::vector<double> 		                  &eigs,
			 RMatrix 			                  &eigenvecs,
			 bool 				                  ipr);

void lanczos_three_band_sym_sector
		     (Ham &h,
		      Simulation_Params &smp,
		      int kx, int ky,
		      std::vector< std::vector<int> > maps, 
		      int nup_holes,
		      int ndn_holes,
                      std::vector<double> &eigs);

void lanczos_spin_hole_requested_sector
		     (Ham &h, Simulation_Params &smp, 
		      int nup_spins, int nup_holes,
		      int ndn_holes,std::vector<double> &eigs);

void lanczos_real_spin_hole_given_map(Ham &h,
                      Simulation_Params &sp, 
		      std::vector<int> const &spin_dets,
		      std::vector<int> const &uphole_dets,
		      std::vector<int> const &dnhole_dets,
		      std::vector<int> const &inverse_map_spin,	
		      std::vector<int> const &inverse_map_uphole,	
		      std::vector<int> const &inverse_map_dnhole,	
                      std::vector<double> &eigs,
                      std::vector< std::vector<double> > &eigenvecs);

void lanczos_spin_hole_all_spin_sectors
		     (Ham &h,
		      int nup_holes,
		      int ndn_holes,
                      int iterations, 
                      std::vector<double> &eigs);
void lanczos(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);


void lanczos(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector<double> > &eigenvecs);

void lanczos_save_ham_hints(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);

void lanczos_sym(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);

void lanczos_sym_evec(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);


#endif
