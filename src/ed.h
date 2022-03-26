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

void normalize(std::vector< complex<double> > & v); 
void normalize(std::vector< double > & v); 
void equatek(vector< complex<double> > &x, vector< complex<double> > &y);
void equatek(vector<double> &x, vector<double> &y);
void zscalk(const int64_t size, complex<double> a, vector< complex< double> > &x);
void dscalk(const int64_t size, double a, vector< double > &x);
void zaxpyk(const int64_t size, complex<double> a, vector< complex< double> > &x, vector< complex< double> > &y);
void daxpyk(const int64_t size, double a, vector< double > &x, vector< double > &y);
complex<double> zdotc(	const int64_t &size, const vector< complex< double> > &v1, const vector< complex< double> > &v2);
double       ddotk(const int64_t &size, vector< double > &v1, vector< double > &v2);
void initialize_alphas_betas(std::vector< complex<double> > &alphas, std::vector< complex<double> > &betas, complex<double>  &beta);

	
void make_basis(std::vector< std::vector<int> > &maps,
   		std::vector< complex<double>  > &characters,
		std::vector< char>    		&reps,
   		std::vector< int64_t> 		&locreps,
   		std::vector< int64_t> 		&ireps,
   		std::vector< char>    		&norms,
   	        std::vector<int64_t>  		&spin_dets,
		int64_t 	      		&hilbert);

void initialize_Lanczos_vectors(std::vector< complex<double> > &v_p, 
			        std::vector< complex<double> > &v_o, 
		                std::vector< complex<double> > &w);


void initialize_alphas_betas(   std::vector< complex<double> > &alphas, 
			        std::vector< complex<double> > &betas, 
		                complex<double>  &beta);

void actHonv(   Ham &h,
   	        std::vector<int64_t>             const &spin_dets,
   		std::vector< complex<double>  >  const &characters,
		std::vector< char>               const &reps,
   		std::vector< int64_t>            const &locreps,
   		std::vector< int64_t> 		 const &ireps,
   		std::vector< char>    		 const &norms,
		std::vector< complex<double> >   const &v_p, 
		std::vector< complex<double> >   &w);

void Lanczos_step_update(complex<double> &beta, 
		         std::vector< complex<double> >  &alphas, 
		         std::vector< complex<double> >  &betas, 
		  	 std::vector< complex<double> > &v_p, 
	                 std::vector< complex<double> > &v_o, 
		         std::vector< complex<double> > &w);

void make_tridiagonal_matrix_and_diagonalize(int size, 
			 std::vector< complex<double> >  const &alphas, 
		         std::vector< complex<double> >  const &betas, 
   			 Matrix 			 &t_mat,
			 std::vector<double>             &eigs,
			 Matrix                          &t_eigenvecs);

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
             std::vector<double> &eigs);

void lanczos_sym(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);

void lanczos_sym_evec(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs);
		
void perform_one_spin_measurements(std::vector< complex<double> > &vec, 
		std::vector<int64_t> &spin_dets, 
   		std::vector< std::vector<int> > &maps,
	        std::vector<complex<double> >  &characters,
		std::vector< char> &reps, std::vector< int64_t> &locreps, 
		std::vector< int64_t> &ireps, std::vector< char> &norms, 
		Simulation_Params &sp);

void perform_two_spin_measurements(std::vector< complex<double> > &vec, 
		std::vector<int64_t> &spin_dets, 
   		std::vector< std::vector<int> > &maps,
	        std::vector<complex<double> >  &characters,
		std::vector< char> &reps, std::vector< int64_t> &locreps, 
		std::vector< int64_t> &ireps, std::vector< char> &norms,
		Simulation_Params &sp);

#endif
