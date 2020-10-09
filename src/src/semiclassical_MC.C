#include"semiclassical_MC.h"
#include"printing_functions.h"
#include<omp.h>
#include"hamiltonian.h"
#include"global.h"
#include"matrix_functions.h"
#include"number_functions.h"
#include"printing_functions.h"
#include"bethe_lapack_interface.h"
#include"math_utilities.h"
#include"hamiltonian_spin_functions.h"

/////////////////////////////////////////////////////////////////////////////////////////
void semiclassical_mc_zeroT(Classical_Spin_Hole_Model &h, 
			    int iterations, 
		            int num_mc_samples, 
			    double start_delta)
{
	// Definitions
	Spin_Config 			   sc_present,sc_proposed;
	std::vector<Spin_Config> 	   best_scs;
        std::vector<double>                best_energies;
	double 				   delta=start_delta;
	double 				   ground_state_energy, ground_state_energy_prev;
	std::vector<double>                energies;
	std::vector< std::vector<double> > eigenvecs;
	std::vector<double>                best_vector;
	
	for (int n=0;n<10;n++)
	{
        //sc_present.init(h.num_sites,h.pairs,h.neighbors,0.5,h.J);
        sc_present.init_random_ising(h.num_sites,h.pairs,h.neighbors,h.eta,0.5,h.J);
	// Run the zero temperature classical MC - heading towards lowest energy state
	if (h.U!=0.0 and h.num_sites<64)   		                // Interacting problem - limited to small systems
	{
        	// Generate uphole dets and downhole dets
        	constrained_dets_i64(h.num_sites,h.nup_holes,h.uphole_dets);
        	constrained_dets_i64(h.num_sites,h.ndn_holes,h.dnhole_dets);
		cout<<"Got constrained dets...."<<endl;
		cout<<"Hilbert space size is "<<h.uphole_dets.size()*h.dnhole_dets.size()<<endl;
		
		for (int i=0;i<num_mc_samples;i++)     // MC steps
		{
			sc_proposed=sc_present;
			if (i>0) sc_proposed.move_random_spin(delta);
			lanczos_real_semiclassical(h, iterations, sc_proposed, energies, 
							   1, eigenvecs,                         
							   true, false,best_vector);
			ground_state_energy=energies[0];	
			if (i>=0)cout<<"Ground state energy of proposed configuration = "<<ground_state_energy<<endl;
			if (i>0) cout<<"Ground state energy of best     configuration = "<<ground_state_energy_prev<<endl;
			if (ground_state_energy<ground_state_energy_prev or i==0)
			{
				sc_present=sc_proposed;
				ground_state_energy_prev=ground_state_energy;
				best_vector=eigenvecs[0];
			}
		}
		cout<<endl;
		cout<<endl;
		cout<<"==============================================================="<<endl;
		cout<<"Final Lanczos...."<<endl;	
		cout<<"==============================================================="<<endl;
		lanczos_real_semiclassical(h, iterations, sc_present, energies, 
						   1, eigenvecs,true, false,best_vector);
		ground_state_energy=energies[0];
	}
        else // Solve non interating problem - can go to very large system sizes and hole numbers
	{
		for (int i=0;i<num_mc_samples;i++)         // MC steps
		{
			sc_proposed=sc_present;
			if (i>0) sc_proposed.move_random_ising_spin();
			diag_t_matrix(h, sc_proposed, ground_state_energy);               
			if (i>=0)cout<<"Ground state energy of proposed configuration = "<<ground_state_energy<<endl;
			if (i>0) cout<<"Ground state energy of best     configuration = "<<ground_state_energy_prev<<endl;
			if (ground_state_energy<ground_state_energy_prev or i==0)
			{
				sc_present=sc_proposed;
				ground_state_energy_prev=ground_state_energy;
			}
		}
		cout<<endl;
		cout<<endl;
		cout<<"==============================================================="<<endl;
		cout<<"Final T matrix...."<<endl;	
		cout<<"==============================================================="<<endl;
		diag_t_matrix(h, sc_present, ground_state_energy);
	}
		best_scs.push_back(sc_present);
		best_energies.push_back(ground_state_energy);
	}

	int ind=0;
	ground_state_energy=best_energies[0];
	for (int n=0;n<best_energies.size();n++)
	{
		if (best_energies[n]<ground_state_energy) {ind=n;}
	}
	
	sc_present=best_scs[ind];
	ground_state_energy=best_energies[ind];

	///////////////////////////////////////////////////
	// Measurements on spin background
        ///////////////////////////////////////////////////
        double stag=0.0;
	for (int i=0;i<h.num_sites;i++)
	{
		for (int j=0;j<h.num_sites;j++)
		{
			for (int k=0;k<3;k++)
			{
				stag+=sc_present.spins(i,k)*sc_present.spins(j,k)*h.eta[i]*h.eta[j];
			}
		}
	}
  	stag=stag/double(h.num_sites*h.num_sites);
  	cout<<"Classical bath spins <M^2>_GS = "<<stag<<endl;

	cout<<"Spin directions (x,y,z)"<<endl;	
	print_real_mat(sc_present.spins);

        double mag_z=0.0;
	for (int i=0;i<h.num_sites;i++) mag_z+=sc_present.spins(i,2);
  	
        cout<<"Classical bath spins <M^z>_GS = "<<mag_z<<endl;
       
}
//////////////////////////////////////////////////////////////////////////////
void diag_t_matrix(Classical_Spin_Hole_Model &h, Spin_Config &sc, double &ground_state_energy)
{
	ground_state_energy=0.0;
	int nsites=h.num_sites;
	RMatrix t_matrix_up(nsites,nsites);
	RMatrix t_matrix_dn(nsites,nsites);
	RMatrix eigenvecs_up(nsites,nsites);
	RMatrix eigenvecs_dn(nsites,nsites);
        std::vector<double> eigs_up(nsites),eigs_dn(nsites);
	cout<<"h.D="<<h.D<<endl;

	for (int i=0;i<h.pairs.size();i++) // Off diagonal terms 
	{
		t_matrix_up(h.pairs[i][0],h.pairs[i][1])=-h.t;
		t_matrix_up(h.pairs[i][1],h.pairs[i][0])=-h.t;
		t_matrix_dn(h.pairs[i][0],h.pairs[i][1])=-h.t;
		t_matrix_dn(h.pairs[i][1],h.pairs[i][0])=-h.t;
	} 
	// Diagonal terms are given by coupling between holes and classical spins
	// Up spins
	for (int i=0;i<nsites;i++)
	{
		double spin_hole_energy=0.0;
		spin_hole_energy=-h.J_h*0.5*sc.spins(i,2);
		for (int k=0;k<h.neighbors_within_rh[i].size();k++)
		{
		int l=h.neighbors_within_rh[i][k];
		double sh_dot_sl=sc.spins(l,2);
		if (sh_dot_sl<1 and sh_dot_sl>0)
		{
		for (int j=0;j<sc.neighbors[l].size();j++)
		{
		int m=sc.neighbors[l][j];
                if (l<m)
		{
		if (std::count(h.neighbors_within_rh[i].begin(),h.neighbors_within_rh[i].end(),m)==1)
		{
			double sh_dot_sm=sc.spins(m,2);
			if (sh_dot_sm<1 and sh_dot_sm>0)
			{
			spin_hole_energy=spin_hole_energy-(h.D*sh_dot_sl*sh_dot_sm);
			}
		}
		}
		}
		}
		}
		t_matrix_up(i,i)=spin_hole_energy;
	}
	// Down spins
	for (int i=0;i<nsites;i++)
	{
		double spin_hole_energy=0.0;
		spin_hole_energy=h.J_h*0.5*sc.spins(i,2);
		for (int k=0;k<h.neighbors_within_rh[i].size();k++)
		{
		int l=h.neighbors_within_rh[i][k];
		double sh_dot_sl=-sc.spins(l,2);
		if (sh_dot_sl<1 and sh_dot_sl>0)
		{
		for (int j=0;j<sc.neighbors[l].size();j++)
		{
		int m=sc.neighbors[l][j];
		if (l<m)
		{
		if (std::count(h.neighbors_within_rh[i].begin(),h.neighbors_within_rh[i].end(),m)==1)
		{
			double sh_dot_sm=-sc.spins(m,2);
			if (sh_dot_sm<1 and sh_dot_sm>0)
			{
			spin_hole_energy=spin_hole_energy-(h.D*sh_dot_sl*sh_dot_sm);
			}
		}
		}
		}
		}
		}
		t_matrix_dn(i,i)=spin_hole_energy;
        }
	print_real_mat(t_matrix_up);
	print_real_mat(t_matrix_dn);
	real_symmetric_diagonalize(t_matrix_up,eigs_up,eigenvecs_up);
	real_symmetric_diagonalize(t_matrix_dn,eigs_dn,eigenvecs_dn);

	for (int i=0;i<h.nup_holes;i++) ground_state_energy+=eigs_up[i];	
	for (int i=0;i<h.ndn_holes;i++) ground_state_energy+=eigs_dn[i];
	
	ground_state_energy+=sc.E_spin;
		
}


//////////////////////////////////////////////////////////////////////////////
void lanczos_real_semiclassical(Classical_Spin_Hole_Model &h, 
		      int iterations,
                      Spin_Config &sc, 
		      std::vector<double> &eigs, int how_many_eigenvecs, 
		      std::vector< std::vector<double> > &eigenvecs,
                      bool store_ham, bool ipr, std::vector<double> &best_vector)
{
   time_t 	      				start,end;
   int 					        n_uphole_dets=h.uphole_dets.size();
   int 						n_dnhole_dets=h.dnhole_dets.size();
   std::vector<int>                             num_states;
   num_states.push_back(n_uphole_dets);
   num_states.push_back(n_dnhole_dets);
   int 		      				hilbert=n_uphole_dets*n_dnhole_dets;
   int 						nd=n_dnhole_dets;
   bool 	      				orth_failed;
   iterations=min(iterations,hilbert);
   double 					dif,tmp;
   double            				q,alpha,beta,norm;
   std::vector<double>				alphas,betas;
   std::vector<double>				w(hilbert),v_p(hilbert),v_o(hilbert), v_p_old(hilbert), ritz_eigenvec(hilbert);
   std::vector< std::vector<int> > 		vec_new_configs;
   std::vector< std::vector<double> >           vs;
   std::vector< std::vector< complex<double> > > vec_hints_list;
   RMatrix 					t_mat,t_eigenvecs;
   bool                                         ham_store;
   double                                       lowest_eigenval_prev;

   eigenvecs.clear();
   for (int i=0;i<how_many_eigenvecs;i++) eigenvecs.push_back(std::vector<double>());

   ipr=true;
   if (store_ham)
   {
	if (hilbert>8000000) {ham_store=false;}
   	else {ham_store=true;}
   }
   else
   {
	ham_store=false;
   }

   // Initializations
   cout<<"TRACE: Initializing start vectors and storing Hamiltonian (if possible)"<<endl;

   if (ham_store) {cout<<"Storing Hamiltonian in sparse format... because we can afford to"<<endl;}
   else           {cout<<"I have decided NOT to store the Hamiltonian"<<endl;}
    
   std::vector< std::vector<int> > list_of_converted_vecs;

   if (best_vector.size()==hilbert)
   {
	   for (int i=0;i<hilbert;i++)
	   {
		v_p[i]=best_vector[i]+(0.05*uniform_rnd());
		v_o[i]=0.0;w[i]=0.0;
		vec_new_configs.push_back(std::vector<int>());
		vec_hints_list.push_back(std::vector< complex<double> >());
		list_of_converted_vecs.push_back(std::vector<int>());
	   }
   }
   else
   {
	   for (int i=0;i<hilbert;i++)
	   {
		v_p[i]=uniform_rnd();
		v_o[i]=0.0;w[i]=0.0;
		vec_new_configs.push_back(std::vector<int>());
		vec_hints_list.push_back(std::vector< complex<double> >());
		list_of_converted_vecs.push_back(std::vector<int>());
	   }
   }
   if (ham_store)
   {
	//#pragma omp parallel for 
   	for (int i=0;i<hilbert;i++)
	{
		std::vector<int> vec=convert_ind_to_vec(i,num_states);
		int i0=vec[0];int i1=vec[1];
		Fermion_Config fc;
		fc.uphole_det=h.uphole_dets[i0];
		fc.dnhole_det=h.dnhole_dets[i1];
		fc.i0=i0; 
		fc.i1=i1;
		fc.nd=nd;
		h(sc,fc,vec_new_configs[i], vec_hints_list[i]); 
	}
        ///////////////////////////////////////////////////////////////////////////////////////////////////////
   }
   else
   {
	#pragma omp parallel for
   	for (int i=0;i<hilbert;i++) {list_of_converted_vecs[i]=convert_ind_to_vec(i,num_states);}
   } 
   norm=sqrt(ddot(hilbert,&*v_p.begin(),1,&*v_p.begin(),1));
   dscal(hilbert,1.0/norm,&*v_p.begin(),1);
  
   beta=0.0;betas.push_back(beta);

   for (int it=0;it<iterations;it++)
   {
       int exit_it=it;
       cout<<"Doing (Real) Lanczos iteration = "<<it<<endl;
       vs.push_back(v_p);     
       time (&start);
       ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
       if (ham_store)
       {
	       #pragma omp parallel 
	       {
		       # pragma omp for
		       for (int ip=0;ip<hilbert;ip++) // Computing H*v_p - This is the bulk of the operation
		       {
			    int mapsize=vec_new_configs[ip].size();
			    for (int kp=0;kp<mapsize;kp++)
			    {
				w[ip]+=(real(vec_hints_list[ip][kp])*v_p[vec_new_configs[ip][kp]]);
			    }
		       }
	       }
       }
       else
       {
	       #pragma omp parallel for	
	       for (int i=0;i<hilbert;i++) // Computing H*v_p - This is the bulk of the operation
	       {
		 std::vector<int>    new_configs;
		 std::vector<int>    new_spin_dets,new_uphole_dets,new_dnhole_dets;
		 std::vector< complex<double> > hints_list;
		 int i0=list_of_converted_vecs[i][0];
		 int i1=list_of_converted_vecs[i][1];
		 Fermion_Config fc;
		 fc.uphole_det=h.uphole_dets[i0];
		 fc.dnhole_det=h.dnhole_dets[i1];
		 fc.i0=i0; 
		 fc.i1=i1;
		 fc.nd=nd;
		 h(sc,fc,new_configs,hints_list); 
		 for (int kp=0;kp<new_configs.size();kp++)
		 {
		   w[i]+=(real(hints_list[kp])*v_p[new_configs[kp]]);
		 }
              }
       }
       ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
       
       daxpy(hilbert,-beta,&*v_o.begin(),1,&*w.begin(),1);
       alpha=ddot(hilbert,&*w.begin(),1,&*v_p.begin(),1);
       alphas.push_back(alpha);
       daxpy(hilbert,-alpha,&*v_p.begin(),1,&*w.begin(),1);
       v_o=v_p;
       beta=sqrt(ddot(hilbert,&*w.begin(),1,&*w.begin(),1));
       v_p=w;
       dscal(hilbert,1.0/beta,&*v_p.begin(),1);
       betas.push_back(beta);
       dscal(hilbert,0.0,&*w.begin(),1);

       // Now reorthogonalize vectors
       if (vs.size()<iterations)
       {
	       orth_failed=false;
	       for (int i=0;i<vs.size();i++)
	       {
		    q=ddot(hilbert,&*vs[i].begin(),1,&*v_p.begin(),1);
		    if (abs(q)>1.0e-8)
		    {
			i=it;
			if (ipr)
			{
				cout<<"q (overlap) ="<<q<<endl;
				cout<<"--------------------------------------------------"<<endl;
				cout<<"Orthogonalization failed... choosing random vector"<<endl;
				cout<<"--------------------------------------------------"<<endl;
			}
			orth_failed=true;
		    }
		    else
		    {
			daxpy(hilbert,-q,&*vs[i].begin(),1,&*v_p.begin(),1);
		    }
	       }
	       norm=sqrt(ddot(hilbert,&*v_p.begin(),1,&*v_p.begin(),1));
	       dscal(hilbert,1.0/norm,&*v_p.begin(),1);
	       
	       norm=0.0;
	       if (orth_failed)
	       {
			while (abs(norm)<1.0e-2)
			{
				for (int j=0;j<hilbert;j++)  {v_p_old[j]=uniform_rnd();}
				norm=sqrt(ddot(hilbert,&*v_p_old.begin(),1,&*v_p_old.begin(),1));
				dscal(hilbert,1.0/norm,&*v_p_old.begin(),1);
				//cout<<"norm="<<norm<<endl;
				v_p=v_p_old;
				for (int i=0;i<vs.size();i++)
				{
					q=ddot(hilbert,&*vs[i].begin(),1,&*v_p_old.begin(),1);
					if (ipr) cout<<"q (after orth fail )="<<q<<endl;
					daxpy(hilbert,-q,&*vs[i].begin(),1,&*v_p.begin(),1);
				}
				norm=sqrt(ddot(hilbert,&*v_p.begin(),1,&*v_p.begin(),1));
			}
			dscal(hilbert,1.0/norm,&*v_p.begin(),1);
	       }
      }
       //vs.push_back(v_p);     
       time (&end);
       dif=difftime(end,start);
       
       if (ipr)
       {
       		cout<<"Time to perform Lanczos iteration "<<it<<" was "<<dif<<" seconds"<<endl;
       		cout<<"================================================================="<<endl;
       }
       if (it %1 ==0) 
       {
	       t_mat.clear();t_eigenvecs.clear();eigs.clear();
	       t_mat.resize(it+1,it+1);t_eigenvecs.resize(it+1,it+1);eigs.resize(it+1);
	       for (int j=0;j<it+1;j++)
	       {
		t_mat(j,j)=alphas[j];
		if (j+1<it+1)
		{t_mat(j,j+1)=betas[j+1];t_mat(j+1,j)=betas[j+1];}
	       }
	       real_symmetric_diagonalize(t_mat,eigs,t_eigenvecs);
	       if (abs(eigs[0]-lowest_eigenval_prev)<1.0e-10 and it>0)  // Converged
	       {it=iterations;}
	       else
	       {lowest_eigenval_prev=eigs[0];}
	       cout<<boost::format("Iteration, Lowest eigenvalue %3i %+.15f") %exit_it %eigs[0]<<endl;
  	}
  }

   how_many_eigenvecs=min(hilbert,how_many_eigenvecs);
   for (int j=0;j<how_many_eigenvecs;j++)
   {
	      eigenvecs[j].resize(hilbert); 
	      if (ipr) cout<<"Making "<<j<<"th Ritz eigenvector"<<endl;
	      # pragma omp parallel for
	      for (int i=0;i<hilbert;i++)
	      {
		eigenvecs[j][i]=0.0;
		for (int k=0;k<vs.size();k++){eigenvecs[j][i]+=vs[k][i]*t_eigenvecs(k,j);}
	      }
	      if (ipr) cout<<"Done Making "<<j<<"th Ritz eigenvector"<<endl;
   }      
   if (hilbert<500 and eigenvecs.size()>0) {print_vec_acc(eigenvecs[0],true);
   cout<<endl;}
}

