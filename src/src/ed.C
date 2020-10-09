#include"rotate.h"
#include"ed.h"
#include"printing_functions.h"
#include<omp.h>
///////////////////////////////////////////////////////////////////////////////////////////
void ed_with_hints_given(std::vector< std::vector<int> > 		const &map,
		      	 std::vector< std::vector< complex<double> > > 	const &hints,
                         std::vector<double> 			        &eigs,
			 RMatrix 					&eigenvecs,
			 bool 					        ipr)
{
   time_t 				start,end;
   int 					hilbert=map.size();
   double 				dif;
   RMatrix                              ham(hilbert,hilbert);
 
   time (&start);
   
   if (ipr) cout<<" Number of states in Matrix diag is "<<hilbert<<endl;
   for (int i=0;i<hilbert*hilbert;i++) ham[i]=0.0;

   eigs.clear();eigenvecs.clear();
   eigs.resize(hilbert);eigenvecs.resize(hilbert,hilbert);    
   
   for (int i=0;i<hilbert;i++) 
   {
	for (int k=0;k<map[i].size();k++)
	{
	  ham(i,map[i][k])+=real(hints[i][k]);
	}
   }
   
   real_symmetric_diagonalize(ham,eigs,eigenvecs);
   if (hilbert<10) print_real_mat(eigenvecs);
   time(&end);
   dif=difftime(end,start);
}

/////////////////////////////////////////////////////////////////////
void lanczos_spin_hole_all_spin_sectors
		     (Ham &h,
		      int nup_holes,
		      int ndn_holes,
                      int iterations, 
                      std::vector<double> &eigs)
{
   int                  how_many_eigenvecs=50;
   int 			size=pow(2,h.num_sites);
   std::vector<int>     spin_dets,uphole_dets,dnhole_dets;
   std::vector<int> 	inverse_map_spin(size),inverse_map_uphole(size),inverse_map_dnhole(size);
   std::vector< std::vector<double> > eigenvecs;

   for (int nup_spins=0;nup_spins<=h.num_sites;nup_spins++)
   {
   double sz=double(nup_spins)-(double(h.num_sites)/2.0);
   constrained_dets(h.num_sites,nup_spins,spin_dets);
   constrained_dets(h.num_sites,nup_holes,uphole_dets);
   constrained_dets(h.num_sites,ndn_holes,dnhole_dets);

   int n_spin_dets   = spin_dets.size();
   int n_uphole_dets = uphole_dets.size();
   int n_dnhole_dets = dnhole_dets.size();
 
   int hilbert=spin_dets.size()*uphole_dets.size()*dnhole_dets.size();
   iterations=min(iterations,hilbert);
 
   for(int i1=0;i1<n_spin_dets;i1++)   inverse_map_spin[spin_dets[i1]]=i1;
   for(int i2=0;i2<n_uphole_dets;i2++) inverse_map_uphole[uphole_dets[i2]]=i2;
   for(int i3=0;i3<n_dnhole_dets;i3++) inverse_map_dnhole[dnhole_dets[i3]]=i3;

   eigs.clear();
   Simulation_Params sp;
   sp.iterations=iterations; 
   sp.store_ham=true;
   sp.ipr=false;
   sp.how_many_eigenvecs=how_many_eigenvecs;
   lanczos_real_spin_hole_given_map(h,sp,spin_dets,uphole_dets,dnhole_dets,
   			            inverse_map_spin,inverse_map_uphole,
				    inverse_map_dnhole,eigs, eigenvecs);

   RMatrix si_sj_spins;
   cout<<"Computing Si dot Sj MATRIX of the bath spins....."<<endl;
   compute_si_sj_spins(h.num_sites,eigenvecs[0],
 		   spin_dets,uphole_dets,dnhole_dets,
   	 	   inverse_map_spin,inverse_map_uphole,
		   inverse_map_dnhole,si_sj_spins);

  /*cout<<endl; 
  cout<<"================================================================="<<endl;
  cout<<"Displaying Si dot Sj MATRIX of the bath spins in sector S_z = "<<sz<<endl;
  cout<<"================================================================="<<endl;
  print_real_mat(si_sj_spins); */
  
  double stag=0.0;
  for (int i=0;i<h.num_sites;i++)
  {
	for (int j=0;j<h.num_sites;j++)
	{
		stag+=si_sj_spins(i,j)*h.eta[i]*h.eta[j];
	}
  }
  stag=stag/double(h.num_sites*h.num_sites);
  cout<<"Staggered Magnetization of the bath spins <M^2>_GS = "<<stag<<" in sector S_z = "<<sz<<endl;
  cout<<"Ground state energy of the hole-spin system E_GS   = "<<eigs[0]<<" in sector S_z = "<<sz<<endl;
  }
}   

/////////////////////////////////////////////////////////////////////////////////////////
void lanczos_spin_hole_requested_sector
		     (Ham &h,
		      Simulation_Params &smp,
		      int nup_spins,
		      int nup_holes,
		      int ndn_holes,
                      std::vector<double> &eigs)
{
   int 			iterations=smp.iterations;
   int                  num_cycles=smp.num_cycles;
   int                  how_many_eigenvecs=smp.how_many_eigenvecs;
   bool                 rotate=smp.rotate;
   int 			size=pow(2,h.num_sites);
   std::vector<int>     spin_dets,uphole_dets,dnhole_dets;
   std::vector<int> 	inverse_map_spin(size),inverse_map_uphole(size),inverse_map_dnhole(size);
   std::vector< std::vector<double> > eigenvecs;

   constrained_dets(h.num_sites,nup_spins,spin_dets);
   constrained_dets(h.num_sites,nup_holes,uphole_dets);
   constrained_dets(h.num_sites,ndn_holes,dnhole_dets);

   int n_spin_dets   = spin_dets.size();
   int n_uphole_dets = uphole_dets.size();
   int n_dnhole_dets = dnhole_dets.size();
 
   int hilbert=spin_dets.size()*uphole_dets.size()*dnhole_dets.size();
   iterations=min(iterations,hilbert);
 
   for(int i1=0;i1<n_spin_dets;i1++)   inverse_map_spin[spin_dets[i1]]=i1;
   for(int i2=0;i2<n_uphole_dets;i2++) inverse_map_uphole[uphole_dets[i2]]=i2;
   for(int i3=0;i3<n_dnhole_dets;i3++) inverse_map_dnhole[dnhole_dets[i3]]=i3;

   eigs.clear();
   Simulation_Params sp;
   sp.iterations=iterations; 
   sp.num_cycles=num_cycles;
   sp.store_ham=true;
   sp.ipr=false;
   sp.how_many_eigenvecs=how_many_eigenvecs;
   lanczos_real_spin_hole_given_map(h,sp,spin_dets,uphole_dets,dnhole_dets,
   			            inverse_map_spin,inverse_map_uphole,
				    inverse_map_dnhole,eigs, eigenvecs);

   cout<<"Energies of state"<<endl;
   print_vec_acc(eigs,true);

   std::vector<RMatrix> one_rdm_up(how_many_eigenvecs),one_rdm_dn(how_many_eigenvecs);
   std::vector<RMatrix> two_rdm_up(how_many_eigenvecs),two_rdm_dn(how_many_eigenvecs);
   std::vector<RMatrix> two_rdm_uddu(how_many_eigenvecs);

   for (int nst=0;nst<how_many_eigenvecs;nst++)
   {
	  cout<<"..................."<<endl;
	  cout<<endl;
	  cout<<endl;
	  cout<<"Computing 1-RDM of the up electrons....."<<endl;
	  compute_one_rdm_up_electrons(h.num_sites,eigenvecs[nst],
			   spin_dets,uphole_dets,dnhole_dets,
			   inverse_map_spin,inverse_map_uphole,
			   inverse_map_dnhole,one_rdm_up[nst]);
	  
	  cout<<"Computing 1-RDM of the down electrons....."<<endl;
	  compute_one_rdm_down_electrons(h.num_sites,eigenvecs[nst],
			   spin_dets,uphole_dets,dnhole_dets,
			   inverse_map_spin,inverse_map_uphole,
			   inverse_map_dnhole,one_rdm_dn[nst]);
	  
	  cout<<"======================================================================"<<endl;
	  cout<<"  Displaying 1-RDM of the UP/DOWN electrons MINIMALIST REPRESENTATION "<<endl;
	  cout<<"======================================================================"<<endl;
	  for (int i=0;i<h.num_sites;i++)
	  {
			for (int k=i;k<h.num_sites;k++)
			{
				cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %k<<"   "<<boost::format("%+5.10f") %one_rdm_up[nst](i,k)<<"   "<<boost::format("%+5.10f") %one_rdm_dn[nst](i,k)<<endl;
			}
	  } 
		 
	  int dim_1=h.num_sites; 
	  RMatrix             one_rdm_up_eigenvecs(dim_1,dim_1),one_rdm_dn_eigenvecs(dim_1,dim_1);
	  std::vector<double> one_rdm_up_eigs(dim_1),one_rdm_dn_eigs(dim_1);
	  real_symmetric_diagonalize(one_rdm_up[nst],one_rdm_up_eigs, one_rdm_up_eigenvecs);
	  real_symmetric_diagonalize(one_rdm_dn[nst],one_rdm_dn_eigs, one_rdm_dn_eigenvecs);

	  cout<<"================================================================="<<endl;
	  cout<<"         Diagonalized the 1-RDM (up).... Displaying eigenvalues  "<<endl;
	  cout<<"================================================================="<<endl;
	  reverse(one_rdm_up_eigs.begin(),one_rdm_up_eigs.end());
	  print_vec_acc(one_rdm_up_eigs,true);
	  double sum=0.0; for (int i=0;i<dim_1;i++) sum+=one_rdm_up_eigs[i];
	  cout<<"Summation of 1-RDM up eigenvalues = "<<sum<<endl;
	  
	  cout<<"================================================================="<<endl;
	  cout<<"         Diagonalized the 1-RDM (dn).... Displaying eigenvalues  "<<endl;
	  cout<<"================================================================="<<endl;
	  reverse(one_rdm_dn_eigs.begin(),one_rdm_dn_eigs.end());
	  print_vec_acc(one_rdm_dn_eigs,true);
		 sum=0.0; for (int i=0;i<dim_1;i++) sum+=one_rdm_dn_eigs[i];
	  cout<<"Summation of 1-RDM down eigenvalues = "<<sum<<endl;
	  
	  /////////////////////////////////////////////////////////////////////////////////////////////////////
	  ///                                  2-RDM (UP)
	  ////////////////////////////////////////////////////////////////////////////////////////////////////
	  if (nup_holes>=2)
	  {
		  cout<<endl;
		  cout<<endl;
		  cout<<"Computing 2-RDM of the up electrons....."<<endl;
		  compute_explicit_two_rdm_up_electrons(h.num_sites,eigenvecs[nst],
				   spin_dets,uphole_dets,dnhole_dets,
				   inverse_map_spin,inverse_map_uphole,
				   inverse_map_dnhole,two_rdm_up[nst]);
		 
		  cout<<endl; 
		  cout<<"=============================================================================="<<endl;
		  cout<<"   Displaying 2-RDM of the up electrons MINIMALIST REPRESENATION (ijlk format)"<<endl;
		  cout<<"=============================================================================="<<endl;
		  double sum_of_diags=0.0;
		  for (int i=0;i<h.num_sites;i++)
		  {
			for (int j=0;j<h.num_sites;j++)
			{
				int ind1=(i*h.num_sites)+j;
				for (int k=0;k<h.num_sites;k++)
				{
					for (int l=0;l<h.num_sites;l++)
					{
						int ind2=(k*h.num_sites)+l;
						cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %j<<" "<<boost::format("%3i") %l<<boost::format("%3i") %k<<"       "<<boost::format("%+5.10f") %two_rdm_up[nst](ind1,ind2)<<endl;
					if (ind2==ind1) {sum_of_diags+=two_rdm_up[nst](ind1,ind2);}
					}
				}
			}
		  }
	  	  cout<<"Sum of diags (uu) = "<<sum_of_diags<<endl;
	  } 
	  /////////////////////////////////////////////////////////////////////////////////////////////////////
	  ///                                  2-RDM (DOWN)
	  ////////////////////////////////////////////////////////////////////////////////////////////////////
	  if (ndn_holes>=2)
	  {
		  cout<<endl;
		  cout<<endl;
		  cout<<"Computing 2-RDM of the down electrons....."<<endl;
		  compute_explicit_two_rdm_dn_electrons(h.num_sites,eigenvecs[nst],
				   spin_dets,uphole_dets,dnhole_dets,
				   inverse_map_spin,inverse_map_uphole,
				   inverse_map_dnhole,two_rdm_dn[nst]);
		 
		  cout<<endl; 
		  cout<<"=============================================================================="<<endl;
		  cout<<"   Displaying 2-RDM of the down electrons MINIMALIST REPRESENATION (ijlk format)"<<endl;
		  cout<<"=============================================================================="<<endl;
		  for (int i=0;i<h.num_sites;i++)
		  {
			for (int j=i+1;j<h.num_sites;j++)
			{
				int ind1=(i*h.num_sites)+j;
				for (int k=0;k<h.num_sites;k++)
				{
					for (int l=k+1;l<h.num_sites;l++)
					{
						int ind2=(k*h.num_sites)+l;
						cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %j<<" "<<boost::format("%3i") %l<<boost::format("%3i") %k<<"       "<<boost::format("%+5.10f") %two_rdm_dn[nst](ind1,ind2)<<endl;
					}
				}
			}
		  } 
	  }

	  /////////////////////////////////////////////////////////////////////////////////////////////////////
	  ///                                  2-RDM (uddu)
	  ////////////////////////////////////////////////////////////////////////////////////////////////////
	  if (ndn_holes>=1 and nup_holes>=1)
	  {
		  cout<<endl;
		  cout<<endl;
		  cout<<"Computing 2-RDM uddu....."<<endl;
		  compute_explicit_two_rdm_uddu(h.num_sites,eigenvecs[nst],
				   spin_dets,uphole_dets,dnhole_dets,
				   inverse_map_spin,inverse_map_uphole,
				   inverse_map_dnhole,two_rdm_uddu[nst]);
		 
		  cout<<endl; 
		  cout<<"=============================================================================="<<endl;
		  cout<<"   Displaying 2-RDM of the uddu electrons MINIMALIST REPRESENATION (ijlk format)"<<endl;
		  cout<<"=============================================================================="<<endl;
		  for (int i=0;i<h.num_sites;i++)
		  {
			for (int j=i+1;j<h.num_sites;j++)
			{
				int ind1=(i*h.num_sites)+j;
				for (int k=0;k<h.num_sites;k++)
				{
					for (int l=k+1;l<h.num_sites;l++)
					{
						int ind2=(k*h.num_sites)+l;
						cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %j<<" "<<boost::format("%3i") %l<<boost::format("%3i") %k<<"       "<<boost::format("%+5.10f") %two_rdm_uddu[nst](ind1,ind2)<<endl;
					}
				}
			}
		  }
	   }
   }
        
  for (int nst=0;nst<how_many_eigenvecs;nst++)
  {
	   cout<<"One RDM for up electrons for state number "<<nst<<endl;
	   print_real_mat(one_rdm_up[nst]);
  }

  RMatrix unitary; 
  std::vector<RMatrix> out_one_rdm;
  if (rotate) optimize_unitary_from_one_rdm(one_rdm_up,unitary,out_one_rdm);
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///                                  3-RDM
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /*if (nup_holes>=3)
  {  
  int                 dim_3=h.num_sites*h.num_sites*h.num_sites;
  RMatrix             three_rdm_up(dim_3,dim_3);
  cout<<endl;
  cout<<endl;
  cout<<"Computing 3-RDM of the up electrons....."<<endl;
  compute_explicit_three_rdm_up_electrons(h.num_sites,eigenvecs[0],
 		   spin_dets,uphole_dets,dnhole_dets,
   	 	   inverse_map_spin,inverse_map_uphole,
		   inverse_map_dnhole,three_rdm_up);
 
  cout<<endl; 
  cout<<"================================================================="<<endl;
  cout<<"   Displaying 3-RDM of the up electrons MINIMALIST REPRESENATION "<<endl;
  cout<<"================================================================="<<endl;
for (int i=0;i<h.num_sites;i++)
{
for (int j=i+1;j<h.num_sites;j++)
{
for (int k=j+1;k<h.num_sites;k++)
{
int ind1=(i*h.num_sites*h.num_sites)+(j*h.num_sites)+k;
for (int l=0;l<h.num_sites;l++)
{
for (int m=l+1;m<h.num_sites;m++)
{
for (int n=m+1;n<h.num_sites;n++)
{
int ind2=(l*h.num_sites*h.num_sites)+(m*h.num_sites)+n;
cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %j<<" "<<boost::format("%3i") %k<<" "<<boost::format("%3i") %n<<" "<<boost::format("%3i") %m<<" "<<boost::format("%3i") %l<<"       "<<boost::format("%+5.10f") %three_rdm_up(ind1,ind2)<<endl;
}
}
}
}
}

} 

  if (dim_3<4800)
  {
	  RMatrix             three_rdm_up_eigenvecs(dim_3,dim_3);
	  std::vector<double> three_rdm_up_eigs(dim_3);

	  cout<<endl;
	  cout<<"================================================================="<<endl;
	  cout<<"         Diagonalizing the 3-RDM.... Displaying eigenvalues      "<<endl;
	  cout<<"================================================================="<<endl;
	  real_symmetric_diagonalize(three_rdm_up,three_rdm_up_eigs, three_rdm_up_eigenvecs);

	  print_vec_acc(three_rdm_up_eigs,true);
	  cout<<endl;  

	  double sum=0.0;
	  for (int i=0;i<dim_3;i++) sum+=three_rdm_up_eigs[i];
	  cout<<"Summation of 3-RDM eigenvalues = "<<sum<<endl;
          std::vector<double> vec(dim_3);

	  for (int i=0;i<dim_3;i++)
	  {
	  	cout<<"============================"<<endl;
		cout<<" Eigenvector number "<<i<<endl;
	  	cout<<"============================"<<endl;
		for (int j=0;j<dim_3;j++) vec[j]=three_rdm_up_eigenvecs(j,i);
		print_vec_acc(vec,true);
	  	cout<<endl;
	  }
  }
}*/

}
//////////////////////////////////////////////////////////////////////////////
void lanczos_real_spin_hole_given_map(Ham &h,
                      Simulation_Params &sp, 
		      std::vector<int> const &spin_dets,
		      std::vector<int> const &uphole_dets,
		      std::vector<int> const &dnhole_dets,
		      std::vector<int> const &inverse_map_spin,	
		      std::vector<int> const &inverse_map_uphole,	
		      std::vector<int> const &inverse_map_dnhole,	
                      std::vector<double> &eigs,
                      std::vector< std::vector<double> > &eigenvecs)
{
   bool                                         store_ham=sp.store_ham;
   bool 					ipr=sp.ipr;
   int   					how_many_eigenvecs=sp.how_many_eigenvecs;
   int 						iterations=sp.iterations;
   int 						num_cycles=sp.num_cycles;
   time_t 	      				start,end;
   int 						n_spin_dets=spin_dets.size();
   int 					        n_uphole_dets=uphole_dets.size();
   int 						n_dnhole_dets=dnhole_dets.size();
   std::vector<int>                             num_states;
   num_states.push_back(n_spin_dets);
   num_states.push_back(n_uphole_dets);
   num_states.push_back(n_dnhole_dets);
   int 		      				hilbert=n_spin_dets*n_uphole_dets*n_dnhole_dets;
   int 						nd=n_dnhole_dets;
   int 						nud=n_uphole_dets*n_dnhole_dets;
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
    
   std::vector< std::vector<int> > list_of_converted_vecs(hilbert);

   cout<<"List of vecs made"<<endl;
   cout<<"hilbert ="<<hilbert<<endl;

   if (ham_store)
   {
   for (int i=0;i<hilbert;i++)
   {
        v_p[i]=uniform_rnd();
        v_o[i]=0.0;w[i]=0.0;
	vec_new_configs.push_back(std::vector<int>());
	vec_hints_list.push_back(std::vector< complex<double> >());
	//list_of_converted_vecs.push_back(std::vector<int>());
   }
   }
   else
   {
   for (int i=0;i<hilbert;i++)
   {
        v_p[i]=uniform_rnd();
        v_o[i]=0.0;w[i]=0.0;
   }
   }

   cout<<"Here..."<<endl;
   if (ham_store)
   {
	#pragma omp parallel for 
   	for (int i=0;i<hilbert;i++)
	{
		std::vector<int>    new_configs;
		std::vector<int>    new_spin_dets,new_uphole_dets,new_dnhole_dets;
		std::vector< complex<double> > hints_list;
		std::vector<int> vec=convert_ind_to_vec(i,num_states);
		int i0=vec[0];int i1=vec[1];int i2=vec[2]; 

		//cout<<"Calculating H..."<<endl;
		h(spin_dets[i0],uphole_dets[i1],dnhole_dets[i2],new_spin_dets,new_uphole_dets,
		  new_dnhole_dets, hints_list); 
		for (int k=0;k<new_spin_dets.size();k++) 
		{
			new_configs.push_back(inverse_map_spin[new_spin_dets[k]]*(nud)+inverse_map_uphole[new_uphole_dets[k]]*(nd)+inverse_map_dnhole[new_dnhole_dets[k]]);
		}
		vec_new_configs[i]=new_configs;     // New configs 
		vec_hints_list[i]=hints_list;       // Hints
	}
	if (hilbert < 4000)
	{
	       cout<<"Performing Matrix Diag"<<endl;
	       RMatrix eigen_mat;
	       ed_with_hints_given(vec_new_configs,vec_hints_list,
                         	   eigs,eigen_mat,ipr);
   	       eigenvecs.clear();
               for (int i=0;i<hilbert;i++) eigenvecs.push_back(std::vector<double>());
	       for (int i=0;i<hilbert;i++)
	       {
	      		eigenvecs[i].resize(hilbert); 
	      		for (int j=0;j<hilbert;j++)
	      		{
				eigenvecs[i][j]=eigen_mat(j,i);
	      		}
	       }
	       return;
	}
        ///////////////////////////////////////////////////////////////////////////////////////////////////////
   }
   else
   {
	#pragma omp parallel for
   	for (int i=0;i<hilbert;i++) {list_of_converted_vecs[i]=convert_ind_to_vec(i,num_states);}
   } 
 
   cout<<"Starting Lanczos..."<<endl; 
   for (int cycle=0;cycle<num_cycles;cycle++)
   { 
   	norm=sqrt(ddot(hilbert,&*v_p.begin(),1,&*v_p.begin(),1));
   	dscal(hilbert,1.0/norm,&*v_p.begin(),1);
   	betas.clear();alphas.clear();  
   	beta=0.0;betas.push_back(beta);
	vs.clear();

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
			 int i2=list_of_converted_vecs[i][2]; 

			 h(spin_dets[i0],uphole_dets[i1],dnhole_dets[i2],new_spin_dets,new_uphole_dets,
			  new_dnhole_dets, hints_list); 
			 for (int k=0;k<new_spin_dets.size();k++) 
			 {
				new_configs.push_back(inverse_map_spin[new_spin_dets[k]]*(nud)+inverse_map_uphole[new_uphole_dets[k]]*(nd)+inverse_map_dnhole[new_dnhole_dets[k]]);
			 }
			
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
			    if (abs(q)>1.0e-10)
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
		       if (abs(eigs[0]-lowest_eigenval_prev)<1.0e-13 and it>500)  // Converged
		       {it=iterations;cycle=num_cycles;}
		       else
		       {lowest_eigenval_prev=eigs[0];}
		       cout<<boost::format("Iteration, Lowest eigenvalue %3i %+.15f") %exit_it %eigs[0]<<endl;
		}
	}
	if (ipr) cout<<"Making best vector from cycle "<<cycle<<" for next cycle...."<<endl;
	# pragma omp parallel for
	for (int i=0;i<hilbert;i++)
	{
		v_p[i]=0.0;
		for (int k=0;k<vs.size();k++){v_p[i]+=vs[k][i]*t_eigenvecs(k,0);}
	}
	if (ipr) cout<<"Done Making best vector for next cycle..."<<endl;
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
   if (hilbert<100) {print_vec_acc(eigenvecs[2],true);
   cout<<endl;}
}
