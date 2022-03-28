#include"ed.h"
#include"printing_functions.h"
#include<omp.h>


/////////////////////////////////////////////////////////////////////////////////
void make_basis(std::vector< std::vector<int> > &maps,
   		std::vector< complex<double>  > &characters,
		std::vector< char>    		&reps,
   		std::vector< int64_t> 		&locreps,
   		std::vector< int64_t> 		&ireps,
   		std::vector< char>    		&norms,
   	        std::vector<int64_t>  		&spin_dets,
		int64_t 	      		&hilbert)
{   
   int64_t fhilbert=reps.size();
   hilbert=0;
   spin_dets.clear();
   #pragma omp parallel for 
   for (int64_t i=0;i<fhilbert;i++)
   {
	get_representative(maps, characters, i, ireps[i], reps[i], norms[i]);
   }
   
   for (int64_t i=0;i<fhilbert;i++)
   {
	if (reps[i]=='0' and norms[i]!='0') 
	{
		//outfile<<" i  = "<<i<<" norms = "<<norms[i]<<endl;
		spin_dets.push_back(i);
		locreps[i]=hilbert;
		hilbert+=1;
	}
   }
}

/////////////////////////////////////////////////////////////////////////////////

void initialize_Lanczos_vectors(std::vector< complex<double> > &v_p, 
			        std::vector< complex<double> > &v_o, 
		                std::vector< complex<double> > &w)
{
   int64_t hilbert=v_p.size();
   //Initialization......
   for (int64_t i=0;i<hilbert;i++)
   {
		v_p[i]=complex<double> (2.0*uniform_rnd()-1.0, 2.0*uniform_rnd()-1.0);
		v_o[i]=0.0;w[i]=0.0;
   }
   normalize(v_p); 
}

/////////////////////////////////////////////////////////////////////////////////
void initialize_alphas_betas(   std::vector< complex<double> > &alphas, 
			        std::vector< complex<double> > &betas, 
		                complex<double>  &beta)
{
   betas.clear();alphas.clear();  
   beta=0.0;betas.push_back(beta);
}

/////////////////////////////////////////////////////////////////////////////////

void actHonv(   Ham &h,
   	        std::vector<int64_t>             const &spin_dets,
   		std::vector< complex<double>  >  const &characters,
		std::vector< char>               const &reps,
   		std::vector< int64_t>            const &locreps,
   		std::vector< int64_t> 		 const &ireps,
   		std::vector< char>    		 const &norms,
		std::vector< complex<double> >   const &v_p, 
		std::vector< complex<double> >   &w)
{
       int64_t hilbert=spin_dets.size();
       int     nsites=h.num_sites;

       #pragma omp parallel for	
       for (int64_t i=0;i<hilbert;i++) // Computing H*v_p - This is the bulk of the operation
       {
		std::vector<int64_t> 	      new_spin_dets(10*nsites+1);
		std::vector<complex<double> > hints(10*nsites+1);
		int64_t orig=spin_dets[i];
		int nconns;
		h(orig,new_spin_dets,hints, nconns);  // Original hamiltonian acted only on the representatives, return num connects
		//outfile<<"nconns = "<<nconns<<endl;
		int repeatket=norms[orig]-'0';
		//cout<<"new_spin_dets.size()="<<new_spin_dets.size()<<endl;
		complex<double> hint;
		double invrepeatket=sqrt(1.0/double(repeatket));
		for (int j=0;j<nconns;j++)  // Connections to state
		{
			int64_t news=new_spin_dets[j];
			char normnews=norms[news];
			if (normnews!='0') // an allowed state 
			{
				// Works only when reps = 0- 9 
				int repeat=normnews-'0'; //will be 0 if normnews='0'
				int op=reps[news]-'0'; // if not allowed it will be 0
				hint=hints[j]*(characters[op])*sqrt(double(repeat))*invrepeatket;
				news=ireps[news]; // faster as rep is stored
				int64_t location=locreps[news];
				w[i]+=(conj(hint)*v_p[location]);
			}
		 }
       }
       //cout<<"Finished Computing HV"<<endl;
}	       
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Lanczos_step_update(complex<double> &beta, 
		         std::vector< complex<double> >  &alphas, 
		         std::vector< complex<double> >  &betas, 
		  	 std::vector< complex<double> > &v_p, 
	                 std::vector< complex<double> > &v_o, 
		         std::vector< complex<double> > &w)
{
       int64_t hilbert=v_p.size();
       zaxpyk(hilbert,-beta,v_o,w);
       complex<double> alpha=conj(zdotc(hilbert,w,v_p));
       alphas.push_back(alpha);
       zaxpyk(hilbert,-alpha,v_p,w);
       equatek(v_p,v_o);
       beta=sqrt(zdotc(hilbert,w,w));
       equatek(w,v_p);
       zscalk(hilbert,1.0/beta,v_p);
       betas.push_back(beta);
       zscalk(hilbert,0.0,w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void make_tridiagonal_matrix_and_diagonalize(int size, 
			 std::vector< complex<double> >  const &alphas, 
		         std::vector< complex<double> >  const &betas, 
   			 Matrix 			 &t_mat,
			 std::vector<double>             &eigs,
			 Matrix                          &t_eigenvecs)
{	       
       t_mat.clear();t_eigenvecs.clear();eigs.clear();
       t_mat.resize(size,size);t_eigenvecs.resize(size,size);eigs.resize(size);
       for (int j=0;j<size*size;j++) {t_mat[j]=0.0;t_eigenvecs[j]=0.0;}
       for (int j=0;j<size;j++)
       {
	t_mat(j,j)=alphas[j];
	if (j+1<size)
	{t_mat(j,j+1)=betas[j+1];t_mat(j+1,j)=betas[j+1];}
       }
       symmetric_diagonalize(t_mat,eigs,t_eigenvecs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int64_t findstate(      int64_t &rep_id,
                        vector<int64_t> &pBasis)
{
        int64_t j = -1;

        int64_t b_min = 0, b_max = pBasis.size();
        do{
                int64_t b = b_min + (b_max - b_min)/2;
                if(rep_id < pBasis[b] )
                        b_max = b - 1;
                else if (rep_id > pBasis[b] )
                        b_min = b + 1;
                else
                {       j = b;  break;}
        }while(b_max >= b_min);

        return j;

}


void equatek(vector< complex<double> > &x, vector< complex<double> > &y)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < x.size(); ++i) {y[i]=x[i];}
}

void zscalk(const int64_t size,
	    complex<double> a,
	    vector< complex< double> > &x)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < size; ++i) {x[i]*=(a);}

}


void zaxpyk(const int64_t size,
	    complex<double> a,
	    vector< complex< double> > &x,
	    vector< complex< double> > &y)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < size; ++i) {y[i]+=(a*x[i]);}

}

complex<double> zdotc(	const int64_t &size,
			const vector< complex< double> > &v1,
			const vector< complex< double> > &v2)
{
	double sumr = 0.0;
	double sumi = 0.0;

	#pragma omp parallel for default(shared) reduction (+ : sumr,sumi)
	for(int64_t i = 0; i < size; ++i)
	{
		complex<double> a=conj(v1[i])*v2[i];
		sumr += real(a);
		sumi += imag(a);
	}

	complex<double> sum=complex<double>(sumr,sumi);

	return sum;

}

void normalize(std::vector< complex<double> > & v) 
{
	int64_t size=v.size();
	complex<double> norminv=1.0/sqrt(real(zdotc(size, v, v)));
	zscalk(size,norminv,v);
}

void orth_wrt_previous_evecs(std::vector< complex<double> > & v, 
		       std::vector< std::vector< complex<double> > > & previous_evecs)
{	
	int64_t size=v.size();
	std::vector< complex<double> > qs;

	for (int i=0; i < previous_evecs.size();i++)
	{
		complex<double> q=conj(zdotc(size, v, previous_evecs[i]));
		qs.push_back(q);
	}
	
	for (int i=0; i < previous_evecs.size();i++)
	{
		zaxpyk(size,-qs[i],previous_evecs[i],v);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
void equatek(vector<double> &x, vector<double> &y)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < x.size(); ++i) {y[i]=x[i];}
}

//////////////////////////////////////////////////////////////////////
void dscalk(const int64_t size,
	    double a,
	    vector< double > &x)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < size; ++i) {x[i]*=(a);}
}

///////////////////////////////////////////////////////////////////////////////////////////
void daxpyk(const int64_t size,
	    double a,
	    vector< double > &x,
	    vector< double > &y)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < size; ++i) {y[i]+=(a*x[i]);}
}


///////////////////////////////////////////////////////////////////////////////////////////
double       ddotk(	const int64_t &size,
			vector< double > &v1,
			vector< double > &v2)
{
	double sumr = 0.0;
	#pragma omp parallel for default(shared) reduction (+ : sumr)
	for(int64_t i = 0; i < size; ++i)
	{
		double a=v1[i]*v2[i];
		sumr += a;
	}
	return sumr;
}

///////////////////////////////////////////////////////////////////////////////////////////
void orth_wrt_previous_evecs(std::vector< double > &v, 
		       std::vector< std::vector< double > > &previous_evecs)
{	
	int64_t size=v.size();
	std::vector< double > qs;

	for (int i=0; i < previous_evecs.size();i++)
	{
		double q=ddotk(size, v, previous_evecs[i]);
		qs.push_back(q);
	}
	
	for (int i=0; i < previous_evecs.size();i++) daxpyk(size,-qs[i],previous_evecs[i],v);
}

///////////////////////////////////////////////////////////////////////////////////////////
void normalize(std::vector< double > & v) 
{
	int64_t size=v.size();
	double norminv=1.0/sqrt((ddotk(size, v, v)));
	dscalk(size,norminv,v);
}

///////////////////////////////////////////////////////////////////////////////////////////
void exact(Ham &h, std::vector<double> &eigs, Matrix &eigenvecs)
{
   time_t						start,end;
   double						dif=0.0;
   int 							nsites=h.num_sites;
   int 							hilbert=pow(2,nsites);
   Matrix						ham(hilbert,hilbert);
   cout<<" Number of states in Matrix diag is "<<hilbert<<endl;
   #pragma omp parallel for
   for (int i=0;i<hilbert*hilbert;i++) ham[i]=0.0;
   for (int i=0;i<hilbert;i++)
   {
   	std::vector<int> 	      new_spin_dets;
   	std::vector<complex<double> > hints;
	h(i,new_spin_dets,hints);
	for (int k=0;k<new_spin_dets.size();k++) ham(new_spin_dets[k],i)+=(hints[k]);
   }
   eigs.clear();eigenvecs.clear();
   eigs.resize(hilbert);eigenvecs.resize(hilbert,hilbert);    
   symmetric_diagonalize(ham,eigs,eigenvecs);
   time(&end);
   dif=difftime(end,start);
}

///////////////////////////////////////////////////////////////////////////////////////////
void ed_with_hints_given(std::vector< std::vector<int> > 		const &map,
		      	 std::vector< std::vector< complex<double> > > 	const &hints,
                         std::vector<double> 			        &eigs,
			 Matrix 					&eigenvecs,
			 bool 					        ipr)
{
   time_t 				start,end;
   int 					hilbert=map.size();
   double 				dif;
   Matrix                               ham(hilbert,hilbert);
 
   time (&start);
   
   if (ipr) cout<<" Number of states in Matrix diag is "<<hilbert<<endl;
   for (int i=0;i<hilbert*hilbert;i++) ham[i]=0.0;

   eigs.clear();eigenvecs.clear();
   eigs.resize(hilbert);eigenvecs.resize(hilbert,hilbert);    
   
   for (int i=0;i<hilbert;i++) 
   {
	for (int k=0;k<map[i].size();k++) ham(i,map[i][k])+=(hints[i][k]);
   }
   
   symmetric_diagonalize(ham,eigs,eigenvecs);
   time(&end);
   dif=difftime(end,start);
}

///////////////////////////////////////////////////////////////////////////////////////////
void ed_with_hints_given_outfile(
		         Simulation_Params &sp,
			 std::vector< std::vector<int> > 		const &newdets,
			 std::vector<int> 				const &inverse_map,
		      	 std::vector< std::vector< complex<double> > > 	const &hints,
                         std::vector<double> 			        &eigs,
			 Matrix 					&eigenvecs)
{
   time_t 				start,end;
   int 					hilbert=newdets.size();
   double 				dif;
   Matrix                               ham(hilbert,hilbert);
   ofstream outfile;
   const char *cstr = (sp.outfile).c_str();
   outfile.open(cstr);

   time (&start);
   
   outfile<<" Number of states in Matrix diag is "<<hilbert<<endl;
   for (int i=0;i<hilbert*hilbert;i++) ham[i]=0.0;

   eigs.clear();eigenvecs.clear();
   eigs.resize(hilbert);eigenvecs.resize(hilbert,hilbert);    
   
   for (int i=0;i<hilbert;i++) 
   {
	for (int k=0;k<newdets[i].size();k++) ham(inverse_map[newdets[i][k]],i)+=(hints[i][k]);
   }
   
   symmetric_diagonalize(ham,eigs,eigenvecs);
   time(&end);
   dif=difftime(end,start);

   for (int nv=0;nv<eigs.size();nv++)
   {		
   	outfile<<"CONVERGED (or lowest available) eigenvalue number "<<nv<<"  =  "<<boost::format ("%+.15f") %eigs[nv]<<endl;
   }
}


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
	for (int k=0;k<map[i].size();k++) ham(i,map[i][k])+=real(hints[i][k]);
   }
   
   real_symmetric_diagonalize(ham,eigs,eigenvecs);
   if (hilbert<10) print_real_mat(eigenvecs);
   time(&end);
   dif=difftime(end,start);
}


///////////////////////////////////////////////////////////////////////////////////////////
//void load_wf_get_corrs(int nsites, 
//		       string wffile, 
//		       Simulation_Params &smp,
//	      	       string type)
//{
//
//          cout<<"nsites ="<<nsites<<endl;	
//	  int 			 size=pow(2,nsites);
//	  std::vector<double>    wf;
//	  load_wf(wffile,wf);
//	  cout<<"wf.size()="<<wf.size()<<endl;
//	  RMatrix corr_fn(nsites,nsites);	  
//	  cout<<"Computing Correlation function of type "<<type<<" ....."<<endl;
//	  if (type==string("S+S-")) 
//	  {
//		cout<<"Computing here S+S-....."<<endl;
//		compute_spsm(nsites,wf,corr_fn);
//	  }
//	  if (type==string("SzSz")) 
//	  {
//		cout<<"Computing here SzSz....."<<endl;
//		compute_szsz(nsites,wf,corr_fn);
//	  }
//	  cout<<"=============================================================================================="<<endl;
//	  cout<<"   Displaying Correlation function "<<type<<endl;
//	  cout<<"=============================================================================================="<<endl;
//	  for (int i=0;i<nsites;i++)
//	  {
//		for (int j=0;j<nsites;j++) cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %j<<" "<<boost::format("%+5.10f") %corr_fn(i,j)<<endl;
//	  }
//}
//
//////////////////////////////////////////////////////////////////////////////
void lanczos_sym(Ham &h, Simulation_Params &sp, std::vector<double> &eigs)
{
   ofstream outfile;
   const char *cstr = (sp.outfile).c_str();
   outfile.open(cstr, ofstream::app);

   std::vector<int> t1=h.t1;
   std::vector<int> t2=h.t2;
   std::vector<int> t3=h.t3;
   double k1=h.k1;
   double k2=h.k2;
   double k3=h.k3;
   int 							nsites=h.num_sites;
   std::string 						analysis=sp.analysis;
   int 							iterations=sp.iterations;
   int 							num_cycles=sp.num_cycles;
   int64_t 						fhilbert=pow(int64_t(2),int64_t(nsites));
   int64_t 						hilbert=0;
   complex<double> 					beta; 
   std::vector< complex<double> >			alphas,betas;
   Matrix 						t_mat,t_eigenvecs;
   std::vector<int64_t>                             	spin_dets;
   //
   int L1,L2,L3;
   get_L1L2L3(t1,t2,t3,L1,L2,L3);
   double pi=3.141592653589793238462643383;
   double twopi=2.00*pi;
   k1=(twopi*k1)/double(L1);
   k2=(twopi*k2)/double(L2);
   k3=(twopi*k3)/double(L3);
   outfile<<"N sites = "<<nsites<<endl;
   outfile<<"L1,L2,L3 = "<<L1<<"  "<<L2<<"  "<<L3<<endl;
   outfile<<"k1,k2,k3 = "<<k1<<"  "<<k2<<"  "<<k3<<endl;
   outfile<<"Full hilbert space ="<<fhilbert<<endl;
   //
   std::vector< std::vector<int> > maps;
   std::vector< complex<double>  > characters;
   std::vector< char> reps(fhilbert);
   std::vector< int64_t> locreps(fhilbert);
   std::vector< int64_t> ireps(fhilbert);
   std::vector< char> norms(fhilbert);
  
   // Make maps, character, basis 
   make_maps_and_characters(L1,L2,L3,t1,t2,t3,k1,k2,k3,maps,characters);
   make_basis(maps,characters,reps,locreps,ireps,norms,spin_dets,hilbert);
   outfile<<"Symmetrized hilbert space ="<<hilbert<<endl;

   std::vector< complex<double> >	w(hilbert),v_p(hilbert),v_o(hilbert);
   outfile.flush(); 
   // Initialize
   initialize_Lanczos_vectors(v_p,v_o,w);

   // Start, initialize alpha, beta 
   outfile<<"Starting Lanczos (complex iterations)..."<<endl; 
   outfile.flush(); 

   // Finite temperature algorithm 
   if (analysis=="finiteT")
   { 
	   initialize_alphas_betas(alphas,betas,beta);  // initial beta will be zero
	   iterations=min(iterations,int(hilbert));
	   outfile<<"(Finite temp) Iterations = "<<iterations<<endl;
	   for (int it=0;it<iterations;it++)
	   {
	       time_t start;
	       time (&start);
	       // Act H on v  to get w
	       actHonv(h,spin_dets,characters,reps,locreps,ireps,norms,v_p,w);
	       // Lanczos orthogonalization, update beta and also the alphas and betas arrays. Also update v_p, v_o, w 
	       Lanczos_step_update(beta, alphas, betas, v_p, v_o, w); // use the current beta and then update it 
	       time_t end;
	       time (&end);
	       double dif=difftime(end,start);
	       outfile<<"Time to perform Lanczos iteration "<<it<<" was "<<dif<<" seconds"<<endl;
	       outfile<<"================================================================="<<endl;
	       outfile.flush();
	      
	       // Make tridiagonal matrix and diagonalize every iteration 
	       if (it %1 ==0) 
	       {
		       make_tridiagonal_matrix_and_diagonalize(it+1,alphas,betas,t_mat,eigs,t_eigenvecs);
		       outfile<<boost::format("#, Iteration, Lowest eigenvalue %3i %+.15f") %it %eigs[0]<<endl;
		       for (int ne=0;ne<eigs.size();ne++) outfile<<boost::format("Energy = %+.15f Matrix el = %+.15f , %+.15f") %eigs[ne] %real(t_eigenvecs(0,ne)) %imag(t_eigenvecs(0,ne))<<endl;
	       }
	   }
   }
   // Ground state algorithm using two Krylov vectors only 
   if (analysis=="gs")
   {
	   std::vector< complex<double> > v_n(hilbert);
	   //num_cycles=min(num_cycles,int(hilbert));
	   outfile<<"(GS) num_cycles = "<<num_cycles<<endl;
	   for (int cycle=0;cycle<num_cycles;cycle++)
	   {
	       initialize_alphas_betas(alphas,betas,beta);
	       // Act H on v_p  to get w
	       actHonv(h,spin_dets,characters,reps,locreps,ireps,norms,v_p,w);
	       complex<double> alpha0=conj(zdotc(hilbert,w,v_p));
	       alphas.push_back(alpha0);

	       // Use the w to obtain v_o 
	       zaxpyk(hilbert,-alpha0,v_p,w);
	       beta=sqrt(zdotc(hilbert,w,w));
	       betas.push_back(beta);
	       equatek(w,v_o);
	       zscalk(hilbert,1.0/beta,v_o);
	       zscalk(hilbert,0.0,w); // Done with w, set it zero so that it can be used the next time

	       // Act H on v_o  to get w and alpha1
	       actHonv(h,spin_dets,characters,reps,locreps,ireps,norms,v_o,w);
       	       zaxpyk(hilbert,-beta,v_p,w);
	       complex<double> alpha1=conj(zdotc(hilbert,w,v_o));
	       alphas.push_back(alpha1);


	       // Use the w to obtain v_n
       	       zaxpyk(hilbert,-alpha1,v_o,w);
	       beta=sqrt(zdotc(hilbert,w,w));
	       betas.push_back(beta);
	       equatek(w,v_n);
	       zscalk(hilbert,1.0/beta,v_n);
	       zscalk(hilbert,0.0,w); // Done with w - set it to zero so that it can be used the next time
	       
	       // Act H on v_n  to get w and alpha2
	       actHonv(h,spin_dets,characters,reps,locreps,ireps,norms,v_n,w);
	       complex<double> alpha2=conj(zdotc(hilbert,w,v_o));
	       alphas.push_back(alpha2);

	      
	       //outfile<<"Overlap = "<<zdotc(hilbert,v_o,v_p)<<endl;

	       // Make 3x3 matrix and diagonalize
	       make_tridiagonal_matrix_and_diagonalize(3,alphas,betas,t_mat,eigs,t_eigenvecs);
	       outfile<<boost::format("#, Cycle, Lowest eigenvalue %3i %+.15f") %cycle %eigs[0]<<endl;
	       for (int ne=0;ne<eigs.size();ne++) outfile<<boost::format("Energy = %+.15f Matrix el = %+.15f , %+.15f") %eigs[ne] %real(t_eigenvecs(0,ne)) %imag(t_eigenvecs(0,ne))<<endl;
	       outfile.flush();

	       // Now find the linear combination of v_p and v_o and v_n that minimizes the energy. Call this linear combination v_p and cycle
	       complex<double> c_p=t_eigenvecs(0,0);
	       complex<double> c_o=t_eigenvecs(1,0);
	       complex<double> c_n=t_eigenvecs(2,0);
	       //outfile<<"Norm = "<<conj(c_p)*c_p + conj(c_o)*c_o<<endl;
	       # pragma omp parallel for
	       for (int64_t i=0;i<hilbert; i++)
	       {
			w[i]=(c_p*v_p[i]) + (c_o*v_o[i]) + (c_n*v_n[i]); 
	       }
	       equatek(w,v_p);  // In principle, need to normalize as transformation is unitary. Save the lowest energy vector as v_p
	       normalize(v_p);  // Seems like we need this else we have issues of roundoff
	       //zscalk(hilbert,0.0,v_o); // Done with v_o - set it to zero so that it can be used the next time
	       //zscalk(hilbert,0.0,v_n); // Done with v_n - set it to zero so that it can be used the next time
	       zscalk(hilbert,0.0,w); // Done with w - set it to zero so that it can be used the next time
	       if (abs(abs(c_p)-1)<1.0e-14 and abs(abs(c_o)-0)<1.0e-14) cycle=num_cycles;
	   }
	   outfile.close();
	   w.clear();
	   v_o.clear();
	   v_n.clear();
	   // Measurements on the ground state
	   perform_one_spin_measurements(v_p, spin_dets, maps, characters, reps, locreps, ireps, norms, sp);
	   perform_two_spin_measurements(v_p, spin_dets, maps, characters, reps, locreps, ireps, norms, sp);
   }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void perform_one_spin_measurements(std::vector< complex<double> > &vec, 
		std::vector<int64_t> &spin_dets, 
   		std::vector< std::vector<int> > &maps,
	        std::vector<complex<double> >  &characters,
		std::vector< char> &reps, std::vector< int64_t> &locreps, 
		std::vector< int64_t> &ireps, std::vector< char> &norms, 
		Simulation_Params &sp)
{
	int64_t hilbert=spin_dets.size();
   	int     nsites=maps[0].size();
   	std::vector<complex<double> > w(hilbert);
   	std::vector<complex<double> > sx(nsites), sy(nsites), sz(nsites);
        ofstream outfile;
        const char *cstr = (sp.outfile).c_str();
        outfile.open(cstr, ofstream::app);
	std::vector< std::vector<int> > groups;

	// Make groups of sites related by symmetry
	for (int m=0;m<nsites;m++)
	{
		std::vector<int> setofsymsites;
		for (int i=0;i<maps.size();i++) {setofsymsites.push_back(maps[i][m]);}  // All symmetry related sites
		if (*min_element(setofsymsites.begin(), setofsymsites.end()) == m)
		{
			groups.push_back(setofsymsites);
			outfile<<"Group "<<groups.size()<<": ";
			for (int n=0;n<setofsymsites.size();n++) outfile<<setofsymsites[n]<<",";
			outfile<<endl;
		}	
	}	
	int numgroups=groups.size();

	for (int which=0;which<3;which++)
	{	
		for (int m=0;m<numgroups;m++)
		{
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			zscalk(hilbert,0.0,w);
			#pragma omp parallel for	
			for (int64_t i=0;i<hilbert;i++) // Computing Sx/y/z*vec - This is the bulk of the operation
			{
				std::vector<int64_t> 	      new_spin_dets;
				std::vector<complex<double> > hints;
				int64_t orig=spin_dets[i];
				int nconns;
				if (which==0) {symmetrized_sx(maps,groups[m],orig,new_spin_dets,hints, nconns);}
				if (which==1) {symmetrized_sy(maps,groups[m],orig,new_spin_dets,hints, nconns);}
				if (which==2) {symmetrized_sz(maps,groups[m],orig,new_spin_dets,hints, nconns);}
				int repeatket=norms[orig]-'0';
				complex<double> hint;
				double invrepeatket=sqrt(1.0/double(repeatket));
				for (int j=0;j<nconns;j++)  // Connections to state
				{
					int64_t news=new_spin_dets[j];
					char normnews=norms[news];
					if (normnews!='0') // an allowed state 
					{
						// Works only when reps = 0- 9 
						int repeat=normnews-'0'; //will be 0 if normnews='0'
						int op=reps[news]-'0'; // if not allowed it will be 0
						hint=hints[j]*(characters[op])*sqrt(double(repeat))*invrepeatket;
						news=ireps[news]; // faster as rep is stored
						int64_t location=locreps[news];
						w[i]+=(conj(hint)*vec[location]);
					}
				 }
			}
			if (which==0) outfile<<"Finished Computing SxV for group m = "<<m<<endl;
			if (which==1) outfile<<"Finished Computing SyV for group m = "<<m<<endl;
			if (which==2) outfile<<"Finished Computing SzV for group m = "<<m<<endl;
			complex<double> value=zdotc(hilbert,vec,w)/double(groups[m].size());
			for (int ind=0;ind<groups[m].size();ind++) 
			{
				int site=groups[m][ind];
				if (which==0) sx[site]=value;
				if (which==1) sy[site]=value;
				if (which==2) sz[site]=value;
			}
	       }
	}	
	outfile<<"==========================================================================="<<endl;
	outfile<<"  m           Smx               Smy                 Smz                     "<<endl;
	outfile<<"==========================================================================="<<endl;
	for (int m=0;m<nsites;m++)
	{
		outfile<<boost::format("%3i  %+.15f  %+.15f  %+.15f") %m %real(sx[m]) %real(sy[m]) %real(sz[m])<<endl;
	}
	outfile.close();
}

/////////////////////////////////////////////////////////////////////////////////////
void perform_two_spin_measurements(std::vector< complex<double> > &vec, 
		std::vector<int64_t> &spin_dets, 
   		std::vector< std::vector<int> > &maps,
	        std::vector<complex<double> >  &characters,
		std::vector< char> &reps, std::vector< int64_t> &locreps, 
		std::vector< int64_t> &ireps, std::vector< char> &norms, 
		Simulation_Params &sp)
{
	int64_t hilbert=spin_dets.size();
   	int     nsites=maps[0].size();
   	std::vector<complex<double> > w(hilbert);
   	Matrix sxsx(nsites,nsites), sxsy_plus_sysx(nsites,nsites), sxsz_plus_szsx(nsites,nsites), sysy(nsites,nsites), sysz_plus_szsy(nsites,nsites), szsz(nsites,nsites);
        ofstream outfile;
        const char *cstr = (sp.outfile).c_str();
        outfile.open(cstr, ofstream::app);
	std::vector< std::vector< std::vector<int> > > groups;

	// Make groups of sites related by symmetry
	for (int m=0;m<nsites;m++)
	{
		for (int n=0;n<nsites;n++)
		{
			bool add=true;
			std::vector< std::vector<int> > setofsymbonds;
			for (int i=0;i<maps.size();i++) 
			{
				std::vector<int> bond;
				int mnew=maps[i][m];
				int nnew=maps[i][n];
				bond.push_back(maps[i][m]);
				bond.push_back(maps[i][n]);
				bool addbond=true;
				for (int l=0;l<setofsymbonds.size();l++)
				{
					if ((bond[0]==setofsymbonds[l][0] and bond[1]==setofsymbonds[l][1]) or (bond[0]==setofsymbonds[l][1] and bond[1]==setofsymbonds[l][0]))
					{
						addbond=false;
					}
				}
				if (addbond) {setofsymbonds.push_back(bond);}
				if (mnew<m) {add=false; i=maps.size();} // This has been accounted elsewhere
				// If bond is present in a previous group then do not add this group
				for (int j=0;j<groups.size();j++)
				{
					for (int k=0;k<groups[j].size();k++)
					{
						if ( (bond[0]==groups[j][k][0] and bond[1]==groups[j][k][1]) or (bond[0]==groups[j][k][1] and bond[1]==groups[j][k][0]))
						{
							k=groups[j].size();
							//j=groups.size();
							//i=maps.size();
							add=false;
						}
					}
					if (add==false) {j=groups.size();i=maps.size();}
				}
			}  

			if (add) 
			{
				for (int i=0;i<setofsymbonds.size();i++) {outfile<<"["<<setofsymbonds[i][0]<<","<<setofsymbonds[i][1]<<"],";}
				outfile<<endl;
				groups.push_back(setofsymbonds);

			}
		}	
	}	

	int numbonds=0;
	int numgroups=groups.size();
	for (int m=0;m<groups.size();m++)
	{	
		numbonds+=groups[m].size();
	}
	outfile<<"Numgroups = "<<numgroups<<endl;
	outfile<<"Numbonds  = "<<numbonds<<endl;

	for (int which=0;which<6;which++)
	{
		for (int m=0;m<numgroups;m++)
		{
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			zscalk(hilbert,0.0,w);
			#pragma omp parallel for	
			for (int64_t i=0;i<hilbert;i++) // Computing Sx/y/z Sx/y/z*vec - This is the bulk of the operation
			{
				std::vector<int64_t> 	      new_spin_dets;
				std::vector<complex<double> > hints;
				int64_t orig=spin_dets[i];
				int nconns;
				if (which==0) symmetrized_sxsx(maps,groups[m],orig,new_spin_dets,hints, nconns); 
				if (which==1) symmetrized_sxsy_plus_sysx(maps,groups[m],orig,new_spin_dets,hints, nconns); 
				if (which==2) symmetrized_sxsz_plus_szsx(maps,groups[m],orig,new_spin_dets,hints, nconns); 
				if (which==3) symmetrized_sysy(maps,groups[m],orig,new_spin_dets,hints, nconns); 
				if (which==4) symmetrized_sysz_plus_szsy(maps,groups[m],orig,new_spin_dets,hints, nconns); 
				if (which==5) symmetrized_szsz(maps,groups[m],orig,new_spin_dets,hints, nconns); 
				int repeatket=norms[orig]-'0';
				complex<double> hint;
				double invrepeatket=sqrt(1.0/double(repeatket));
				for (int j=0;j<nconns;j++)  // Connections to state
				{
					int64_t news=new_spin_dets[j];
					char normnews=norms[news];
					if (normnews!='0') // an allowed state 
					{
						// Works only when reps = 0- 9 
						int repeat=normnews-'0'; //will be 0 if normnews='0'
						int op=reps[news]-'0'; // if not allowed it will be 0
						hint=hints[j]*(characters[op])*sqrt(double(repeat))*invrepeatket;
						news=ireps[news]; // faster as rep is stored
						int64_t location=locreps[news];
						w[i]+=(conj(hint)*vec[location]);
					}
				 }
			}
			if (which==0) outfile<<"Finished Computing SxSxV for group m = "<<m<<endl;
			if (which==1) outfile<<"Finished Computing (SxSy + SySx) V for group m = "<<m<<endl;
			if (which==2) outfile<<"Finished Computing (SxSz + SzSx) V for group m = "<<m<<endl;
			if (which==3) outfile<<"Finished Computing SySyV for group m = "<<m<<endl;
			if (which==4) outfile<<"Finished Computing (SySz + SzSy) V for group m = "<<m<<endl;
			if (which==5) outfile<<"Finished Computing SzSzV for group m = "<<m<<endl;
			complex<double> value=zdotc(hilbert,vec,w)/double(groups[m].size());
			for (int ind=0;ind<groups[m].size();ind++) 
			{
				int site1=groups[m][ind][0];
				int site2=groups[m][ind][1];
				if (which==0) {sxsx(site1,site2)=value; sxsx(site2,site1)=value;}
				if (which==1) {sxsy_plus_sysx(site1,site2)=value; sxsy_plus_sysx(site2,site1)=value;}
				if (which==2) {sxsz_plus_szsx(site1,site2)=value; sxsz_plus_szsx(site2,site1)=value;}
				if (which==3) {sysy(site1,site2)=value; sysy(site2,site1)=value;}
				if (which==4) {sysz_plus_szsy(site1,site2)=value; sysz_plus_szsy(site2,site1)=value;}
				if (which==5) {szsz(site1,site2)=value; szsz(site2,site1)=value;}
			}
		}
	}

	outfile<<"========================================================================================================================================================================="<<endl;
	outfile<<" m     n         SmxSnx         SmxSny+SmySnx         SmxSnz+SmzSnx         SmySny          SmySnz+SmzSny           SmzSnz    "<<endl;
	outfile<<"========================================================================================================================================================================="<<endl;
	for (int m=0;m<nsites;m++)
	{
		for (int n=0;n<nsites;n++)
		{	
			outfile<<boost::format("%3i  %3i  %+.15f  %+.15f  %+.15f  %+.15f  %+.15f  %+.15f") %m %n %real(sxsx(m,n)) %real(sxsy_plus_sysx(m,n)) %real(sxsz_plus_szsx(m,n)) %real(sysy(m,n)) %real(sysz_plus_szsy(m,n)) %real(szsz(m,n))<<endl;
		}
	}
	outfile.close();
}

/*
////////////////////////////////////////////////////////////////////////////////////////////
void perform_one_spin_measurements(std::vector< complex<double> > &vec, 
		std::vector<int64_t> &spin_dets, 
   		std::vector< std::vector<int> > &maps,
	        std::vector<complex<double> >  &characters,
		std::vector< char> &reps, std::vector< int64_t> &locreps, 
		std::vector< int64_t> &ireps, std::vector< char> &norms, 
		Simulation_Params &sp)
{
   ofstream outfile;
   const char *cstr = (sp.outfile).c_str();
   outfile.open(cstr, ofstream::app);

   int64_t hilbert=vec.size();
   int nsites=maps[0].size();
   std::vector<complex<double> > sx(nsites), sy(nsites), sz(nsites);
   int numsyms=maps.size();
   for (int64_t i=0;i<hilbert;i++)
   {
   	  if (i%100000 == 0) {outfile<< i << endl;}
	  int repeatket=norms[spin_dets[i]]-'0';
	  double factor1=1.0/sqrt(double(numsyms)*double(repeatket));
	  for (int j=0;j<numsyms;j++)
	  {
		int64_t translatedstate1=spin_dets[i];
		translateT(translatedstate1,maps[j], nsites);
		complex<double> character1=characters[j]*factor1;
		#pragma omp parallel for
		for (int m=0;m<nsites;m++)
		{
                         	std::vector<int64_t> newstatesx, newstatesy, newstatesz;
                         	std::vector< complex<double> > hints_listx, hints_listy, hints_listz;
				calc_hints_sx(1.0,m,translatedstate1,newstatesx,hints_listx);
				calc_hints_sy(1.0,m,translatedstate1,newstatesy,hints_listy);
				calc_hints_sz(1.0,m,translatedstate1,newstatesz,hints_listz);
				for (int k=0;k<newstatesx.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesx[k]; int64_t irep=ireps[newstate]; int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sx[m]+=(character1*character2*hints_listx[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstatesy.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesy[k]; int64_t irep=ireps[newstate]; int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sy[m]+=(character1*character2*hints_listy[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstatesz.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesz[k]; int64_t irep=ireps[newstate]; int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sz[m]+=(character1*character2*hints_listz[k]*vec[i]*conj(vec[loc]));	
					}
				}
		}
	  }	
   }

   for (int m=0;m<nsites;m++)
   {
   	outfile<<boost::format("# m, Smx, Smy, Smz %3i %+.15f %+.15f %+.15f") %m %sx[m] %sy[m] %sz[m]<<endl;
   }
}
////////////////////////////////////////////////////////////////////////////////////////////
void perform_two_spin_measurements(std::vector< complex<double> > &vec, 
		std::vector<int64_t> &spin_dets, 
   		std::vector< std::vector<int> > &maps,
	        std::vector<complex<double> >  &characters,
		std::vector< char> &reps, std::vector< int64_t> &locreps, 
		std::vector< int64_t> &ireps, std::vector< char> &norms, 
		Simulation_Params &sp)
{
   ofstream outfile;
   const char *cstr = (sp.outfile).c_str();
   outfile.open(cstr, ofstream::app);

   int64_t hilbert=vec.size();
   int nsites=maps[0].size();
   Matrix sxsx(nsites,nsites), sxsy(nsites,nsites), sxsz(nsites,nsites), sysy(nsites,nsites), sysz(nsites,nsites), szsz(nsites,nsites);
   int numsyms=maps.size();
   for (int64_t i=0;i<hilbert;i++)
   {
	  int repeatket=norms[spin_dets[i]]-'0';
	  double factor1=1.0/sqrt(double(numsyms)*double(repeatket));
	  for (int j=0;j<numsyms;j++)
	  {
		int64_t translatedstate1=spin_dets[i];
		translateT(translatedstate1,maps[j], nsites);
		complex<double> character1=characters[j]*factor1;
		#pragma omp parallel for
		for (int m=0;m<nsites;m++)
		{
			for (int n=0;n<nsites;n++)
			{
                         	std::vector<int64_t> newstatesxx, newstatesxy, newstatesxz, newstatesyy, newstatesyz, newstateszz;
                         	std::vector< complex<double> > hints_listxx, hints_listxy, hints_listxz, hints_listyy, hints_listyz, hints_listzz;
				calc_hints_sxsx(1.0,m,n,translatedstate1,newstatesxx,hints_listxx);
				calc_hints_sxsy(1.0,m,n,translatedstate1,newstatesxy,hints_listxy);
				calc_hints_sxsz(1.0,m,n,translatedstate1,newstatesxz,hints_listxz);
				calc_hints_sysy(1.0,m,n,translatedstate1,newstatesyy,hints_listyy);
				calc_hints_sysz(1.0,m,n,translatedstate1,newstatesyz,hints_listyz);
				calc_hints_szsz(1.0,m,n,translatedstate1,newstateszz,hints_listzz);
				for (int k=0;k<newstatesxx.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesxx[k]; int64_t irep=ireps[newstate]; int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sxsx(m,n)+=(character1*character2*hints_listxx[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstatesxy.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesxy[k]; int64_t irep=ireps[newstate];int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sxsy(m,n)+=(character1*character2*hints_listxy[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstatesxz.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesxz[k]; int64_t irep=ireps[newstate];int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sxsz(m,n)+=(character1*character2*hints_listxz[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstatesyy.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesyy[k]; int64_t irep=ireps[newstate]; int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sysy(m,n)+=(character1*character2*hints_listyy[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstatesyz.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstatesyz[k]; int64_t irep=ireps[newstate];int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						sysz(m,n)+=(character1*character2*hints_listyz[k]*vec[i]*conj(vec[loc]));	
					}
				}
				for (int k=0;k<newstateszz.size();k++)
				{
					// get representative of the new state
					int64_t newstate=newstateszz[k]; int64_t irep=ireps[newstate]; int64_t loc=locreps[irep];
					// Now see which symm operations on this rep give the new state
					complex<double> character2=0.0;
					for (int j2=0;j2<numsyms;j2++)
					{
						int64_t translatedstate2=irep;
						translateT(translatedstate2,maps[j2], nsites);
						if (translatedstate2==newstate) {character2+=characters[j2];}	
					}
					if (norms[irep]!='0')
					{
	  					int repeatbra=norms[irep]-'0';
	  					character2=conj(character2)/sqrt(double(numsyms)*double(repeatbra));
						szsz(m,n)+=(character1*character2*hints_listzz[k]*vec[i]*conj(vec[loc]));	
					}
				}
			}
		}
	  }	
   }

   for (int m=0;m<nsites;m++)
   {
	for (int n=0;n<nsites;n++)
	{	
   		outfile<<boost::format("# m,n, SmxSnx, SmxSny, SmxSnz, SmySny, SmySnz, SmzSnz %3i %3i %+.15f %+.15f %+.15f %+.15f %+.15f %+.15f") %m %n %sxsx(m,n) %sxsy(m,n) %sxsz(m,n) %sysy(m,n) %sysz(m,n) %szsz(m,n)<<endl;
	}
   }
}
*/
