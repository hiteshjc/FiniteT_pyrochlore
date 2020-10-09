#include"ed.h"
#include"printing_functions.h"
#include<omp.h>

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


/////////////////////////////////////////////////////////////////////////////////////////
void load_wf_get_corrs(int nsites, 
		       string wffile, 
		       Simulation_Params &smp,
	      	       string type)
{

          cout<<"nsites ="<<nsites<<endl;	
	  int 			 size=pow(2,nsites);
	  std::vector<double>    wf;
	  load_wf(wffile,wf);
	  cout<<"wf.size()="<<wf.size()<<endl;
	  RMatrix corr_fn(nsites,nsites);	  
	  cout<<"Computing Correlation function of type "<<type<<" ....."<<endl;
	  if (type==string("S+S-")) 
	  {
		cout<<"Computing here S+S-....."<<endl;
		compute_spsm(nsites,wf,corr_fn);
	  }
	  if (type==string("SzSz")) 
	  {
		cout<<"Computing here SzSz....."<<endl;
		compute_szsz(nsites,wf,corr_fn);
	  }
	  cout<<"=============================================================================================="<<endl;
	  cout<<"   Displaying Correlation function "<<type<<endl;
	  cout<<"=============================================================================================="<<endl;
	  for (int i=0;i<nsites;i++)
	  {
		for (int j=0;j<nsites;j++) cout<<boost::format("%3i") % i<<" "<<boost::format("%3i") %j<<" "<<boost::format("%+5.10f") %corr_fn(i,j)<<endl;
	  }
}

//////////////////////////////////////////////////////////////////////////////
void lanczos_sym(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs)
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
   outfile<<"N sites = "<<nsites<<endl;
   int   						how_many_eigenvecs=sp.how_many_eigenvecs;
   int 							iterations=sp.iterations;
   int 							num_cycles=sp.num_cycles;
   int64_t 						fhilbert=pow(int64_t(2),int64_t(nsites));
   std::vector< complex<double> >			alphas,betas;
   Matrix 						t_mat,t_eigenvecs;
   std::vector<int64_t>                             	spin_dets;
   int ctr=0; 
   //
   int L1,L2,L3;
   get_L1L2L3(t1,t2,t3,L1,L2,L3);
   double pi=3.141592653589793238462643383;
   double twopi=2.00*pi;
   k1=(twopi*k1)/double(L1);
   k2=(twopi*k2)/double(L2);
   k3=(twopi*k3)/double(L3);
   outfile<<"L1,L2,L3 = "<<L1<<"  "<<L2<<"  "<<L3<<endl;
   outfile<<"k1,k2,k3 = "<<k1<<"  "<<k2<<"  "<<k3<<endl;
   outfile<<"Full hilbert space ="<<fhilbert<<endl;
   //
   //
   int64_t hilbert=0;
   std::vector< std::vector<int> > maps;
   std::vector< complex<double>  > characters;
   make_maps_and_characters(L1,L2,L3,t1,t2,t3,k1,k2,k3,maps,characters);
   std::vector< char> reps(fhilbert);
   std::vector< int64_t> locreps(fhilbert);
   std::vector< int64_t> ireps(fhilbert);
   std::vector< char> norms(fhilbert);
  
   #pragma omp parallel for 
   for (int64_t i=0;i<fhilbert;i++)
   {
	//char crep;
	int64_t irep;
	//get_representative(maps, characters, i, irep, reps[i], norms[i]);
	get_representative(maps, characters, i, ireps[i], reps[i], norms[i]);
   }
   
   for (int64_t i=0;i<fhilbert;i++)
   {
	//outfile<<" i  = "<<i<<" norms = "<<norms[i]<<endl;
	if (reps[i]=='0' and norms[i]!='0') 
	{
		//outfile<<" i  = "<<i<<" norms = "<<norms[i]<<endl;
		spin_dets.push_back(i);
		locreps[i]=hilbert;
		hilbert+=1;
	}
   }
   //int hilbert=spin_dets.size();
   outfile<<"Symmetrized hilbert space ="<<hilbert<<endl;
   //return;

   std::vector< complex<double> >	w(hilbert),v_p(hilbert),v_o(hilbert);
   outfile.flush(); 
   //Initialization......
   for (int64_t i=0;i<hilbert;i++)
   {
		v_p[i]=complex<double> (2.0*uniform_rnd()-1.0, 2.0*uniform_rnd()-1.0);
		v_o[i]=0.0;w[i]=0.0;
   }
   
   // Start 
   outfile<<"Starting Lanczos (complex iterations)..."<<endl; 
   outfile.flush(); 
   normalize(v_p); 
   betas.clear();alphas.clear();  
   complex<double> beta=0.0;betas.push_back(beta);
   ctr=0;
   iterations=min(iterations,int(hilbert));
   outfile<<"Iterations = "<<iterations<<endl;
   for (int it=0;it<iterations;it++)
   {
       time_t start;
       time (&start);
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
				//hints[j]=hints[j]*(characters[op]*normket)/norm;
				// Symmetrized Hamiltonian matrix element from unsymmetrized number
				//cout<<repeatket<<"  "<<repeat<<endl;
				//hint=hints[j]*(characters[op]*sqrt(double(repeat)/double(repeatket)));
				hint=hints[j]*(characters[op])*sqrt(double(repeat))*invrepeatket;
				//outfile<<"hint = "<<hint<<endl;
				//translateT(news,maps[op],nsites); // after this news becomes its representative
				news=ireps[news]; // faster as rep is stored
				//int64_t location=findstate(news,spin_dets);
				int64_t location=locreps[news];
				//outfile<<"location = "<<location<<endl;
				//if (location!=-1) w[i]+=(hint*v_p[location]);
				w[i]+=(hint*v_p[location]);
			}
		 }
       }
       //cout<<"Finished Computing HV"<<endl;
       ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	       
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
       time_t end;
       time (&end);
       double dif=difftime(end,start);
       outfile<<"Time to perform Lanczos iteration "<<it<<" was "<<dif<<" seconds"<<endl;
       outfile<<"================================================================="<<endl;
       outfile.flush();
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
	       symmetric_diagonalize(t_mat,eigs,t_eigenvecs);
	       outfile<<boost::format("#, Iteration, Lowest eigenvalue %3i %+.15f") %it %eigs[0]<<endl;
	       for (int ne=0;ne<eigs.size();ne++) outfile<<boost::format("Energy = %+.15f Matrix el = %+.15f , %+.15f") %eigs[ne] %real(t_eigenvecs(0,ne)) %imag(t_eigenvecs(0,ne))<<endl;

	       //for (int ne=0;ne<eigs.size();ne++) outfile<<boost::format("%+.15f") %eigs[ne]<<endl;
	}
   }

}

////////////////////////////////////////////////////////////////////////////////////
void lanczos_sym_evec(Ham &h,
             Simulation_Params &sp, 
             std::vector<double> &eigs,
             std::vector< std::vector< complex<double> > > &eigenvecs)
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
   outfile<<"N sites = "<<nsites<<endl;
   int   						how_many_eigenvecs=sp.how_many_eigenvecs;
   int 							iterations=sp.iterations;
   int 							num_cycles=sp.num_cycles;
   int64_t 						fhilbert=pow(int64_t(2),int64_t(nsites));
   std::vector< complex<double> >			alphas,betas;
   Matrix 						t_mat,t_eigenvecs;
   std::vector<int64_t>                             	spin_dets;
   int ctr=0; 
   //
   int L1,L2,L3;
   get_L1L2L3(t1,t2,t3,L1,L2,L3);
   double pi=3.141592653589793238462643383;
   double twopi=2.00*pi;
   k1=(twopi*k1)/double(L1);
   k2=(twopi*k2)/double(L2);
   k3=(twopi*k3)/double(L3);
   outfile<<"L1,L2,L3 = "<<L1<<"  "<<L2<<"  "<<L3<<endl;
   outfile<<"k1,k2,k3 = "<<k1<<"  "<<k2<<"  "<<k3<<endl;
   outfile<<"Full hilbert space ="<<fhilbert<<endl;
   //
   //
   int64_t hilbert=0;
   std::vector< std::vector<int> > maps;
   std::vector< complex<double>  > characters;
   make_maps_and_characters(L1,L2,L3,t1,t2,t3,k1,k2,k3,maps,characters);
   std::vector< char> reps(fhilbert);
   std::vector< int64_t> locreps(fhilbert);
   std::vector< int64_t> ireps(fhilbert);
   std::vector< char> norms(fhilbert);
  
   #pragma omp parallel for 
   for (int64_t i=0;i<fhilbert;i++)
   {
	//char crep;
	int64_t irep;
	get_representative(maps, characters, i, ireps[i], reps[i], norms[i]);
   }
   
   for (int64_t i=0;i<fhilbert;i++)
   {
	//outfile<<" i  = "<<i<<" norms = "<<norms[i]<<endl;
	if (reps[i]=='0' and norms[i]!='0') 
	{
		//outfile<<" i  = "<<i<<" norms = "<<norms[i]<<endl;
		spin_dets.push_back(i);
		locreps[i]=hilbert;
		hilbert+=1;
	}
   }
   //int hilbert=spin_dets.size();
   outfile<<"Symmetrized hilbert space ="<<hilbert<<endl;
   //return;

   std::vector< complex<double> >	w(hilbert),v_p(hilbert),v_o(hilbert);
   outfile.flush(); 
   //Initialization......
   for (int64_t i=0;i<hilbert;i++)
   {
		v_p[i]=complex<double> (2.0*uniform_rnd()-1.0, 0.0);
		v_o[i]=0.0;w[i]=0.0;
   }
   
   // Start 
   outfile<<"Starting Lanczos (complex iterations)..."<<endl; 
   outfile.flush(); 
   normalize(v_p); 
   betas.clear();alphas.clear();  
   complex<double> beta=0.0;betas.push_back(beta);
   ctr=0;
   iterations=min(iterations,int(hilbert));
   outfile<<"Iterations = "<<iterations<<endl;
   for (int it=0;it<iterations;it++)
   {
       time_t start;
       time (&start);
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
		outfile<<"repeatket = "<<repeatket<<endl;
		//outfile<<"new_spin_dets.size()="<<new_spin_dets.size()<<endl;
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
				//hints[j]=hints[j]*(characters[op]*normket)/norm;
				// Symmetrized Hamiltonian matrix element from unsymmetrized number
				//cout<<repeatket<<"  "<<repeat<<endl;
				//hint=hints[j]*(characters[op]*sqrt(double(repeat)/double(repeatket)));
				hint=hints[j]*(characters[op])*sqrt(double(repeat))*invrepeatket;
				//outfile<<"hint = "<<hint<<endl;
				//translateT(news,maps[op],nsites); // after this news becomes its representative
				news=ireps[news]; // faster as rep is stored
				//int64_t location=findstate(news,spin_dets);
				int64_t location=locreps[news];
				//outfile<<"location = "<<location<<endl;
				//if (location!=-1) w[i]+=(hint*v_p[location]);
				w[i]+=(hint*v_p[location]);
			}
		 }
       }
       //cout<<"Finished Computing HV"<<endl;
       ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
       complex<double> energy=conj(zdotc(hilbert,w,v_p));
       equatek(w,v_p);
       normalize(v_p);
       outfile<<boost::format("#, Iteration, Lowest eigenvalue %3i %+.15f") %it %energy<<endl;
   }
   for (int64_t i=0; i<hilbert; i++) outfile<<boost::format("%10i %+.15f") %spin_dets[i] %v_p[i]<<endl;
}


