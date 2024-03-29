#include"number_functions.h"
#include"matrix_functions.h"
#include"printing_functions.h"

using namespace std;

/////////////////////////////////////////////////////////////////////
void is_present(std::vector<int> &indices, std::vector< std::vector<int> > &maps, bool &present, int &where)
{
	where=-1;
	present=false;
	for (int i=0;i<maps.size();i++)
	{
		if (indices==maps[i]) {present=true; where=i; return;}
	}
	return;

}
/////////////////////////////////////////////////////////////////////
void act_map(std::vector<int> &indices, std::vector<int> &t)
{
	int nsites=t.size();
	std::vector<int> newindices(nsites);
	for (int i=0;i<nsites;i++)
	{
		newindices[i]=indices[t[i]];	
	}
	indices=newindices;
}

/////////////////////////////////////////////////////////////////////
void make_maps_and_characters(int L1, int L2, int L3, 
			      std::vector<int> &t1, 
			      std::vector<int> &t2, 
			      std::vector<int> &t3, 
			      double &k1, double &k2, double &k3, 
			      std::vector< std::vector<int> > &maps, 
			      std::vector<complex<double> >   &characters)
{
	int where;
	bool present; 

	maps.clear();
	characters.clear();
	int t1v=0;int t2v=0;int t3v=0;
	std::vector<int> indices;
	int nsites=t1.size();
	for (int i=0;i<nsites;i++) indices.push_back(i); // Identity transformation
	maps.push_back(indices);
	print_vec(indices);
	complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
	characters.push_back(character);
	
	for (int i=0;i<L1;i++)
	{
		if (i>0)
		{
			act_map(indices,t1);
			t1v=(t1v+1)%L1;
			is_present(indices,maps,present, where);
			if (present==false)
			{
				maps.push_back(indices);
				cout<<"t1,t2,t3 = "<<t1v<<"  "<<t2v<<"  "<<t3v<<"  map = ";
				print_vec(indices);
				complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
				characters.push_back(character);
			}
			/*else
			{
				complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
				characters[where]+=character;
			}*/
		}
		for (int j=0;j<L2;j++)
		{
			if (j>0)
			{
				act_map(indices,t2);	
				t2v=(t2v+1)%L2;
				is_present(indices,maps,present, where);
				if (present==false)
				{
					maps.push_back(indices);
					cout<<"t1,t2,t3 = "<<t1v<<"  "<<t2v<<"  "<<t3v<<"  map = ";
					print_vec(indices);
					complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
					characters.push_back(character);
				}
				/*else
				{
					complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
					characters[where]+=character;
				}*/
			}
			for (int k=0; k<L3; k++)
			{
			   if (k>0)
			   {
				act_map(indices,t3);
				t3v=(t3v+1)%L3;
				is_present(indices,maps,present, where);
				if (present==false)
				{
					maps.push_back(indices);
					cout<<"t1,t2,t3 = "<<t1v<<"  "<<t2v<<"  "<<t3v<<"  map = ";
					print_vec(indices);
					complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
					characters.push_back(character);
				}
				/*else
				{
					complex<double> character=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));
					characters[where]+=character;
				}*/
			   }
			}
		}
	}
	//cout<<"Number of symmetries = "<<maps.size()<<endl;
}


///////////////////////////////////////////////////////////////
void translateT(int64_t &state, std::vector<int> &t, int nsites)
{
	int64_t newstate=0;
	for (int n=0;n<nsites;n++)
	{
		if (btest64(state,n)==1) {newstate=ibset64(newstate,t[n]);}	
	}
	state=newstate;
}

///////////////////////////////////////////////////////////////
void get_L1L2L3(std::vector<int> &t1, 
		std::vector<int> &t2, 
		std::vector<int> &t3, 
		int &L1, int &L2, int &L3)
{
	int nsites=t1.size();
	L1=0;L2=0;L3=0;
	int64_t tmp_id = 1;
	do{
		 translateT(tmp_id, t1, nsites);
		 L1++;
	  }while(tmp_id != 1);

	tmp_id = 1;

	do{
		translateT(tmp_id, t2, nsites);
		L2++;
	}while(tmp_id != 1);

	tmp_id = 1;

	do{
		translateT(tmp_id, t3, nsites);
		L3++;
	}while(tmp_id != 1);

}

///////////////////////////////////////////////////////////////
void get_representative(std::vector< std::vector<int> >  &maps, 
			std::vector<complex<double> > 	 &characters,
			int64_t 		         &state, 
			int64_t 		         &repstate, 
			char 			 	 &rep, 
			char 			 	 &norm) 
{
  int ispecial=0;
  int numsyms=maps.size();
  std::vector<int64_t> newstates(numsyms);
  complex<double> normc=0.0;
  int nsites=maps[0].size();
  int repeat=0;
  repstate=state;
  //#pragma omp parallel for
  for (int i=0;i<numsyms;i++)
  {
  	newstates[i]=state;
	translateT(newstates[i],maps[i], nsites);
  }

  for (int i=0;i<numsyms;i++) 
  {
	if (newstates[i]<repstate) {repstate=newstates[i]; ispecial=i;} 
  }
  
  for (int i=0;i<numsyms;i++) 
  {
	if (newstates[i]==state) {normc+=characters[i];repeat+=1;} 
  }
  
  rep=(char)(ispecial+48);
  //cout<<"Rep = "<<rep<<endl;
  if (abs(normc)>1.0e-10) norm=(char)(repeat+48); 
  else {norm='0'; rep='x';}
  return;
}

///////////////////////////////////////////////////////////////
void get_representative(std::vector< std::vector<int> >  &maps, 
			std::vector<complex<double> > 	 &characters,
			int64_t 		         &state, 
			int64_t 			 &rep, 
			complex<double> &phase, complex<double> &norm)
{
  int ispecial=0;
  int numsyms=maps.size();
  //cout<<"numsyms = "<<numsyms<<endl;
  std::vector<int64_t> newstates(numsyms);
  norm=0.0;
  int nsites=maps[0].size();
  int repeat=0;
  rep=state;
  //#pragma omp parallel for
  for (int i=0;i<numsyms;i++)
  {
  	newstates[i]=state;
	translateT(newstates[i],maps[i], nsites);
  }

  for (int i=0;i<numsyms;i++) 
  {
	if (newstates[i]<rep) {rep=newstates[i]; ispecial=i;} 
  }
  
  for (int i=0;i<numsyms;i++) 
  {
	if (newstates[i]==state) {norm+=characters[i];repeat+=1;} 
  }
  phase=characters[ispecial];
  if (abs(norm)>1.0e-10) norm=sqrt(abs(double(numsyms)/double(repeat))); 
  else norm=0.0;
  return;
}


///////////////////////////////////////////////////////////////
void get_representative_and_ignore(std::vector< std::vector<int> >  &maps, 
			std::vector<complex<double> > 	 	    &characters,
			int64_t 				    &state, 
			int64_t 				    &rep, 
			complex<double> 			    &phase, 
			complex<double> 			    &norm, 
			bool 					    &ignore)
{
  int ispecial=0;
  int numsyms=maps.size();
  //cout<<"numsyms = "<<numsyms<<endl;
  std::vector<int64_t> newstates(numsyms);
  ignore=true;
  norm=0.0;
  int nsites=maps[0].size();
  int repeat=0;
  rep=state;
  //#pragma omp parallel for schedule(static)
  for (int i=0;i<numsyms;i++)
  {
  	int64_t newstate=state;
	translateT(newstate,maps[i], nsites);
	if (newstate<state) return;
	newstates[i]=newstate;
  }

  /*for (int i=0;i<numsyms;i++) 
  {
	if (newstates[i]<rep) {rep=newstates[i]; ispecial=i; return;} 
  }*/
  
  for (int i=0;i<numsyms;i++) 
  {
	if (newstates[i]==state) {norm+=characters[i];repeat+=1;} 
  }

  phase=characters[ispecial];
  if (abs(norm)>1.0e-10) {norm=sqrt(abs(double(numsyms)/double(repeat)));}
  else {norm=0.0;}
  
  ignore=false;
  return;
}


///////////////////////////////////////////////////////////////
void get_representative_and_ignore(int L1, int L2, int L3, 
			std::vector<int> &t1, 
			std::vector<int> &t2, 
			std::vector<int> &t3,
			double k1, double k2, double k3, 
			int64_t &state, int64_t &rep, 
			complex<double> &phase, 
			complex<double> &norm, 
			bool &ignore)
{
  ignore=1;
  //cout<<"L1,L2 "<<L1<<"  "<<L2<<endl;
  norm=1.0;
  int nsites=t1.size();
  rep=state;
  int64_t newstate=state;
  int t1v=0;
  int t2v=0;
  int t3v=0;
  int t1v_special=0;
  int t2v_special=0;
  int t3v_special=0;
  int repeat=1;
  for (int i=0;i<L1;i++)
  {
  	if (i>0)
	{
		translateT(newstate,t1, nsites);
		t1v=t1v+1;
		//cout<<"New state ="<<newstate<<endl;
		if (newstate<rep) {rep=newstate;t1v_special=t1v;t2v_special=t2v;t3v_special=t3v;return;}
		//if (newstate==state) {norm+=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));repeat+=1;}
	}
	for (int j=0;j<L2;j++)
	{
		if (j>0)
		{
		     translateT(newstate,t2, nsites);
		     t2v=t2v+1;
		     //cout<<"New state ="<<newstate<<endl;
		     if (newstate<rep) {rep=newstate;t1v_special=t1v;t2v_special=t2v;t3v_special=t3v;return;}
		     //if (newstate==state) {norm+=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));repeat+=1;}
		}
		for (int k=0;k<L3;k++)
		{
			if (k>0)
			{
			 translateT(newstate,t3, nsites);
			 t3v=t3v+1;
			 //cout<<"New state ="<<newstate<<endl;
			 if (newstate<rep) {rep=newstate;t1v_special=t1v;t2v_special=t2v;t3v_special=t3v;return;}
		     	 //if (newstate==state) {norm+=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));repeat+=1;}
			}
		}
	}
  }
  phase=exp(complex<double>(0, (-k1*double(t1v_special)) + (-k2*double(t2v_special)) + (-k3*double(t3v_special))));
  if (abs(norm)>1.0e-10) 
  {
  	norm=sqrt(abs(double(L1*L2*L3)/double(repeat))); 
  }
  else
  {
	norm=0.0;
  }
  ignore=0;
  return;
}

///////////////////////////////////////////////////////////////

void get_representative(int L1, int L2, int L3, 
			std::vector<int> &t1, std::vector<int> &t2, std::vector<int> &t3,
			double k1, double k2, double k3, int64_t & state, int64_t &rep, 
			complex<double> &phase, complex<double> &norm)
{
  //cout<<"L1,L2 "<<L1<<"  "<<L2<<endl;
  norm=1.0;
  int nsites=t1.size();
  rep=state;
  int64_t newstate=state;
  int t1v=0;
  int t2v=0;
  int t3v=0;
  int t1v_special=0;
  int t2v_special=0;
  int t3v_special=0;
  int repeat=1;
  for (int i=0;i<L1;i++)
  {
  	if (i>0)
	{
		translateT(newstate,t1, nsites);
		t1v=t1v+1;
		//cout<<"New state ="<<newstate<<endl;
		if (newstate<rep) {rep=newstate;t1v_special=t1v;t2v_special=t2v;t3v_special=t3v;}
		if (newstate==state) {norm+=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));repeat+=1;}
	}
	for (int j=0;j<L2;j++)
	{
		if (j>0)
		{
		     translateT(newstate,t2, nsites);
		     t2v=t2v+1;
		     //cout<<"New state ="<<newstate<<endl;
		     if (newstate<rep) {rep=newstate;t1v_special=t1v;t2v_special=t2v;t3v_special=t3v;}
		     if (newstate==state) {norm+=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));repeat+=1;}
		}
		for (int k=0;k<L3;k++)
		{
			if (k>0)
			{
			 translateT(newstate,t3, nsites);
			 t3v=t3v+1;
			 //cout<<"New state ="<<newstate<<endl;
			 if (newstate<rep) {rep=newstate;t1v_special=t1v;t2v_special=t2v;t3v_special=t3v;}
		     	 if (newstate==state) {norm+=exp(complex<double>(0, (-k1*double(t1v)) + (-k2*double(t2v)) + (-k3*double(t3v))));repeat+=1;}
			}
		}
	}
  }
  phase=exp(complex<double>(0, (-k1*double(t1v_special)) + (-k2*double(t2v_special)) + (-k3*double(t3v_special))));
  if (abs(norm)>1.0e-10) 
  {
  	norm=sqrt(abs(double(L1*L2*L3)/double(repeat))); 
  }
  else
  {
	norm=0.0;
  }
}


//////////////////////////////////////////////////////////////////////
void dump_wfs(string mfile,
              std::vector<double> &eigs,
	      const std::vector< std::vector<double> > &wfs)
{
	int hilbert=wfs[0].size();
	int num_wfs=wfs.size();
	for (int n=0;n<num_wfs;n++)
	{
		ofstream file;
		string combined=mfile+string("_wf_")+to_string(n);
        	const char *cstr = combined.c_str();
		file.open(cstr);
		for (int i=0;i<hilbert;i++) file<<boost::format("%12d") %i<<"  "<<boost::format("%+.15f") %wfs[n][i]<<endl;
		file.close();
	}
}

//////////////////////////////////////////////////////////////////////
void load_wf(string mfile,
	     std::vector<double> &wf)
{
	wf.clear();
	cout<<"Wavefunction being read...."<<endl;
	const char *cstr = mfile.c_str();
	ifstream file(cstr);
	double wf_val;
	int spin_det;
	int x;
	if (file.is_open())
	{
		while (file>>spin_det>>wf_val)	
		{
			wf.push_back(wf_val);
			if (!file) break;
		}
	}
	file.close();
}
//////////////////////////////////////////////////////////////////////
// Look in a sorted list to find location of an integer
int binary_search(int64_t det, std::vector<int64_t> &sorted_dets_list) 
{
   // function:
   //   Searches sortedArray[first]..sortedArray[last] for key.  
   // returns: index of the matching element if it finds key, 
   //         otherwise  -(index where it could be inserted)-1.
   // parameters:
   //   sortedArray in  array of sorted (ascending) values.
   //   first, last in  lower and upper subscript bounds
   //   key         in  value to search for.
   // returns:
   //   index of key, or -insertion_position -1 if key is not 
   //                 in the array. This value can easily be
   //                 transformed into the position to insert it.
   int first=0;
   int last=sorted_dets_list.size()-1;   
   while (first <= last) {
       int mid = (first + last) / 2;  // compute mid point.
       if (det > sorted_dets_list[mid]) 
           first = mid + 1;  // repeat search in top half.
       else if (det < sorted_dets_list[mid]) 
           last = mid - 1; // repeat search in bottom half.
       else
           return mid;     // found it. return position /////
   }
   return -(first + 1);    // failed to find key
}

//////////////////////////////////////////////////////////////////////////////
// Useful bit functions  
//////////////////////////////////////////////////////////////////////////////

int btest64(int64_t a,int pos)
{ 
  bitset<64> b;
  b=a;
  return b[pos];
}

int64_t ibset64(int64_t a,int pos)
{return a| int64_t(1)<<pos;}

int64_t ibclr64(int64_t a,int pos)
{return a & ~(int64_t(1)<<pos);}

int btest(int a,int pos)
{ int b=a & 1<<pos;
  if (b!=0) return 1;
  else      return 0;}

int ibset(int a,int pos)
{return a| 1<<pos;}

int ibclr(int a,int pos)
{return a & ~(1<<pos);}

//////////////////////////////////////////////////////////////////////////////
// Useful number functions  
//////////////////////////////////////////////////////////////////////////////

int n_choose_k(int n,int k)
{
//    !---------------------------------------------------------------------------
//    ! Description : Generate binomial coefficient
    int         nck;
    int   	i;
    double	log_n_factorial, log_k_factorial, log_n_minus_k_factorial;

    if (n < k) return 0;
    
    log_n_factorial = 0.0;
    log_k_factorial = 0.0;
    log_n_minus_k_factorial = 0.0;

    for (i=2;i<=n;i++) log_n_factorial = log_n_factorial + log(double(i));
    for (i=2;i<=k;i++) log_k_factorial = log_k_factorial + log(double(i));
    for (i=2;i<=n-k;i++) log_n_minus_k_factorial = log_n_minus_k_factorial + log(double(i));

    nck = int((exp(log_n_factorial - log_k_factorial - log_n_minus_k_factorial))+0.5);

    return nck;
}
//////////////////////////////////////////////////////////////////////////////

void constrained_dets(int num_sites,int num_ones,std::vector<int> &dets)
{
// ----------------------------------------------------------------------------------------------
//    ! Description   : Gives us a neat way of generating all configs with a fixed
//    !                 particle number
//    ! Author        : F. Petruzielo's code used by H.J. Changlani (ref. Bit Twiddling Hacks)
//    ! ----------------------------------------------------------------------------------------------

    int                      temp,temp1,temp2,temp3,temp4,temp5,temp6;
    int                      i;
    int 		     num_configs;

    if (num_sites<num_ones) 
    {
	cout<<"ERROR: Num sites < Num_ones"<<endl;
        return;
    }

    dets.clear();
    num_configs=n_choose_k(num_sites,num_ones);

    dets.push_back(pow(2,num_ones) - 1);
    for( i=1;i<num_configs;i++)
    {  
       temp  = (dets[i-1] | dets[i-1] - 1) + 1;
       temp2 = (temp) & (-temp);
       temp3=  (dets[i-1]) & (-dets[i-1]);
       temp4=temp2/temp3;
       temp5=temp4>>1;
       temp6=temp5-1;
       dets.push_back(temp|temp6);
    }
}

void constrained_dets_i64(int num_sites,int num_ones,std::vector<int64_t> &dets)
{
// ----------------------------------------------------------------------------------------------
//    ! Description   : Gives us a neat way of generating all configs with a fixed
//    !                 particle number
//    ! Author        : F. Petruzielo's code used by H.J. Changlani (ref. Bit Twiddling Hacks)
//    ! ----------------------------------------------------------------------------------------------

    int64_t                  temp,temp1,temp2,temp3,temp4,temp5,temp6;
    int 		     num_configs;

    if (num_sites<num_ones) 
    {
	cout<<"ERROR: Num sites < Num_ones"<<endl;
        return;
    }

    dets.clear();
    num_configs=n_choose_k(num_sites,num_ones);

    dets.push_back(pow(int64_t(2),int64_t(num_ones)) - int64_t(1));
    for( int i=1;i<num_configs;i++)
    {  
       temp  = (dets[i-1] | dets[i-1] - int64_t(1)) + int64_t(1);
       temp2 = (temp) & (-temp);
       temp3=  (dets[i-1]) & (-dets[i-1]);
       temp4=temp2/temp3;
       temp5=temp4>>int64_t(1);
       temp6=temp5-int64_t(1);
       dets.push_back(temp|temp6);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
int64_t representative(int64_t det, std::vector< std::vector<int> > maps)
{
	int64_t rep=det;
	
        for (int i=0;i<maps.size();i++)
	{
		int64_t newdet=0;
		for (int j=0;j<maps[i].size();j++) 
		{
			if (btest64(det,j)==1) newdet=ibset64(newdet,maps[i][j]);
		}
		if (newdet<rep) rep=newdet;
	}
	return rep;
}
int64_t convert_locs_to_int64(std::vector<int> &locs)
{
	int64_t temp=0;
	for (int i=0;i<locs.size();i++)
	{
		temp=ibset64(temp,locs[i]);
	}
	return temp;
}
/////////////////////////////////////////////////////////////////////////////////
//bool isrepudloc(std::vector<int> &locu, std::vector<int> &locd, std::vector< std::vector<int> > &maps)
//{
//	std::vector<int> replocu=locu;
//	std::vector<int> replocd=locd;
//	int64_t repu=convert_locs_to_int64(replocu);
//	int64_t repd=convert_locs_to_int64(replocu);
//	
//        for (int i=0;i<maps.size();i++)
//	{
//		std::vector<int> newlocu;
//		std::vector<int> newlocd;
//		int64_t nu=0,nd=0;
//		for (int j=0;j<maps[i].size();j++) 
//		{
//			newlocu.push_back(maps[i][locu[j]]);
//			newlocd.push_back(maps[i][locd[j]]);
//		}
//		nu=convert_locs_to_int64(newlocu);
//		nd=convert_locs_to_int64(newlocu);
//		if (nu<repu or (newdetu==repu and newdetd<repd)) 
//		{
//			return false;
//			repu=nu;
//			repd=nd;
//		}
//	}
//	return true;
//}

/////////////////////////////////////////////////////////////////////////////////////////
bool isrepud(int64_t &detu, int64_t &detd, std::vector<int> &locsup,std::vector<int> &locsdn, std::vector< std::vector<int> > &maps)
{
	int64_t repu=detu;
	int64_t repd=detd;
	//std::vector<int> locsup,locsdn;
	/*for (int j=0;j<maps[0].size();j++) 
	{
			if (btest64(detu,j)==1) locsup.push_back(j);
			if (btest64(detd,j)==1) locsdn.push_back(j);
	}*/
        for (int i=0;i<maps.size();i++)
	{
		int64_t newdetu=0;
		int64_t newdetd=0;
		//for (int j=0;j<maps[i].size();j++) 
		//{
			//if (btest64(detu,j)==1) newdetu=ibset64(newdetu,maps[i][j]);
			//if (btest64(detd,j)==1) newdetd=ibset64(newdetd,maps[i][j]);
		//}
		for (int j=0;j<locsup.size();j++) newdetu=ibset64(newdetu,maps[i][locsup[j]]);
		for (int j=0;j<locsdn.size();j++) newdetd=ibset64(newdetd,maps[i][locsdn[j]]);
		if (newdetu<repu or (newdetu==repu and newdetd<repd)) 
		{
			return false;
			//repu=newdetu;
			//repd=newdetd;
		}
	}
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
void representativeud(int64_t &detu, int64_t &detd, std::vector< std::vector<int> > &maps, int64_t &repu, int64_t &repd)
{
	repu=detu;
	repd=detd;
	
        for (int i=0;i<maps.size();i++)
	{
		int64_t newdetu=0;
		int64_t newdetd=0;
		for (int j=0;j<maps[i].size();j++) 
		{
			if (btest64(detu,j)==1) newdetu=ibset64(newdetu,maps[i][j]);
			if (btest64(detd,j)==1) newdetd=ibset64(newdetd,maps[i][j]);
		}
		if (newdetu<repu or (newdetu==repu and newdetd<repd)) 
		{
			repu=newdetu;
			repd=newdetd;
		}
	}
}
//////////////////////////////////////////////////////////////////////////////
RMatrix translate(std::vector<int> T, int ntimes)
{
	int sites=T.size();
	RMatrix Tmat(sites,sites);
	RMatrix eye(sites,sites);
	for (int i=0;i<sites;i++) 
	{
		for (int j=0;j<sites;j++)
		{
			if (i!=j) {eye(i,j)=0;}
			else {eye(i,i)=1;}
			if (j==T[i]) {Tmat(i,j)=1;}
			else {Tmat(i,j)=0;}
		}
	}
	RMatrix Tmatn=Tmat;
	if (ntimes==0) return eye;
	if (ntimes==1) return Tmat;
	
	for (int n=0;n<ntimes-1;n++)
	{
		RMatrix Tmatnp1;
		real_matrix_multiply(Tmatn,Tmat,Tmatnp1);	
		Tmatn=Tmatnp1;
	}
	return Tmatn; 
}
//////////////////////////////////////////////////////////////////////////////
std::vector<int> make_vector_from_tmat(RMatrix Tmat)
{
	int sites=Tmat.NRows();
	std::vector<int> map;
	for (int i=0;i<sites;i++)
	{
		for (int j=0;j<sites;j++) 
		{
			if (abs(Tmat(i,j)-1)<1.0e-10) map.push_back(j);
		}
	}
	return map;
}
//////////////////////////////////////////////////////////////////////////////
std::vector< std::vector<int> > make_maps(std::vector<int> T1, std::vector<int> T2)
{
   // Write T1 and T2 as a matrix of 0's and 1's
   std::vector< std::vector<int> > maps;
   int lx,ly;

   if (T1.size()==12) {lx=2;ly=2;}
   if (T1.size()==24) {lx=4;ly=4;}

   cout<<"lx,ly ="<<lx<<" "<<ly<<endl;
   for (int nx=0;nx<lx;nx++)
   {
	RMatrix Tmatx=translate(T1,nx);
	for (int ny=0;ny<ly;ny++)
	{
		RMatrix Tmaty=translate(T2,ny);
		RMatrix Tmat;
	        real_matrix_multiply(Tmatx,Tmaty,Tmat);	
		std::vector<int> map=make_vector_from_tmat(Tmat);
		maps.push_back(map);
	}
   }
   cout<<"Maps.size() ="<<maps.size()<<endl;
   return maps; 
}
//////////////////////////////////////////////////////////////////////////////
std::vector<int> convert_ind_to_vec(int index, 
				    std::vector<int> nstates)
{
	int 			coord;
	int 			n=1;
	int 			dim=nstates.size();
	std::vector<int> 	vec(dim);

	n=1;
	for (int i=0;i<dim;i++)
	{
	   n=nstates[dim-1-i];
	   coord=index%n;
	   vec[dim-1-i]=coord;
	   index=index/n;
	}
	return vec;
}  

//////////////////////////////////////////////////////////////////////////////

int convert_vec_to_ind(std::vector<int> const &vec, 
		       std::vector<int> nstates)

{
	int 		ind,n;
	int 		dim=nstates.size();
	
	n=1; ind=0;
	for (int i=0;i<dim;i++)
	{
           ind+=(n*vec[dim-1-i]);
	   n=n*nstates[dim-1-i];
	}
	return ind;
}  


//////////////////////////////////////////////////////////////////////////////
void get_adj_list_from_pairs(std::vector< std::vector<int> > const &pairs,
			     std::vector< std::vector<int> > &adj_list)
{
	int max=0;

	for (int i=0;i<pairs.size();i++)
	{
		if (pairs[i][0]>max) {max=pairs[i][0];}
		if (pairs[i][1]>max) {max=pairs[i][1];}
		
	}
 
	adj_list.clear();
	adj_list.resize(max+1);
		
	for (int i=0;i<pairs.size();i++)
	{
		adj_list[pairs[i][0]].push_back(pairs[i][1]);
		adj_list[pairs[i][1]].push_back(pairs[i][0]);
	}
}
//////////////////////////////////////////////////////////////////////////////
void convert_num_to_vec(int num, 
                        int base, 
                        int num_bits, 
                        std::vector<int> &config)
{
    config.resize(num_bits);
    for (int i=0;i<num_bits;i++) config[i]=0;

    if (pow(base,num_bits)<num) cout<<"Input number greater than representable by num_bits, Expect errors"<<" "<<num<<"\n";
    for (int ctr=1; ctr<=num_bits; ctr++)
    {
            config[num_bits-ctr]=num%base; // reversed bits needed
            num=num/base;
    }
}
//////////////////////////////////////////////////////////////////////////////
int convert_vec_to_num(std::vector<int> &config, 
		       int base)
{
    int 	num=0;
    int 	mult=1;

    for (int i=config.size()-1; i>=0; i--)
    {
           num+= (config[i]*mult);
           mult=mult*base;
    }
    return num;   
}
//////////////////////////////////////////////////////////////////////////////
void sort_by_energy(std::vector<double> &eigs,std::vector<double> &szs)
{
   for (int j=0;j<eigs.size()-1;j++)
   {	
	   for (int i=j+1;i<eigs.size();i++)
	   {
		if (eigs[j]>eigs[i])
		{
		   swap(eigs[j],eigs[i]);
		   swap(szs[j],szs[i]);
		}
		
		if (abs(eigs[j]-eigs[i])<1.0e-8)
		{
		  if (szs[j]>szs[i]) {swap(szs[j],szs[i]);}  	
		}
	   }
   }
}
//////////////////////////////////////////////////////////////////////////////
int str_to_int(string str)
{ return boost::lexical_cast<int>(str);}

//////////////////////////////////////////////////////////////////////////////
double str_to_d(string str)
{ return boost::lexical_cast<long double>(str);}

//////////////////////////////////////////////////////////////////////////////
bool str_to_bool(string str)
{       bool b=0;
        if (str==string("t") or str==string("T") or str==string("true") or
            str==string("True") or str==string("Y") or str==string("Yes") 
            or str==string("1")){b=1;}
        return b;
}

//////////////////////////////////////////////////////////////////////////////
std::string dtos(double dbl)
{
    char buf[BUFSIZ];
    sprintf(buf, "%+lf", dbl);
    return buf;
}
//////////////////////////////////////////////////////////////////////////////
template <class T>
std::string to_string (T const &t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}
template std::string to_string<int>(int const &);
template std::string  to_string<double>(double const &);

//////////////////////////////////////////////////////////////////////////////
std::vector<int> convert_string_to_vec(std::string const &s)
{
    // Eg s=[1,0] would get converted to vec[1,0]
    std::vector<int> v;
    int num;
    
    std::string temp="";
    for (int i=1;i<s.size();i++)
    {
        if (s.substr(i,1)!=string(",") and s.substr(i,1)!=string("]") 
            and s.substr(i,1)!=string(")")
            and s.substr(i,1)!=string("}"))
        {   temp+=s.substr(i,1);}    
        else
        {   num=str_to_int(temp);
            v.push_back(num);
            temp="";}
    }
    return v;
}
///////////////////////////////////////////////////////////////////////
std::vector<double> convert_string_to_vec_double(std::string const &s)
{
    // Eg s=[1,0] would get converted to vec[1,0]
    std::vector<double> v;
    double num;
    
    std::string temp="";
    for (int i=1;i<s.size();i++)
    {
        if (s.substr(i,1)!=string(",") and s.substr(i,1)!=string("]") 
            and s.substr(i,1)!=string(")")
            and s.substr(i,1)!=string("}"))
        {   temp+=s.substr(i,1);}    
        else
        {   num=str_to_d(temp);
            v.push_back(num);
            temp="";}
    }
    return v;
}


//////////////////////////////////////////////////////////////////////////////

std::vector< std::vector<int> > convert_string_to_vec_of_vec(std::string const &s)
{
    // Eg s=[[1,0],[1,1]] would get converted to vec[1,0]+vec[1,1]
    std::vector<int> temp_vec;
    std::vector< vector<int> > v;

    string temp="";
    for (int i=1;i<s.size();i++)
    {
        if (s.substr(i,1)!=string("]") and i!=s.size())
        {   temp+=s.substr(i,1);
        }    
        else
        {
            temp+=s.substr(i,1);
            temp_vec=convert_string_to_vec(temp);
            v.push_back(temp_vec);
            temp="";
            i++;
        }
    }
    return v;
}
//////////////////////////////////////////////////////////////////////////////

