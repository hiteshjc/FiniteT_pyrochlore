#include"global.h"
#include"ybmg.h"
#include"hamiltonian_spin_functions.h"
#include"number_functions.h"
#include"search_for.h"
#include"printing_functions.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                 YbMgGaO4 MODEL
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
void Spin_Half_Ybmg::operator()
                    (int64_t spin_det,
                     std::vector<int64_t> &new_spin_dets, 
                     std::vector< complex<double> > &hints_list, 
		     int &ctr)
{
	complex<double> im=complex<double>(0.0,1.0);
	complex<double>  diaghint=0.0;
	ctr=0;
	
	// Save bits 
	std::vector<int> setofbits(this->num_sites);
	for (int i=0;i<this->num_sites;i++) setofbits[i]=btest64(spin_det,i);
	// NN pairs
	for (int i=0;i<this->pairs1.size();i++)
	{
		int64_t 	newdet=spin_det;
		complex<double> hint;
		int site1=this->pairs1[i][0];
		int site2=this->pairs1[i][1];
		if (setofbits[site1]==0 and setofbits[site2]==0)
		{
		     newdet=ibset64(newdet,site1);
		     newdet=ibset64(newdet,site2);
		     hint=this->J1ppmm*this->phases[i];
		     diaghint+=(0.25*this->J1zz);
		}
		else if (setofbits[site1]==0 and setofbits[site2]==1)
		{
		     newdet=ibset64(newdet,site1);
		     newdet=ibclr64(newdet,site2);
		     hint=(this->J1pm); // No 0.5, this is the notation by Mourigal et al
		     diaghint-=(0.25*this->J1zz);
		}
		else if (setofbits[site1]==1 and setofbits[site2]==0)
		{
		     newdet=ibclr64(newdet,site1);
		     newdet=ibset64(newdet,site2);
		     hint=(this->J1pm); // No 0.5, this is the notation by Mourigal et al
		     diaghint-=(0.25*this->J1zz);
		}
		else if (setofbits[site1]==1 and setofbits[site2]==1)
		{
		     newdet=ibclr64(newdet,site1);
		     newdet=ibclr64(newdet,site2);
		     hint=this->J1ppmm*conj(this->phases[i]);
		     diaghint+=(0.25*this->J1zz);
		}
		new_spin_dets[ctr]=newdet;
		hints_list[ctr]=hint;
		ctr+=1;
	}
	//cout<<"NN done " <<endl;	
	// NNN pairs
	for (int i=0;i<this->pairs2.size();i++)
	{
		int64_t 	newdet=spin_det;
		complex<double> hint;
		int site1=this->pairs2[i][0];
		int site2=this->pairs2[i][1];
		if (setofbits[site1]==0 and setofbits[site2]==0)
		{
		     diaghint+=(0.25*this->J2zz);
		}
		else if (setofbits[site1]==0 and setofbits[site2]==1)
		{
		     newdet=ibset64(newdet,site1);
		     newdet=ibclr64(newdet,site2);
		     hint=(this->J2pm);  // No 0.5, this is the notation by Mourigal et al
		     diaghint-=(0.25*this->J2zz);
		}
		else if (setofbits[site1]==1 and setofbits[site2]==0)
		{
		     newdet=ibclr64(newdet,site1);
		     newdet=ibset64(newdet,site2);
		     hint=(this->J2pm);  //  No 0.5, this is the notation by Mourigal et al
		     diaghint-=(0.25*this->J2zz);
		}
		else if (setofbits[site1]==1 and setofbits[site2]==1)
		{
		     diaghint+=(0.25*this->J2zz);
		}
		new_spin_dets[ctr]=newdet;
		hints_list[ctr]=hint;
		ctr+=1;
	}
	///cout<<"NNN done " <<endl;	

	// One spin flipped
	// hx, hy, hz already rescaled by mub
	complex<double> imi = complex<double> (0,1.0);

	for (int i=0;i<this->num_sites;i++)
	{
		diaghint-=(this->gz*this->hz*double(setofbits[i]-0.5));
		int64_t 	newdet=spin_det;
		complex<double> hint=0.0;
		if (setofbits[i]==0)
		{
			newdet=ibset64(newdet,i);
			for (int n=0;n<this->neighbors[i].size();n++)
			{
				int j=this->neighbors[i][n];
				hint+=(double(setofbits[j]-0.5)*conj(this->phasematrix(i,j))*complex<double>(0,-0.5)*this->J1zpm);
			}
			hint-=(this->gxy*(this->hx - imi*this->hy)*0.5);
		}
		else
		{
			newdet=ibclr64(newdet,i);
			for (int n=0;n<this->neighbors[i].size();n++)
			{
				int j=this->neighbors[i][n];
				hint+=(double(setofbits[j]-0.5)*this->phasematrix(i,j)*complex<double>(0,+0.5)*this->J1zpm);
			}
			hint-=(this->gxy*(this->hx + imi*this->hy)*0.5);
		}
		new_spin_dets[ctr]=newdet;
		hints_list[ctr]=hint;
		ctr+=1;
	}
	//cout<<"Spin flip done " <<endl;	
	new_spin_dets[ctr]=spin_det;
	hints_list[ctr]=diaghint;
	ctr+=1;
	//cout<<"Diag done " <<endl;	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void ybmg_setup(string filename, 
               Spin_Half_Ybmg &ybmg)

{    
    double J1zz,J1pm,J1ppmm,J1zpm,J2zz,J2pm, gxy,gz, hx,hy,hz;
    bool found;
    string str_ret;
    std::vector< std::vector<int> > pairs1;
    std::vector< std::vector<int> > pairs2;
    std::vector<int> 		    bondtypes;
    
    search_for(string("pairs1"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) pairs1=convert_string_to_vec_of_vec(str_ret);}    
    search_for(string("pairs2"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) pairs2=convert_string_to_vec_of_vec(str_ret);}    
    search_for(string("bondtypes"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) bondtypes=convert_string_to_vec(str_ret);}    

    search_for(string("J1zz"),filename,str_ret,found);
    if (found){J1zz=str_to_d(str_ret);} else{J1zz=0.0;}
    search_for(string("J1pm"),filename,str_ret,found);
    if (found){J1pm=str_to_d(str_ret);} else{J1pm=0.0;}
    search_for(string("J1ppmm"),filename,str_ret,found);
    if (found){J1ppmm=str_to_d(str_ret);} else{J1ppmm=0.0;}
    search_for(string("J1zpm"),filename,str_ret,found);
    if (found){J1zpm=str_to_d(str_ret);} else{J1zpm=0.0;}
    search_for(string("J2zz"),filename,str_ret,found);
    if (found){J2zz=str_to_d(str_ret);} else{J2zz=0.0;}
    search_for(string("J2pm"),filename,str_ret,found);
    if (found){J2pm=str_to_d(str_ret);} else{J2pm=0.0;}
    search_for(string("gxy"),filename,str_ret,found);
    if (found){gxy=str_to_d(str_ret);} else{gxy=0.0;}
    search_for(string("gz"),filename,str_ret,found);
    if (found){gz=str_to_d(str_ret);} else{gz=0.0;}
    search_for(string("hx"),filename,str_ret,found);
    if (found){hx=str_to_d(str_ret);} else{hx=0.0;}
    search_for(string("hy"),filename,str_ret,found);
    if (found){hy=str_to_d(str_ret);} else{hy=0.0;}
    search_for(string("hz"),filename,str_ret,found);
    if (found){hz=str_to_d(str_ret);} else{hz=0.0;}
    
    ybmg.init(pairs1, pairs2, bondtypes, J1zz, J1pm, J1ppmm, J1zpm, J2zz, J2pm, gxy,gz,hx,hy,hz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void ybmg_sym_setup(string filename, 
               Spin_Half_Ybmg &ybmg)

{    
    double J1zz,J1pm,J1ppmm,J1zpm,J2zz,J2pm, gxy,gz, hx,hy,hz;
    bool found;
    string str_ret;
    std::vector< std::vector<int> > pairs1;
    std::vector< std::vector<int> > pairs2;
    std::vector<int> 		    bondtypes;
    
    search_for(string("pairs1"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) pairs1=convert_string_to_vec_of_vec(str_ret);}    
    search_for(string("pairs2"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) pairs2=convert_string_to_vec_of_vec(str_ret);}    
    search_for(string("bondtypes"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) bondtypes=convert_string_to_vec(str_ret);}    

    search_for(string("J1zz"),filename,str_ret,found);
    if (found){J1zz=str_to_d(str_ret);} else{J1zz=0.0;}
    search_for(string("J1pm"),filename,str_ret,found);
    if (found){J1pm=str_to_d(str_ret);} else{J1pm=0.0;}
    search_for(string("J1ppmm"),filename,str_ret,found);
    if (found){J1ppmm=str_to_d(str_ret);} else{J1ppmm=0.0;}
    search_for(string("J1zpm"),filename,str_ret,found);
    if (found){J1zpm=str_to_d(str_ret);} else{J1zpm=0.0;}
    search_for(string("J2zz"),filename,str_ret,found);
    if (found){J2zz=str_to_d(str_ret);} else{J2zz=0.0;}
    search_for(string("J2pm"),filename,str_ret,found);
    if (found){J2pm=str_to_d(str_ret);} else{J2pm=0.0;}
    search_for(string("gxy"),filename,str_ret,found);
    if (found){gxy=str_to_d(str_ret);} else{gxy=0.0;}
    search_for(string("gz"),filename,str_ret,found);
    if (found){gz=str_to_d(str_ret);} else{gz=0.0;}
    search_for(string("hx"),filename,str_ret,found);
    if (found){hx=str_to_d(str_ret);} else{hx=0.0;}
    search_for(string("hy"),filename,str_ret,found);
    if (found){hy=str_to_d(str_ret);} else{hy=0.0;}
    search_for(string("hz"),filename,str_ret,found);
    if (found){hz=str_to_d(str_ret);} else{hz=0.0;}
    
    std::vector<int> t1,t2,t3; 
    double	     k1,k2,k3;

    search_for(string("t1"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) t1=convert_string_to_vec(str_ret);
    }    
    search_for(string("t2"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) t2=convert_string_to_vec(str_ret);
    }    
    search_for(string("t3"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) t3=convert_string_to_vec(str_ret);
    }    
    search_for(string("k1"),filename,str_ret,found);
    if (found){k1=str_to_d(str_ret);} else{k1=0.0;}
    search_for(string("k2"),filename,str_ret,found);
    if (found){k2=str_to_d(str_ret);} else{k2=0.0;}
    search_for(string("k3"),filename,str_ret,found);
    if (found){k3=str_to_d(str_ret);} else{k3=0.0;}

    ybmg.init(pairs1, pairs2, bondtypes, J1zz, J1pm, J1ppmm, J1zpm, J2zz, J2pm, gxy,gz,hx,hy,hz, t1, t2, t3, k1, k2, k3);
    cout<<ybmg.num_sites<<endl;
}


