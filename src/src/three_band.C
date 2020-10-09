#include"three_band.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 Spin Hole MODEL
/////////////////////////////////////////////////////////////////////////////////////////////////////
void Three_Band_Model::operator()
                    (std::vector<int> const &config_spins,
                     std::vector<int> const &config_up_holes,
                     std::vector<int> const &config_down_holes,
                     std::vector< std::vector<int> > &touched_sites_list_spins, 
                     std::vector< std::vector<int> > &vals_on_touched_list_spins,
                     std::vector< std::vector<int> > &touched_sites_list_upholes, 
                     std::vector< std::vector<int> > &vals_on_touched_list_upholes,
                     std::vector< std::vector<int> > &touched_sites_list_dnholes, 
                     std::vector< std::vector<int> > &vals_on_touched_list_dnholes,
                     std::vector<double> &hints_list)
{
	// J 
	
	// D f(S_hole dot Si, S_hole dot Sj)

	// Hubbard t

	// Hubbard onsite U + J Siz Sjz
}

void Three_Band_Model::operator()
                   (int spin_det,
                    int uphole_det,
                    int dnhole_det,
                    std::vector<int> &new_spin_dets,
                    std::vector<int> &new_uphole_dets,
                    std::vector<int> &new_dnhole_dets,
                    std::vector< complex<double> > &hints_list)
{
	//cout<<"Here 1.."<<endl;
	std::vector<int> upsigns(this->num_sites),dnsigns(this->num_sites);
	int ctru=0;
	for (int i=0;i<this->num_sites;i++)
	{
		upsigns[i]=ctru;	
		if (btest(uphole_det,i)==1) ctru=ctru+1;
	}
	//cout<<"Here 2.."<<endl;
	
	int ctrd=0;
	for (int i=0;i<this->num_sites;i++)
	{
		dnsigns[i]=ctrd;	
		if (btest(dnhole_det,i)==1) ctrd=ctrd+1;
	}
	
	//cout<<"Here 3.."<<endl;

	for (int i=0;i<this->CuOpairs.size();i++)
	{
		int su=1;
		int sd=1;
		int first=this->CuOpairs[i][0];
		int second=this->CuOpairs[i][1];
		
		// Hubbard tpd
		int mx=max(first,second);
		int mn=min(first,second);
		if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
		if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
		calc_hints_fermion_hop(-1.0*this->tpd*this->signsCuOpairs[i], 
				       first, second, su, sd, spin_det, uphole_det, dnhole_det,
				       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
				       hints_list);
	}
	
	//cout<<"Here 4.."<<endl;
	
	for (int i=0;i<this->OOpairs.size();i++)
	{
		int su=1;
		int sd=1;
		int first=this->OOpairs[i][0];
		int second=this->OOpairs[i][1];
		
		// Hubbard tpp
		int mx=max(first,second);
		int mn=min(first,second);
		if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
		if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
		calc_hints_fermion_hop(1.0*this->tpp*this->signsOOpairs[i], 
				       first, second, su, sd, spin_det, uphole_det, dnhole_det,
				       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
				       hints_list);
	}
	//cout<<"Here 5.."<<endl;
	
	double hint=0.0;	    
        for (int i=0;i<this->Cusites.size();i++)
	{ 
		int j=this->Cusites[i];
		hint+=(this->Udd)*double(btest(uphole_det,j))*double(btest(dnhole_det,j));
		hint+=(this->epsd)*double(btest(uphole_det,j)+btest(dnhole_det,j));
		hint+=(this->etaCu[i]*this->hst*0.5)*double(btest(uphole_det,j)-btest(dnhole_det,j));
	}
        for (int i=0;i<this->Osites.size();i++)
	{ 
		int j=this->Osites[i];
		hint+=(this->Upp)*double(btest(uphole_det,j))*double(btest(dnhole_det,j));
		hint+=(this->epsp)*double(btest(uphole_det,j)+btest(dnhole_det,j));
	}
	new_spin_dets.push_back(spin_det);
	new_uphole_dets.push_back(uphole_det);
	new_dnhole_dets.push_back(dnhole_det);
        hints_list.push_back( complex<double> (hint) );

	//cout<<"Here 6.."<<endl;
	// Upd ni nj
	calc_hints_V_only(this->Upd, this->CuOpairs, 
			  spin_det, uphole_det, dnhole_det,
			  new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			  hints_list);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void three_band_setup(string filename, 
               Three_Band_Model &three_band)

{    
    ///////////////////////////////////////////
    // Set the 3-band Hamiltonian
    ///////////////////////////////////////////
    bool found=true;;
    string str,str_ret;
    int nup_holes,ndn_holes;
    double Upp,Upd,Udd,tpd,tpp,epsp,epsd,hst;
    std::vector< std::vector<int> > CuOpairs,OOpairs;
    std::vector<int> Cusites,Osites,etaCu, signsCuOpairs,signsOOpairs;

    // coupling parameters search
    search_for("nup_holes",filename,str_ret,found);
    if (found){nup_holes=str_to_int(str_ret);} else{nup_holes=0;}
    
    search_for("ndn_holes",filename,str_ret,found);
    if (found){ndn_holes=str_to_int(str_ret);} else{ndn_holes=0;}
    
    search_for("epsp",filename,str_ret,found);
    if (found){epsp=str_to_d(str_ret);} else{epsp=0.0;}
    
    search_for("epsd",filename,str_ret,found);
    if (found){epsd=str_to_d(str_ret);} else{epsd=0.0;}
    
    search_for("hst",filename,str_ret,found);
    if (found){hst=str_to_d(str_ret);} else{hst=0.0;}
    
    search_for("tpd",filename,str_ret,found);
    if (found){tpd=str_to_d(str_ret);} else{tpd=0.0;}
    
    search_for("tpp",filename,str_ret,found);
    if (found){tpp=str_to_d(str_ret);} else{tpp=0.0;}
    
    search_for("Upp",filename,str_ret,found);
    if (found){Upp=str_to_d(str_ret);} else{Upp=0.0;}
    
    search_for("Upd",filename,str_ret,found);
    if (found){Upd=str_to_d(str_ret);} else{Upd=0.0;}
    
    search_for("Udd",filename,str_ret,found);
    if (found){Udd=str_to_d(str_ret);} else{Udd=0.0;}
  
    search_for("CuOpairs",filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) CuOpairs=convert_string_to_vec_of_vec(str_ret);}
     
    search_for("signsCuOpairs",filename,str_ret,found);
    if (found){if (str_ret.substr(0,1)==string("[")) signsCuOpairs=convert_string_to_vec(str_ret);}
    
    search_for("OOpairs",filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) OOpairs=convert_string_to_vec_of_vec(str_ret);}
    
    search_for("signsOOpairs",filename,str_ret,found);
    if (found){if (str_ret.substr(0,1)==string("[")) signsOOpairs=convert_string_to_vec(str_ret);}
 
     search_for("Cusites",filename,str_ret,found);
     if (found){if (str_ret.substr(0,1)==string("[")) Cusites=convert_string_to_vec(str_ret);}
     
     search_for("Osites",filename,str_ret,found);
     if (found){if (str_ret.substr(0,1)==string("[")) Osites=convert_string_to_vec(str_ret);}
     
     search_for("etaCu",filename,str_ret,found);
     if (found){if (str_ret.substr(0,1)==string("[")) etaCu=convert_string_to_vec(str_ret);}
 
    three_band.init(nup_holes,ndn_holes,tpd,tpp,Upp,Upd,Udd,epsp, epsd, hst, CuOpairs,OOpairs, Cusites, Osites, etaCu, signsCuOpairs,signsOOpairs);

}
