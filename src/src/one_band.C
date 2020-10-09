#include"one_band.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 Spin Hole MODEL
/////////////////////////////////////////////////////////////////////////////////////////////////////
void One_Band_Model::operator()
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

void One_Band_Model::operator()
                   (int spin_det,
                    int uphole_det,
                    int dnhole_det,
                    std::vector<int> &new_spin_dets,
                    std::vector<int> &new_uphole_dets,
                    std::vector<int> &new_dnhole_dets,
                    std::vector< complex<double> > &hints_list)
{

	std::vector<int> upsigns(this->num_sites),dnsigns(this->num_sites);
	int ctru=0;
	for (int i=0;i<this->num_sites;i++)
	{
		upsigns[i]=ctru;	
		if (btest(uphole_det,i)==1) ctru=ctru+1;
	}
	
	int ctrd=0;
	for (int i=0;i<this->num_sites;i++)
	{
		dnsigns[i]=ctrd;	
		if (btest(dnhole_det,i)==1) ctrd=ctrd+1;
	}

	for (int i=0;i<this->CuCupairs.size();i++)
	{
		int su=1;
		int sd=1;
		int first=this->CuCupairs[i][0];
		int second=this->CuCupairs[i][1];
		
		// Hubbard tCu
		int mx=max(first,second);
		int mn=min(first,second);
		if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
		if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
		calc_hints_fermion_hop(-1.0*this->tCuCu*this->signsCuCupairs[i], 
				       first, second, su, sd, spin_det, uphole_det, dnhole_det,
				       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
				       hints_list);
	}
	
	double hint=0.0;	    
        for (int i=0;i<this->Cusites.size();i++)
	{ 
		int j=this->Cusites[i];
		hint+=(this->Udd)*double(btest(uphole_det,j))*double(btest(dnhole_det,j));
		hint+=(this->epsd)*double(btest(uphole_det,j)+btest(dnhole_det,j));
		hint+=(this->etaCu[i]*this->hst*0.5)*double(btest(uphole_det,j)-btest(dnhole_det,j));
	}
        new_spin_dets.push_back(spin_det);
	new_uphole_dets.push_back(uphole_det);
	new_dnhole_dets.push_back(dnhole_det);
        hints_list.push_back( complex<double> (hint) );

	// UCu Cu 
	calc_hints_V_only(this->UCuCu, this->CuCupairs, 
			  spin_det, uphole_det, dnhole_det,
			  new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			  hints_list);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void one_band_setup(string filename, 
                    One_Band_Model &one_band)

{    
    ///////////////////////////////////////////
    // Set the 3-band Hamiltonian
    ///////////////////////////////////////////
    bool found=true;;
    string str,str_ret;
    int nup_holes,ndn_holes;
    double UCuCu,Udd,tCuCu,epsd,hst;
    std::vector< std::vector<int> > CuCupairs;
    std::vector<int> Cusites,etaCu, signsCuCupairs;

    // coupling parameters search
    search_for("nup_holes",filename,str_ret,found);
    if (found){nup_holes=str_to_int(str_ret);} else{nup_holes=0;}
    
    search_for("ndn_holes",filename,str_ret,found);
    if (found){ndn_holes=str_to_int(str_ret);} else{ndn_holes=0;}
    
    search_for("epsd",filename,str_ret,found);
    if (found){epsd=str_to_d(str_ret);} else{epsd=0.0;}
    
    search_for("hst",filename,str_ret,found);
    if (found){hst=str_to_d(str_ret);} else{hst=0.0;}
    
    search_for("tCuCu",filename,str_ret,found);
    if (found){tCuCu=str_to_d(str_ret);} else{tCuCu=0.0;}
    
    search_for("UCuCu",filename,str_ret,found);
    if (found){UCuCu=str_to_d(str_ret);} else{UCuCu=0.0;}
    
    search_for("Udd",filename,str_ret,found);
    if (found){Udd=str_to_d(str_ret);} else{Udd=0.0;}
  
    search_for("CuCupairs",filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) CuCupairs=convert_string_to_vec_of_vec(str_ret);}
     
    search_for("signsCuCupairs",filename,str_ret,found);
    if (found){if (str_ret.substr(0,1)==string("[")) signsCuCupairs=convert_string_to_vec(str_ret);}
    
     search_for("Cusites",filename,str_ret,found);
     if (found){if (str_ret.substr(0,1)==string("[")) Cusites=convert_string_to_vec(str_ret);}
     
     search_for("etaCu",filename,str_ret,found);
     if (found){if (str_ret.substr(0,1)==string("[")) etaCu=convert_string_to_vec(str_ret);}
 
     one_band.init(nup_holes,ndn_holes, tCuCu, Udd, epsd, hst, UCuCu, CuCupairs, Cusites, etaCu, signsCuCupairs);

}
