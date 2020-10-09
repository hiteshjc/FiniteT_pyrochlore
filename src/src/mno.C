#include"mno.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 MnO MODEL
/////////////////////////////////////////////////////////////////////////////////////////////////////
void MnO_Model::operator()
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
}

void MnO_Model::operator()
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

	int su=1;int sd=1;
	
        // Tpi
	int first=3;int second=4;
	int mx=max(first,second); int mn=min(first,second);
	if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
	if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
	calc_hints_fermion_hop(this->Tpi, first, second, su, sd, spin_det, uphole_det, dnhole_det,
			       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			       hints_list);

	su=1;sd=1;
	first=5;second=6;
	mx=max(first,second); mn=min(first,second);
	if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
	if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
	calc_hints_fermion_hop(this->Tpi, first, second, su, sd, spin_det, uphole_det, dnhole_det,
			       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			       hints_list);

	// T01
	su=1;sd=1;
	first=0;second=1;
	mx=max(first,second); mn=min(first,second);
	if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
	if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
	calc_hints_fermion_hop(this->T01, first, second, su, sd, spin_det, uphole_det, dnhole_det,
			       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			       hints_list);

	// T02
	su=1;sd=1;
	first=0;second=2;
	mx=max(first,second); mn=min(first,second);
	if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
	if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
	calc_hints_fermion_hop(this->T02, first, second, su, sd, spin_det, uphole_det, dnhole_det,
			       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			       hints_list);

	// T12
	su=1;sd=1;
	first=1;second=2;
	mx=max(first,second); mn=min(first,second);
	if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
	if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
	calc_hints_fermion_hop(this->T12, first, second, su, sd, spin_det, uphole_det, dnhole_det,
			       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			       hints_list);

	
	double hint=0.0;	
	// 2p on site and Hubbard    
        for (int i=0;i<this->psites.size();i++)
	{ 
		int j=this->psites[i];
		hint+=(1.0*(this->U2p)*double(btest(uphole_det,j))*double(btest(dnhole_det,j)));
		if (j==3 or j==5) hint+=(this->E2p)*double(btest(uphole_det,j)+btest(dnhole_det,j));
	}
	// 3d on site and Hubbard    
        for (int i=0;i<this->dsites.size();i++)
	{ 
		int j=this->dsites[i];
		hint+=(1.0*this->U3d)*double(btest(uphole_det,j))*double(btest(dnhole_det,j));
		hint+=(this->E3d)*double(btest(uphole_det,j)+btest(dnhole_det,j));
	}
	// 4s on site and Hubbard    
        for (int i=0;i<this->ssites.size();i++)
	{ 
		int j=this->ssites[i];
		hint+=(1.0*this->U4s)*double(btest(uphole_det,j))*double(btest(dnhole_det,j));
		hint+=(this->E4s)*double(btest(uphole_det,j)+btest(dnhole_det,j));
	}
        // intrasite 2p-2p
	for (int i=0;i<this->psites.size();i++)
	{
		int one=this->psites[i];
		//for (int j=i+1;j<this->psites.size();j++)
		for (int j=i+1;j<this->psites.size();j++)
		{
			if (i!=j) {int two=this->psites[j];
			hint+=(this->U2p2p)*double(btest(uphole_det,one)+btest(dnhole_det,one))*double(btest(uphole_det,two)+btest(dnhole_det,two));}
		
		}
	}
	
        // intrasite 3d-3d
	for (int i=0;i<this->dsites.size();i++)
	{
		int one=this->dsites[i];
		//for (int j=i+1;j<this->dsites.size();j++)
		for (int j=i+1;j<this->dsites.size();j++)
		{
			if (i!=j) {int two=this->dsites[j];
			hint+=(this->U3d3d)*double(btest(uphole_det,one)+btest(dnhole_det,one))*double(btest(uphole_det,two)+btest(dnhole_det,two));}
		
		}
	}
	
        // intrasite 3d-4s
	/*for (int i=0;i<this->dsites.size();i++)
	{
		int one=this->dsites[i];
		for (int j=0;j<this->ssites.size();j++)
		{
			int two=this->ssites[j];
			hint+=(this->U3d4s)*double(btest(uphole_det,one)+btest(dnhole_det,one))*double(btest(uphole_det,two)+btest(dnhole_det,two));
		}
	}*/
	
	int one=1;
	for (int j=0;j<this->ssites.size();j++)
	{
		int two=this->ssites[j];
		hint+=(1.0*this->U3d4s)*double(btest(uphole_det,one)+btest(dnhole_det,one))*double(btest(uphole_det,two)+btest(dnhole_det,two));
	}
	
 	// inter-site 2p-4s
	one=0;
	for (int j=0;j<this->ssites.size();j++)
	{
		int two=this->ssites[j];
		hint+=(1.0*this->U2p4s)*double(btest(uphole_det,one)+btest(dnhole_det,one))*double(btest(uphole_det,two)+btest(dnhole_det,two));
	}
		
 	// inter-site 2p-3d
	one=0;
	int two=1;
	//for (int j=0;j<this->dsites.size();j++)
	//{
	//int two=this->dsites[j];
	hint+=(1.0*this->U2p3d)*double(btest(uphole_det,one)+btest(dnhole_det,one))*double(btest(uphole_det,two)+btest(dnhole_det,two));
	//}
		
		
	one=1;
	two=0;
        hint+=(0.25*this->U1001)*double(btest(uphole_det,one)-btest(dnhole_det,one))*double(btest(uphole_det,two)-btest(dnhole_det,two));
        one=2; two=0;
	hint+=(0.25*this->U2002)*double(btest(uphole_det,one)-btest(dnhole_det,one))*double(btest(uphole_det,two)-btest(dnhole_det,two));
        one=2; two=1;
	hint+=(0.25*this->U2112)*double(btest(uphole_det,one)-btest(dnhole_det,one))*double(btest(uphole_det,two)-btest(dnhole_det,two));
	
	new_spin_dets.push_back(spin_det);
	new_uphole_dets.push_back(uphole_det);
	new_dnhole_dets.push_back(dnhole_det);
        hints_list.push_back( complex<double> (hint) );
	// 2-EXCHANGE U1001
	if (btest(uphole_det,0)==1 and btest(uphole_det,1)==0 and btest(dnhole_det,0)==0 and btest(dnhole_det,1)==1)
	{
		new_spin_dets.push_back(spin_det);
		int new_uphole_det=uphole_det;
		new_uphole_det=ibclr(new_uphole_det,0);
		new_uphole_det=ibset(new_uphole_det,1);
		new_uphole_dets.push_back(new_uphole_det);
		
		int new_dnhole_det=dnhole_det;
		new_dnhole_det=ibclr(new_dnhole_det,1);
		new_dnhole_det=ibset(new_dnhole_det,0);
		new_dnhole_dets.push_back(new_dnhole_det);
        	hints_list.push_back( complex<double> (-0.5*this->U1001) );
	}
	
	if (btest(uphole_det,0)==0 and btest(uphole_det,1)==1 and btest(dnhole_det,0)==1 and btest(dnhole_det,1)==0)
	{
		new_spin_dets.push_back(spin_det);
		int new_uphole_det=uphole_det;
		new_uphole_det=ibclr(new_uphole_det,1);
		new_uphole_det=ibset(new_uphole_det,0);
		new_uphole_dets.push_back(new_uphole_det);
		
		int new_dnhole_det=dnhole_det;
		new_dnhole_det=ibclr(new_dnhole_det,0);
		new_dnhole_det=ibset(new_dnhole_det,1);
		new_dnhole_dets.push_back(new_dnhole_det);
        	hints_list.push_back( complex<double> (-0.5*this->U1001) );
	}

	// 2-EXCHANGE U2002
	if (btest(uphole_det,0)==1 and btest(uphole_det,2)==0 and btest(dnhole_det,0)==0 and btest(dnhole_det,2)==1)
	{
		double gamma=1.0;
		if (btest(uphole_det,1)==1) gamma=gamma*-1.0;
		if (btest(dnhole_det,1)==1) gamma=gamma*-1.0;

		new_spin_dets.push_back(spin_det);
		int new_uphole_det=uphole_det;
		new_uphole_det=ibclr(new_uphole_det,0);
		new_uphole_det=ibset(new_uphole_det,2);
		new_uphole_dets.push_back(new_uphole_det);
		
		int new_dnhole_det=dnhole_det;
		new_dnhole_det=ibclr(new_dnhole_det,2);
		new_dnhole_det=ibset(new_dnhole_det,0);
		new_dnhole_dets.push_back(new_dnhole_det);
        	hints_list.push_back( complex<double> (-0.5*this->U2002*gamma) );
	}
	
	if (btest(uphole_det,0)==0 and btest(uphole_det,2)==1 and btest(dnhole_det,0)==1 and btest(dnhole_det,2)==0)
	{
		double gamma=1.0;
		if (btest(uphole_det,1)==1) gamma=gamma*-1.0;
		if (btest(dnhole_det,1)==1) gamma=gamma*-1.0;
		
		new_spin_dets.push_back(spin_det);
		int new_uphole_det=uphole_det;
		new_uphole_det=ibclr(new_uphole_det,2);
		new_uphole_det=ibset(new_uphole_det,0);
		new_uphole_dets.push_back(new_uphole_det);
		
		int new_dnhole_det=dnhole_det;
		new_dnhole_det=ibclr(new_dnhole_det,0);
		new_dnhole_det=ibset(new_dnhole_det,2);
		new_dnhole_dets.push_back(new_dnhole_det);
        	hints_list.push_back( complex<double> (-0.5*this->U2002*gamma) );
	}
	
	// 2-EXCHANGE U2112
	if (btest(uphole_det,1)==1 and btest(uphole_det,2)==0 and btest(dnhole_det,1)==0 and btest(dnhole_det,2)==1)
	{
		new_spin_dets.push_back(spin_det);
		int new_uphole_det=uphole_det;
		new_uphole_det=ibclr(new_uphole_det,1);
		new_uphole_det=ibset(new_uphole_det,2);
		new_uphole_dets.push_back(new_uphole_det);
		
		int new_dnhole_det=dnhole_det;
		new_dnhole_det=ibclr(new_dnhole_det,2);
		new_dnhole_det=ibset(new_dnhole_det,1);
		new_dnhole_dets.push_back(new_dnhole_det);
        	hints_list.push_back( complex<double> (-0.5*this->U2112) );
	}
	
	if (btest(uphole_det,1)==0 and btest(uphole_det,2)==1 and btest(dnhole_det,1)==1 and btest(dnhole_det,2)==0)
	{
		new_spin_dets.push_back(spin_det);
		int new_uphole_det=uphole_det;
		new_uphole_det=ibclr(new_uphole_det,2);
		new_uphole_det=ibset(new_uphole_det,1);
		new_uphole_dets.push_back(new_uphole_det);
		
		int new_dnhole_det=dnhole_det;
		new_dnhole_det=ibclr(new_dnhole_det,1);
		new_dnhole_det=ibset(new_dnhole_det,2);
		new_dnhole_dets.push_back(new_dnhole_det);
        	hints_list.push_back( complex<double> (-0.5*this->U2112) );
	}


}
////////////////////////////////////////////////////////////////////////////////////////////////////
void mno_setup(string filename, 
               MnO_Model &mno)

{    
    ///////////////////////////////////////////
    // Set the 3-band Hamiltonian
    ///////////////////////////////////////////
    bool found=true;;
    string str,str_ret;
    int nup_holes,ndn_holes;
    double scale,E2p,E3d,E4s,Tpi,T01,T02,T12,U2p,U3d,U4s,U2p2p,U3d3d,U3d4s,U2p3d,U2p4s,U1001,U2002,U2112;

    // coupling parameters search
    search_for("nup_holes",filename,str_ret,found);
    if (found){nup_holes=str_to_int(str_ret);} else{nup_holes=0;}
    
    search_for("ndn_holes",filename,str_ret,found);
    if (found){ndn_holes=str_to_int(str_ret);} else{ndn_holes=0;}
    
    search_for("scale",filename,str_ret,found);
    if (found){scale=str_to_d(str_ret);} else{scale=1.0;}
    
    search_for("E2p",filename,str_ret,found);
    if (found){E2p=str_to_d(str_ret);} else{E2p=0.0;}
    
    search_for("E3d",filename,str_ret,found);
    if (found){E3d=str_to_d(str_ret);} else{E3d=0.0;}
    
    search_for("E4s",filename,str_ret,found);
    if (found){E4s=str_to_d(str_ret);} else{E4s=0.0;}
    
    search_for("Tpi",filename,str_ret,found);
    if (found){Tpi=str_to_d(str_ret);} else{Tpi=0.0;}
    
    search_for("T01",filename,str_ret,found);
    if (found){T01=str_to_d(str_ret);} else{T01=0.0;}
    
    search_for("T02",filename,str_ret,found);
    if (found){T02=str_to_d(str_ret);} else{T02=0.0;}
    
    search_for("T12",filename,str_ret,found);
    if (found){T12=str_to_d(str_ret);} else{T12=0.0;}
    
    search_for("U2p",filename,str_ret,found);
    if (found){U2p=str_to_d(str_ret);} else{U2p=0.0;}
    
    search_for("U3d",filename,str_ret,found);
    if (found){U3d=str_to_d(str_ret);} else{U3d=0.0;}
    
    search_for("U4s",filename,str_ret,found);
    if (found){U4s=str_to_d(str_ret);} else{U4s=0.0;}
    
    search_for("U2p2p",filename,str_ret,found);
    if (found){U2p2p=str_to_d(str_ret);} else{U2p2p=0.0;}
    
    search_for("U3d3d",filename,str_ret,found);
    if (found){U3d3d=str_to_d(str_ret);} else{U3d3d=0.0;}
    
    search_for("U3d4s",filename,str_ret,found);
    if (found){U3d4s=str_to_d(str_ret);} else{U3d4s=0.0;}
    
    search_for("U2p3d",filename,str_ret,found);
    if (found){U2p3d=str_to_d(str_ret);} else{U2p3d=0.0;}
    
    search_for("U2p4s",filename,str_ret,found);
    if (found){U2p4s=str_to_d(str_ret);} else{U2p4s=0.0;}
    
    search_for("U1001",filename,str_ret,found);
    if (found){U1001=str_to_d(str_ret);} else{U1001=0.0;}
    
    search_for("U2002",filename,str_ret,found);
    if (found){U2002=str_to_d(str_ret);} else{U2002=0.0;}
    
    search_for("U2112",filename,str_ret,found);
    if (found){U2112=str_to_d(str_ret);} else{U2112=0.0;}
    
    mno.init(nup_holes, ndn_holes, scale, E2p, E3d, E4s, Tpi, T01, T02, T12, U2p, U3d, U4s, U2p2p, U3d3d, U3d4s, U2p3d, U2p4s, U1001, U2002, U2112);

}
