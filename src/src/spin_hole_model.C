#include"spin_hole_model.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 Spin Hole MODEL
/////////////////////////////////////////////////////////////////////////////////////////////////////
void Spin_Hole_Model::operator()
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

void Spin_Hole_Model::operator()
                   (int spin_det,
                    int uphole_det,
                    int dnhole_det,
                    std::vector<int> &new_spin_dets,
                    std::vector<int> &new_uphole_dets,
                    std::vector<int> &new_dnhole_dets,
                    std::vector< complex<double> > &hints_list)
{
	if (this->J!=0)
	{
	double szsz=0.0;
	std::vector<int> bits(this->num_sites);
	for (int i=0;i<this->num_sites;i++) bits[i]=btest(spin_det,i);

	for (int i=0;i<this->pairs.size();i++)
	{
		int first=this->pairs[i][0];
		int second=this->pairs[i][1];
		int a=bits[first];
		int b=bits[second];
		// J/2 Si+Sj- + J/2 Si-Sj+
		if (a==0 and b==1)
		{
		    int new_state=ibset(spin_det,first);
		    new_state=ibclr(new_state,second);
	            new_spin_dets.push_back(new_state);
	            new_uphole_dets.push_back(uphole_det);
	            new_dnhole_dets.push_back(dnhole_det);
	            hints_list.push_back(0.5*this->J);
		    szsz+=-0.25*this->J;
		}
		else if(a==1 and b==0)
		{
		    int new_state=ibset(spin_det,second);
		    new_state=ibclr(new_state,first);
	            new_spin_dets.push_back(new_state);
	            new_uphole_dets.push_back(uphole_det);
	            new_dnhole_dets.push_back(dnhole_det);
	            hints_list.push_back(0.5*this->J);
		    szsz+=-0.25*this->J;
		}
		else
		{
		    szsz+=0.25*this->J;
		}
	}
	new_spin_dets.push_back(spin_det);
	new_uphole_dets.push_back(uphole_det);
	new_dnhole_dets.push_back(dnhole_det);
	hints_list.push_back(szsz);

	}

	if (this->t!=0)
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

		for (int i=0;i<this->pairs.size();i++)
		{
			int su=1;
			int sd=1;
			int first=this->pairs[i][0];
			int second=this->pairs[i][1];
			
			// Hubbard t
			int mx=max(first,second);
			int mn=min(first,second);
			if (abs(upsigns[mx]-upsigns[mn+1])%2 !=0 ) su=-1;
			if (abs(dnsigns[mx]-dnsigns[mn+1])%2 !=0 ) sd=-1;
			calc_hints_fermion_hop(this->t, first, second, su, sd, spin_det, uphole_det, dnhole_det,
					       new_spin_dets, new_uphole_dets, new_dnhole_dets, 
					       hints_list);
		}
	}

	if (this->U!=0.0 or this->V!=0.0)
	{
		// Hubbard onsite U + V ni nj
        	calc_hints_U_plus_V(this->U, this->V,this->num_sites, this->pairs, this->neighbors, 
			            spin_det, uphole_det, dnhole_det,
			            new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			            hints_list);
	}
	
	if (this->D!=0.0)
	{
		// D - hole
        	calc_hints_U_plus_D(0.0, this->D, this->num_sites, this->pairs, this->neighbors, 
			            this->neighbors_within_rh,spin_det, uphole_det, dnhole_det,
			            new_spin_dets, new_uphole_dets, new_dnhole_dets, 
			            hints_list);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void Spin_Hole_Model::set_pairs_square_lattice()
{
	std::vector<int> pair(2);
	int l_x=this->l_x;
	int l_y=this->l_y;
	bool pbc=this->pbc;
	this->pairs.clear();
	this->neighbors.clear();
	this->neighbors_within_rh.clear();
 	this->eta.clear();

	for(int i=0;i<this->num_sites;i++) 
	{
		this->neighbors.push_back(std::vector<int>());
		this->neighbors_within_rh.push_back(std::vector<int>());
	}
	
 	// Nearest neighbor pairs 
	for(int i=0;i<this->num_sites;i++)
	{
		int xi=i%l_x;
		int yi=(i-xi)/l_x;
		if ( (xi+yi) %2 ==0 ) {this->eta.push_back(1);}
		else                  {this->eta.push_back(-1);}

		int ind_l,ind_r,ind_u,ind_d;

		// Right
		if (xi==l_x-1 and pbc)
		{ 
			int xr=0; 
			ind_r=(yi*l_x) + xr;
			if (i<ind_r){pair[0]=i;pair[1]=ind_r;this->pairs.push_back(pair);}
		}
		
		if (xi<l_x-1)
		{
			int xr=xi+1;
			ind_r=(yi*l_x) + xr;
			if (i<ind_r){pair[0]=i;pair[1]=ind_r;this->pairs.push_back(pair);}
		}	
	
		// Left
		if (xi==0 and pbc)
		{ 
			int xl=l_x-1; 
			ind_l=(yi*l_x) + xl;
			if (i<ind_l and ind_l!=ind_r){pair[0]=i;pair[1]=ind_l;this->pairs.push_back(pair);}
		}
		
		if (xi>0)
		{
			int xl=xi-1;
			ind_l=(yi*l_x) + xl;
			if (i<ind_l and ind_l!=ind_r){pair[0]=i;pair[1]=ind_l;this->pairs.push_back(pair);}
		}	

		// Up 
		if (yi==l_y-1 and pbc)
		{ 
			int yu=0; 
			ind_u=(yu*l_x) + xi;
			if (i<ind_u){pair[0]=i;pair[1]=ind_u;this->pairs.push_back(pair);}
		}
		
		if (yi<l_y-1)
		{
			int yu=yi+1;
			ind_u=(yu*l_x) + xi;
			if (i<ind_u){pair[0]=i;pair[1]=ind_u;this->pairs.push_back(pair);}
		}	
	
		// Down	
		if (yi==0 and pbc)
		{ 
			int yd=l_y-1; 
			ind_d=(yd*l_x) + xi;
			if (i<ind_d and ind_d!=ind_u){pair[0]=i;pair[1]=ind_d;this->pairs.push_back(pair);}
		}
		
		if (yi>0)
		{
			int yd=yi-1;
			ind_d=(yd*l_x) + xi;
			if (i<ind_d and ind_d!=ind_u){pair[0]=i;pair[1]=ind_d;this->pairs.push_back(pair);}
		}	
	}

	// Nearest neighbors from pairs constructed
	for (int i=0;i<this->pairs.size();i++)
	{
		this->neighbors[pairs[i][0]].push_back(pairs[i][1]);
		this->neighbors[pairs[i][1]].push_back(pairs[i][0]);
	}	

	// Set distance matrix
	this->distance.resize(this->num_sites,this->num_sites);
	
	for (int i=0;i<this->num_sites;i++)
	{
		int xi=i%l_x;
		int yi=(i-xi)/l_x;
		for (int j=0;j<this->num_sites;j++)
		{
			int xj=j%l_x;
			int yj=(j-xj)/l_x;
			if (pbc)
			{
				int d=0,dmin=0;
				for (int a=-1;a<=1;a++)
				{
					for (int b=-1;b<=1;b++)
					{
					   d=((xi-xj-a*l_x)*(xi-xj-a*l_x)) + ((yi-yj-b*l_y)*(yi-yj-b*l_y));
					   if (d<dmin or (a==-1 and b==-1)) dmin=d; 
					}
				}
				this->distance(i,j)=sqrt(double(dmin));
			}
			else
			{
				this->distance(i,j)=double((xi-xj)*(xi-xj)) + double((yi-yj)*(yi-yj));
				this->distance(i,j)=sqrt(this->distance(i,j));
			}
		}
	}
	// Look at distance matrix
	for (int i=0;i<this->num_sites;i++)
	{
		for (int j=0;j<this->num_sites;j++)
		{
			if (this->distance(i,j)<=(rh+1.0e-6)) neighbors_within_rh[i].push_back(j); 
		}
	}

	// Print everything
	cout<<"Pairs"<<endl;
	print_mathematica_pairs(this->pairs);
	cout<<endl;
	cout<<"Neighbors"<<endl;
	print_mat_int(this->neighbors);
	cout<<endl;
	cout<<"Neighbors with rh="<<rh<<endl;
	print_mat_int(this->neighbors_within_rh);
	cout<<endl;
	
}



////////////////////////////////////////////////////////////////////////////////////////////////////
void spin_hole_setup(string filename, 
               Spin_Hole_Model &spin_hole)

{    
    ///////////////////////////////////////////
    // Set the Boson Hamiltonian
    ///////////////////////////////////////////

    bool found=true,pbc;
    string str,str_ret;
    int l_x,l_y,nup_holes,ndn_holes;
    double J,t,D,U,V,rh,sz;

    // coupling parameters search
    search_for("l_x",filename,str_ret,found);
    if (found){l_x=str_to_int(str_ret);} else {l_x=1;}

    search_for("l_y",filename,str_ret,found);
    if (found){l_y=str_to_int(str_ret);} else {l_y=1;}
    
    search_for("sz",filename,str_ret,found);
    if (found){sz=str_to_d(str_ret);} else{sz=double(l_x*l_y)/2.0;}
    
    search_for("nup_holes",filename,str_ret,found);
    if (found){nup_holes=str_to_int(str_ret);} else{nup_holes=0;}
    
    search_for("ndn_holes",filename,str_ret,found);
    if (found){ndn_holes=str_to_int(str_ret);} else{ndn_holes=0;}
    
    search_for("J",filename,str_ret,found);
    if (found){J=str_to_d(str_ret);} else{J=0.0;}
    
    search_for("D",filename,str_ret,found);
    if (found){D=str_to_d(str_ret);} else{D=0.0;}
    
    search_for("t",filename,str_ret,found);
    if (found){t=str_to_d(str_ret);} else{t=0.0;}
    
    search_for("pbc",filename,str_ret,found);
    if (found){pbc=str_to_bool(str_ret);} else{pbc=true;}
 
    search_for("U",filename,str_ret,found);
    if (found){U=str_to_d(str_ret);} else{U=0.0;}
    
    search_for("V",filename,str_ret,found);
    if (found){V=str_to_d(str_ret);} else{V=0.0;}
    
    search_for("rh",filename,str_ret,found);
    if (found){rh=str_to_d(str_ret);} else{rh=1.0;}

    if (nup_holes>l_x*l_y or ndn_holes>l_x*l_y) 
    {cout<<"nholes (UP or DOWN ) can NOT exceed number of sites"<<endl;cout<<endl;exit(1);}    
    
    spin_hole.init(l_x,l_y,pbc,sz,nup_holes,ndn_holes,J,D,t,U,V,rh);
	
}


