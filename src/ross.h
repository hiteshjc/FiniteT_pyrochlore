#ifndef ROSS_MODEL_HEADER
#define ROSS_MODEL_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 ROSS MODEL
//
//////////////////////////////////////////////////////////////////////////////////////////
#include"hamiltonian.h"

void make_J_mats(double J1,double J2,double J3,double J4, 
		 std::vector<RMatrix> &Jmats);

void make_g_mats(double gxy,double gz, std::vector<RMatrix> &gmats); 

class Spin_Half_Ross: public Ham
{
    public:
    double J1,J2,J3,J4,gxy,gz;
    double hx,hy,hz;
    double hxTesla,hyTesla,hzTesla;
    std::vector<RMatrix> Jmats;
    std::vector<RMatrix> gmats;
    std::vector<int> sublattice;
    std::vector<int> bondtype;
    std::vector< std::vector<int> > neighbors;
    IMatrix 	     btmatrix;

    void init(std::vector< std::vector<int> > &filepairs, std::vector<int> & sublattice, 
	      double J1, double J2, double J3, double J4, double gxy, double gz, double hx, double hy, double hz)
    {
	this->bondtype.clear();
	this->pairs.clear();
	this->sublattice=sublattice;
	double mub=5.7883818012*0.01;	
	// To get number of sites, and to bring pairs in i<j format
	int maxs=0;
	for (int i=0;i<filepairs.size();i++)
	{
		int site1=filepairs[i][0];
		int site2=filepairs[i][1];
		std::vector<int> pair;
		int minv=min(site1,site2);
		int maxv=max(site1,site2);
		pair.push_back(minv);
		pair.push_back(maxv);
		this->pairs.push_back(pair);
		if (site1>maxs) maxs=site1;
		if (site2>maxs) maxs=site2;
	}
        this->num_sites=maxs+1;
	this->neighbors.resize(this->num_sites);
	this->btmatrix.resize(this->num_sites,this->num_sites);
	for(int i=0;i<this->num_sites*this->num_sites;i++) this->btmatrix[i]=-1;

	for (int i=0;i<this->pairs.size();i++)
	{
		int site1=this->pairs[i][0];
		int site2=this->pairs[i][1];
		this->neighbors[site1].push_back(site2);
		this->neighbors[site2].push_back(site1);

		//pairs=[[i,j],....] have been brought in i<j format
		if (sublattice[site1]==0 and sublattice[site2]==1) 
		{this->bondtype.push_back(0); this->btmatrix(site1,site2)=0; this->btmatrix(site2,site1)=1;}
		
		if (sublattice[site1]==1 and sublattice[site2]==0)
		{this->bondtype.push_back(1); this->btmatrix(site1,site2)=1; this->btmatrix(site2,site1)=0;}
		
		if (sublattice[site1]==0 and sublattice[site2]==2)
		{this->bondtype.push_back(2); this->btmatrix(site1,site2)=2; this->btmatrix(site2,site1)=3;}
		
		if (sublattice[site1]==2 and sublattice[site2]==0)
		{this->bondtype.push_back(3); this->btmatrix(site1,site2)=3; this->btmatrix(site2,site1)=2;}
		
		if (sublattice[site1]==0 and sublattice[site2]==3)
		{this->bondtype.push_back(4); this->btmatrix(site1,site2)=4; this->btmatrix(site2,site1)=5;}
		
		if (sublattice[site1]==3 and sublattice[site2]==0)
		{this->bondtype.push_back(5); this->btmatrix(site1,site2)=5; this->btmatrix(site2,site1)=4;}
		
		if (sublattice[site1]==1 and sublattice[site2]==2)
		{this->bondtype.push_back(6); this->btmatrix(site1,site2)=6; this->btmatrix(site2,site1)=7;}
		
		if (sublattice[site1]==2 and sublattice[site2]==1) 
		{this->bondtype.push_back(7); this->btmatrix(site1,site2)=7; this->btmatrix(site2,site1)=6;}
		
		if (sublattice[site1]==1 and sublattice[site2]==3)
		{this->bondtype.push_back(8); this->btmatrix(site1,site2)=8; this->btmatrix(site2,site1)=9;}
		
		if (sublattice[site1]==3 and sublattice[site2]==1)
		{this->bondtype.push_back(9); this->btmatrix(site1,site2)=9; this->btmatrix(site2,site1)=8;}
		
		if (sublattice[site1]==2 and sublattice[site2]==3)
		{this->bondtype.push_back(10); this->btmatrix(site1,site2)=10; this->btmatrix(site2,site1)=11;}
		
		if (sublattice[site1]==3 and sublattice[site2]==2)
		{this->bondtype.push_back(11); this->btmatrix(site1,site2)=11; this->btmatrix(site2,site1)=10;}
	}

	this->hxTesla=hx;
	this->hyTesla=hy;
	this->hzTesla=hz;
	this->hx=hx*mub;
	this->hy=hy*mub;
	this->hz=hz*mub;
	this->J1=J1;
	this->J2=J2;
	this->J3=J3;
	this->J4=J4;
	this->gxy=gxy;
	this->gz=gz;
	make_J_mats(J1,J2,J3,J4,this->Jmats);
	make_g_mats(gxy,gz,this->gmats);
    }

    void init(std::vector< std::vector<int> > &filepairs, std::vector<int> & sublattice, 
	      double J1, double J2, double J3, double J4, double gxy, double gz, double hx, double hy, double hz, 
	      std::vector<int> &t1, std::vector<int> &t2, std::vector<int> &t3, double k1, double k2, double k3)
    {
	this->bondtype.clear();
	this->pairs=pairs;
	this->sublattice=sublattice;
	double mub=5.7883818012*0.01;	
	// To get number of sites, and to bring pairs in i<j format
	int maxs=0;
	for (int i=0;i<filepairs.size();i++)
	{
		int site1=filepairs[i][0];
		int site2=filepairs[i][1];
		std::vector<int> pair;
		int minv=min(site1,site2);
		int maxv=max(site1,site2);
		pair.push_back(minv);
		pair.push_back(maxv);
		this->pairs.push_back(pair);
		if (site1>maxs) maxs=site1;
		if (site2>maxs) maxs=site2;
	}
        this->num_sites=maxs+1;
	this->neighbors.resize(this->num_sites);
	this->btmatrix.resize(this->num_sites,this->num_sites);
	for(int i=0;i<this->num_sites*this->num_sites;i++) this->btmatrix[i]=-1;

	for (int i=0;i<this->pairs.size();i++)
	{
		int site1=this->pairs[i][0];
		int site2=this->pairs[i][1];
		this->neighbors[site1].push_back(site2);
		this->neighbors[site2].push_back(site1);

		//pairs=[[i,j],....] have been brought in i<j format
		if (sublattice[site1]==0 and sublattice[site2]==1) 
		{this->bondtype.push_back(0); this->btmatrix(site1,site2)=0; this->btmatrix(site2,site1)=1;}
		
		if (sublattice[site1]==1 and sublattice[site2]==0)
		{this->bondtype.push_back(1); this->btmatrix(site1,site2)=1; this->btmatrix(site2,site1)=0;}
		
		if (sublattice[site1]==0 and sublattice[site2]==2)
		{this->bondtype.push_back(2); this->btmatrix(site1,site2)=2; this->btmatrix(site2,site1)=3;}
		
		if (sublattice[site1]==2 and sublattice[site2]==0)
		{this->bondtype.push_back(3); this->btmatrix(site1,site2)=3; this->btmatrix(site2,site1)=2;}
		
		if (sublattice[site1]==0 and sublattice[site2]==3)
		{this->bondtype.push_back(4); this->btmatrix(site1,site2)=4; this->btmatrix(site2,site1)=5;}
		
		if (sublattice[site1]==3 and sublattice[site2]==0)
		{this->bondtype.push_back(5); this->btmatrix(site1,site2)=5; this->btmatrix(site2,site1)=4;}
		
		if (sublattice[site1]==1 and sublattice[site2]==2)
		{this->bondtype.push_back(6); this->btmatrix(site1,site2)=6; this->btmatrix(site2,site1)=7;}
		
		if (sublattice[site1]==2 and sublattice[site2]==1) 
		{this->bondtype.push_back(7); this->btmatrix(site1,site2)=7; this->btmatrix(site2,site1)=6;}
		
		if (sublattice[site1]==1 and sublattice[site2]==3)
		{this->bondtype.push_back(8); this->btmatrix(site1,site2)=8; this->btmatrix(site2,site1)=9;}
		
		if (sublattice[site1]==3 and sublattice[site2]==1)
		{this->bondtype.push_back(9); this->btmatrix(site1,site2)=9; this->btmatrix(site2,site1)=8;}
		
		if (sublattice[site1]==2 and sublattice[site2]==3)
		{this->bondtype.push_back(10); this->btmatrix(site1,site2)=10; this->btmatrix(site2,site1)=11;}
		
		if (sublattice[site1]==3 and sublattice[site2]==2)
		{this->bondtype.push_back(11); this->btmatrix(site1,site2)=11; this->btmatrix(site2,site1)=10;}
	}
		
	this->hxTesla=hx;
	this->hyTesla=hy;
	this->hzTesla=hz;
	this->hx=hx*mub;
	this->hy=hy*mub;
	this->hz=hz*mub;
	this->J1=J1;
	this->J2=J2;
	this->J3=J3;
	this->J4=J4;
	this->gxy=gxy;
	this->gz=gz;
	this->t1=t1;
	this->t2=t2;
	this->t3=t3;
	this->k1=k1;
	this->k2=k2;
	this->k3=k3;
	make_J_mats(J1,J2,J3,J4,this->Jmats);
	make_g_mats(gxy,gz,this->gmats);
    }
    
    void operator()(int64_t spin_det,
		    std::vector<int64_t> &new_spin_dets,
                    std::vector< complex<double> > &hints_list, int &nconns);
    
    Ham* clone() const
    {
        return new Spin_Half_Ross(*this);
    }    
};

void ross_setup(std::string filename, 
               Spin_Half_Ross &ross);

void ross_sym_setup(std::string filename, 
               Spin_Half_Ross &ross);


#endif
