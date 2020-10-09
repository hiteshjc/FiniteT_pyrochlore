#ifndef YBMG_MODEL_HEADER
#define YBMG_MODEL_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 YbMg MODEL
//
//////////////////////////////////////////////////////////////////////////////////////////
#include"hamiltonian.h"


class Spin_Half_Ybmg: public Ham
{
    public:
    double J1zz, J1pm, J1ppmm, J1zpm, J2zz, J2pm;
    double gxy,gz;
    double hx,hy,hz;
    double hxTesla,hyTesla,hzTesla;
    std::vector< std::vector<int> > neighbors;
    std::vector<int> bondtypes;
    std::vector<complex<double> > phases;
    Matrix phasematrix; 
 
    void init(std::vector< std::vector<int> > &pairs1, 
    	      std::vector< std::vector<int> > &pairs2, 
	      std::vector<int> &bondtypes, 
	      double J1zz, double J1pm, double J1ppmm, double J1zpm, 
	      double J2zz, double J2pm, 
	      double gxy, double gz, 
	      double hx, double hy, double hz) 
    {
	double pi=3.141592653589793;
	complex<double> omega=complex<double>(cos(2.0*pi/3.0),sin(2.0*pi/3.0));
	complex<double> omega2=complex<double>(cos(2.0*pi/3.0),-sin(2.0*pi/3.0));

	this->phases.clear();
	this->neighbors.clear();
	this->pairs1=pairs1;
	this->pairs2=pairs2;
	this->bondtypes=bondtypes;
	double mub=5.7883818012*0.01;	
	
	// To get number of sites
	int maxs=0;
	for (int i=0;i<pairs1.size();i++)
	{
		int site1=pairs1[i][0];
		int site2=pairs1[i][1];
		if (site1>maxs) maxs=site1;
		if (site2>maxs) maxs=site2;
		if (this->bondtypes[i]==0) this->phases.push_back(1.0);
		if (this->bondtypes[i]==1) this->phases.push_back(omega);
		if (this->bondtypes[i]==2) this->phases.push_back(omega2);
	}
        this->num_sites=maxs+1;
	cout<<" Number of sites = "<<this->num_sites<<endl;
	for (int i=0;i<this->num_sites;i++) this->neighbors.push_back(std::vector<int>());
	this->phasematrix.resize(this->num_sites,this->num_sites);

	// Make neighbors	
	for (int i=0;i<pairs1.size();i++)
	{
		int site1=pairs1[i][0];
		int site2=pairs1[i][1];
		this->neighbors[site1].push_back(site2);
		this->neighbors[site2].push_back(site1);
		this->phasematrix(site1,site2)=this->phases[i];
		this->phasematrix(site2,site1)=this->phases[i];
	}

	this->hxTesla=hx;
	this->hyTesla=hy;
	this->hzTesla=hz;
	this->hx=hx*mub;
	this->hy=hy*mub;
	this->hz=hz*mub;
	this->J1zz=J1zz;
	this->J1pm=J1pm;
	this->J1ppmm=J1ppmm;
	this->J1zpm=J1zpm;
	this->J2zz=J2zz; 
	this->J2pm=J2pm; 
	this->gxy=gxy;
	this->gz=gz;
    }

    void init(std::vector< std::vector<int> > &pairs1, 
    	      std::vector< std::vector<int> > &pairs2, 
	      std::vector<int> &bondtypes, 
	      double J1zz, double J1pm, double J1ppmm, double J1zpm, 
	      double J2zz, double J2pm, 
	      double gxy, double gz, 
	      double hx, double hy, double hz, 
	      std::vector<int> &t1, std::vector<int> &t2, std::vector<int> &t3, 
	      double k1, double k2, double k3)
    {
	double pi=3.141592653589793;
	complex<double> omega=complex<double>(cos(2.0*pi/3.0),sin(2.0*pi/3.0));
	complex<double> omega2=complex<double>(cos(2.0*pi/3.0),-sin(2.0*pi/3.0));

	this->phases.clear();
	this->neighbors.clear();
	this->pairs1=pairs1;
	this->pairs2=pairs2;
	this->bondtypes=bondtypes;
	double mub=5.7883818012*0.01;	
	
	// To get number of sites
	int maxs=0;
	for (int i=0;i<pairs1.size();i++)
	{
		int site1=pairs1[i][0];
		int site2=pairs1[i][1];
		if (site1>maxs) maxs=site1;
		if (site2>maxs) maxs=site2;
		if (this->bondtypes[i]==0) this->phases.push_back(1.0);
		if (this->bondtypes[i]==1) this->phases.push_back(omega);
		if (this->bondtypes[i]==2) this->phases.push_back(omega2);
	}
        this->num_sites=maxs+1;
	cout<<" Number of sites = "<<this->num_sites<<endl;
	for (int i=0;i<this->num_sites;i++) this->neighbors.push_back(std::vector<int>());
	this->phasematrix.resize(this->num_sites,this->num_sites);

	// Make neighbors	
	for (int i=0;i<pairs1.size();i++)
	{
		int site1=pairs1[i][0];
		int site2=pairs1[i][1];
		this->neighbors[site1].push_back(site2);
		this->neighbors[site2].push_back(site1);
		this->phasematrix(site1,site2)=this->phases[i];
		this->phasematrix(site2,site1)=this->phases[i];
	}

	this->hxTesla=hx;
	this->hyTesla=hy;
	this->hzTesla=hz;
	this->hx=hx*mub;
	this->hy=hy*mub;
	this->hz=hz*mub;
	this->J1zz=J1zz;
	this->J1pm=J1pm;
	this->J1ppmm=J1ppmm;
	this->J1zpm=J1zpm;
	this->J2zz=J2zz; 
	this->J2pm=J2pm; 
	this->gxy=gxy;
	this->gz=gz;
	this->t1=t1;
	this->t2=t2;
	this->t3=t3;
	this->k1=k1;
	this->k2=k2;
	this->k3=k3;
    }
    
    void operator()(int64_t spin_det,
		    std::vector<int64_t> &new_spin_dets,
                    std::vector< complex<double> > &hints_list, int &nconns);
    
    Ham* clone() const
    {
        return new Spin_Half_Ybmg(*this);
    }    
};

void ybmg_setup(std::string filename, 
               Spin_Half_Ybmg &ybmg);

void ybmg_sym_setup(std::string filename, 
               Spin_Half_Ybmg &ybmg);


#endif
