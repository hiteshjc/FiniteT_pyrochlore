#ifndef OLEG_MODEL_HEADER
#define OLEG_MODEL_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 OLEG MODEL
//
//////////////////////////////////////////////////////////////////////////////////////////

#include"hamiltonian.h"
class Spin_Half_Oleg: public Ham
{
    public:
    int fhilbert,hilbert; 
    double jut,jh,jh2;
    
    void init(std::vector< std::vector<int> > hexagons,
    	      std::vector< std::vector<int> > up_triangles,
	      double jh, double jut, double jh2)
    {
        this->hexagons=hexagons;
        this->up_triangles=up_triangles;
	int max=0;
        for (int i=0;i<hexagons.size();i++)
        {
            for (int j=0;j<6;j++)
            {
                if (hexagons[i][j]>max) {max=hexagons[i][j];}
            }
        }
        this->num_sites=max+1;
	this->hilbert=pow(2,this->num_sites);
	this->jut=jut;
	this->jh=jh;
	this->jh2=jh2;
    }

    void init(std::vector< std::vector<int> > hexagons,
    	      std::vector< std::vector<int> > up_triangles,
	      double jh, double jut)
    {
        this->hexagons=hexagons;
        this->up_triangles=up_triangles;
	int max=0;
        for (int i=0;i<hexagons.size();i++)
        {
            for (int j=0;j<6;j++)
            {
                if (hexagons[i][j]>max) {max=hexagons[i][j];}
            }
        }
        this->num_sites=max+1;
	this->hilbert=pow(2,this->num_sites);
	this->jut=jut;
	this->jh=jh;
	this->jh2=0.0;
    }
    
    void init(std::vector< std::vector<int> > hexagons,
    	      std::vector< std::vector<int> > up_triangles,
	      double jh, double jut,
	      std::vector<int> t1,
	      std::vector<int> t2,
	      double k1, double k2)
    {
        this->hexagons=hexagons;
        this->up_triangles=up_triangles;
	int max=0;
        for (int i=0;i<hexagons.size();i++)
        {
            for (int j=0;j<6;j++)
            {
                if (hexagons[i][j]>max) {max=hexagons[i][j];}
            }
        }
        this->num_sites=max+1;
	this->fhilbert=pow(2,this->num_sites);
	this->hilbert=pow(2,this->num_sites);
	this->jut=jut;
	this->jh=jh;
	this->t1=t1;
	this->t2=t2;
	this->k1=k1;
	this->k2=k2;
	this->jh2=0.0;
    }
    
    void operator()(int spin_det,
		    std::vector<int> &new_spin_dets,
                    std::vector< complex<double> > &hints_list);
    
    Ham* clone() const
    {
        return new Spin_Half_Oleg(*this);
    }    
};

void oleg_setup(std::string filename, 
               Spin_Half_Oleg &oleg);

void oleg_sym_setup(std::string filename, 
               Spin_Half_Oleg &oleg);


#endif
