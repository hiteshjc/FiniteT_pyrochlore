#ifndef ONE_BAND_HEADER
#define ONE_BAND_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 ONE BAND MODEL
//
//////////////////////////////////////////////////////////////////////////////////////////
#include"hamiltonian.h"
#include"global.h"
#include"hamiltonian_spin_functions.h"
#include"number_functions.h"
#include"search_for.h"
#include"printing_functions.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////// 
class One_Band_Model: public Ham
{
    public:
    int 			    nup_holes,ndn_holes,nup_spins;
    double 			    tCuCu, Udd, UCuCu, epsd, sz, hst;
    std::vector< std::vector<int> > neighbors;
    int 			    hilbert_spins;
    int 			    hilbert_upholes;
    int 			    hilbert_dnholes;
    int 			    hilbert;
    std::vector< std::vector<int> > CuCupairs;
    std::vector<int>                Cusites,etaCu;
    std::vector<int>                signsCuCupairs;

    public:
    void init(int nup_holes,int ndn_holes, double tCuCu, double Udd, 
	      double epsd, double hst, double UCuCu, std::vector< std::vector<int> > CuCupairs, std::vector<int> Cusites, std::vector<int> etaCu, std::vector<int> signsCuCupairs)
    {
	this->signsCuCupairs=signsCuCupairs;
	this->hst=hst;
	this->epsd=epsd;
	this->CuCupairs=CuCupairs;
	this->Cusites=Cusites;
	this->etaCu=etaCu;
	this->num_sites=this->Cusites.size();
	this->nup_holes=nup_holes;
	this->ndn_holes=ndn_holes;
	this->sz=double(this->num_sites/2.0);
	this->nup_spins=int(double(this->num_sites/2.0)+(this->sz));
	this->hilbert_spins=n_choose_k(this->num_sites,this->nup_spins);
	this->hilbert_upholes=n_choose_k(this->num_sites,this->nup_holes);
	this->hilbert_dnholes=n_choose_k(this->num_sites,this->ndn_holes);
	this->hilbert=this->hilbert_spins*this->hilbert_upholes*this->hilbert_dnholes;
        this->tCuCu=tCuCu;this->UCuCu=UCuCu;this->Udd=Udd;
        cout<<endl;
        cout<<"--------------------------------------------"<<endl;
        cout<<"N_sites    ="<<this->num_sites<<endl;
        cout<<"N_CuCu_bonds="<<this->CuCupairs.size()<<endl;
        cout<<"nup_holes  ="<<this->nup_holes<<endl;
        cout<<"ndn_holes  ="<<this->ndn_holes<<endl;
        cout<<"tCuCu      ="<<this->tCuCu<<endl;
        cout<<"UCuCu      ="<<this->UCuCu<<endl;
        cout<<"Udd        ="<<this->Udd<<endl;
        cout<<"epsd       ="<<this->epsd<<endl;
        cout<<"H_staggered="<<this->hst<<endl;
        cout<<endl;
        cout<<"--------------------------------------------"<<endl;
	cout<<endl;
    }
    
/////////////////////////////////////////////////////////////////////////// 
void operator()     (std::vector<int> const &config_spins,
                     std::vector<int> const &config_up_holes,
                     std::vector<int> const &config_down_holes,
                     std::vector< std::vector<int> > &touched_sites_list_spins, 
                     std::vector< std::vector<int> > &vals_on_touched_list_spins,
                     std::vector< std::vector<int> > &touched_sites_list_upholes, 
                     std::vector< std::vector<int> > &vals_on_touched_list_upholes,
                     std::vector< std::vector<int> > &touched_sites_list_dnholes, 
                     std::vector< std::vector<int> > &vals_on_touched_list_dnholes,
                     std::vector<double> &hints_list);

void operator()
                   (int spin_det,
                    int up_hole_det,
                    int dn_hole_det,
                    std::vector<int> &new_spin_dets,
                    std::vector<int> &new_uphole_dets,
                    std::vector<int> &new_dnhole_dets,
                    std::vector< complex<double> > &hints_list);

    Ham* clone() const
    {
	return new One_Band_Model(*this);
    }    
};

void one_band_setup(std::string filename, 
               One_Band_Model &one_band);

#endif
