#ifndef THREE_BAND_HEADER
#define THREE_BAND_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 THREE BAND MODEL
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
class Three_Band_Model: public Ham
{
    public:
    int 			    nup_holes,ndn_holes,nup_spins;
    double 			    J,D,tpd,tpp,Upp,Udd,Upd,epsp,epsd,sz,hst;
    std::vector< std::vector<int> > neighbors;
    int 			    hilbert_spins;
    int 			    hilbert_upholes;
    int 			    hilbert_dnholes;
    int 			    hilbert;
    std::vector< std::vector<int> > CuOpairs,OOpairs;
    std::vector<int>                Cusites,Osites,etaCu;
    std::vector<int>                signsCuOpairs,signsOOpairs;

    public:
    void init(int nup_holes,int ndn_holes, double tpd, double tpp, double Upp, double Upd, double Udd,
	      double epsp, double epsd, double hst,
	      std::vector< std::vector<int> > CuOpairs, std::vector< std::vector<int> > OOpairs,
	      std::vector<int> Cusites, std::vector<int> Osites, std::vector<int> etaCu,
 	      std::vector<int> signsCuOpairs, std::vector<int> signsOOpairs)
    {
	this->signsCuOpairs=signsCuOpairs;
	this->signsOOpairs=signsOOpairs;
	this->hst=hst;
	this->epsp=epsp;
	this->epsd=epsd;
	this->CuOpairs=CuOpairs;
	this->OOpairs=OOpairs;
	this->Cusites=Cusites;
	this->etaCu=etaCu;
	this->Osites=Osites;
	this->num_sites=this->Cusites.size()+this->Osites.size();
	this->nup_holes=nup_holes;
	this->ndn_holes=ndn_holes;
	this->sz=double(this->num_sites/2.0);
	this->nup_spins=int(double(this->num_sites/2.0)+(this->sz));
	this->hilbert_spins=n_choose_k(this->num_sites,this->nup_spins);
	this->hilbert_upholes=n_choose_k(this->num_sites,this->nup_holes);
	this->hilbert_dnholes=n_choose_k(this->num_sites,this->ndn_holes);
	this->hilbert=this->hilbert_spins*this->hilbert_upholes*this->hilbert_dnholes;
        this->J=0.0;this->D=0.0;this->tpd=tpd;this->tpp=tpp;this->Upp=Upp;this->Upd=Upd;this->Udd=Udd;
        cout<<endl;
        cout<<"--------------------------------------------"<<endl;
        cout<<"N_sites    ="<<this->num_sites<<endl;
        cout<<"N_CuO_bonds="<<this->CuOpairs.size()<<endl;
        cout<<"N_OO_bonds ="<<this->OOpairs.size()<<endl;
        cout<<"nup_holes  ="<<this->nup_holes<<endl;
        cout<<"ndn_holes  ="<<this->ndn_holes<<endl;
        cout<<"tpd        ="<<this->tpd<<endl;
        cout<<"tpp        ="<<this->tpp<<endl;
        cout<<"Upp        ="<<this->Upp<<endl;
        cout<<"Upd        ="<<this->Upd<<endl;
        cout<<"Udd        ="<<this->Udd<<endl;
        cout<<"epsp       ="<<this->epsp<<endl;
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
	return new Three_Band_Model(*this);
    }    
};

void three_band_setup(std::string filename, 
               Three_Band_Model &three_band);

#endif
