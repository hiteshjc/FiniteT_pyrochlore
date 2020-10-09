#ifndef MNO_HEADER
#define MNO_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 MnO MODEL
//
///////////`///////////////////////////////////////////////////////////////////////////////
#include"hamiltonian.h"
#include"global.h"
#include"hamiltonian_spin_functions.h"
#include"number_functions.h"
#include"search_for.h"
#include"printing_functions.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////// 
class MnO_Model: public Ham
{
    public:
    int 			    nup_holes,ndn_holes,nup_spins;
    double 			    scale,E2p,E3d,E4s,Tpi,T01,T02,T12,U2p,U3d,U4s,U2p2p,U3d3d,U3d4s,U2p3d,U2p4s,U1001,U2002,U2112;
    int 			    hilbert_spins;
    int 			    hilbert_upholes;
    int 			    hilbert_dnholes;
    int 			    hilbert;
    std::vector<int>                psites,dsites,ssites;

    public:
    void init(int nup_holes,int ndn_holes, double scale, double E2p, double E3d, double E4s, double Tpi, double T01,
	      double T02, double T12, double U2p, double U3d, double U4s, double U2p2p, double U3d3d,
              double U3d4s, double U2p3d, double U2p4s, double U1001, double U2002, double U2112)
{
	this->psites.clear();
	this->psites.push_back(0);this->psites.push_back(3);this->psites.push_back(5);
	this->dsites.clear();
	this->dsites.push_back(1);this->dsites.push_back(4);this->dsites.push_back(6);
	this->dsites.push_back(7);this->dsites.push_back(8);
	this->ssites.clear();
	this->ssites.push_back(2);
	this->scale=scale;
	this->E2p=E2p*scale;
	this->E3d=E3d*scale;
	this->E4s=E4s*scale;
	this->Tpi=Tpi*scale;
	this->T01=T01*scale;
	this->T02=T02*scale;
	this->T12=T12*scale;
	this->U2p=U2p*scale;
        this->U3d=U3d*scale;
        this->U4s=U4s*scale;
        this->U2p2p=U2p2p*scale;
        this->U3d3d=U3d3d*scale;
        this->U3d4s=U3d4s*scale;
        this->U2p3d=U2p3d*scale;
        this->U2p4s=U2p4s*scale;
        this->U1001=U1001*scale;
        this->U2002=U2002*scale;
        this->U2112=U2112*scale;
	this->num_sites=9;
	this->nup_holes=nup_holes;
	this->ndn_holes=ndn_holes;
	this->nup_spins=0;
	this->hilbert_spins=n_choose_k(this->num_sites,this->nup_spins);
	this->hilbert_upholes=n_choose_k(this->num_sites,this->nup_holes);
	this->hilbert_dnholes=n_choose_k(this->num_sites,this->ndn_holes);
	this->hilbert=this->hilbert_spins*this->hilbert_upholes*this->hilbert_dnholes;
        cout<<endl;
        cout<<"--------------------------------------------"<<endl;
        cout<<"N_sites    ="<<this->num_sites<<endl;
        cout<<"nup_holes  ="<<this->nup_holes<<endl;
        cout<<"ndn_holes  ="<<this->ndn_holes<<endl;
        cout<<"E2p        ="<<this->E2p<<endl;
        cout<<"E3d        ="<<this->E3d<<endl;
        cout<<"E4s        ="<<this->E4s<<endl;
        cout<<"Tpi        ="<<this->Tpi<<endl;
        cout<<"T01        ="<<this->T01<<endl;
        cout<<"T02        ="<<this->T02<<endl;
        cout<<"T12        ="<<this->T12<<endl;
        cout<<"U2p        ="<<this->U2p<<endl;
        cout<<"U3d        ="<<this->U3d<<endl;
        cout<<"U4s        ="<<this->U4s<<endl;
        cout<<"U2p2p      ="<<this->U2p2p<<endl;
        cout<<"U3d3d      ="<<this->U3d3d<<endl;
        cout<<"U2p3d      ="<<this->U2p3d<<endl;
        cout<<"U3d4s      ="<<this->U3d4s<<endl;
        cout<<"U1001      ="<<this->U1001<<endl;
        cout<<"U2002      ="<<this->U2002<<endl;
        cout<<"U2112      ="<<this->U2112<<endl;
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
	return new MnO_Model(*this);
    }    
};

void mno_setup(std::string filename, 
               MnO_Model &mno);

#endif
