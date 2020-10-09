#ifndef SPIN_HOLE_MODEL_HEADER
#define SPIN_HOLE_MODEL_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 SPIN HOLE MODEL
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
class Spin_Hole_Model: public Ham
{
    public:
    int 			    l_x,l_y;
    int 			    nup_holes,ndn_holes,nup_spins;
    double 			    J,D,t,U,V,rh,sz;
    std::vector< std::vector<int> > pairs;
    std::vector< std::vector<int> > neighbors;
    std::vector< std::vector<int> > neighbors_within_rh;
    bool			    pbc;
    int 			    hilbert_spins;
    int 			    hilbert_upholes;
    int 			    hilbert_dnholes;
    int 			    hilbert;
    RMatrix                         distance;

    public:
    void init(int l_x,int l_y, bool pbc, double sz, int nup_holes,int ndn_holes, 
	      double J, double D, double t, double U, double V,double rh)
    {
	this->l_x=l_x;
	this->l_y=l_y;
	this->pbc=pbc;
	this->rh=rh;
	this->num_sites=(this->l_x*this->l_y);
	this->nup_holes=nup_holes;
	this->ndn_holes=ndn_holes;
	this->sz=sz;
	this->nup_spins=int(double(this->num_sites/2.0)+(this->sz));
	this->hilbert_spins=n_choose_k(this->num_sites,this->nup_spins);
	this->hilbert_upholes=n_choose_k(this->num_sites,this->nup_holes);
	this->hilbert_dnholes=n_choose_k(this->num_sites,this->ndn_holes);
	this->hilbert=this->hilbert_spins*this->hilbert_upholes*this->hilbert_dnholes;
        this->J=J;this->D=D;this->t=t;this->U=U;this->V=V;
	this->set_pairs_square_lattice();
        cout<<endl;
        cout<<"--------------------------------------------"<<endl;
        cout<<"l_x        ="<<this->l_x<<endl;
        cout<<"l_y        ="<<this->l_y<<endl;
        cout<<"pbc        ="<<this->pbc<<endl;
        cout<<"N_sites    ="<<this->num_sites<<endl;
        cout<<"N_bonds    ="<<this->pairs.size()<<endl;
        cout<<"sz         ="<<this->sz<<endl;
        cout<<"nup_holes  ="<<this->nup_holes<<endl;
        cout<<"ndn_holes  ="<<this->ndn_holes<<endl;
        cout<<"J          ="<<this->J<<endl;
        cout<<"D          ="<<this->D<<endl;
        cout<<"t          ="<<this->t<<endl;
        cout<<"U          ="<<this->U<<endl;
        cout<<"V          ="<<this->V<<endl;
        cout<<"rh         ="<<this->rh<<endl;
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

void set_pairs_square_lattice();

    Ham* clone() const
    {
	return new Spin_Hole_Model(*this);
    }    
};

void spin_hole_setup(std::string filename, 
               Spin_Hole_Model &spin_hole);


#endif
