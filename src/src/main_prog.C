#include"global.h"
#include"mtrand.h"

// Search
#include"search_for.h"

// Hamiltonian
#include"hamiltonian.h"

// Spin-Hole Model, One Band, Three Band, MNO
#include"mno.h"
#include"three_band.h"
#include"one_band.h"
#include"spin_hole_model.h"
#include"classical_spin_hole_model.h"

// ED
#include"ed.h"
#include"semiclassical_MC.h"

using namespace std;


int main(int argc, char *argv[])
{
    time_t start,end;
    double dif;
    bool found;
    int seed;
    MTRand irand;
    
    std::vector<int> dets;
    string str_ret,neigs_str,nkrylov_str,filename;

    if (argc <= 1)
    {
        cout << "Usage: " << argv[0] << " <Filename>" << endl;
        exit(1);
    }
 
    filename=argv[1];
    search_for(string("seed"),filename,str_ret,found);
    if (found) 
    {
		if (found) {seed=str_to_int(str_ret);}
		else {seed=1;}
    }	
    irand.seed(seed);

 /////////////////////////////////////////////////////////////////////////////
 //                    HAMILTONIAN READ AND SETUP calls
 /////////////////////////////////////////////////////////////////////////////
    string hamiltonian;
    Ham *ham=NULL;
    Spin_Hole_Model 	      spin_hole;
    Three_Band_Model 	      three_band;
    One_Band_Model 	      one_band;
    MnO_Model 	              mno;
    Classical_Spin_Hole_Model classical_spin_hole;
    bool ham_found;
    int neigs;

    cout<<endl;
    cout<<"========================================="<<endl;
    cout<<"I am reading the Hamiltonian information "<<endl;
    cout<<"========================================="<<endl;

    search_for(string("hamiltonian"),filename,hamiltonian,ham_found);
    if (ham_found) 
    {
        if (hamiltonian.compare("spin_hole")==0)
	{
	    cout<<"Setting up spin-hole model"<<endl;
            spin_hole_setup(filename,spin_hole);cout<<endl;
            ham=spin_hole.clone();
	    cout<<"spin-hole model setup completed"<<endl;
	}
        else if (hamiltonian.compare("one_band")==0)
	{
	    cout<<"Setting up one band Hubbard model"<<endl;
            one_band_setup(filename,one_band);cout<<endl;
            ham=one_band.clone();
	    cout<<"One Band Hubbard model setup completed"<<endl;
	}
        else if (hamiltonian.compare("three_band")==0)
	{
	    cout<<"Setting up three band Hubbard model"<<endl;
            three_band_setup(filename,three_band);cout<<endl;
            ham=three_band.clone();
	    cout<<"Three Band Hubbard model setup completed"<<endl;
	}
        else if (hamiltonian.compare("mno")==0)
	{
	    cout<<"Setting up MnO model"<<endl;
            mno_setup(filename,mno);cout<<endl;
            ham=mno.clone();
	    cout<<"MnO model setup completed"<<endl;
	}
        else if (hamiltonian.compare("classical_spin_hole")==0)
	{
	    cout<<"Setting up classical spin-hole model"<<endl;
            classical_spin_hole_setup(filename,classical_spin_hole);cout<<endl;
            ham=classical_spin_hole.clone();
	    cout<<"classical spin-hole model setup completed"<<endl;
	}
	else
        {
            cout<<endl;cout<<"I could not find the requested hamiltonian"<<endl;
            //return 0;
        }
    }
    bool neigs_found;
    search_for(string("neigs"),filename,neigs_str,neigs_found);
    if (neigs_found){neigs=str_to_int(neigs_str);}
    else{neigs=1;}

/////////////////////////////////////////////////////////////////////////////
//                              EXACT DIAGONALIZATION
/////////////////////////////////////////////////////////////////////////////
    std::vector<double> eigs;
    int hilbert_space;
    int nkrylov,num_cycles,num_vecs;
    bool nkrylov_found,num_cycles_found, num_vecs_found, rotate_found;
    bool rotate;
    
    search_for(string("nkrylov"),filename,str_ret,nkrylov_found);
    if (nkrylov_found){nkrylov=str_to_int(str_ret);}
    else{nkrylov=20;}
    
    search_for(string("num_cycles"),filename,str_ret,num_cycles_found);
    if (num_cycles_found){num_cycles=str_to_int(str_ret);}
    else{num_cycles=1;}
    
    search_for(string("num_vecs"),filename,str_ret,num_vecs_found);
    if (num_vecs_found){num_vecs=str_to_int(str_ret);}
    else{num_vecs=1;}
    
    search_for(string("rotate"),filename,str_ret,rotate_found);
    if (rotate_found){rotate=str_to_bool(str_ret);}
    else{rotate=false;}

    if (ham_found)
    {
            search_for(string("diagonalize"),filename,str_ret,found);
	    if (str_to_bool(str_ret))
	    {		
		if (hamiltonian.compare("spin_hole")==0)
		{
		   hilbert_space=spin_hole.hilbert;
		   cout<<"Hilbert space size is "<<hilbert_space<<endl;
		   cout<<"TRACE: Calling Lanczos for requested number of particles"<<endl;
		   cout<<endl;
		   Simulation_Params sp;
		   sp.iterations=nkrylov;
	           sp.num_cycles=num_cycles;
		   lanczos_spin_hole_requested_sector(*ham,sp,spin_hole.nup_spins,spin_hole.nup_holes,spin_hole.ndn_holes,eigs);
		   //lanczos_spin_hole_all_spin_sectors(*ham,spin_hole.nup_holes,spin_hole.ndn_holes,nkrylov,eigs);
		   cout<<endl;
		   cout<<"--------------------------------"<<endl;
		   cout<<"The Eigenvalues (Spin-Hole) are"<<endl;
		   cout<<"--------------------------------"<<endl;
		   print_vec_acc(eigs,true,neigs);
		   cout<<endl;
		   cout<<"TRACE: Finished Lanczos for requested number of particles"<<endl;
		   cout<<endl;
	        }		
		if (hamiltonian.compare("one_band")==0)
		{
		   hilbert_space=one_band.hilbert;
		   cout<<"Hilbert space size is "<<hilbert_space<<endl;
		   cout<<"TRACE: Calling Lanczos for requested number of particles"<<endl;
		   cout<<endl;
		   Simulation_Params sp;
		   sp.iterations=nkrylov;
	           sp.num_cycles=num_cycles;
	           sp.how_many_eigenvecs=num_vecs;
	           sp.rotate=rotate;
		   lanczos_spin_hole_requested_sector(*ham,sp,one_band.nup_spins,one_band.nup_holes,one_band.ndn_holes,
						       eigs);
		   cout<<endl;
		   cout<<"----------------------------"<<endl;
		   cout<<"The Eigenvalues (1-band) are"<<endl;
		   cout<<"----------------------------"<<endl;
		   print_vec_acc(eigs,true,neigs);
		   cout<<endl;
		   cout<<"TRACE: Finished Lanczos for requested number of particles"<<endl;
		   cout<<endl;
	        }		
		if (hamiltonian.compare("three_band")==0)
		{
		   hilbert_space=three_band.hilbert;
		   cout<<"Hilbert space size is "<<hilbert_space<<endl;
		   cout<<"TRACE: Calling Lanczos for requested number of particles"<<endl;
		   cout<<endl;
		   Simulation_Params sp;
		   sp.iterations=nkrylov;
	           sp.num_cycles=num_cycles;
	           sp.how_many_eigenvecs=num_vecs;
	           sp.rotate=rotate;
		   lanczos_spin_hole_requested_sector(*ham,sp,three_band.nup_spins,three_band.nup_holes,three_band.ndn_holes,
						       eigs);
		   cout<<endl;
		   cout<<"----------------------------"<<endl;
		   cout<<"The Eigenvalues (3-band) are"<<endl;
		   cout<<"----------------------------"<<endl;
		   print_vec_acc(eigs,true,neigs);
		   cout<<endl;
		   cout<<"TRACE: Finished Lanczos for requested number of particles"<<endl;
		   cout<<endl;
	        }		
		if (hamiltonian.compare("mno")==0)
		{
		   hilbert_space=mno.hilbert;
		   cout<<"Hilbert space size is "<<hilbert_space<<endl;
		   cout<<"TRACE: Calling Lanczos for requested number of particles"<<endl;
		   cout<<endl;
		   Simulation_Params sp;
		   sp.iterations=nkrylov;
	           sp.num_cycles=num_cycles;
		   lanczos_spin_hole_requested_sector(*ham,sp,mno.nup_spins,mno.nup_holes,mno.ndn_holes,
						       eigs);
		   cout<<endl;
		   cout<<"----------------------------"<<endl;
		   cout<<"The Eigenvalues (MnO) are"<<endl;
		   cout<<"----------------------------"<<endl;
		   print_vec_acc(eigs,true,neigs);
		   cout<<endl;
		   cout<<"TRACE: Finished Lanczos for requested number of particles"<<endl;
		   cout<<endl;
	        }		
	    }   
    }

/////////////////////////////////////////////////////////////////////////////
//           CLASSICAL SPINS + EXACT DIAGONALIZATION OF HOLES
/////////////////////////////////////////////////////////////////////////////
    bool num_mc_found=false;
    int  num_mc_samples;
    search_for(string("num_mc_samples"),filename,str_ret,num_mc_found);
    if (num_mc_found){num_mc_samples=str_to_int(str_ret);}
    else{num_mc_samples=1000;}
    
    bool   delta_found=false;
    double start_delta;
    search_for(string("start_delta"),filename,str_ret,delta_found);
    if (delta_found){start_delta=str_to_d(str_ret);}
    else{start_delta=0.02;}


    if (ham_found)
    {
            search_for(string("semiclassical_mc"),filename,str_ret,found);
	    if (str_to_bool(str_ret))
	    {		
		if (hamiltonian.compare("classical_spin_hole")==0)
		{
		   hilbert_space=classical_spin_hole.hilbert;
		   cout<<"TRACE: Calling MC+Lanczos for requested number of holes"<<endl;
		   semiclassical_mc_zeroT(classical_spin_hole,nkrylov,num_mc_samples,start_delta);
		   cout<<"TRACE: Finished MC+Lanczos for requested number of holes"<<endl;
	        }		
	    }   
    }


/////////////////////////////////////////////////////////////////////////////
//                              CLEAN UP 
/////////////////////////////////////////////////////////////////////////////

    if (ham!=NULL) {delete ham;ham=NULL;}
    return 0;

}

/////////////////////////////////////////////////////////////////////////////
