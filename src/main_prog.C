#include"global.h"
#include"mtrand.h"

// Search
#include"search_for.h"

// Hamiltonian
#include"hamiltonian.h"
#include"ross.h"
#include"ybmg.h"
// ED
#include"ed.h"
using namespace std;


int main(int argc, char *argv[])
{
    time_t start,end;
    double dif;
    bool found;
    int seed;
    string str_ret,neigs_str,nkrylov_str,filename,outfilename;
    if (argc <= 2)
    {
        cout << "Usage: " << argv[0] << " <Filename>  <Filename>" << endl;
        exit(1);
    }
 
    filename=argv[1];
    outfilename=argv[2];
    ofstream outfile;
    const char *cstr = (outfilename).c_str();
    outfile.open(cstr);
    outfile.close();
    outfile.open(cstr,ofstream::app);

    search_for(string("seed"),filename,str_ret,found);
    if (found) {seed=str_to_int(str_ret);}
    else {seed=1;}
    MTRand irand((unsigned long)seed);

 /////////////////////////////////////////////////////////////////////////////
 //                    HAMILTONIAN READ AND SETUP calls
 /////////////////////////////////////////////////////////////////////////////
    string hamiltonian;
    Ham *ham=NULL;
    Spin_Half_Ross 	      ross;
    Spin_Half_Ybmg 	      ybmg;
    bool ham_found;
    int neigs;

    outfile<<endl;
    outfile<<"========================================="<<endl;
    outfile<<"I am reading the Hamiltonian information "<<endl;
    outfile<<"========================================="<<endl;
    outfile.flush();

    search_for(string("hamiltonian"),filename,hamiltonian,ham_found);
    if (ham_found) 
    {
        if (hamiltonian.compare("ross_sym")==0)
	{
	    outfile<<"Setting up Ross model with symmetries"<<endl;
            ross_sym_setup(filename,ross,outfilename);outfile<<endl;
	    outfile<<"N sites Ross before clone = "<<ross.num_sites<<endl;
            ham=ross.clone();
	    outfile<<"N sites Ross after clone = "<<(*ham).num_sites<<endl;
	    outfile<<"Ross setup with symmetries completed"<<endl;
	    outfile.flush();
	}
	else
        {
            outfile<<endl;outfile<<"I could not find the requested hamiltonian"<<endl;
        }
    }
    bool neigs_found;
    search_for(string("neigs"),filename,neigs_str,neigs_found);
    if (neigs_found){neigs=str_to_int(neigs_str);}
    else{neigs=1;}

   //for (int i=0;i<100;i++) outfile<<" i = "<<i<<" random = "<<irand()<<endl;
/////////////////////////////////////////////////////////////////////////////
//                              EXACT DIAGONALIZATION
/////////////////////////////////////////////////////////////////////////////
    std::vector<double> eigs;
    int hilbert_space;
    int nkrylov,num_cycles,num_vecs, nones;
    bool nkrylov_found, nones_found, num_cycles_found, num_vecs_found, rotate_found,wffile_found,loadwffile_found, use_sym_found, analysis_found;
    bool rotate,use_sym;
    std::string wffile,loadwffile,analysis; 

    search_for(string("nones"),filename,str_ret,nones_found);
    if (nones_found){nones=str_to_int(str_ret);}
    else{nones=0;}
    
    search_for(string("use_sym"),filename,str_ret,use_sym_found);
    if (use_sym_found){use_sym=str_to_bool(str_ret);}
    else{use_sym=false;}
    
    search_for(string("nkrylov"),filename,str_ret,nkrylov_found);
    if (nkrylov_found){nkrylov=str_to_int(str_ret);}
    else{nkrylov=20;}
    
    search_for(string("num_cycles"),filename,str_ret,num_cycles_found);
    if (num_cycles_found){num_cycles=str_to_int(str_ret);}
    else{num_cycles=1;}
    
    search_for(string("num_vecs"),filename,str_ret,num_vecs_found);
    if (num_vecs_found){num_vecs=str_to_int(str_ret);}
    else{num_vecs=1;}
    
    search_for(string("wffile"),filename,str_ret,wffile_found);
    if (wffile_found){wffile=str_ret; outfile<<"Wffile = "<<wffile<<endl;}
    else{wffile=string("random");}
    
    search_for(string("loadwffile"),filename,str_ret,loadwffile_found);
    if (loadwffile_found){loadwffile=str_ret; outfile<<"LoadWffile = "<<loadwffile<<endl;}
    else{loadwffile=string("random");}
    
    search_for(string("analysis"),filename,str_ret,analysis_found);
    if (analysis_found){analysis=str_ret; outfile<<"Analysis = "<<analysis<<endl;}
    else{analysis=string("finiteT");}
   
    Simulation_Params sp;
    sp.outfile=outfilename;
    sp.analysis=analysis;
    sp.use_sym=use_sym;
    sp.iterations=nkrylov;
    sp.nones=nones;
    sp.num_cycles=num_cycles;
    sp.how_many_eigenvecs=num_vecs;
    sp.wffile=wffile;
    sp.loadwffile=loadwffile;
    std::vector< std::vector< complex<double> > > evecs;
    outfile<<"sp.use_sym = "<<sp.use_sym<<endl;

    if (ham_found)
    {
            search_for(string("diagonalize"),filename,str_ret,found);
	    if (str_to_bool(str_ret))
	    {		
		if(hamiltonian.compare("ross_sym")==0)
		{
		   lanczos_sym(*ham,sp,eigs);
		   //outfile<<endl;
		   //outfile<<"---------------------------------------"<<endl;
		   //outfile<<"The Eigenvalues (Ross_sym) from Lanczos are"<<endl;
		   //outfile<<"---------------------------------------"<<endl;
		   //print_vec_acc(eigs,true,neigs);
		   //outfile<<endl;
	        }		
	    }   
            /*search_for(string("analysis"),filename,str_ret,found);
	    if (str_ret==string("S+S-"))
	    {
		outfile<<"Reading wf and calculating S+S-"<<endl;
		load_wf_get_corrs((*ham).num_sites,loadwffile,sp,string("S+S-"));	
	    }
	    if (str_ret==string("SzSz"))
	    {
		outfile<<"Reading wf and calculating SzSz"<<endl;
		load_wf_get_corrs((*ham).num_sites,loadwffile,sp,string("SzSz"));	
	    }*/
    }
/////////////////////////////////////////////////////////////////////////////
//                              CLEAN UP 
/////////////////////////////////////////////////////////////////////////////
    if (ham!=NULL) {delete ham;ham=NULL;}
    outfile.close();
    return 0;

}

/////////////////////////////////////////////////////////////////////////////
