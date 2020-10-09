#include"global.h"
#include"oleg.h"
#include"hamiltonian_spin_functions.h"
#include"number_functions.h"
#include"search_for.h"
#include"printing_functions.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                 Oleg's MODEL
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

void Spin_Half_Oleg::operator()
                    (int spin_det,
                     std::vector<int> &new_spin_dets, 
                     std::vector< complex<double> > &hints_list)
{
    std::vector< std::vector<int> >:: iterator p;

    if (abs(this->jh)>1.0e-15)
    {
	    for (p=this->hexagons.begin();p!=this->hexagons.end();p++)
	    {
		int first=(*p)[0]; int second=(*p)[1];
		int third=(*p)[2]; int fourth=(*p)[3];
		int fifth=(*p)[4]; int sixth=(*p)[5];
		calc_hints_xyzxyz(-1.0*this->jh, first, second, third, fourth, fifth, sixth, 
				     spin_det, new_spin_dets, hints_list);

		// Note H = - Sum W
	    }
    }
    
    if (abs(this->jh2)>1.0e-15)
    {
	    for (p=this->hexagons.begin();p!=this->hexagons.end();p++)
	    {
		int first=(*p)[0]; int second=(*p)[1];
		int third=(*p)[2]; int fourth=(*p)[3];
		int fifth=(*p)[4]; int sixth=(*p)[5];
		//calc_hints_xyzxyz(-1.0*this->jh2, first, second, fourth, sixth, third, fifth, 
		//		     spin_det, new_spin_dets, hints_list);
		calc_hints_xyyzzx(-1.0*this->jh2, first, second, third, fourth, fifth, sixth, 
				     spin_det, new_spin_dets, hints_list);

		// Note H = - Sum W
	    }
    }

    if (abs(this->jut)>1.0e-15)
    {
	    for (p=this->up_triangles.begin();p!=this->up_triangles.end();p++)
	    {
		int first=(*p)[0]; int second=(*p)[1];int third=(*p)[2];
		calc_hints_xyz(-1.0*this->jut, first, second, third, spin_det, new_spin_dets, hints_list);
		// Note H = - Sum W
	    }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void oleg_setup(string filename, 
               Spin_Half_Oleg &oleg)

{    
    double jut,jh,jh2;
    bool found=true;
    string str_ret;
    std::vector< std::vector<int> > hexagons, up_triangles;
    search_for(string("hexagons"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) hexagons=convert_string_to_vec_of_vec(str_ret);
    }    
    search_for(string("up_triangles"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) up_triangles=convert_string_to_vec_of_vec(str_ret);
    }    
    search_for(string("jut"),filename,str_ret,found);
    if (found){jut=str_to_d(str_ret);} else{jut=0.0;}
    search_for(string("jh"),filename,str_ret,found);
    if (found){jh=str_to_d(str_ret);} else{jh=0.0;}
    search_for(string("jh2"),filename,str_ret,found);
    if (found){jh2=str_to_d(str_ret);} else{jh2=0.0;}
    
    oleg.init(hexagons,up_triangles,jh,jut,jh2);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void oleg_sym_setup(string filename, 
               Spin_Half_Oleg &oleg)

{    
    double jut,jh;
    bool found=true;
    string str_ret;
    std::vector< std::vector<int> > hexagons, up_triangles;
    std::vector<int> t1,t2; 
    double	     k1,k2;

    search_for(string("hexagons"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) hexagons=convert_string_to_vec_of_vec(str_ret);
    }    
    search_for(string("up_triangles"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) up_triangles=convert_string_to_vec_of_vec(str_ret);
    }    
    search_for(string("jut"),filename,str_ret,found);
    if (found){jut=str_to_d(str_ret);} else{jut=0.0;}
    search_for(string("jh"),filename,str_ret,found);
    if (found){jh=str_to_d(str_ret);} else{jh=0.0;}
    search_for(string("k1"),filename,str_ret,found);
    if (found){k1=str_to_d(str_ret);} else{k1=0.0;}
    search_for(string("k2"),filename,str_ret,found);
    if (found){k2=str_to_d(str_ret);} else{k2=0.0;}
    
    search_for(string("t1"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) t1=convert_string_to_vec(str_ret);
    }    
    search_for(string("t2"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) t2=convert_string_to_vec(str_ret);
    }    
    oleg.init(hexagons,up_triangles,jh,jut,t1,t2,k1,k2);
}


