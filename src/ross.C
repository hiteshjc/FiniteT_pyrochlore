#include"global.h"
#include"ross.h"
#include"hamiltonian_spin_functions.h"
#include"number_functions.h"
#include"search_for.h"
#include"printing_functions.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                 Ross's MODEL
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
void make_g_mats(double gxy,double gz, std::vector<RMatrix> &gmats)
{
	gmats.clear();
	RMatrix gmat0;
	RMatrix gmat1;
	RMatrix gmat2;
	RMatrix gmat3;

	gmat0.resize(3,3);
	gmat1.resize(3,3);
	gmat2.resize(3,3);
	gmat3.resize(3,3);

	double gp=(2.0*gxy+ gz)/3.0;
	double gm=(gxy - gz)/3.0;

	gmat0(0,0)=gp;gmat0(0,1)=-gm;gmat0(0,2)=-gm;
	gmat0(1,0)=-gm;gmat0(1,1)=gp;gmat0(1,2)=-gm;
	gmat0(2,0)=-gm;gmat0(2,1)=-gm;gmat0(2,2)=gp;

	gmat1(0,0)=gp;gmat1(0,1)=gm;gmat1(0,2)=gm;
	gmat1(1,0)=gm;gmat1(1,1)=gp;gmat1(1,2)=-gm;
	gmat1(2,0)=gm;gmat1(2,1)=-gm;gmat1(2,2)=gp;

	gmat2(0,0)=gp;gmat2(0,1)=gm;gmat2(0,2)=-gm;
	gmat2(1,0)=gm;gmat2(1,1)=gp;gmat2(1,2)=gm;
	gmat2(2,0)=-gm;gmat2(2,1)=gm;gmat2(2,2)=gp;

	gmat3(0,0)=gp;gmat3(0,1)=-gm;gmat3(0,2)=gm;
	gmat3(1,0)=-gm;gmat3(1,1)=gp;gmat3(1,2)=gm;
	gmat3(2,0)=gm;gmat3(2,1)=gm;gmat3(2,2)=gp;

	gmats.push_back(gmat0);
	gmats.push_back(gmat1);
	gmats.push_back(gmat2);
	gmats.push_back(gmat3);
}


void make_J_mats(double J1,double J2,double J3,double J4, 
		 std::vector<RMatrix> &Jmats)
{
	Jmats.clear();
	RMatrix Jmat01, Jmat10;
	RMatrix Jmat02, Jmat20;
	RMatrix Jmat03, Jmat30;
	RMatrix Jmat12, Jmat21;
	RMatrix Jmat13, Jmat31;
	RMatrix Jmat23, Jmat32;

	Jmat01.resize(3,3);
	Jmat10.resize(3,3);
	Jmat02.resize(3,3);
	Jmat20.resize(3,3);
	Jmat03.resize(3,3);
	Jmat30.resize(3,3);
	Jmat12.resize(3,3);
	Jmat21.resize(3,3);
	Jmat13.resize(3,3);
	Jmat31.resize(3,3);
	Jmat23.resize(3,3);
	Jmat32.resize(3,3);

	Jmat01(0,0)=+J2; Jmat01(0,1)=+J4; Jmat01(0,2)=+J4;
	Jmat01(1,0)=-J4; Jmat01(1,1)=+J1; Jmat01(1,2)=+J3;
	Jmat01(2,0)=-J4; Jmat01(2,1)=+J3; Jmat01(2,2)=+J1;
	
	Jmat10(0,0)=+J2;  Jmat10(0,1)=-J4; Jmat10(0,2)=-J4;
	Jmat10(1,0)=+J4;  Jmat10(1,1)=+J1; Jmat10(1,2)=+J3;
	Jmat10(2,0)=+J4;  Jmat10(2,1)=+J3; Jmat10(2,2)=+J1;

	Jmat02(0,0)=+J1; Jmat02(0,1)=-J4; Jmat02(0,2)=+J3;
	Jmat02(1,0)=+J4; Jmat02(1,1)=+J2; Jmat02(1,2)=+J4;
	Jmat02(2,0)=+J3; Jmat02(2,1)=-J4; Jmat02(2,2)=+J1;

	Jmat20(0,0)=+J1;  Jmat20(0,1)=+J4; Jmat20(0,2)=+J3;
	Jmat20(1,0)=-J4;  Jmat20(1,1)=+J2; Jmat20(1,2)=-J4;
	Jmat20(2,0)=+J3;  Jmat20(2,1)=+J4; Jmat20(2,2)=+J1;

	Jmat03(0,0)=+J1; Jmat03(0,1)=+J3; Jmat03(0,2)=-J4;
	Jmat03(1,0)=+J3; Jmat03(1,1)=+J1; Jmat03(1,2)=-J4;
	Jmat03(2,0)=+J4; Jmat03(2,1)=+J4; Jmat03(2,2)=+J2;
	
        Jmat30(0,0)=+J1;  Jmat30(0,1)=+J3; Jmat30(0,2)=+J4;
	Jmat30(1,0)=+J3;  Jmat30(1,1)=+J1; Jmat30(1,2)=+J4;
	Jmat30(2,0)=-J4;  Jmat30(2,1)=-J4; Jmat30(2,2)=+J2;

	Jmat12(0,0)=+J1; Jmat12(0,1)=-J3; Jmat12(0,2)=+J4;
	Jmat12(1,0)=-J3; Jmat12(1,1)=+J1; Jmat12(1,2)=-J4;
	Jmat12(2,0)=-J4; Jmat12(2,1)=+J4; Jmat12(2,2)=+J2;
	
        Jmat21(0,0)=+J1;  Jmat21(0,1)=-J3; Jmat21(0,2)=-J4;
	Jmat21(1,0)=-J3;  Jmat21(1,1)=+J1; Jmat21(1,2)=+J4;
	Jmat21(2,0)=+J4;  Jmat21(2,1)=-J4; Jmat21(2,2)=+J2;

	Jmat13(0,0)=+J1; Jmat13(0,1)=+J4; Jmat13(0,2)=-J3;
	Jmat13(1,0)=-J4; Jmat13(1,1)=+J2; Jmat13(1,2)=+J4;
	Jmat13(2,0)=-J3; Jmat13(2,1)=-J4; Jmat13(2,2)=+J1;

	Jmat31(0,0)=+J1;  Jmat31(0,1)=-J4; Jmat31(0,2)=-J3;
	Jmat31(1,0)=+J4;  Jmat31(1,1)=+J2; Jmat31(1,2)=-J4;
	Jmat31(2,0)=-J3;  Jmat31(2,1)=+J4; Jmat31(2,2)=+J1;

	Jmat23(0,0)=+J2; Jmat23(0,1)=-J4; Jmat23(0,2)=+J4;
	Jmat23(1,0)=+J4; Jmat23(1,1)=+J1; Jmat23(1,2)=-J3;
	Jmat23(2,0)=-J4; Jmat23(2,1)=-J3; Jmat23(2,2)=+J1;
	
	Jmat32(0,0)=+J2;  Jmat32(0,1)=+J4; Jmat32(0,2)=-J4;
	Jmat32(1,0)=-J4;  Jmat32(1,1)=+J1; Jmat32(1,2)=-J3;
	Jmat32(2,0)=+J4;  Jmat32(2,1)=-J3; Jmat32(2,2)=+J1;

	Jmats.push_back(Jmat01);
	Jmats.push_back(Jmat10);
	Jmats.push_back(Jmat02);
	Jmats.push_back(Jmat20);
	Jmats.push_back(Jmat03);
	Jmats.push_back(Jmat30);
	Jmats.push_back(Jmat12);
	Jmats.push_back(Jmat21);
	Jmats.push_back(Jmat13);
	Jmats.push_back(Jmat31);
	Jmats.push_back(Jmat23);
	Jmats.push_back(Jmat32);
}

void Spin_Half_Ross::operator()
                    (int64_t spin_det,
                     std::vector<int64_t> &new_spin_dets, 
                     std::vector< complex<double> > &hints_list, 
		     int &ctr)
{
	complex<double> im=complex<double>(0.0,1.0);
	complex<double>  diaghint=0.0;
	ctr=0;
	
	// Save bits 
	std::vector<int> setofbits(this->num_sites);
	for (int i=0;i<this->num_sites;i++) setofbits[i]=btest64(spin_det,i);
	// 1st term
	for (int i=0;i<this->pairs.size();i++)
	{
		int64_t 	newdet=spin_det;
		complex<double> hint;
		int site1=this->pairs[i][0];
		int site2=this->pairs[i][1];
		int bt=this->btmatrix(site1,site2);
		//if (btest64(spin_det,site1)==0 and btest64(spin_det,site2)==0)
		if (setofbits[site1]==0 and setofbits[site2]==0)
		{
		     newdet=ibset64(newdet,site1);
		     newdet=ibset64(newdet,site2);
		     hint=0.25*(this->Jmats[bt](0,0)+(this->Jmats[bt](0,1)/im)+(this->Jmats[bt](1,0)/im) - this->Jmats[bt](1,1));
		     diaghint+=(this->Jmats[bt](2,2)*0.25);
		}
		//else if (btest64(spin_det,site1)==0 and btest64(spin_det,site2)==1)
		else if (setofbits[site1]==0 and setofbits[site2]==1)
		{
		     newdet=ibset64(newdet,site1);
		     newdet=ibclr64(newdet,site2);
		     hint=0.25*(this->Jmats[bt](0,0)-(this->Jmats[bt](0,1)/im)+(this->Jmats[bt](1,0)/im) + this->Jmats[bt](1,1));
		     diaghint-=(this->Jmats[bt](2,2)*0.25);
		}
		//if (btest64(spin_det,site1)==1 and btest64(spin_det,site2)==0)
		else if (setofbits[site1]==1 and setofbits[site2]==0)
		{
		     newdet=ibclr64(newdet,site1);
		     newdet=ibset64(newdet,site2);
		     hint=0.25*(this->Jmats[bt](0,0)+(this->Jmats[bt](0,1)/im)-(this->Jmats[bt](1,0)/im) + this->Jmats[bt](1,1));
		     diaghint-=(this->Jmats[bt](2,2)*0.25);
		}
		//if (btest64(spin_det,site1)==1 and btest64(spin_det,site2)==1)
		else
		{
		     newdet=ibclr64(newdet,site1);
		     newdet=ibclr64(newdet,site2);
		     hint=0.25*(this->Jmats[bt](0,0)-(this->Jmats[bt](0,1)/im)-(this->Jmats[bt](1,0)/im) - this->Jmats[bt](1,1));
		     diaghint+=(this->Jmats[bt](2,2)*0.25);
		}
		//new_spin_dets.push_back(newdet);
		//hints_list.push_back(hint);
		new_spin_dets[ctr]=newdet;
		hints_list[ctr]=hint;
		ctr+=1;
	}

	// 2nd term - Can combine 2nd and 3rd to reduce number of connections!!!!
	// hx, hy, hz already rescaled by mub
	for (int i=0;i<this->num_sites;i++)
	{
		int w=this->sublattice[i];
		diaghint-=(double(setofbits[i]-0.5)*((this->gmats[w](0,2)*this->hx) + (this->gmats[w](1,2)*this->hy) + (this->gmats[w](2,2)*this->hz)));
		int64_t 	newdet=spin_det;
		complex<double> hint=0.0;
		//if (btest64(spin_det,i)==0)
		if (setofbits[i]==0)
		{
			newdet=ibset64(newdet,i);
			for (int n=0;n<this->neighbors[i].size();n++)
			{
				int j=neighbors[i][n];
				if (i<j)
				{
					int bt=this->btmatrix(i,j);
					//hint+=(double(btest64(spin_det,j)-0.5)*0.5*(this->Jmats[bt](0,2) + (this->Jmats[bt](1,2)/im)));
					//hint+=(double(setofbits[j]-0.5)*0.5*(this->Jmats[bt](0,2) + (this->Jmats[bt](1,2)/im))); x 0.5 later
					hint+=(double(setofbits[j]-0.5)*(this->Jmats[bt](0,2) + (this->Jmats[bt](1,2)/im)));
				}
				else
				{
					int bt=this->btmatrix(j,i);
					//hint+=(double(setofbits[j]-0.5)*0.5*(this->Jmats[bt](2,0) + (this->Jmats[bt](2,1)/im))); x 0.5 later
					hint+=(double(setofbits[j]-0.5)*(this->Jmats[bt](2,0) + (this->Jmats[bt](2,1)/im)));
				}
			}
			hint-=((this->gmats[w](0,0)+ (this->gmats[w](0,1)/im))*this->hx); // x 0.5 later
			hint-=((this->gmats[w](1,0)+ (this->gmats[w](1,1)/im))*this->hy); // x 0.5 later
			hint-=((this->gmats[w](2,0)+ (this->gmats[w](2,1)/im))*this->hz); // x 0.5 later
		}
		else
		{
			newdet=ibclr64(newdet,i);
			for (int n=0;n<this->neighbors[i].size();n++)
			{
				int j=neighbors[i][n];
				if (i<j)
				{
					int bt=this->btmatrix(i,j);
					//hint+=(double(btest64(spin_det,j)-0.5)*0.5*(this->Jmats[bt](0,2) - (this->Jmats[bt](1,2)/im)));
					//hint+=(double(setofbits[j]-0.5)*0.5*(this->Jmats[bt](0,2) - (this->Jmats[bt](1,2)/im))); x 0.5 later
					hint+=(double(setofbits[j]-0.5)*(this->Jmats[bt](0,2) - (this->Jmats[bt](1,2)/im)));
				}
				else
				{
					int bt=this->btmatrix(j,i);
					//hint+=(double(setofbits[j]-0.5)*0.5*(this->Jmats[bt](2,0) - (this->Jmats[bt](2,1)/im)));
					hint+=(double(setofbits[j]-0.5)*(this->Jmats[bt](2,0) - (this->Jmats[bt](2,1)/im)));
				}
			}
			hint-=((this->gmats[w](0,0)- (this->gmats[w](0,1)/im))*this->hx); // x 0.5 later
			hint-=((this->gmats[w](1,0)- (this->gmats[w](1,1)/im))*this->hy); // x 0.5 later
			hint-=((this->gmats[w](2,0)- (this->gmats[w](2,1)/im))*this->hz); // x 0.5 later
		}
		//new_spin_dets.push_back(newdet);
		//hints_list.push_back(hint);
		new_spin_dets[ctr]=newdet;
		hints_list[ctr]=hint*0.5;
		ctr+=1;
	}
	
	/*// 3rd term - now combined with second
	for (int j=0;j<this->num_sites;j++)
	{
		int64_t 	newdet=spin_det;
		complex<double> hint=0.0;
		//if (btest64(spin_det,j)==0)
		if (setofbits[j]==0)
		{
			newdet=ibset64(newdet,j);
			for (int n=0;n<this->neighbors[j].size();n++)
			{
				int i=neighbors[j][n];
				if (i<j)
				{
					int bt=this->btmatrix(i,j);
					//hint+=(double(btest64(spin_det,i)-0.5)*0.5*(this->Jmats[bt](2,0) + (this->Jmats[bt](2,1)/im)));
					hint+=(double(setofbits[i]-0.5)*0.5*(this->Jmats[bt](2,0) + (this->Jmats[bt](2,1)/im)));
				}
			}
		}
		else
		{
			newdet=ibclr64(newdet,j);
			for (int n=0;n<this->neighbors[j].size();n++)
			{
				int i=neighbors[j][n];
				if (i<j)
				{
					int bt=this->btmatrix(i,j);
					//hint+=(double(btest64(spin_det,i)-0.5)*0.5*(this->Jmats[bt](2,0) - (this->Jmats[bt](2,1)/im)));
					hint+=(double(setofbits[i]-0.5)*0.5*(this->Jmats[bt](2,0) - (this->Jmats[bt](2,1)/im)));
				}
			}
		}
		//new_spin_dets.push_back(newdet);
		//hints_list.push_back(hint);
		new_spin_dets[ctr]=newdet;
		hints_list[ctr]=hint;
		ctr+=1;
	}*/
	// 4th term in diaghint
	//new_spin_dets.push_back(spin_det);
	//hints_list.push_back(diaghint);
	new_spin_dets[ctr]=spin_det;
	hints_list[ctr]=diaghint;
	ctr+=1;

}
////////////////////////////////////////////////////////////////////////////////////////////////////
void ross_setup(string filename, 
               Spin_Half_Ross &ross)

{    
    double J1,J2,J3,J4,gxy,gz, hx,hy,hz;
    bool found;
    string str_ret;
    std::vector< std::vector<int> > pairs;
    std::vector<int> 		    sublattice;
    search_for(string("pairs"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) pairs=convert_string_to_vec_of_vec(str_ret);}    
    search_for(string("sublattice"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) sublattice=convert_string_to_vec(str_ret);
    }    
    search_for(string("J1"),filename,str_ret,found);
    if (found){J1=str_to_d(str_ret);} else{J1=0.0;}
    search_for(string("J2"),filename,str_ret,found);
    if (found){J2=str_to_d(str_ret);} else{J2=0.0;}
    search_for(string("J3"),filename,str_ret,found);
    if (found){J3=str_to_d(str_ret);} else{J3=0.0;}
    search_for(string("J4"),filename,str_ret,found);
    if (found){J4=str_to_d(str_ret);} else{J4=0.0;}
    search_for(string("gxy"),filename,str_ret,found);
    if (found){gxy=str_to_d(str_ret);} else{gxy=0.0;}
    search_for(string("gz"),filename,str_ret,found);
    if (found){gz=str_to_d(str_ret);} else{gz=0.0;}
    search_for(string("hx"),filename,str_ret,found);
    if (found){hx=str_to_d(str_ret);} else{hx=0.0;}
    search_for(string("hy"),filename,str_ret,found);
    if (found){hy=str_to_d(str_ret);} else{hy=0.0;}
    search_for(string("hz"),filename,str_ret,found);
    if (found){hz=str_to_d(str_ret);} else{hz=0.0;}
    
    ross.init(pairs,sublattice,J1,J2,J3,J4,gxy,gz,hx,hy,hz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void ross_sym_setup(string filename, 
               Spin_Half_Ross &ross)

{    
    double J1,J2,J3,J4,gxy,gz, hx,hy,hz;
    bool found;
    string str_ret;
    std::vector< std::vector<int> > pairs;
    std::vector<int> 		    sublattice;
    search_for(string("pairs"),filename,str_ret,found);
    if (found) {if (str_ret.substr(0,1)==string("[")) pairs=convert_string_to_vec_of_vec(str_ret);}    
    
    print_mathematica_pairs(pairs);
    search_for(string("sublattice"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) sublattice=convert_string_to_vec(str_ret);
    }    
    search_for(string("J1"),filename,str_ret,found);
    if (found){J1=str_to_d(str_ret);} else{J1=0.0;}
    search_for(string("J2"),filename,str_ret,found);
    if (found){J2=str_to_d(str_ret);} else{J2=0.0;}
    search_for(string("J3"),filename,str_ret,found);
    if (found){J3=str_to_d(str_ret);} else{J3=0.0;}
    search_for(string("J4"),filename,str_ret,found);
    if (found){J4=str_to_d(str_ret);} else{J4=0.0;}
    search_for(string("gxy"),filename,str_ret,found);
    if (found){gxy=str_to_d(str_ret);} else{gxy=0.0;}
    search_for(string("gz"),filename,str_ret,found);
    if (found){gz=str_to_d(str_ret);} else{gz=0.0;}
    search_for(string("hx"),filename,str_ret,found);
    if (found){hx=str_to_d(str_ret);} else{hx=0.0;}
    search_for(string("hy"),filename,str_ret,found);
    if (found){hy=str_to_d(str_ret);} else{hy=0.0;}
    search_for(string("hz"),filename,str_ret,found);
    if (found){hz=str_to_d(str_ret);} else{hz=0.0;}
    
    std::vector<int> t1,t2,t3; 
    double	     k1,k2,k3;

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
    search_for(string("t3"),filename,str_ret,found);
    if (found)
    {
          if (str_ret.substr(0,1)==string("[")) t3=convert_string_to_vec(str_ret);
    }    
    search_for(string("k1"),filename,str_ret,found);
    if (found){k1=str_to_d(str_ret);} else{k1=0.0;}
    search_for(string("k2"),filename,str_ret,found);
    if (found){k2=str_to_d(str_ret);} else{k2=0.0;}
    search_for(string("k3"),filename,str_ret,found);
    if (found){k3=str_to_d(str_ret);} else{k3=0.0;}

    ross.init(pairs,sublattice,J1,J2,J3,J4,gxy,gz,hx,hy,hz, t1,t2,t3,k1,k2,k3);
    cout<<ross.num_sites<<endl;
}


