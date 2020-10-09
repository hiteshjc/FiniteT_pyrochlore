#include"number_functions.h"

using namespace std;

// Look in a sorted list to find location of an integer
int binary_search(int64_t det, std::vector<int64_t> &sorted_dets_list) 
{
   // function:
   //   Searches sortedArray[first]..sortedArray[last] for key.  
   // returns: index of the matching element if it finds key, 
   //         otherwise  -(index where it could be inserted)-1.
   // parameters:
   //   sortedArray in  array of sorted (ascending) values.
   //   first, last in  lower and upper subscript bounds
   //   key         in  value to search for.
   // returns:
   //   index of key, or -insertion_position -1 if key is not 
   //                 in the array. This value can easily be
   //                 transformed into the position to insert it.
   int first=0;
   int last=sorted_dets_list.size()-1;   
   while (first <= last) {
       int mid = (first + last) / 2;  // compute mid point.
       if (det > sorted_dets_list[mid]) 
           first = mid + 1;  // repeat search in top half.
       else if (det < sorted_dets_list[mid]) 
           last = mid - 1; // repeat search in bottom half.
       else
           return mid;     // found it. return position /////
   }
   return -(first + 1);    // failed to find key
}

//////////////////////////////////////////////////////////////////////////////
// Useful bit functions  
//////////////////////////////////////////////////////////////////////////////

int btest64(int64_t a,int pos)
{ 
  bitset<64> b;
  b=a;
  return b[pos];
}

int64_t ibset64(int64_t a,int pos)
{return a| int64_t(1)<<pos;}

int64_t ibclr64(int64_t a,int pos)
{return a & ~(int64_t(1)<<pos);}

int btest(int a,int pos)
{ int b=a & 1<<pos;
  if (b!=0) return 1;
  else      return 0;}

int ibset(int a,int pos)
{return a| 1<<pos;}

int ibclr(int a,int pos)
{return a & ~(1<<pos);}

//////////////////////////////////////////////////////////////////////////////
// Useful number functions  
//////////////////////////////////////////////////////////////////////////////

int n_choose_k(int n,int k)
{
//    !---------------------------------------------------------------------------
//    ! Description : Generate binomial coefficient
    int         nck;
    int   	i;
    double	log_n_factorial, log_k_factorial, log_n_minus_k_factorial;

    if (n < k) return 0;
    
    log_n_factorial = 0.0;
    log_k_factorial = 0.0;
    log_n_minus_k_factorial = 0.0;

    for (i=2;i<=n;i++) log_n_factorial = log_n_factorial + log(double(i));
    for (i=2;i<=k;i++) log_k_factorial = log_k_factorial + log(double(i));
    for (i=2;i<=n-k;i++) log_n_minus_k_factorial = log_n_minus_k_factorial + log(double(i));

    nck = int((exp(log_n_factorial - log_k_factorial - log_n_minus_k_factorial))+0.5);

    return nck;
}
//////////////////////////////////////////////////////////////////////////////

void constrained_dets(int num_sites,int num_ones,std::vector<int> &dets)
{
// ----------------------------------------------------------------------------------------------
//    ! Description   : Gives us a neat way of generating all configs with a fixed
//    !                 particle number
//    ! Author        : F. Petruzielo's code used by H.J. Changlani (ref. Bit Twiddling Hacks)
//    ! ----------------------------------------------------------------------------------------------

    int                      temp,temp1,temp2,temp3,temp4,temp5,temp6;
    int                      i;
    int 		     num_configs;

    if (num_sites<num_ones) 
    {
	cout<<"ERROR: Num sites < Num_ones"<<endl;
        return;
    }

    dets.clear();
    num_configs=n_choose_k(num_sites,num_ones);

    dets.push_back(pow(2,num_ones) - 1);
    for( i=1;i<num_configs;i++)
    {  
       temp  = (dets[i-1] | dets[i-1] - 1) + 1;
       temp2 = (temp) & (-temp);
       temp3=  (dets[i-1]) & (-dets[i-1]);
       temp4=temp2/temp3;
       temp5=temp4>>1;
       temp6=temp5-1;
       dets.push_back(temp|temp6);
    }
}

void constrained_dets_i64(int num_sites,int num_ones,std::vector<int64_t> &dets)
{
// ----------------------------------------------------------------------------------------------
//    ! Description   : Gives us a neat way of generating all configs with a fixed
//    !                 particle number
//    ! Author        : F. Petruzielo's code used by H.J. Changlani (ref. Bit Twiddling Hacks)
//    ! ----------------------------------------------------------------------------------------------

    int64_t                  temp,temp1,temp2,temp3,temp4,temp5,temp6;
    int 		     num_configs;

    if (num_sites<num_ones) 
    {
	cout<<"ERROR: Num sites < Num_ones"<<endl;
        return;
    }

    dets.clear();
    num_configs=n_choose_k(num_sites,num_ones);

    dets.push_back(pow(int64_t(2),int64_t(num_ones)) - int64_t(1));
    for( int i=1;i<num_configs;i++)
    {  
       temp  = (dets[i-1] | dets[i-1] - int64_t(1)) + int64_t(1);
       temp2 = (temp) & (-temp);
       temp3=  (dets[i-1]) & (-dets[i-1]);
       temp4=temp2/temp3;
       temp5=temp4>>int64_t(1);
       temp6=temp5-int64_t(1);
       dets.push_back(temp|temp6);
    }
}


//////////////////////////////////////////////////////////////////////////////

std::vector<int> convert_ind_to_vec(int index, 
				    std::vector<int> nstates)
{
	int 			coord;
	int 			n=1;
	int 			dim=nstates.size();
	std::vector<int> 	vec(dim);

	n=1;
	for (int i=0;i<dim;i++)
	{
	   n=nstates[dim-1-i];
	   coord=index%n;
	   vec[dim-1-i]=coord;
	   index=index/n;
	}
	return vec;
}  

//////////////////////////////////////////////////////////////////////////////

int convert_vec_to_ind(std::vector<int> const &vec, 
		       std::vector<int> nstates)

{
	int 		ind,n;
	int 		dim=nstates.size();
	
	n=1; ind=0;
	for (int i=0;i<dim;i++)
	{
           ind+=(n*vec[dim-1-i]);
	   n=n*nstates[dim-1-i];
	}
	return ind;
}  


//////////////////////////////////////////////////////////////////////////////
void get_adj_list_from_pairs(std::vector< std::vector<int> > const &pairs,
			     std::vector< std::vector<int> > &adj_list)
{
	int max=0;

	for (int i=0;i<pairs.size();i++)
	{
		if (pairs[i][0]>max) {max=pairs[i][0];}
		if (pairs[i][1]>max) {max=pairs[i][1];}
		
	}
 
	adj_list.clear();
	adj_list.resize(max+1);
		
	for (int i=0;i<pairs.size();i++)
	{
		adj_list[pairs[i][0]].push_back(pairs[i][1]);
		adj_list[pairs[i][1]].push_back(pairs[i][0]);
	}
}
//////////////////////////////////////////////////////////////////////////////
void convert_num_to_vec(int num, 
                        int base, 
                        int num_bits, 
                        std::vector<int> &config)
{
    config.resize(num_bits);
    for (int i=0;i<num_bits;i++) config[i]=0;

    if (pow(base,num_bits)<num) cout<<"Input number greater than representable by num_bits, Expect errors"<<" "<<num<<"\n";
    for (int ctr=1; ctr<=num_bits; ctr++)
    {
            config[num_bits-ctr]=num%base; // reversed bits needed
            num=num/base;
    }
}
//////////////////////////////////////////////////////////////////////////////
int convert_vec_to_num(std::vector<int> &config, 
		       int base)
{
    int 	num=0;
    int 	mult=1;

    for (int i=config.size()-1; i>=0; i--)
    {
           num+= (config[i]*mult);
           mult=mult*base;
    }
    return num;   
}
//////////////////////////////////////////////////////////////////////////////
void sort_by_energy(std::vector<double> &eigs,std::vector<double> &szs)
{
   for (int j=0;j<eigs.size()-1;j++)
   {	
	   for (int i=j+1;i<eigs.size();i++)
	   {
		if (eigs[j]>eigs[i])
		{
		   swap(eigs[j],eigs[i]);
		   swap(szs[j],szs[i]);
		}
		
		if (abs(eigs[j]-eigs[i])<1.0e-8)
		{
		  if (szs[j]>szs[i]) {swap(szs[j],szs[i]);}  	
		}
	   }
   }
}
//////////////////////////////////////////////////////////////////////////////
int str_to_int(string str)
{ return boost::lexical_cast<int>(str);}

//////////////////////////////////////////////////////////////////////////////
double str_to_d(string str)
{ return boost::lexical_cast<long double>(str);}

//////////////////////////////////////////////////////////////////////////////
bool str_to_bool(string str)
{       bool b=0;
        if (str==string("t") or str==string("T") or str==string("true") or
            str==string("True") or str==string("Y") or str==string("Yes") 
            or str==string("1")){b=1;}
        return b;
}

//////////////////////////////////////////////////////////////////////////////
std::string dtos(double dbl)
{
    char buf[BUFSIZ];
    sprintf(buf, "%+lf", dbl);
    return buf;
}
//////////////////////////////////////////////////////////////////////////////
template <class T>
std::string to_string (T const &t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}
template std::string to_string<int>(int const &);
template std::string  to_string<double>(double const &);

//////////////////////////////////////////////////////////////////////////////
std::vector<int> convert_string_to_vec(std::string const &s)
{
    // Eg s=[1,0] would get converted to vec[1,0]
    std::vector<int> v;
    int num;
    
    std::string temp="";
    for (int i=1;i<s.size();i++)
    {
        if (s.substr(i,1)!=string(",") and s.substr(i,1)!=string("]") 
            and s.substr(i,1)!=string(")")
            and s.substr(i,1)!=string("}"))
        {   temp+=s.substr(i,1);}    
        else
        {   num=str_to_int(temp);
            v.push_back(num);
            temp="";}
    }
    return v;
}
///////////////////////////////////////////////////////////////////////
std::vector<double> convert_string_to_vec_double(std::string const &s)
{
    // Eg s=[1,0] would get converted to vec[1,0]
    std::vector<double> v;
    double num;
    
    std::string temp="";
    for (int i=1;i<s.size();i++)
    {
        if (s.substr(i,1)!=string(",") and s.substr(i,1)!=string("]") 
            and s.substr(i,1)!=string(")")
            and s.substr(i,1)!=string("}"))
        {   temp+=s.substr(i,1);}    
        else
        {   num=str_to_d(temp);
            v.push_back(num);
            temp="";}
    }
    return v;
}


//////////////////////////////////////////////////////////////////////////////

std::vector< std::vector<int> > convert_string_to_vec_of_vec(std::string const &s)
{
    // Eg s=[[1,0],[1,1]] would get converted to vec[1,0]+vec[1,1]
    std::vector<int> temp_vec;
    std::vector< vector<int> > v;

    string temp="";
    for (int i=1;i<s.size();i++)
    {
        if (s.substr(i,1)!=string("]") and i!=s.size())
        {   temp+=s.substr(i,1);
        }    
        else
        {
            temp+=s.substr(i,1);
            temp_vec=convert_string_to_vec(temp);
            v.push_back(temp_vec);
            temp="";
            i++;
        }
    }
    return v;
}
//////////////////////////////////////////////////////////////////////////////

