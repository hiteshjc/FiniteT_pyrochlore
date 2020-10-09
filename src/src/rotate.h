#ifndef ROTATE_HEADER
#define ROTATE_HEADER

///////////////////////////////////////////////////////////////////////////////////////////
//
//                                 ROTATION
//
//////////////////////////////////////////////////////////////////////////////////////////
#include"global.h"
#include"number_functions.h"
#include"search_for.h"
#include"printing_functions.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

using namespace std;

double cost(const gsl_vector * x, void *params);
void rotate_one_rdm(RMatrix &in_one_rdm,
		    RMatrix &unitary,
                    RMatrix &out_one_rdm);

void optimize_unitary_from_one_rdm(std::vector<RMatrix> &in_one_rdm, RMatrix &unitary, std::vector<RMatrix> &out_one_rdm);  

#endif
