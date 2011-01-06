#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tnc.h"

static tnc_function function;

int simple_tnc(int n, double x[], /*double _xopt[],*/ double *f, double g[], tnc_function *function, void* state, double ubound, double lbound) {
	int i, rc, maxCGit = 5, maxnfeval = 100, nfeval;
	double fopt = 1.0,
	*low, *up,
    eta = -1.0, stepmx = 10.0,
    accuracy = -1.0, fmin = 0.0, ftol = -1.0, xtol = -1.0, pgtol = -1.0,
    rescale = -1.0, maxv = -1.0;
	
	//xopt = (double*)malloc(sizeof(double)*n);
	//for(i=0;i<n;i++) xopt[i] = _xopt[i];
	
	up = (double*)malloc(sizeof(double)*n);
	low = (double*)malloc(sizeof(double)*n);
	
	//for(i=0;i<n;i++) { if(x[i] > maxv) maxv = x[i]; }
	for(i=0;i<n;i++) { up[i] = ubound; low[i] = lbound; }
	
	rc = tnc(n, x, f, g, function, state, low, up, NULL, NULL, TNC_MSG_NONE,
			 maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
			 rescale, &nfeval);
	
	free(up);
	free(low);
	
	return rc;
}