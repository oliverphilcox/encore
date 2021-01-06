#ifndef WEIGHT_FUNCTIONS_H
#define WEIGHT_FUNCTIONS_H

// maximum order for 3PCF/4PCF
#define MAXORDER 10
// maximum order for 5PCF
#define MAXORDER5 4

#define NLM_MAX ((MAXORDER+1)*(MAXORDER+2)/2)
#define NLM ((ORDER+1)*(ORDER+2)/2)

// Some global constants for the 3PCF a_lm*a_lm.conj() normalizations.
// From: http://en.wikipedia.org/wiki/Table_of_spherical_harmonics

// All factors are of the form a*sqrt(b/pi), so let's use that:
#define YNORM(a,b) ((1.0*a)*(1.0*a)*(1.0*b)/M_PI)
// These are the normalizations for [a_lm]^2.
// For a_lm normalizations, we take the square root of these, and add a factor (-1)^m
// NB: this does *not* include a factor of 2 for the m<0 / m>0 symmetry; this is added in elsewhere.

static Float almnorm[NLM_MAX] = {
    YNORM(1/2,1),

    YNORM(1/2,3),
    YNORM(1/2,3/2),

    YNORM(1/4,5),
    YNORM(1/2,15/2),
    YNORM(1/4,15/2),

    YNORM(1/4,7),
    YNORM(1/8,21),
    YNORM(1/4,105/2),
    YNORM(1/8,35),

    YNORM(3/16,1),
    YNORM(3/8,5),
    YNORM(3/8,5/2),
    YNORM(3/8, 35),
    YNORM(3/16, 35/2),

    YNORM(1/16, 11),
    YNORM(1/16, 165/2),
    YNORM(1/8, 1155/2),
    YNORM(1/32, 385),
    YNORM(3/16, 385/2),
    YNORM(3/32, 77),

    YNORM(1/32, 13),
    YNORM(1/16, 273/2),
    YNORM(1/64, 1365),
    YNORM(1/32, 1365),
    YNORM(3/32, 91/2),
    YNORM(3/32, 1001),
    YNORM(1/64, 3003),

    YNORM(1/32, 15),
    YNORM(1/64, 105/2),
    YNORM(3/64, 35),
    YNORM(3/64, 35/2),
    YNORM(3/32, 385/2),
    YNORM(3/64, 385/2),
    YNORM(3/64, 5005),
    YNORM(3/64, 715/2),

    YNORM(1/256, 17),
    YNORM(3/64, 17/2),
    YNORM(3/128, 595),
    YNORM(1/64, 19635/2),
    YNORM(3/128, 1309/2),
    YNORM(3/64, 17017/2),
    YNORM(1/128, 7293),
    YNORM(3/64, 12155/2),
    YNORM(3/256, 12155/2),

    YNORM(1/256, 19),
    YNORM(3/256, 95/2),
    YNORM(3/128, 1045),
    YNORM(1/256, 21945),
    YNORM(3/128, 95095/2),
    YNORM(3/256, 2717),
    YNORM(1/128, 40755),
    YNORM(3/512, 13585),
    YNORM(3/256, 230945/2),
    YNORM(1/512, 230945),

    YNORM(1/512, 21),
    YNORM(1/256, 1155/2),
    YNORM(3/512, 385/2),
    YNORM(3/256, 5005),
    YNORM(3/256, 5005/2),
    YNORM(3/256, 1001),
    YNORM(3/1024, 5005),
    YNORM(3/512, 85085),
    YNORM(1/512, 255255/2),
    YNORM(1/512, 4849845),
    YNORM(1/1024, 969969)
};

// Create array for 3PCF weights (to be filled at runtime from almnorm)
Float weight3pcf[NLM];

// Include the full coupling matrix up to ell = MAXORDER
// This is defined as C_l^m = (-1)^(l-m)/Sqrt[2l+1]
// Format is an array of dimension NLM_MAX
#include "coupling3PCF.h"


#ifdef FOURPCF
// Create array for 4PCF weights (to be filled at runtime from almnorm and the coupling matrices)
// We need two sets of matrices to do the summation efficiently
// Note these size allocations are somewhat overestimated, so will have zeros at the end
// NB: the second matrix has zeros for m1 and/or m2 = 0
Float weight4pcf1[NLM*NLM*(ORDER+1)];
Float weight4pcf2[NLM*NLM*(ORDER+1)];

// Full 4PCF weighting matrix
// This contains (-1)^{l1+l2+l3} ThreeJ[(l1, m1) (l2, m2) (l3, m3)]
// Data-type is a 3D array indexing {(l1,m1), (l2,m2), (l3)} with the (l1,m1) and (l2,m2) flattened.
// This should be read-in and converted to whatever length is necessary as a 1D array
#include "coupling4PCF.h"
#endif

#ifdef FIVEPCF
// Create array for 5PCF weights (to be filled at runtime from almnorm and the coupling matrices)
// These are of slightly different format to the 4PCF matrices, using just a single array (to cut down on memory usage)
// Note these size allocations are somewhat overestimated, since we drop any multipoles disallowed by the triangle conditions
// Notably they need both odd and even m_1, m_2, m_3 to be stored.

Float weight5pcf[int(pow(ORDER+1,8))];

// Full 5PCF weighting matrix
// This contains (-1)^{l1+l2+l3+l4} Sum_{m12} (-1)^{l12-m12} ThreeJ[(l1, m1) (l2, m2) (l12, -m12)]ThreeJ[(l12, m12) (l3, m3) (l4, m4)]
// Data-type is a 5D array indexing {(l1,m1), (l2,m2), l12, (l3,m3), l4} with the (l1,m1), (l2,m2) and (l3,m3) flattened.
// This should be read-in and converted to whatever length is necessary as a 1D array

#include "coupling5PCF.h"

#endif


#endif
