#ifndef WEIGHT_FUNCTIONS_H
#define WEIGHT_FUNCTIONS_H

#define MAXORDER 10
#define NLM_MAX ((MAXORDER+1)*(MAXORDER+2)/2)

// Some global constants for the 3PCF a_lm*a_lm.conj() normalizations.
// From: http://en.wikipedia.org/wiki/Table_of_spherical_harmonics

// All factors are of the form a*sqrt(b/pi), so let's use that:
#define YNORM(a,b) ((1.0*a)*(1.0*a)*(1.0*b)/M_PI)
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

// Create array for 3PCF weights (to be filled later from almnorm)
static Float weight3pcf[NLM_MAX];

#ifdef FOURPCF

/// INCLUDE 4PCF WEIGHTS HERE
// needs many more weights for this...
static Float weight4pcf[NLM_MAX];

#endif

#endif
