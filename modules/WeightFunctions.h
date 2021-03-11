#ifndef WEIGHT_FUNCTIONS_H
#define WEIGHT_FUNCTIONS_H

// maximum order for 3PCF/4PCF
#define MAXORDER 10
// maximum order for 5PCF
#define MAXORDER5 5
// maximum order for 6PCF
#define MAXORDER6 3

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

// Create array for 3PCF weights (to be filled at runtime from almnorm and the coupling matrices)
Float weight3pcf[NLM];
Float threepcf_coupling[NLM_MAX];

void load_3pcf_coupling(){
  // Load the full coupling matrix up to ell = MAXORDER from file
  // This is defined as C_l^m = (-1)^(l-m)/Sqrt[2l+1]
  // Format is an array of dimension NLM_MAX

  char line[100000];
  FILE *fp;
  char filename [100];
  snprintf(filename,sizeof(filename), "coupling_matrices/weights_3pcf_LMAX%d.txt",MAXORDER);

  fp = fopen(filename,"r");
  if (fp==NULL){
     fprintf(stderr,"Coupling matrix file %s not found - this should be recomputed!\n",filename);
     abort();
  }

  // Read in values from file (straightforward as file has no comment strings and one value per line)
  int line_count = 0;
  while (fgets(line,1000000,fp)!=NULL) threepcf_coupling[line_count++]=atof(line);
  assert(line_count==NLM_MAX);
};

void generate_3pcf_weights(){
  // Generate the 3PCF weight array for the specific LMAX used here.
  // This includes the additional normalization factors
  // We fill up the one-dimensional weight3pcf array

  // First initialize this to zero for safety
  for(int x=0; x<NLM; x++) weight3pcf[x] = 0.;

  for(int ell=0, n=0; ell<=ORDER; ell++){
    for(int m=0; m<=ell; m++, n++){
        // Add coupling weight, alm normalizations and symmetry factor of 2 unless m1=m2=0
        // The (-1)^m factor comes from replacing a_{l-m} with its conjugate a_{lm}* later.
        // This cancels with another (-1)^m factor in the coupling
        weight3pcf[n] = 2.*almnorm[n]*threepcf_coupling[n]*pow(-1.,m);
        if (m==0) weight3pcf[n] /= 2.;
      }
  }
}

#ifdef FOURPCF
// Create array for 4PCF weights (to be filled at runtime from almnorm and the coupling matrices)
// This just uses a single array (to cut down on memory usage)
// Note these size allocations are somewhat overestimated, since we drop any multipoles disallowed by the triangle conditions
// We need both odd and even m_1, m_2 to be stored.

Float *weight4pcf;

// array for all possible weights up to MAX_ORDER
Float fourpcf_coupling[(MAXORDER+1)*(MAXORDER+1)][(MAXORDER+1)*(MAXORDER+1)][(MAXORDER+1)];

void load_4pcf_coupling(){

  // Load the full coupling matrix up to ell = MAXORDER from file
  // This is defined as C_m^Lambda = (-1)^{l1+l2+l3} ThreeJ[(l1, m1) (l2, m2) (l3, -m3)]
  // Data-type is a 3D array indexing {(l1,m1), (l2,m2), l3} with the (l1,m1) and (l2,m2) flattened.
  // It will be trimmed to the relevant l_max at runtime.

  // Allocate memory
  weight4pcf = (Float *)malloc(sizeof(Float)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1));

  char line[100000];
  FILE *fp;
  char filename [100];
  snprintf(filename,sizeof(filename), "coupling_matrices/weights_4pcf_LMAX%d.txt",MAXORDER);

  fp = fopen(filename,"r");
  if (fp==NULL){
     fprintf(stderr,"Coupling matrix file %s not found - this should be recomputed!\n",filename);
     abort();
  }

  // First count number of lines in file, and allocate memory (straightforward as file has no comment strings and one value per line)
  int line_count=0;
  while (fgets(line,1000,fp)!=NULL) line_count++;
  rewind(fp);
  Float* tmp_arr = (Float *)malloc(sizeof(Float)*line_count); // array to hold flattened weights

  // Read in values from file
  line_count = 0;
  while (fgets(line,1000,fp)!=NULL) tmp_arr[line_count++]=atof(line);

  // Now reconstruct array using the triangle conditions to pick out the relevant elements
  // note that we don't need to initialize the other elements as they're never used
  for(int l1=0, n=0;l1<=MAXORDER;l1++){
    for(int l2=0;l2<=MAXORDER;l2++){
      for(int l3=abs(l1-l2);l3<=fmin(MAXORDER,l1+l2);l3++){
        if(pow(-1,l1+l2+l3)==-1) continue; // skip odd parity
        for(int m1=-l1;m1<=l1;m1++){
          for(int m2=-l2;m2<=l2;m2++){
            if(abs(m1+m2)>l3) continue; // m3 condition
            fourpcf_coupling[l1*l1+l1+m1][l2*l2+l2+m2][l3] = tmp_arr[n++];
          }
        }
      }
    }
  }
};

void generate_4pcf_weights(){
  // Generate the 4PCF weight array for the specific LMAX used here.
  // This includes the additional normalization factors
  // We fill up the one-dimensional weight4pcf array

  // We start by initializing all elements to zero
  for(int x=0; x<int(pow(ORDER+1,5)); x++){
    weight4pcf[x] = 0.0;
  }

  int m3, n=0;
  // Now load 4PCF weight matrices, only filling values that don't violate triangle conditions
  for(int l1=0; l1<=ORDER; l1++){ // n indexes the (l1,l2,l3,m1,m2) quintuplet (m3 is specified by triangle conditions)
    for(int l2=0; l2<=ORDER; l2++){
      for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2);l3++){
        if(pow(-1,l1+l2+l3)==-1) continue; // skip odd parity combinationss
        // NB: we sum m_i from -li to li here. m3>=0 however.
        for(int m1=-l1; m1<=l1; m1++){
          for(int m2=-l2; m2<=l2; m2++){
            m3 = -m1-m2;
            if(m3<0) continue; // only need to use m3>=0
            if (m3>l3) continue; // this violates triangle conditions
            // Now add in the weights. This is 2 * coupling[l1, l2, l3, m1, m2, -m1-m2] S(m1+m2) with S(M) = 1/2 if M=0 and unity else.
            // We also add in alm normalization factors, including the extra factor of (-1)^{m1+m2+m3} (which is trivial)
            weight4pcf[n] = 2.*sqrt(almnorm[l1*(1+l1)/2+abs(m1)]*almnorm[l2*(1+l2)/2+abs(m2)]*almnorm[l3*(1+l3)/2+m3]);
            weight4pcf[n] *= fourpcf_coupling[l1*l1+l1+m1][l2*l2+l2+m2][l3];
            if(m3==0) weight4pcf[n] /= 2;
            // also add in factors from complex conjugations of m1,m2
            if(m1<0) weight4pcf[n] *= pow(-1.,m1);
            if(m2<0) weight4pcf[n] *= pow(-1.,m2);
            n++;
          }
        }
      }
    }
  }
};
#endif

#ifdef DISCONNECTED
// Create array for disconnected weights

Float weightdiscon[(ORDER+1)*(ORDER+1)];

void generate_discon_weights(){
  // Generate the disconnected weight arrays for the specific LMAX used here.
  // This includes the additional normalization factors and (-1)^m from complex conjugation

  for (int ell1=0, n1=0; ell1<=ORDER; ell1++) {
    for (int mm1=-ell1; mm1<=ell1; mm1++, n1++) {
      // Define normalization, including the extra factor of (-1)^m1
      weightdiscon[n1] = sqrt(almnorm[ell1*(ell1+1)/2+abs(mm1)])*pow(-1.,mm1);
      // also add in factors from complex conjugations of m1,m2
      if(mm1<0) weightdiscon[n1] *= pow(-1.,mm1);
    }
  }
};
#endif

#ifdef FIVEPCF
// Create array for 5PCF weights (to be filled at runtime from almnorm and the coupling matrices)
// These are of slightly different format to the 4PCF matrices, using just a single array (to cut down on memory usage)
// Note these size allocations are somewhat overestimated, since we drop any multipoles disallowed by the triangle conditions
// Notably they need both odd and even m_1, m_2, m_3 to be stored.
// We note that intermediate momenta can go up to 2*ell_max by 3j conditions

Float *weight5pcf;

// array for all possible weights up to MAX_ORDER
Float fivepcf_coupling[(MAXORDER5+1)*(MAXORDER5+1)][(MAXORDER5+1)*(MAXORDER5+1)][(2*MAXORDER5+1)][(MAXORDER5+1)*(MAXORDER5+1)][(MAXORDER5+1)];

void load_5pcf_coupling(){

  // Load the full coupling matrix up to ell = MAXORDER5 from file
  // This is defined as C_m^Lambda = (-1)^{l1+l2+l3+l4} Sum_{m12} (-1)^{l12-m12} ThreeJ[(l1, m1) (l2, m2) (l12, -m12)]ThreeJ[(l12, m12) (l3, m3) (l4, m4)]
  // Data-type is a 3D array indexing {(l1,m1), (l2,m2), l12, (l3,m3), l4} with the (l1,m1), (l2,m2) and (l3,m3) flattened.
  // It will be trimmed to the relevant l_max at runtime.

  // Assign memory to array
  weight5pcf = (Float *)malloc(sizeof(Float)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(2*ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1));

  char line[100000];
  FILE *fp;
  char filename [100];
  snprintf(filename,sizeof(filename), "coupling_matrices/weights_5pcf_LMAX%d.txt",MAXORDER5);

  fp = fopen(filename,"r");
  if (fp==NULL){
     fprintf(stderr,"Coupling matrix file %s not found - this should be recomputed!\n",filename);
     abort();
  }

  // First count number of lines in file, and allocate memory (straightforward as file has no comment strings and one value per line)
  int line_count=0;
  while (fgets(line,1000,fp)!=NULL) line_count++;
  rewind(fp);
  Float* tmp_arr = (Float *)malloc(sizeof(Float)*line_count); // array to hold flattened weights

  // Read in values from file
  line_count = 0;
  while (fgets(line,1000000,fp)!=NULL) tmp_arr[line_count++]=atof(line);

  // Now reconstruct array using the triangle conditions to pick out the relevant elements
  // note that we don't need to initialize the other elements as they're never used
  for(int l1=0, n=0;l1<=MAXORDER5;l1++){
    for(int l2=0;l2<=MAXORDER5;l2++){
      for(int l12=abs(l1-l2);l12<=l1+l2;l12++){ // allow any l12!
        for(int l3=0;l3<=MAXORDER5;l3++){
          for(int l4=abs(l12-l3);l4<=fmin(MAXORDER5,l12+l3);l4++){
            if(pow(-1,l1+l2+l3+l4)==-1) continue; // skip odd parity
            for(int m1=-l1;m1<=l1;m1++){
              for(int m2=-l2;m2<=l2;m2++){
                if(abs(m1+m2)>l12) continue; // m12 condition
                for(int m3=-l3;m3<=l3;m3++){
                  if(abs(m1+m2+m3)>l4) continue; // m4 condition
                  fivepcf_coupling[l1*l1+l1+m1][l2*l2+l2+m2][l12][l3*l3+l3+m3][l4] = tmp_arr[n++];
                }
              }
            }
          }
        }
      }
    }
  }
};

void generate_5pcf_weights(){
    // Generate the 5PCF weight array for the specific LMAX used here.
    // This includes the additional normalization factors
    // We fill up the one-dimensional weight5pcf array

    // We start by initializing all elements to zero
    for(int x=0; x<int(pow(ORDER+1,7)*(2*ORDER+1)); x++){
      weight5pcf[x] = 0.0;
    }

    int m4, n=0;
    // Now load 5PCF weight matrices, only filling values that don't violate triangle conditions
    for(int l1=0; l1<=ORDER; l1++){ // n indexes the (l1,l2,l12,l3,l4,m1,m2,m3) octuplet (m4 is specified by triangle conditions)
      for(int l2=0; l2<=ORDER; l2++){
        for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){ // allow any ell12!
          for(int l3=0; l3<=ORDER; l3++){
            for(int l4=fabs(l12-l3);l4<=fmin(ORDER,l12+l3);l4++){
              if(pow(-1,l1+l2+l3+l4)==-1) continue; // skip odd parity combinationss
              // NB: we sum m_i from -li to li here. m4>=0 however.
              for(int m1=-l1; m1<=l1; m1++){
                for(int m2=-l2; m2<=l2; m2++){
                  if(abs(m1+m2)>l12) continue; // m12 condition
                  for(int m3=-l3; m3<=l3; m3++){
                    m4 = -m1-m2-m3;
                    if (m4<0) continue; // only need to use m4>=0
                    if (m4>l4) continue; // this violates triangle conditions
                    // Now add in the weights. This is 2 * coupling[l1, l2, l12, l3, l4, m1, m2, m3, -m1-m2-m3] * (-1)^{m1+m2+m3} * S(m1+m2+m3) with S(M) = 1/2 if M=0 and unity else.
                    // We also add in alm normalization factors, including the extra factor of (-1)^{m1+m2+m3+m4} (which is trivial)
                    weight5pcf[n] = 2.*sqrt(almnorm[l1*(1+l1)/2+abs(m1)]*almnorm[l2*(1+l2)/2+abs(m2)]*almnorm[l3*(1+l3)/2+abs(m3)]*almnorm[l4*(1+l4)/2+m4]);
                    weight5pcf[n] *= fivepcf_coupling[l1*l1+l1+m1][l2*l2+l2+m2][l12][l3*l3+l3+m3][l4];
                    if(m4==0) weight5pcf[n] /= 2;
                    // also add in factors from complex conjugations of m1->m3
                    if(m1<0) weight5pcf[n] *= pow(-1.,m1);
                    if(m2<0) weight5pcf[n] *= pow(-1.,m2);
                    if(m3<0) weight5pcf[n] *= pow(-1.,m3);
                    n++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

#endif

#ifdef SIXPCF
// Create array for 6PCF weights (to be filled at runtime from almnorm and the coupling matrices)
// These are of the same form as the 5PCF matrices, and store both odd and even m1, m2, m3, m4.
// Note these size allocations are somewhat overestimated, since we drop any multipoles disallowed by the triangle conditions

Float *weight6pcf;

// array for all possible weights up to MAX_ORDER
Float sixpcf_coupling[(MAXORDER6+1)*(MAXORDER6+1)][(MAXORDER6+1)*(MAXORDER6+1)][(2*MAXORDER6+1)][(MAXORDER6+1)*(MAXORDER6+1)][(2*MAXORDER6+1)][(MAXORDER6+1)*(MAXORDER6+1)][(MAXORDER6+1)];

void load_6pcf_coupling(){
  // Load the full coupling matrix up to ell = MAXORDER6 from file
  // This is defined as C_m^Lambda = (-1)^{l1+l2+l3+l4+l5} Sum_{m12, m123} (-1)^{l12-m12+l123-m123} ThreeJ[(l1, m1) (l2, m2) (l12, -m12)]ThreeJ[(l12, m12) (l3, m3) (l123, -m123)]ThreeJ[(l123, m123) (l4, m4) (l5, m5)]
  // Data-type is a 5D array indexing {(l1,m1), (l2,m2), l12, (l3,m3), l123, (l4, m4), l5} with the (l_i,m_i) sections flattened.
  // It will be trimmed to the relevant l_max at runtime.

  // Assign memory to array
  weight6pcf = (Float *)malloc(sizeof(Float)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(2*ORDER+1)*(ORDER+1)*(ORDER+1)*(2*ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1));

  char line[100000];
  FILE *fp;
  char filename [100];
  snprintf(filename,sizeof(filename), "coupling_matrices/weights_6pcf_LMAX%d.txt",MAXORDER6);

  fp = fopen(filename,"r");
  if (fp==NULL){
     fprintf(stderr,"Coupling matrix file %s not found - this should be recomputed!\n",filename);
     abort();
  }

  // First count number of lines in file, and allocate memory (straightforward as file has no comment strings and one value per line)
  int line_count=0;
  while (fgets(line,1000,fp)!=NULL) line_count++;
  rewind(fp);
  Float* tmp_arr = (Float *)malloc(sizeof(Float)*line_count); // array to hold flattened weights

  // Read in values from file
  line_count = 0;
  while (fgets(line,1000000,fp)!=NULL) tmp_arr[line_count++]=atof(line);

  // Now reconstruct array using the triangle conditions to pick out the relevant elements
  // note that we don't need to initialize the other elements as they're never used
  for(int l1=0,n=0;l1<=MAXORDER6;l1++){
    for(int l2=0;l2<=MAXORDER6;l2++){
      for(int l12=abs(l1-l2);l12<=l1+l2;l12++){ // no ell-max conditions here!
        for(int l3=0;l3<=MAXORDER6;l3++){
          for(int l123=abs(l12-l3);l123<=l12+l3;l123++){ // no ell-max conditions here!
            for(int l4=0;l4<=MAXORDER6;l4++){
              for(int l5=abs(l123-l4);l5<=fmin(MAXORDER6,l123+l4);l5++){
                if(pow(-1,l1+l2+l3+l4+l5)==-1) continue; // skip odd parity
                for(int m1=-l1;m1<=l1;m1++){
                  for(int m2=-l2;m2<=l2;m2++){
                    if(abs(m1+m2)>l12) continue; // m12 condition
                    for(int m3=-l3;m3<=l3;m3++){
                      if(abs(m1+m2+m3)>l123) continue; // m123 condition
                      for(int m4=-l4;m4<=l4;m4++){
                        if(abs(m1+m2+m3+m4)>l5) continue; // m4 condition
                        sixpcf_coupling[l1*l1+l1+m1][l2*l2+l2+m2][l12][l3*l3+l3+m3][l123][l4*l4+l4+m4][l5] = tmp_arr[n++];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

void generate_6pcf_weights(){
  // Generate the 6PCF weight array for the specific LMAX used here.
  // This includes the additional normalization factors
  // We fill up the one-dimensional weight5pcf array

  // We start by initializing them to zero
  for(int x=0; x<int(pow(ORDER+1,9)*(2*ORDER+1)*(2*ORDER+1)); x++){
    weight6pcf[x] = 0.0;
  }

  int m5, n=0;
  // Now load 6PCF weight matrices, only filling values that don't violate triangle conditions
  for(int l1=0; l1<=ORDER; l1++){ // n indexes the (l1,l2,l12,l3,123,l4,l5,m1,m2,m3,m4) undecuplet (m5 is specified by triangle conditions)
    for(int l2=0; l2<=ORDER; l2++){
      for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){ // no ell-max conditions here!
        for(int l3=0; l3<=ORDER; l3++){
          for(int l123=fabs(l12-l3);l123<=l12+l3;l123++){ // no ell-max conditions here!
            for(int l4=0; l4<=ORDER; l4++){
              for(int l5=fabs(l123-l4);l5<=fmin(ORDER,l123+l4);l5++){
                if(pow(-1,l1+l2+l3+l4+l5)==-1) continue; // skip odd parity combinationss
                // NB: we sum m_i from -li to li here. m5>=0 however.
                for(int m1=-l1; m1<=l1; m1++){
                  for(int m2=-l2; m2<=l2; m2++){
                    if(abs(m1+m2)>l12) continue; // m12 condition
                    for(int m3=-l3; m3<=l3; m3++){
                      if(abs(m1+m2+m3)>l123) continue; // m123 condition
                      for(int m4=-l4; m4<=l4; m4++){
                        m5 = -m1-m2-m3-m4;
                        if (m5<0) continue; // only need to use m4>=0
                        if (m5>l5) continue; // this violates triangle conditions
                        // Now add in the weights. This is 2 * coupling[l1, l2, l12, l3, l123, l4, l5, m1, m2, m3, m4, -m1-m2-m3-m4] * (-1)^{m1+m2+m3+m4} * S(m1+m2+m3+m4) with S(M) = 1/2 if M=0 and unity else.
                        // We also add in alm normalization factors, including the extra factor of (-1)^{m1+m2+m3+m4} (which is trivial)
                        weight6pcf[n] = 2.*sqrt(almnorm[l1*(1+l1)/2+abs(m1)]*almnorm[l2*(1+l2)/2+abs(m2)]*almnorm[l3*(1+l3)/2+abs(m3)]*almnorm[l4*(1+l4)/2+abs(m4)]*almnorm[l5*(1+l5)/2+m5]);
                        weight6pcf[n] *= sixpcf_coupling[l1*l1+l1+m1][l2*l2+l2+m2][l12][l3*l3+l3+m3][l123][l4*l4+l4+m4][l5];
                        if(m5==0) weight6pcf[n] /= 2;
                        // also add in factors from complex conjugations of m1->m4
                        if(m1<0) weight6pcf[n] *= pow(-1.,m1);
                        if(m2<0) weight6pcf[n] *= pow(-1.,m2);
                        if(m3<0) weight6pcf[n] *= pow(-1.,m3);
                        if(m4<0) weight6pcf[n] *= pow(-1.,m4);
                        n++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  };
#endif


#endif
