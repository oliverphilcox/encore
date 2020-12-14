#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define FOR(a,b,c) for(a=b;a<=c;a++) 

int main(void) {
    FILE *fp;
    
    fp = fopen("CMASM.c","w"); assert(fp!=NULL);

#ifdef AVXMULTIPOLES
    int i,j,k,ORDER,maxm,m;

    fprintf(fp,"#include \"avxsseabrev.h\"  \n");

    fprintf(fp," typedef struct { double v[4]; } d4; \n");

    fprintf(fp,"#define X      \"%%ymm0\" \n");
    fprintf(fp,"#define Y      \"%%ymm1\" \n");
    fprintf(fp,"#define Z      \"%%ymm2\" \n");

    fprintf(fp,"#define XI     \"%%ymm3\" \n");
    fprintf(fp,"#define XIJ    \"%%ymm4\" \n");
    fprintf(fp,"#define XIJK   \"%%ymm5\" \n");

    fprintf(fp,"#define CX     \"%%ymm4\" \n");
    fprintf(fp,"#define CY     \"%%ymm5\" \n");
    fprintf(fp,"#define CZ     \"%%ymm12\" \n");

    fprintf(fp,"#define X2      \"%%ymm7\" \n");
    fprintf(fp,"#define Y2      \"%%ymm8\" \n");
    fprintf(fp,"#define Z2      \"%%ymm9\" \n");

    fprintf(fp,"#define XI2     \"%%ymm11\" \n");
    fprintf(fp,"#define XIJ2    \"%%ymm12\" \n");
    fprintf(fp,"#define XIJK2   \"%%ymm13\"  \n");

    fprintf(fp,"#define P      \"%%ymm15\" \n");

    
    for(ORDER=1;ORDER<=16;ORDER++) {
        maxm = ((ORDER+1)*(ORDER+2)*(ORDER+3))/6;
        m = 0;

        fprintf(fp, "void MultipoleKernel%d(d4 *ip1x, d4 *ip2x, d4 *ip1y, d4 *ip2y, d4 *ip1z, d4 *ip2z, d4 *cx, d4 *cy, d4 *cz, d4 *globalM, d4 *masses1, d4 *masses2) {\n", ORDER); 

        fprintf(fp, " VLOADPD(*cx->v, CX); VLOADPD(*cy->v, CY); VLOADPD(*cz->v, CZ); \n");
        fprintf(fp, " VLOADPD(*masses1->v,XI); VLOADPD(*masses2->v,XI2); \n");

        fprintf(fp, " VLOADPD(*ip1x->v, X); VLOADPD(*ip1y->v, Y); VLOADPD(*ip1z->v, Z); \n");
        fprintf(fp, " VLOADPD(*ip2x->v, X2); VLOADPD(*ip2y->v, Y2); VLOADPD(*ip2z->v, Z2);  \n");

        fprintf(fp, " VSUBPD(CX,X,X); VSUBPD(CX,X2,X2); \n");
        fprintf(fp, " VSUBPD(CY,Y,Y); VSUBPD(CY,Y2,Y2); \n");
        fprintf(fp, " VSUBPD(CZ,Z,Z); VSUBPD(CZ,Z2,Z2); \n");

        fprintf(fp, "VLOADPD(*globalM->v,P);\n");

        FOR(i,0,ORDER) {
            fprintf(fp, "VMOVAPD(XI,XIJ);\n");
            fprintf(fp, "VMOVAPD(XI2,XIJ2);\n");
        
            FOR(j,0,ORDER-i) {
                fprintf(fp, "VMOVAPD(XIJ,XIJK);\n");
                fprintf(fp, "VMOVAPD(XIJ2,XIJK2);\n");

                FOR(k,0,ORDER-i-j) { 
    
                    fprintf(fp, "VADDPD(XIJK,P,P);\n");
                    fprintf(fp, "VADDPD(XIJK2,P,P);\n");

                    fprintf(fp, "VSTORPD(P,*globalM->v);\n");
                    m++; 
                    if(m!=maxm) {
                        fprintf(fp, "globalM++;\n");
                        fprintf(fp, "VLOADPD(*globalM->v,P);\n");
                    }

                    if(k<ORDER-i-j) fprintf(fp, "VMULPD(Z,XIJK,XIJK);\n");
                    if(k<ORDER-i-j) fprintf(fp, "VMULPD(Z2,XIJK2,XIJK2);\n");
                }
                if(j<ORDER-i) fprintf(fp, "VMULPD(Y,XIJ,XIJ);\n");
                if(j<ORDER-i) fprintf(fp, "VMULPD(Y2,XIJ2,XIJ2);\n");
            }
            if(i<ORDER) fprintf(fp, "VMULPD(X,XI,XI);\n");
            if(i<ORDER) fprintf(fp, "VMULPD(X2,XI2,XI2);\n");
        }
        fprintf(fp, "}\n");
    }

    fprintf(fp,"#undef X       \n");
    fprintf(fp,"#undef Y       \n");
    fprintf(fp,"#undef Z       \n");
    fprintf(fp,"#undef XI      \n");
    fprintf(fp,"#undef XIJ     \n");
    fprintf(fp,"#undef XIJK    \n");

    fprintf(fp,"#undef X2       \n");
    fprintf(fp,"#undef Y2       \n");
    fprintf(fp,"#undef Z2       \n");
    fprintf(fp,"#undef XI2      \n");
    fprintf(fp,"#undef XIJ2     \n");
    fprintf(fp,"#undef XIJK2    \n");

    fprintf(fp,"#undef P        \n");
#endif
    fclose(fp);

    return 0;
}
