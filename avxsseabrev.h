#ifndef __AVX_MACRO__
#define __AVX_MACRO__

#define YMM00 "%ymm0"
#define YMM01 "%ymm1"
#define YMM02 "%ymm2"
#define YMM03 "%ymm3"
#define YMM04 "%ymm4"
#define YMM05 "%ymm5"
#define YMM06 "%ymm6"
#define YMM07 "%ymm7"
#define YMM08 "%ymm8"
#define YMM09 "%ymm9"
#define YMM10 "%ymm10"
#define YMM11 "%ymm11"
#define YMM12 "%ymm12"
#define YMM13 "%ymm13"
#define YMM14 "%ymm14"
#define YMM15 "%ymm15"

#define VXORPS(reg1, reg2, dst)  asm("vxorps " reg1 "," reg2 "," dst);
#define VXORPD(reg1, reg2, dst)  asm("vxorpd " reg1 "," reg2 "," dst);

#define VMOVAPS(src, dst)        asm("vmovaps " src "," dst);

#define VLOADPS(mem, reg)        asm("vmovaps %0, %"reg::"m"(mem));
#define VSTORPS(reg, mem)        asm("vmovaps %"reg ", %0" ::"m"(mem));
#define VLOADPD(mem, reg)        asm("vmovapd %0, %"reg::"m"(mem));
#define VSTORPD(reg, mem)        asm("vmovapd %"reg ", %0" ::"m"(mem));
#define VMOVAPD(src,dst)         asm("vmovapd " src "," dst );


#define VBROADCASTSS(mem, reg)   asm("vbroadcastss %0, %"reg::"m"(mem));
#define VBROADCASTSD(mem, reg)   asm("vbroadcastsd %0, %"reg::"m"(mem));

#define VBROADCASTF128(mem, reg) asm("vbroadcastf128 %0, %"reg::"m"(mem));

#define VADDPS(reg1, reg2, dst)  asm("vaddps " reg1 "," reg2 "," dst);
#define VADDPD(reg1, reg2, dst)  asm("vaddpd " reg1 "," reg2 "," dst);
#define VADDPS_M(mem, reg, dst)  asm("vaddps %0, %"reg ", %"dst " "::"m"(mem));
#define VADDPD_M(mem, reg, dst)  asm("vaddpd %0, %"reg ", %"dst " "::"m"(mem));

#define VSUBPS(reg1, reg2, dst)  asm("vsubps " reg1 "," reg2 "," dst);
#define VSUBPD(reg1, reg2, dst)  asm("vsubpd " reg1 "," reg2 "," dst);
#define VSUBPS_M(mem, reg, dst)  asm("vsubps %0, %"reg ", %"dst " "::"m"(mem));
#define VSUBPD_M(mem, reg, dst)  asm("vsubpd %0, %"reg ", %"dst " "::"m"(mem));

#define VMULPS(reg1, reg2, dst)  asm("vmulps " reg1 "," reg2 "," dst);
#define VMULPD(reg1, reg2, dst)  asm("vmulpd " reg1 "," reg2 "," dst);
#define VMULPS_M(mem, reg, dst)  asm("vmulps %0, %"reg ", %"dst " "::"m"(mem));
#define VMULPD_M(mem, reg, dst)  asm("vmulpd %0, %"reg ", %"dst " "::"m"(mem));

#define VDIVPS(reg1, reg2, dst)  asm("vdivps " reg1 "," reg2 "," dst);
#define VDIVPD(reg1, reg2, dst)  asm("vdivpd " reg1 "," reg2 "," dst);
#define VDIVPS_M(mem, reg, dst)  asm("vdivps %0, %"reg ", %"dst " "::"m"(mem));
#define VDIVPD_M(mem, reg, dst)  asm("vdivpd %0, %"reg ", %"dst " "::"m"(mem));

// src2_0 + src2_1 -> dst_0
// src1_0 + src1_1 -> dst_1
// src2_2 + src2_3 -> dst_2
// src1_2 + src1_3 -> dst_3
#define VHADDPD(src1, src2, dst) asm("vhaddpd " src1 "," src2 "," dst);

#define VRSQRTPS(reg, dst)       asm("vrsqrtps " reg "," dst);

#define VSQRTPS(reg, dst)        asm("vsqrtps " reg "," dst);
#define VSQRTPD(reg, dst)        asm("vsqrtpd " reg "," dst);

#define VZEROALL                 asm("vzeroall");

#define VCVTPD2PS(reg, dst)      asm("vcvtpd2ps " reg "," dst);
#define VCVTPS2PD(reg, dst)      asm("vcvtps2pd " reg "," dst);

#define VPERM2F128(src1, src2, dest, imm) asm("vperm2f128 %0, %"src2 ", %"src1 ", %"dest " "::"g"(imm));
#define VEXTRACTF128(src, dest, imm)      asm("vextractf128 %0, %"src ", %"dest " "::"g"(imm));
#define VSHUFPS(src1, src2, dest, imm)    asm("vshufps %0, %"src2 ", %"src1 ", %"dest " "::"g"(imm));
#define VSHUFPD(src1, src2, dest, imm)    asm("vshufpd %0, %"src2 ", %"src1 ", %"dest " "::"g"(imm));

#define VANDPS(reg1, reg2, dst)      asm("vandps " reg1 "," reg2 "," dst);
#define VCMPPS(reg1, reg2, dst, imm) asm("vcmpps %0, %"reg1 ", %"reg2 ", %"dst " "::"g"(imm));

#define PREFETCH(mem)            asm ("prefetcht0 %0"::"m"(mem))

#endif /* __AVX_MACRO__ */

#ifndef __SSE_MACRO__
#define __SSE_MACRO__

#define XMM00 "%xmm0"
#define XMM01 "%xmm1"
#define XMM02 "%xmm2"
#define XMM03 "%xmm3"
#define XMM04 "%xmm4"
#define XMM05 "%xmm5"
#define XMM06 "%xmm6"
#define XMM07 "%xmm7"
#define XMM08 "%xmm8"
#define XMM09 "%xmm9"
#define XMM10 "%xmm10"
#define XMM11 "%xmm11"
#define XMM12 "%xmm12"
#define XMM13 "%xmm13"
#define XMM14 "%xmm14"
#define XMM15 "%xmm15"


#define XORPS(a, b)         asm("xorps "  a  ","  b );
#define LOADPS(mem, reg)    asm("movaps %0, %"reg::"m"(mem));
#define LOADLPS(mem, reg)   asm("movlps %0, %"reg::"m"(mem));
#define LOADHPS(mem, reg)   asm("movhps %0, %"reg::"m"(mem));
#define LDDQU(mem, reg)     asm("lddqu %0, %"reg::"m"(mem));
#define MOVAPSX(mem, reg)   asm("movaps %0, %"reg::"x"(mem));
#define STORPS(reg, mem)    asm("movaps %"reg " , %0"::"m"(mem));
#define MOVAPS(src, dst)    asm("movaps " src "," dst);
#define MOVQ(src, dst)      asm("movq " src "," dst);
#define BCAST0(reg)         asm("shufps $0x00, " reg ","  reg);
#define BCAST1(reg)         asm("shufps $0x55, " reg ","  reg);
#define BCAST2(reg)         asm("shufps $0xaa, " reg ","  reg);
#define BCAST3(reg)         asm("shufps $0xff, " reg ","  reg);
#define MULPS(src, dst)     asm("mulps " src "," dst);
#define MULPS_M(mem, reg)   asm("mulps %0, %"reg::"m"(mem));
#define ADDPS(src, dst)     asm("addps " src ","  dst);
#define ADDPS_M(mem, reg)   asm("addps %0, %"reg::"m"(mem));
#define SUBPS(src, dst)     asm("subps "  src "," dst);
#define SUBPS_M(mem, reg)   asm("subps %0, %"reg::"m"(mem));
#define MINPS(src,dst)      asm("minps  " src "," dst);
#define MINPS_M(mem, reg)   asm("minps %0, %"reg ::"m"(mem));
#define RSQRTPS(src, dst)   asm("rsqrtps " src "," dst);
#define RSQRTPS_M(mem, reg) asm("rsqrtps %0, %"reg ::"m"(mem));
#define SQRTPS(src, dst)    asm("sqrtps " src "," dst);
#define SQRTPS_M(mem, reg)  asm("sqrtps %0, %"reg ::"m"(mem));
#define RCPPS(src, dst)     asm("rcpps " src "," dst);
#define RCPPS_M(mem, reg)   asm("rcpps %0, %"reg ::"m"(mem));
#define CVTTPS2PI(src, dst) asm("cvttps2pi " src "," dst);
#define CVTTPS2DQ(src, dst) asm("cvttps2dq " src "," dst);
#define CVTDQ2PS(src, dst)  asm("cvtdq2ps " src "," dst);
#define UNPCKLPS(src, dst)  asm("unpcklps "  src "," dst);
#define UNPCKHPS(src, dst)  asm("unpckhps "  src "," dst);
#define CMPLTPS(src, dst)   asm("cmpltps "  src "," dst);
#define CMPLTPS_M(mem, reg) asm("cmpltps %0, %"reg ::"m"(mem));
#define ANDPS(src, dst)     asm("andps " src "," dst);
#define ANDNPS(src, dst)    asm("andnps " src "," dst);
#define EXTINT0(reg, ireg)  asm volatile("pextrw $0, %"reg " , %0":"=r"(ireg));
#define EXTINT1(reg, ireg)  asm volatile("pextrw $2, %"reg " , %0":"=r"(ireg));
#define EXTINT2(reg, ireg)  asm volatile("pextrw $4, %"reg " , %0":"=r"(ireg));
#define EXTINT3(reg, ireg)  asm volatile("pextrw $6, %"reg " , %0":"=r"(ireg));
#define PREFETCH(mem)       asm ("prefetcht0 %0"::"m"(mem))
#define NOP                 asm("nop");

#endif /* __SSE_MACRO__ */
