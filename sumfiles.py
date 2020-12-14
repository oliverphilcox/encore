import sys
import re
import string
import numpy as np
import scipy.misc as sm

# Syntax: list all of the D-R files first, with the RRR file last
# For example, the glob r* will put r00 before rrr.

# Average all of the D-R files, then divide by the RRR file

nfile = 0
bins = -1
order = -1

out_dir = str(sys.argv[1])

numfiles = len(sys.argv[2:])
print "# Looking at %d files" % (numfiles)

for filename in sys.argv[2:]:
    f = open(filename,'r')

    for line in f:
	if (re.match("Bins = ",line)):
	    if (bins<0):
		bins = string.atoi(line.split()[-1])
		print "# Using %d bins" % (bins)
	if (re.match("Order = ",line)):
	    if (order<0):
		order = string.atoi(line.split()[-1])
		print "# Using order %d" % (order)

		pairs = np.zeros((bins,numfiles))
		power = np.zeros((bins,bins,order+1,numfiles))

	if (re.match("#", line)):
	    continue
	if (re.search("=", line)):
	    continue
	if (re.match("Multipole", line)):
	    continue
	if (re.match("^$", line)):
	    continue
	if (re.match("There", line)):
	    continue
	if (re.match("Pairs",line)):
        # We've found a pairs line.  Process it.
	    b,cnt = (line.split()[1:-1])
	    pairs[string.atoi(b),nfile] = string.atoi(cnt)
	    continue

	# Otherwise, we have a power line, so process that
	s = line.split()
	b1 = string.atoi(s[0])
	b2 = string.atoi(s[1])
	p0 = string.atof(s[2])
	for p in range(order+1):
	    if (p==0):
		power[b1,b2,p,nfile] = p0
	    else:
		power[b1,b2,p,nfile] = p0*string.atof(s[2+p])


    # End loop over lines in a file
    nfile+=1
# End loop over files

# Sum up over the D-R files.
# Also compute the stdev so we can monitor convergence.

nfile -= 1    # Now [nfile] is RRR and the other loops can be normal

# Fix the normalization of the C_ell
for ell in range(0,order+1):
    power[:,:,ell,:] *= (2.*ell+1.)/2.0

pairsD = np.average(pairs[:,0:-1],axis=1)
powerD = np.average(power[:,:,:,0:-1],axis=3)

sqdof = np.sqrt(nfile)   # Convert to error on the mean
if (sqdof==0): sqdof = 1e30
pairsDsig = np.std(pairs[:,0:-1],axis=1,ddof=1)/sqdof
powerDsig = np.std(power[:,:,:,0:-1],axis=3,ddof=1)/sqdof

pairsR = pairs[:,nfile]
powerR = power[:,:,:,nfile]

zeta = np.copy(powerD)   # Just a place-holder

# Now combine to make our desired output

print
print "Multipole RRR correction factors (f_ell): "
f = np.copy(powerR)

for b1 in range(bins):
    for b2 in range(b1):
	f[b1,b2,1:] /= f[b1,b2,0]
	print "%2d %2d "% (b1,b2),
	np.set_printoptions(precision=5,suppress=True,linewidth=1000,formatter={'float': '{: 0.5f}'.format})
	print f[b1,b2,1:]
	np.set_printoptions()
    print

# Now we have the RRR corrections.  For each bin, these adjust
# the final zeta_ell's.

def triplefact(j1,j2,j3):
    jhalf = (j1+j2+j3)/2.0
    return sm.factorial(jhalf) /sm.factorial(jhalf-j1) /sm.factorial(jhalf-j2) /sm.factorial(jhalf-j3)

def threej(j1,j2,j3):
    # Compute {j1,j2,j3; 0,0,0} 3j symbol
    j = j1+j2+j3
    if (j%2>0): return 0     # Must be even
    if (j1+j2<j3): return 0  # Check legal triangle
    if (j2+j3<j1): return 0  # Check legal triangle
    if (j3+j1<j2): return 0  # Check legal triangle
    return (-1)**(j/2.0) * triplefact(j1,j2,j3) / (triplefact(2*j1,2*j2,2*j3)*(j+1))**0.5
    # DJE did check this against Wolfram

def Mjl_calc(j,ell,flist):
    # WARNING: Check this setup of the matrix!
    s=0.
    for ellprime in np.arange(1,len(flist)):
	s += (threej(ell,ellprime,j))**2*flist[ellprime]
    s*=(2.*j+1.)
    return s

powerD_unnorm = powerD.copy()
powerR_unnorm = powerR.copy()

print
print "Three-point Function (no multipole RRR corrections), with errors due to randoms: "
for b1 in range(bins):
    for b2 in range(b1):
	# Get zeta's by dividing by RRR_0
	powerD[b1,b2,:] /= powerR[b1,b2,0]
	powerDsig[b1,b2,:] /= powerR[b1,b2,0]

	# Now adjust for the f_ell
	Mjl = np.zeros((order+1,order+1))
	for j in range(order+1):
	    for k in range(order+1):
		Mjl[j][k] = Mjl_calc(j,k,f[b1,b2,:])

	# WARNING: The following lines may be the wrong math!!!
	geometry = np.linalg.inv(np.identity(order+1)+Mjl)
	zeta[b1,b2,:] = geometry.dot(powerD[b1,b2,:])

	# Now normalize to ell=0 for printing
	zeta[b1,b2,1:] /= zeta[b1,b2,0]
	powerD[b1,b2,1:] /= powerD[b1,b2,0]
	powerDsig[b1,b2,0:] /= powerD[b1,b2,0]
	print "%2d %2d %10.7f "% (b1,b2,zeta[b1,b2,0]),
	np.set_printoptions(precision=5,suppress=True,linewidth=1000,formatter={'float': '{: 0.5f}'.format})
	print zeta[b1,b2,1:], "zeta"
	print "%2d %2d %10.7f "% (b1,b2,powerD[b1,b2,0]),
	print powerD[b1,b2,1:], "raw"
	print "%2d %2d %10.7f " % (b1,b2, powerDsig[b1,b2,0]),
	print powerDsig[b1,b2,1:], "sigma"
	print
	np.set_printoptions()


print
# print "Two-point Correlation Monopole:"
# xi = pairsD/pairsR
# print xi
#
# print "Two-point Correlation Monopole Error due to Randoms:"
# xisig = pairsDsig/pairsR
# print xisig

outfile = out_dir+'/3pcf_output.npz'
print "Saving output to %s"%outfile
np.savez(outfile,zeta=zeta,powerD=powerD,powerDsig=powerDsig,pairsD=pairsD,pairsR=pairsR,
         pairsDsig=pairsDsig,f=f,bins=bins,order=order,numfiles=numfiles,powerR=powerR,
         powerD_unnorm=powerD_unnorm,powerR_unnorm=powerR_unnorm) #xi=xi,xisig=xisig,
