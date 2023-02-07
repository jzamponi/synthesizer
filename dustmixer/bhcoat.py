import numpy as np


# *********************
# compute coated spherical grain optical properties.
# All Bessel functions computed by upward recurrence.
# x: 2 * pi * r_core / wlen
# y: 2 * pi * r_mantle / wlen
# r_core: core radius, same units as wlen
# r_mantle: mantle radius, same units as wlen
# rfrel1: refractive index core (complex)
# rfrel2: refractive index mantle (complex)
# wlen: wavelength, same units as r_core and r_mantle
# output is Qext, Qsca, Qbak
#
# Adapted from Fortran version obtained from C.L.Joseph
# Routine BHCOAT is taken from Bohren & Huffman (1983)
def bhcoat(x, y, rfrel1, rfrel2):
    ii = complex(0e0, 1e0)
    ddel = 1e-8  # inner sphere convergence criterion

    x1 = rfrel1 * x
    x2 = rfrel2 * x
    y2 = rfrel2 * y
    ystop = y + 4e0 * y**(1./3.) + 2e0
    refrel = rfrel2 / rfrel1

    # series terminated after nstop terms
    nstop = ystop

    d0x1 = np.cos(x1) / np.sin(x1)
    d0x2 = np.cos(x2) / np.sin(x2)
    d0y2 = np.cos(y2) / np.sin(y2)
    psi0y = np.cos(y)
    psi1y = np.sin(y)
    chi0y = -np.sin(y)
    chi1y = np.cos(y)
    xi0y = psi0y - ii * chi0y
    xi1y = psi1y - ii * chi1y
    chi0y2 = -np.sin(y2)
    chi1y2 = np.cos(y2)
    chi0x2 = -np.sin(x2)
    chi1x2 = np.cos(x2)
    qsca = 0e0
    qext = 0e0
    xback = complex(0e0, 0e0)
    n = 1
    iflag = 0

    # set variables to None to trigger errors
    # if never initialized
    brack = crack = chipy2 = chiy2 = None
    chix2 = d1x1 = d1x2 = None

    counter = 1
    while True:
        rn = n  # LABEL:200
        psiy = (2e0 * rn - 1e0) * psi1y / y - psi0y
        chiy = (2e0 * rn - 1e0) * chi1y / y - chi0y
        xiy = psiy - ii * chiy
        d1y2 = 1e0 / (rn / y2 - d0y2) - rn / y2
        if iflag != 1:
            d1x1 = 1e0 / (rn / x1 - d0x1) - rn / x1
            d1x2 = 1e0 / (rn / x2 - d0x2) - rn / x2
            chix2 = (2e0 * rn - 1e0) * chi1x2 / x2 - chi0x2
            chiy2 = (2e0 * rn - 1e0) * chi1y2 / y2 - chi0y2
            chipx2 = chi1x2 - rn*chix2 / x2
            chipy2 = chi1y2 - rn*chiy2 / y2
            ancap = refrel * d1x1 - d1x2
            ancap = ancap / (refrel * d1x1 * chix2 - chipx2)
            ancap = ancap / (chix2 * d1x2 - chipx2)
			# ancap sometimes become -inf
            #print(f'ancap: {ancap}  chix2: {chix2}  d1x2: {d1x2}  chipx2: {chipx2}')
            brack = ancap * (chiy2 * d1y2 - chipy2)
            bncap = refrel * d1x2 - d1x1
            bncap = bncap / (refrel * chipx2 - d1x1 * chix2)
            bncap = bncap / (chix2 * d1x2 - chipx2)
            crack = bncap * (chiy2 * d1y2 - chipy2)
            amess1 = brack * chipy2
            amess2 = brack * chiy2
            amess3 = crack * chipy2
            amess4 = crack * chiy2

            # set of conditions
            cond1 = abs(amess1) <= ddel * abs(d1y2)
            cond2 = abs(amess2) <= ddel
            cond3 = abs(amess3) <= ddel * abs(d1y2)
            cond4 = abs(amess4) <= ddel

            # check all conditions
            if all([cond1, cond2, cond3, cond4]):
                brack = complex(0e0, 0e0)
                crack = complex(0e0, 0e0)
                iflag = 1

        dnbar = d1y2 - brack * chipy2  # LABEL:999
        dnbar = dnbar / (1e0 - brack * chiy2)
        gnbar = d1y2 - crack * chipy2
        gnbar = gnbar / (1e0 - crack * chiy2)
        an = (dnbar / rfrel2 + rn / y) * psiy - psi1y
        an = an / ((dnbar / rfrel2 + rn / y) * xiy - xi1y)
        bn = (rfrel2 * gnbar + rn / y) * psiy - psi1y
        bn = bn / ((rfrel2 * gnbar + rn / y) * xiy - xi1y)
        qsca = qsca + (2e0 * rn + 1e0) * (abs(an) * abs(an) + abs(bn) * abs(bn))
        xback = xback + (2e0 * rn + 1e0) * (-1e0)**n * (an - bn)
        qext = qext + (2e0 * rn + 1e0) * (np.real(an) + np.real(bn))
        psi0y = psi1y
        psi1y = psiy
        chi0y = chi1y
        chi1y = chiy
        xi1y = psi1y - ii * chi1y
        chi0x2 = chi1x2
        chi1x2 = chix2
        chi0y2 = chi1y2
        chi1y2 = chiy2
        d0x1 = d1x1
        d0x2 = d1x2
        d0y2 = d1y2
        n = n + 1

        counter = counter + 1 
        # main loop breaking condition
        if n - 1 - nstop >= 0e0:
            break

    iyy = 1e0 / (y * y)
    qqsca = 2e0 * qsca * iyy
    qqext = 2e0 * qext * iyy
    qqabs = qqext - qqsca
    qback = (abs(xback))**2
    qback = qback * iyy

    return qqext, qqsca, qqabs, qback


# *********************
# compute coated spherical grain optical properties,
# using input in physical units instead of relative
# r_core: core radius, cm
# r_mantle: mantle radius, cm
# ref_core: refractive index core (complex)
# ref_mantle: refractive index mantle (complex)
# wlen: wavelength, cm
# output is Qext, Qsca, Qbak
def bhcoat_ph(r_core, r_mantle, ref_core, ref_mantle, wlen):
    xx = 2e0 * np.pi * r_core / wlen
    yy = 2e0 * np.pi * r_mantle / wlen
    return bhcoat(xx, yy, ref_core, ref_mantle)
