import numpy as np
from scipy import interpolate, special, stats
from numpy import log10

def period_distribution_CDF(power1=2.311111111111111, power2=-0.08, cutoff=5.75):
        """
        Creates CDF for underlying period distribution (see Figure 2 Rogers & Owen 2021)
        """

        P_range = np.logspace(0,2,100)
        pdf = []

        # create pdf using double power law
        for i in range(len(P_range)):
            if P_range[i] <= cutoff:
                pdf_i = P_range[i] ** power1
                pdf.append(pdf_i)
            else:
                pdf_i = (cutoff ** (power1-power2)) * (P_range[i] ** power2)
                pdf.append(pdf_i)

        # normalise pdf, calculate cdf
        pdf = pdf / np.sum(pdf)
        cdf = np.cumsum(pdf)
        cdf[0] = 0.0
        cdf[-1] = 1.0

        # remove repeats
        cdf, mask = np.unique(cdf, return_index=True)
        P_mask = P_range[mask]

        # interpolate
        cdf_interp = interpolate.interp1d(cdf, P_range)
        return cdf_interp

def bernstein_CDF(coeffs, lims=[0.6,100]):
    """
    Creates CDF for Bernstein polynomial, used for core mass distribution. See appendix of Rogers & Owen 2021
    """

    x = np.linspace(log10(lims[0]), log10(lims[1]), 50)
    bern_x = np.linspace(0.0,1.0,50)
    bern_pdf = [bernstein_poly(i, len(coeffs)-1, coeffs) for i in bern_x]

    # normalise pdf, calculate cdf
    bern_pdf = bern_pdf / np.sum(bern_pdf)
    bern_cdf = np.cumsum(bern_pdf)
    bern_cdf[0] = 0.0
    bern_cdf[-1] = 1.0

    # remove repeats
    bern_cdf, mask = np.unique(bern_cdf, return_index=True)
    x_mask = x[mask]

    # interpolate
    cdf_interp = interpolate.interp1d(bern_cdf, x_mask)

    return cdf_interp

def bernstein_poly(x, order, coefficients):

    """
    Bernstein polynomials. See appendix of Rogers & Owen 2021
    """

    coefficients = np.array(coefficients)
    poly_array = np.array([special.binom(order, i)*(x**i)*((1-x)**(order-i)) for i in range(order+1)])
    B = np.dot(coefficients, poly_array)

    return B