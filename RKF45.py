import numpy as np


####################### INDIVIDUAL RUNGE-KUTTA METHODS #########################

def RKCashKarp_Step45(t, var_i, dydx_functions, dt, parameters):
    """
    This function uses a Runge-Kutta 4th and 5th order scheme to calculate the
    derivatives of a given system of equations. Both the 5th order solution and
    the error are returned.
    """

    # calculate the individual conributions to prediction
    k1 = dydx_functions(t, var_i, parameters)
    if k1 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k1 = k1*dt
    var1 = var_i + 0.2*k1
    if var1 < 0: # a negative contribution should not be obtained
        return (None, None)

    k2 = dydx_functions(t + 0.2 * dt, var1, parameters)
    if k2 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k2 = k2*dt
    var2 = var_i + (3/40)*k1 + (9/40)*k2
    if var2 < 0: # a negative contribution should not be obtained
        return (None, None)

    k3 = dydx_functions(t + 0.3 * dt, var2, parameters)
    if k3 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k3 = k3*dt
    var3 = var_i + (3/10)*k1 - (9/10)*k2 + (6/5)*k3
    if var3 < 0: # a negative contribution should not be obtained
        return (None, None)

    k4 = dydx_functions(t + 0.6 * dt, var3, parameters)
    if k4 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k4 = k4*dt
    var4 = var_i - (11/54)*k1 + (5/2)*k2 - (70/27)*k3 + (35/27)*k4
    if var4 < 0: # a negative contribution should not be obtained
        return (None, None)

    k5 = dydx_functions(t +       dt, var4, parameters)
    if k5 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k5 = k5*dt
    var5 = var_i + (1631/55296)*k1 + (175/512)*k2 + (575/13824)*k3 \
         + (44275/110592)*k4 + (253/4096)*k5
    if var5 < 0: # a negative contribution should not be obtained
        return (None, None)

    k6 = dydx_functions(t + (7/8)*dt, var5, parameters)
    if k6 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k6 = k6*dt

    # calculate the 5th order solution
    RK5_sol = var_i + (37/378)*k1 + (250/621)*k3 + (125/594)*k4 + (512/1771)*k6
    # calculate the 4th order solution
    RK4_sol = var_i + (2825/27648)*k1 + (18575/48384)*k3 + (13525/55296)*k4 + (277/14336)*k5 + 0.25*k6

    # estimate the error, given by the difference between the 4th and 5th order solutions
    err = RK5_sol - RK4_sol

    return (RK5_sol, err)


def RKFehlberg12(t, var_i, dydx_functions, dt, parameters):
    """
    This function uses a Runge-Kutta 2nd and 1st order scheme to calculate the
    derivatives of a given system of equations. Both the 2nd order solution and
    the error are returned.
    """

    k1 = dt * dydx_functions(t, var_i, parameters)
    var1 = var_i + 0.5*k1
    if var1 < 0: # a negative contribution should not be obtained
        return (None, None)

    k2 = dt * dydx_functions(t + 0.5 * dt, var1, parameters)
    var2 = var_i + (1/256)*k1 + (255/256)*k2
    if var2 < 0: # a negative contribution should not be obtained
        return (None, None)

    k3 = dt * dydx_functions(t + dt, var2, parameters)

    # calculate the 1st order solution
    RK1_sol = var_i + (1/256)*k1 + (255/256)*k2
    # calculate the 2nd order solution
    RK2_sol = var_i + (1/512)*k1 + (255/256)*k2 + (1/512)*k3

    # estimate the error, given by the difference between the 2nd and 1st order solutions
    err = RK2_sol - RK1_sol

    return (RK2_sol, err)



######################## CHOOSE THE APPROPRIATE STEPSIZE #######################
def step_control(t, var_i, dt_try, dydx_functions, accuracy, parameters):

    """
    This function controls the step size used in the RKCashKarp_Step function. If
    a reasonable step size has been used, the solution, new time point and next time
    step are returned.
    """

    # calculate the derivatives
    dydx = dydx_functions(t, var_i, parameters)

    # the scaling factor is used to ensure the relative error is bounded
    yscal = abs(var_i) + abs(dydx * dt_try) + 1e-3

    # try a time step
    dt = dt_try
    while True:
        # take an RK45 step
        (var_new, var_err) = RKCashKarp_Step45(t, var_i, dydx_functions, dt, parameters)
        if (var_new, var_err) == (None, None):
            dt = 0.01*dt
            continue
        if (var_new, var_err) == ("BRENTQ ERROR", "BRENTQ ERROR"):
            return [0.0, 0.0, 0.0]
        # find the maximum error of all ODEs, also scale to required tolerance
        err_max = abs(var_err/yscal) / accuracy

        # step succeeded i.e. error <= required accuracy
        if err_max <= 1.0:
            break

        # propose new stepsize; in this case truncation was too large so reduce dt
        dt_new = 0.9*dt/err_max**0.25


        # if the stepsize reduced too much by above, only reduce by factor of 10 instead
        if abs(dt_new) < 0.1*abs(dt):
            dt_new = 0.1*dt

        # update dt
        dt = dt_new
        # error if there's been no update to variables
        if t+dt == t:
            print("ERROR: step size too small")

    # if the maximum error was far too small, increase by factor of 5
    if err_max < 2e-4:
        dt_next = 5.0*dt
    # if step was too small, but in the right region - increase by **0.2 next step
    else:
        dt_next = 0.9*(dt/err_max)**0.2

    return [t+dt, var_new, dt_next]
