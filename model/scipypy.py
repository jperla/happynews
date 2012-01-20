import math

def gammaln(xx):
    """Returns the value ln(gamma(xx)) for xx > 0.
        From: Numerical Recipes in C.
    """
    cof = [57.1562356658629235, -59.5979603554754912, 14.1360979747417471, -0.491913816097620199, 0.33994649984811887e-4, 0.465236289270485756e-4, -0.983744753048795646e-4, 0.158088703224912494e-3, -0.210264441724104883e-3, 0.217439618115212643e-3, -0.164318106536763890e-3, 0.844182239838527433e-4, -0.26190838405184087e-4, 0.368991826595316234e-5]
    assert len(cof) == 14
    if xx <= 0:
        return float('inf')
    else:
        y = x = xx
        tmp = x + 5.2421875000000000
        tmp = (x + 0.5) * math.log(tmp) - tmp
        ser = 0.999999999999997092
        for j in xrange(14):
            y += 1
            ser += (cof[j] / y)
        return tmp + math.log(2.5066282746310005 * ser / x)

def psi(x):
    """
    #define el 0.5772156649015329

    double psi(double x)
    {
        double s,ps,xa,x2;
        int n,k;
        static double a[] = {
            -0.8333333333333e-01,
            0.83333333333333333e-02,
            -0.39682539682539683e-02,
            0.41666666666666667e-02,
            -0.75757575757575758e-02,
            0.21092796092796093e-01,
            -0.83333333333333333e-01,
            0.4432598039215686};

        xa = fabs(x);
        s = 0.0;
        if ((x == (int)x) && (x <= 0.0)) {
            ps = 1e308;
            return ps;
        }
        if (xa == (int)xa) {
            n = xa;
            for (k=1;k<n;k++) {
                s += 1.0/k;
            }
            ps =  s-el;
        }
        else if ((xa+0.5) == ((int)(xa+0.5))) {
            n = xa-0.5;
            for (k=1;k<=n;k++) {
                s += 1.0/(2.0*k-1.0);
            }
            ps = 2.0*s-el-1.386294361119891;
        }
        else {
            if (xa < 10.0) {
                n = 10-(int)xa;
                for (k=0;k<n;k++) {
                    s += 1.0/(xa+k);
                }
                xa += n;
            }
            x2 = 1.0/(xa*xa);
            ps = log(xa)-0.5/xa+x2*(((((((a[7]*x2+a[6])*x2+a[5])*x2+
                a[4])*x2+a[3])*x2+a[2])*x2+a[1])*x2+a[0]);
            ps -= s;
        }
        if (x < 0.0)
            ps = ps - M_PI*cos(M_PI*x)/sin(M_PI*x)-1.0/x;
            return ps;
    }
    """
    el = 0.57721566490153287
    a = [-0.8333333333333e-1, 0.83333333333333333e-2, -0.39682539682539683e-2, 0.41666666666666667e-2, -0.75757575757575758e-2, 0.21092796092796093e-1, -0.83333333333333333e-1, 0.4432598039215686]
    xa = abs(x)
    s = 0.0
    ps = 0.0
    if x % 1 == 0.0 and x <= 0.0:
        return 1.7976931348623157e+308;
    elif x == 1.0:
        return -el
    else:
        if xa % 1 == 0.0:
            n = xa
            for k in xrange(1,int(n+1)):
                if k < n:
                    s += 1.0/k;
                    ps = s - el;
        elif (xa + 0.5) % 1 == 0.0:
            n = xa - 0.5;
            for k in xrange(1,int(n+1)):
                if k <= n:
                    s += 1.0/(2.0*k-1.0);
            ps = 2.0 * s - el - 1.386294361119891;
        else:
            if xa < 10.0:
                n = 10 - int(xa);
                for k in xrange(0,int(n+1)):
                    if k < n:
                        s += 1.0 / (xa + k);
                xa += n;
                
            x2 = 1.0 / (xa * xa);
            ps = math.log(xa) - (0.5 / xa) + x2 * (((((((a[7]*x2+a[6])*x2+a[5])*x2+
                a[4])*x2+a[3])*x2+a[2])*x2+a[1])*x2+a[0]);
            ps -= s;

        M_PI = math.pi
        if x < 0.0:
            ps = ps - (M_PI * math.cos(M_PI * x) / math.sin(M_PI * x)) - (1.0 / x);
        return ps

