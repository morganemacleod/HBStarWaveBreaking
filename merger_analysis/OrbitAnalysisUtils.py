# normal stuff
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table
#from merger_analysis import athena_read as ar
try:
    from . import athena_read as ar
except:
    import athena_read as ar
#import athena_read as ar
from glob import glob
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid


def read_trackfile(fn,triple=False,m1=0,m2=0):
    orb=ascii.read(fn)
    if triple==False:
        print ("reading orbit file for binary simulation...")
        if m1==0:
            m1 = orb['m1']
        if m2==0:
            m2 = orb['m2']
        orb['lgoz'] = np.cumsum( np.gradient(orb['time']) * orb['ldoz'] )
        orb['ltz'] = orb['lpz'] + orb['lgz'] + orb['lgoz']
        
        orb['sep'] = np.sqrt(orb['x']**2 + orb['y']**2 + orb['z']**2)
        
        orb['r'] = np.array([orb['x'],orb['y'],orb['z']]).T
        orb['rhat'] = np.array([orb['x']/orb['sep'],orb['y']/orb['sep'],orb['z']/orb['sep']]).T
        
        orb['v'] = np.array([orb['vx'],orb['vy'],orb['vz']]).T
        orb['vmag'] = np.linalg.norm(orb['v'],axis=1)
        orb['vhat'] = np.array([orb['vx']/orb['vmag'],orb['vy']/orb['vmag'],orb['vz']/orb['vmag']]).T
        
        orb['agas1'] = np.array([orb['agas1x'],orb['agas1y'],orb['agas1z']]).T
        orb['agas2'] = np.array([orb['agas2x'],orb['agas2y'],orb['agas2z']]).T
        
        orb['rcom'] = np.array([orb['xcom'],orb['ycom'],orb['zcom']]).T
        orb['vcom'] = np.array([orb['vxcom'],orb['vycom'],orb['vzcom']]).T
    
        F12 = - m1*m2/orb['sep']**2
        # aceel of 1 by 2
        orb['a21'] = np.array([-F12/m1*orb['x']/orb['sep'],
                               -F12/m1*orb['y']/orb['sep'],
                               -F12/m1*orb['z']/orb['sep']]).T
        # accel of 2 by 1
        orb['a12'] = np.array([F12/m2*orb['x']/orb['sep'],
                               F12/m2*orb['y']/orb['sep'],
                               F12/m2*orb['z']/orb['sep']]).T
    

    else:
        print ("reading orbit file for triple simulation... (note:ignoring m1,m2)")
        orb['rcom'] = np.array([orb['xcom'],orb['ycom'],orb['zcom']]).T
        orb['vcom'] = np.array([orb['vxcom'],orb['vycom'],orb['vzcom']]).T

    # clean to remove restarts
    #orb_clean_sel = orb['time'][1:] > orb['time'][:-1] 
    #orb_clean = orb[1:][orb_clean_sel].copy()
    orb_clean = orb.copy()   
    
    return orb_clean


def get_orb_hst(base_dir,filestart="HSE"):

    orb = read_trackfile(base_dir+"pm_trackfile.dat")

    print ("ORB: ... ", orb.colnames)

    hst = ascii.read(base_dir+filestart+".hst",
                     names=['time','dt','mass','1-mom','2-mom','3-mom','1-KE','2-KE','3-KE','tot-E','mxOmegaEnv','mEnv','mr1','mr12','scalar'])
    print ("\nHSE: ...", hst.colnames)

    mg = hst['mr12'][0]

    orb['M1'] = np.interp(orb['time'],hst['time'],hst['mr1']) + orb['m1']
    orb['M2'] = orb['m2']

    orb['x1'] = -orb['xcom']
    orb['y1'] = -orb['ycom']
    orb['z1'] = -orb['zcom']
    orb['x2'] = orb['x']-orb['xcom']
    orb['y2'] = orb['y']-orb['ycom']
    orb['z2'] = orb['z']-orb['zcom']

    orb['v1x'] = -orb['vxcom']
    orb['v1y'] = -orb['vycom']
    orb['v1z'] = -orb['vzcom']
    orb['v2x'] = orb['vx']-orb['vxcom']
    orb['v2y'] = orb['vy']-orb['vycom']
    orb['v2z'] = orb['vz']-orb['vzcom']

    orb['PE'] = -orb['M1']*orb['M2']/orb['sep']
    orb['KE1'] = 0.5*orb['M1']*(orb['v1x']**2 + orb['v1y']**2 + orb['v1z']**2)
    orb['KE2'] = 0.5*orb['M2']*(orb['v2x']**2 + orb['v2y']**2 + orb['v2z']**2)
    orb['E'] = orb['KE1'] + orb['KE2'] + orb['PE']
    orb['a'] = -orb['M1']*orb['M2']/(2*orb['E'])

    orb['Lz'] = orb['M1']*(orb['x1']*orb['v1y']-orb['y1']*orb['v1x']) + orb['M2']*(orb['x2']*orb['v2y']-orb['y2']*orb['v2x'])
    orb['e'] = np.sqrt( 1.0 - orb['Lz']**2 *(orb['M1']+orb['M2'])/(orb['a']*(orb['M1']*orb['M2'])**2) )
    
    return orb,hst



def get_midplane_theta(myfile,level=0):
    dblank=ar.athdf(myfile,level=level,quantities=[],subsample=True)

    # get closest to midplane value
    return dblank['x2v'][ np.argmin(np.abs(dblank['x2v']-np.pi/2.) ) ]


def get_t1(orb,skip=1):
    sel = orb['sep']<1.5
    return np.interp(1.0,np.flipud(orb[sel]['sep'][::skip]),np.flipud(orb[sel]['time'][::skip]) )




def get_Omega_env_dist(fn,dv=0.05,G=1,rho_thresh=1.e-2,level=2):
    """ Get the mass-average Omega within r<1 """
    d = ar.athdf(fn,level=level)

    # MAKE grid based coordinates
    d['gx1v'] = np.zeros_like(d['rho'])
    for i in range((d['rho'].shape)[2]):
        d['gx1v'][:,:,i] = d['x1v'][i]
    
    d['gx2v'] = np.zeros_like(d['rho'])
    for j in range((d['rho'].shape)[1]):
        d['gx2v'][:,j,:] = d['x2v'][j]

    d['gx3v'] = np.zeros_like(d['rho'])
    for k in range((d['rho'].shape)[0]):
        d['gx3v'][k,:,:] = d['x3v'][k]
    
    
    ## dr, dth, dph
    d['d1'] = d['x1f'][1:] - d['x1f'][:-1]
    d['d2'] = d['x2f'][1:] - d['x2f'][:-1]
    d['d3'] = d['x3f'][1:] - d['x3f'][:-1]

    # grid based versions
    d['gd1'] = np.zeros_like(d['rho'])
    for i in range((d['rho'].shape)[2]):
        d['gd1'][:,:,i] = d['d1'][i]
    
    d['gd2'] = np.zeros_like(d['rho'])
    for j in range((d['rho'].shape)[1]):
        d['gd2'][:,j,:] = d['d2'][j]

    d['gd3'] = np.zeros_like(d['rho'])
    for k in range((d['rho'].shape)[0]):
        d['gd3'][k,:,:] = d['d3'][k]
    
    
    # VOLUME 
    d['dvol'] = d['gx1v']**2 * np.sin(d['gx2v']) * d['gd1']* d['gd2']* d['gd3']
    # cell masses
    d['dm'] = d['rho']*d['dvol']


    select = (d['rho']>rho_thresh) #(d['gx1v']<1.0)
    # Omega = vphi/(r sin th)
    vpf =  (d['vel3'][select] / (d['gx1v'][select]* np.sin(d['gx2v'][select])) ).flatten()
    #vpf =  d['vel3'][select].flatten()
    dmf = d['dm'][select].flatten()
    
    mybins = np.arange(-0.1,0.1,dv)
    mydist = np.histogram(vpf,weights=dmf,bins=mybins)
    GMtot=G*np.sum(mydist[0])
    
    print ("Total mass GM = ",G*np.sum(dmf))
    print ("Total mass GM (distribution) = ", GMtot)
    
    return mydist, np.average(vpf,weights=dmf)


def read_data(fn,orb,
              m1=0,m2=0,
              G=1,rsoft2=0.1,level=0,
              get_cartesian=True,get_cartesian_vel=True,get_torque=False,get_energy=False,
             x1_min=None,x1_max=None,
             x2_min=None,x2_max=None,
             x3_min=None,x3_max=None,
             profile_file="hse_profile.dat",
             gamma=5./3.,
             triple=False):
    """ Read spherical data and reconstruct cartesian mesh for analysis/plotting """
    
    print ("read_data...reading file",fn)
    
    
    d = ar.athdf(fn,level=level,subsample=True,
                 x1_min=x1_min,x1_max=x1_max,
                 x2_min=x2_min,x2_max=x2_max,
                 x3_min=x3_min,x3_max=x3_max,
                 return_levels=True) # approximate arrays by subsampling if level < max
    print (" ...file read, constructing arrays")
    print (" ...gamma=",gamma)
    
    # current time
    t = d['Time']
    # get properties of orbit
    rcom,vcom = rcom_vcom(orb,t)

    if m1==0:
        m1 = np.interp(t,orb['time'],orb['m1'])
    if m2==0:
        m2 = np.interp(t,orb['time'],orb['m2'])

    data_shape = (len(d['x3v']),len(d['x2v']),len(d['x1v']))
   
       
    # MAKE grid based coordinates
    d['gx1v']=np.broadcast_to(d['x1v'],(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    d['gx2v']=np.swapaxes(np.broadcast_to(d['x2v'],(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    d['gx3v']=np.swapaxes(np.broadcast_to(d['x3v'],(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    
    ####
    # GET THE VOLUME 
    ####
    
    ## dr, dth, dph
    d1 = d['x1f'][1:] - d['x1f'][:-1]
    d2 = d['x2f'][1:] - d['x2f'][:-1]
    d3 = d['x3f'][1:] - d['x3f'][:-1]
    
    # grid based versions
    gd1=np.broadcast_to(d1,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    gd2=np.swapaxes(np.broadcast_to(d2,(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    gd3=np.swapaxes(np.broadcast_to(d3,(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    # AREA / VOLUME 
    sin_th = np.sin(d['gx2v'])
    d['dA'] = d['gx1v']**2 * sin_th * gd2*gd3
    d['dvol'] = d['dA'] * gd1
    
    # free up d1,d2,d3
    del d1,d2,d3
    del gd1,gd2,gd3
    
    
    ### 
    # CARTESIAN VALUES
    ###
    if(get_cartesian or get_torque or get_energy):
        print ("...getting cartesian arrays...")
        # angles
        cos_th = np.cos(d['gx2v'])
        sin_ph = np.sin(d['gx3v'])
        cos_ph = np.cos(d['gx3v']) 
        
        # cartesian coordinates
        d['x'] = d['gx1v'] * sin_th * cos_ph 
        d['y'] = d['gx1v'] * sin_th * sin_ph 
        d['z'] = d['gx1v'] * cos_th

        if(get_cartesian_vel or get_torque or get_energy):
            # cartesian velocities
            d['vx'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
            d['vy'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
            d['vz'] = cos_th*d['vel1'] - sin_th*d['vel2']  
            
        del cos_th, sin_th, cos_ph, sin_ph
    
    
    if(get_torque & (triple==False)):
        print ("...getting torque arrays...")
        x2,y2,z2 = pos_secondary(orb,t)
        
        # define grav forces
        dist2 = np.sqrt( (d['x']-x2)**2 + (d['y']-y2)**2 + (d['z']-z2)**2 )
        dist1c = d['gx1v']**3
    
        soft_grav = fspline(dist2,rsoft2)
        
        fdens2x = G*m2*d['rho']*soft_grav * (d['x']-x2)
        fdens2y = G*m2*d['rho']*soft_grav * (d['y']-y2)
        #fdens2z = G*m2*d['rho']*soft_grav * (d['z']-z2)

        fdens1x = G*m1*d['rho']/dist1c * d['x']
        fdens1y = G*m1*d['rho']/dist1c * d['y']
        #fdens1z = G*m1*d['rho']/dist1c * d['z']

        del dist1c,dist2

        d['torque_dens_2_z'] = (x2-rcom[0])*fdens2y - (y2-rcom[1])*fdens2x
        d['torque_dens_1_z'] = (-rcom[0])*fdens1y - (-rcom[1])*fdens1x
        
        del fdens2x,fdens2y #,fdens2z
        del fdens1x,fdens1y #,fdens1z

    if(get_energy & (triple==False)):
        print ("...getting energy arrays...")
        x2,y2,z2 = pos_secondary(orb,t)
        
        #hse_prof = ascii.read(profile_file,
        #                      names=['r','rho','p','m'])
        #M1r = np.interp(d['gx1v'],hse_prof['r'],hse_prof['m'])
        
        #energy (monopole self grav)
        dist2 = np.sqrt( (d['x']-x2)**2 + (d['y']-y2)**2 + (d['z']-z2)**2 )

        #NGRAV = 100
        #rmin = 0.3
        #rmax = 100.0
        #logr = np.linspace(np.log10(rmin),np.log10(rmax),NGRAV)
        
        #menc_logr = np.zeros_like(logr)
        #for i,lr in enumerate(logr):
        #    sel = np.log10(d['gx1v'])<lr
        #    menc_logr[i] = np.sum( (d['rho']*d['dvol'])[sel] )

        #mencr_gas = np.interp(np.log10(d['x1v']),logr,menc_logr)
        #menc_gas = np.broadcast_to(mencr_gas,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) ) 
        
        #mencr_x1fp_gas = np.cumsum( np.sum(d['rho']*d['dvol'],axis=(0,1)) )
        #mencr_gas = np.interp(d['x1v'],d['x1f'][1:],mencr_x1fp_gas )
        #menc_gas = np.broadcast_to(mencr_x1fp_gas,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )  # CHANGE

        #rm = d['x1f'][0:-1]
        #rp = d['x1f'][1:]
        #coord_area2_i = 0.5*(rp**2 - rm**2)
        #coord_vol_i = (rp**3 - rm**3)/3.
        #coord_src1 = np.broadcast_to(coord_area2_i / coord_vol_i,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
        
        #d['epotg'] = -G*menc_gas*d['rho']*coord_src1
        #d['epot1'] = -G*m1*d['rho']*coord_src1
        d['epotp2'] = - G*m2*d['rho']*pspline(dist2,rsoft2)
        d['ek'] = 0.5*d['rho']*((d['vx']-vcom[0])**2 +
                           (d['vy']-vcom[1])**2 + 
                           (d['vz']-vcom[2])**2)
        d['ei'] = d['press']/(gamma-1)
        d['etot'] = -d['r7'] +d['epotp2']+ d['ei'] + d['ek']
        #d['h'] = gamma*d['press']/((gamma-1)*d['rho'])
        d['bern'] = (d['etot']+d['press'])/d['rho']
        d['ek_star'] = 0.5*d['rho']*(d['vel1']**2 + d['vel2']**2 + d['vel3']**2)

        d['etot_star'] = -d['r7'] + (d['ei'] + d['ek_star'])/d['rho']
        d['etot_star_0'] = (d['r4'] + d['r5'] - d['r6'])
        d['dE'] = d['etot_star'] - d['etot_star_0']
        
        #del hse_prof,dist2,M1r
    
    return d
    
# get time from filename
def time_fn(fn,dt=1):
    """ returns a float time from parsing the fn string (assumes dt=1 in fn convention)"""
    return dt*float(fn[-11:-6])


def rcom_vcom(orb,t):
    """pass a pm_trackfile.dat that has been read, time t"""
    rcom =  np.array([np.interp(t,orb['time'],orb['rcom'][:,0]),
                  np.interp(t,orb['time'],orb['rcom'][:,1]),
                  np.interp(t,orb['time'],orb['rcom'][:,2])])
    vcom =  np.array([np.interp(t,orb['time'],orb['vcom'][:,0]),
                  np.interp(t,orb['time'],orb['vcom'][:,1]),
                  np.interp(t,orb['time'],orb['vcom'][:,2])])
    
    return rcom,vcom

def pos_secondary(orb,t):
    x2 = np.interp(t,orb['time'],orb['x'])
    y2 = np.interp(t,orb['time'],orb['y'])
    z2 = np.interp(t,orb['time'],orb['z'])
    return x2,y2,z2

    
# individual grav force 
def fspline(r,eps):
    """Hernquist & Katz 1989, Eq A2 """
    u=r/eps
    condlist = [u<1,u<2,u>=2]
    resultlist = [1/eps**3 * (4./3. - 1.2*u**2 + 0.5*u**3),
                  1/r**3 * (-1./15. + 8./3.*u**3 - 3.*u**4 + 1.2*u**5 - 1./6.*u**6),
                  1/r**3]
    return np.select(condlist,resultlist)


def pspline(r,eps):
    """Hernquist & Katz 1989, Eq A1 """
    u=r/eps
    condlist = [u<1,u<2,u>=2]
    resultlist = [-2./eps * (1./3.*u**2 -0.15*u**4 + 0.05*u**5) + 7./(5.*eps),
                  -1./(15.*r) - 1/eps*( 4./3.*u**2 - u**3 + 0.3*u**4 -1./30.*u**5) + 8./(5.*eps),
                  1./r]
    return np.select(condlist,resultlist)





def get_plot_array_midplane(arr):
    return np.append(arr,[arr[0]],axis=0)



from scipy.signal import argrelextrema

class makebinary:
    """assumes orientation along x-axis and G=1"""
    def __init__(self,m1,m2,a):
        self.q = m1/m2
        self.m1 = m1
        self.m2 = m2
        self.M = m1+m2
        self.a = a
        self.x1 = -self.m2/self.M*self.a
        self.x2 = self.m1/self.M*self.a
        self.r1 = np.array([self.x1,0.0,0.0])
        self.r2 = np.array([self.x2,0.0,0.0])
        self.omega = np.sqrt(self.M*self.a**-3)
        self.omega_vec = self.omega * np.array([0,0,1.0])
    
    def phi_roche(self,x,y,z):
        r = np.array([x,y,z])
        phi1 = - self.m1 / np.linalg.norm(r-self.r1)
        phi2 = - self.m2 / np.linalg.norm(r-self.r2)
        cor = - 0.5 * np.linalg.norm( np.cross(self.omega_vec,r)  )**2
        return phi1+phi2+cor
    
    def get_xL_phiL(self,points=1001):
        phi_temp = self.get_phi_function()
        x = np.linspace(-3*self.a,3*self.a,points)
        Lind = argrelextrema(phi_temp(x,0,0),np.greater)[0]
        xL   = x[Lind]
        phiL = phi_temp(x,0,0)[Lind]
        return xL,phiL
    
    def get_phi_function(self):
        return np.vectorize(self.phi_roche)
    
def get_roche_function(orb,time,M1=1,M2=0.3):
    a = np.interp(time,orb['time'],orb['sep'])
    b = makebinary(M1,M2,a)
    xL,phiL = b.get_xL_phiL(points=100)
    return xL, phiL, b.get_phi_function()



def get_plot_array_vertical(quantity,phislice,
                            myfile,profile_file,orb,m1,m2,
                           G=1,rsoft2=0.1,level=0,
                            x1_max=None):
    
    dblank=ar.athdf(myfile,level=level,quantities=[],subsample=True)
    
    
    
    get_cartesian=True
    get_torque=False
    get_energy=False
    if quantity in ['torque_dens_1_z','torque_dens_2_z']:
        get_torque=True
    if quantity in ['ek','ei','etot','epot','epotg','epotp','h','bern']:
        get_energy=True

    x3slicevalue = dblank['x3v'][np.argmin(np.abs(dblank['x3v']+phislice))]
    d=read_data(myfile,orb,m1,m2,G=G,rsoft2=rsoft2,level=level,
                get_cartesian=get_cartesian,
                get_torque=get_torque,
                get_energy=get_energy,
                x1_max=x1_max,x3_min=x3slicevalue,x3_max=x3slicevalue,
                profile_file=profile_file)
    
    x1 = d['gx1v'][0,:,:]*np.sin(d['gx2v'][0,:,:])
    z1 = d['z'][0,:,:]
    val1 = d[quantity][0,:,:]
    
    if(x3slicevalue<0):
        x3slicevalue += np.pi
    else:
        x3slicevalue -= np.pi
    
    d=read_data(myfile,orb,m1,m2,G=G,rsoft2=rsoft2,level=level,
                get_cartesian=get_cartesian,
                get_torque=get_torque,
                get_energy=get_energy,
                x1_max=x1_max,x3_min=x3slicevalue,x3_max=x3slicevalue,
                profile_file=profile_file)
    
    x2 = -d['gx1v'][0,:,:]*np.sin(d['gx2v'][0,:,:])
    z2 = d['z'][0,:,:]
    val2 = d[quantity][0,:,:]
    
    # Combine the arrays
    x = np.concatenate((x2,np.flipud(x1)))
    z = np.concatenate((z2,np.flipud(z1)))
    val = np.concatenate((val2,np.flipud(val1)))

    x = np.concatenate((x,x2[0:1]) ,axis=0)
    z = np.concatenate((z,z2[0:1]) ,axis=0)
    val = np.concatenate((val,val2[0:1]) ,axis=0)
    
    return x,z,val

