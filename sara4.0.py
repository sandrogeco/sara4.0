from scipy import ndimage
from obspy import read_inventory
from obspy.geodetics import  gps2dist_azimuth
import numpy as np
from matplotlib import pyplot as plt
from obspy.clients.filesystem.sds import Client
from obspy import Stream
from obspy import UTCDateTime
from obspy.signal.filter import envelope
import pickle
from skimage import measure



def dist_12(s1,s2,r):
    s1_coord=r.get_coordinates(s1)
    s2_coord = r.get_coordinates(s2)
    d=gps2dist_azimuth(s1_coord['latitude'],s1_coord['longitude'],s2_coord['latitude'],s2_coord['longitude'])
    return d


def dist_xyz(seed_id,lat,lon,z,r):
    x=seed_id.split('.')
    s1=r.select(network=x[0], station=x[1]).get_contents()['channels'][0]
    s1_coord=r.get_coordinates(s1)
    d=gps2dist_azimuth(s1_coord['latitude'],s1_coord['longitude'],lat,lon)
    dz=s1_coord['elevation']+s1_coord['local_depth']-z
    return np.sqrt(d[0]**2+dz**2)


def grid(latmin,latmax,lonmin,lonmax,depthmin,depthmax,step_lat,step_lon,step_z):
    lats=np.linspace(latmin,latmax,step_lat)
    lons = np.linspace(lonmin, lonmax, step_lon)
    depths=np.linspace(depthmin,depthmax,step_z)
    # gr=np.meshgrid(lats,lons,depths)
    return lats,lons,depths

def dists(lats,lons,depths,seed_id,r):
    d=np.zeros([len(lats),len(lons),len(depths)])

    for la in range(0,len(lats)):
        for lo in range(0,len(lons)):
            for ds in range(0,len(depths)):
                d[la,lo,ds]=dist_xyz(seed_id,lats[la],lons[lo],depths[ds],r)
    return d

def rGrid(seed_ids,lats,lons,depths,r):
    ratios={}
    sa=np.asarray(seed_ids)
    for id1 in range(0,len(sa)):
        for id2 in range(id1+1,len(sa)):
            d1=dists(lats,lons,depths,sa[id1],r)
            d2 = dists(lats, lons, depths,sa[id2],r)
            ratios[sa[id1],sa[id2]]=d2/d1
    return ratios

def get_extent(inv):
    lats=[inv.get_coordinates(x)['latitude'] for x in inv.get_contents()['channels']]
    lons=[inv.get_coordinates(x)['longitude'] for x in inv.get_contents()['channels']]
    depths = [inv.get_coordinates(x)['elevation']-inv.get_coordinates(x)['local_depth'] for x in inv.get_contents()['channels']]
    return np.min(lats),np.max(lats),np.min(lons),np.max(lons),np.min(depths),np.max(depths)

def plotR(s1,s2,inv,lats,lons,depths,z):
    inv1=inv.select(network=s1.split('.')[0],station=s1.split('.')[1])
    X, Y = np.meshgrid(lats, lons)
    d = np.where(depths > z)[0][0]
    # ind = np.unravel_index(np.argmin(r[s1, s2][:, :, d], axis=None), r[s1, s2][:, :, d].shape)
    # print(lats[ind[0]],lons[ind[1]])
    plt.contour(Y, X, np.transpose(r[s1, s2][:, :, d]),20)
    x=inv1.get_contents()['channels'][0]
    print(inv1.get_coordinates(x)['latitude'], inv1.get_coordinates(x)['longitude'])
    # print(np.diff(lats)[0],np.diff(lons)[0])
    plt.scatter(inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude'] )
    plt.annotate(s1,( inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude']))
    inv1 = inv.select(network=s2.split('.')[0], station=s2.split('.')[1])
    x = inv1.get_contents()['channels'][0]
    plt.scatter(inv1.get_coordinates(x)['longitude'], inv1.get_coordinates(x)['latitude'])
    plt.annotate(s2,(inv1.get_coordinates(x)['longitude'], inv1.get_coordinates(x)['latitude']))
    plt.show()

def plotErrs(s1,s2,ratios,inv,lats,lons,depths,z):
    inv1=inv.select(network=s1.split('.')[0],station=s1.split('.')[1])
    X, Y = np.meshgrid(lats, lons)
    d = np.where(depths > z)[0][0]
    # ind = np.unravel_index(np.argmin(r[s1, s2][:, :, d], axis=None), r[s1, s2][:, :, d].shape)
    # print(lats[ind[0]],lons[ind[1]])
    plt.contour(Y, X, np.transpose(np.abs(r[s1, s2][:, :, d]-ratios[s1,s2])),20)
    x=inv1.get_contents()['channels'][0]
    print(inv1.get_coordinates(x)['latitude'], inv1.get_coordinates(x)['longitude'])
    # print(np.diff(lats)[0],np.diff(lons)[0])
    plt.scatter(inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude'] )
    plt.annotate(s1,( inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude']))
    inv1 = inv.select(network=s2.split('.')[0], station=s2.split('.')[1])
    x = inv1.get_contents()['channels'][0]
    plt.scatter(inv1.get_coordinates(x)['longitude'], inv1.get_coordinates(x)['latitude'])
    plt.annotate(s2,(inv1.get_coordinates(x)['longitude'], inv1.get_coordinates(x)['latitude']))
    plt.show()

def plotErr(err,seed_ids,inv,lats,lons,depths):
    X, Y = np.meshgrid(lats, lons)
    ind = np.unravel_index(np.argmin(err, axis=None),err.shape)
    d = np.where(depths > ind[2])[0][0]
    plt.pcolor(Y, X, np.transpose(err[:,:,d]))#,100)
    plt.colorbar()
    for s in seed_ids:
        inv1 = inv.select(network=s.split('.')[0], station=s.split('.')[1])
        x = inv1.get_contents()['channels'][0]
        plt.scatter(inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude'] )
        plt.annotate(s,( inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude']))

    plt.show()

def minErr(err,seed_ids,inv,lats,lons,depths,z,pplt=True,thr=0.9):

    errc=err.copy()
    m=np.max(err)
    errc[err<thr*m]=0
    ind = np.unravel_index(np.argmax(err, axis=None), err.shape)
    xx=np.where(errc>0)
    dLat=[xx[0][0],xx[0][-1]]
    dLon=[xx[1][0],xx[1][-1]]
    dZ=[xx[2][0],xx[2][-1]]
    errx=gps2dist_azimuth(lats[dLat[0]],lons[ind[1]],lats[dLat[1]],lons[ind[1]])
    erry = gps2dist_azimuth(lats[ind[0]], lons[dLon[0]], lats[ind[0]], lons[dLon[1]])
    errz = np.abs(dZ)

    X, Y = np.meshgrid(lats, lons)
    d = np.where(depths > z)[0][0]
    if pplt:
        plt.pcolor(Y, X, np.transpose(errc[:,:,d]))#,100)
        plt.colorbar()
        for s in seed_ids:
            inv1 = inv.select(network=s.split('.')[0], station=s.split('.')[1])
            x = inv1.get_contents()['channels'][0]
            plt.scatter(inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude'] )
            plt.annotate(s,( inv1.get_coordinates(x)['longitude'],inv1.get_coordinates(x)['latitude']))

        plt.show()
    return m,lats[ind[0]],lons[ind[1]],depths[ind[2]],errx[0],erry[0],errz



def sgnElab(a,seed_ids,ts,te,fmin,fmax,c=Client('/mnt/ide/seed/')):
    s=Stream()
    try:
        for x in seed_ids:
            xs=x.split('.')
            s.append(c.get_waveforms(xs[0],xs[1],xs[2],xs[3],ts-1,te+1)[0])
        s.filter('bandpass',freqmin=fmin,freqmax=fmax)
        s.trim(ts,te)
        ls=[sx.stats['npts'] for sx in s]
        if np.min(ls)!=np.max(ls):
            raise 'different trace length'
        ampls=[envelope(sx.data)**2 for sx in s]
        ampl=np.sqrt(ampls[0]+ampls[1]+ampls[2])
        max_ampl=np.max(ampl)
        # argm_ampl=np.argmax(ampl)
    except Exception as e:
        print(e)
        max_ampl=np.nan
    a[xs[0]+'.'+xs[1]]=max_ampl

def ratioSgn(seed_ids,ampl):
    ratios={}
    sa=np.asarray(seed_ids)
    for id1 in range(0,len(sa)):
        for id2 in range(id1+1,len(sa)):
            ratios[sa[id1],sa[id2]]=ampl[sa[id1]]/ampl[sa[id2]]
    return ratios


def errMap(seed_ids,ratios,expRatios):
    errs={}
    relErrs={}
    sa=np.asarray(seed_ids)
    err=np.zeros(expRatios[sa[0],sa[1]].shape)
    relErr = np.zeros(expRatios[sa[0], sa[1]].shape)
    i=0

    for id1 in range(0,len(sa)):
        for id2 in range(id1+1,len(sa)):
            if not np.isnan(np.sum(ratios[sa[id1],sa[id2]])):
                #TODO 2 valutare se escludere coppie con alti errori (basso relErrs)
                relErrs[sa[id1], sa[id2]]=(expRatios[sa[id1], sa[id2]]-ratios[sa[id1],sa[id2]])**2/\
                                       (expRatios[sa[id1], sa[id2]]+ratios[sa[id1],sa[id2]])**2
                relErr+=relErrs[sa[id1], sa[id2]]
                relErrs[sa[id1], sa[id2]]=1-relErrs[sa[id1], sa[id2]]
                errs[sa[id1], sa[id2]] = (expRatios[sa[id1], sa[id2]] - ratios[sa[id1], sa[id2]]) ** 2
                err += errs[sa[id1], sa[id2]]
                i+=1
    relErr=1-(relErr/i)
    err=np.sqrt(err/i)
    return err,relErr,relErrs

st=['LK.BRK0','LK.BRK1','LK.BRK2','LK.BRK3','LK.BRK4']

inv=read_inventory('/mnt/ide/resp.xml')

minLat,maxLat,minLon,maxLon,minDepth,maxDepth=get_extent(inv)
minLat=-9.62
maxLat=-9.65
minLon=-35.76
maxLon=-35.73

#TODO 1 salvare la griglia sotto assieme a r calcolato sotto da rGrid

lats,lons,depths=grid(minLat,maxLat,minLon,maxLon,minDepth,maxDepth,30,30,10)

try:
    with open('r.pickle', 'rb') as handle:
        r = pickle.load(handle)
    # r=np.load('tmp.npz',allow_pickle=True)['r']
except:
    r = rGrid(st, lats, lons, depths, inv)
    with open('r.pickle', 'wb') as handle:
        pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)

# r = rGrid(st, lats, lons, depths, inv)

ampl={}
t0=UTCDateTime('2022-09-11T17:12:34')
rr=[]
for tt in range(0,60,2):
    t=t0+tt
    for x in st:
        sgnElab(ampl,[x+'..EHZ',x+'..EHN',x+'..EHE'],t,t+5,2,12)

    rSgn=ratioSgn(st,ampl)

    err,relerr,relerrs=errMap(st,rSgn,r)
    u=minErr(relerr,st,inv,lats,lons,depths,-1,False)
    rr.append(u)

    print(gps2dist_azimuth(inv.get_coordinates('LK.BRK0..EHZ')['latitude'],
                           inv.get_coordinates('LK.BRK0..EHZ')['longitude'],u[1],u[2] ))
    print(rSgn)
    print('  ')





print('pippo')