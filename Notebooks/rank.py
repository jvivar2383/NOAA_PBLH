import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
# Some code from utilhysplit module.


class TestTalagrand():

    def __init__(self):
        self.enumber=10 #number of ensemble members
        self.mean=2
        self.num=100

    def make_normal_obs(self,xcenter,ycenter,varname='obs'):
        nm = self.num
        mean = self.mean
        obsval = np.random.normal(mean, 1, nm)
        xval = np.arange(xcenter-self.num, xcenter+self.num, 1)
        yval = np.arange(ycenter-self.num, ycenter+self.num, 1)
        #print(xval)
        #print(yval)
        df = pd.DataFrame(zip(obsval,xval,yval),columns=['val','x','y'])
        df = df.set_index(['x','y'])
        dset = df.to_xarray() 
        dset = xr.where(dset<0,self.mean,dset)
        #dset = dset.fillna(0)
        dset = dset.val
        dset = dset.assign_coords(ens=varname)
        dset = dset.expand_dims('ens')
        #dset = dset.val
        #dset['ens'] = varname
        return dset
       
    def non_overlapping_ensembleA(self):
        # non-overlapping with more observations
        self.num=15
        xlist = []
        obs = self.make_normal_obs(1,15,'obs') 
        xlist.append(obs)
        enames = ['e{}'.format(x) for x in np.arange(1,self.enumber)]
        #print(enames)
        ycenterlist = np.arange(21,21+5*self.enumber,5)
        xcenterlist = np.arange(1,1+self.enumber)
        self.num=10
        for ens in zip(enames,xcenterlist,ycenterlist):
            xlist.append(self.make_normal_obs(ens[1],ens[2],ens[0]))
        dset = xr.concat(xlist,dim='ens')
        return dset

    def overlapping_ensembleA(self):
        # perfectly overlapping ensembles.
        # need a large number to get the sampling right.
        self.num=100
        xlist = []
        obs = self.make_normal_obs(1,15,'obs') 
        xlist.append(obs)
        enames = ['e{}'.format(x) for x in np.arange(1,self.enumber)]
        for ens in zip(enames):
            xlist.append(self.make_normal_obs(1,15,ens[0]))
        dset = xr.concat(xlist,dim='ens')
        return dset

    def overlapping_ensembleB(self):
        # half are overlapping and half are not.
        self.num=100
        onum = self.num
        xlist = []
        ox = 1
        oy = 15
        obs = self.make_normal_obs(ox,oy,'obs') 
        xlist.append(obs)
        enames = ['e{}'.format(x) for x in np.arange(1,self.enumber)]
        #print(enames)
        ycenterlist = np.arange(21,21+5*self.enumber,5)
        xcenterlist = np.arange(1,1+self.enumber)
        nlist = np.ones([self.enumber]) * self.num
        iii=0
        for ens in zip(enames,xcenterlist,ycenterlist,nlist):
            if iii < self.enumber/2:
                xlist.append(self.make_normal_obs(ens[1],ens[2],ens[0]))
            else:
                xlist.append(self.make_normal_obs(ox,oy,ens[0]))
            iii+=1
        dset = xr.concat(xlist,dim='ens')
        return dset

    def non_overlapping_ensembleC(self):
        # non-overlapping with more observations in half the ensemble members
        self.num=16
        onum = self.num
        xlist = []
        obs = self.make_normal_obs(1,15,'obs') 
        xlist.append(obs)
        enames = ['e{}'.format(x) for x in np.arange(1,self.enumber)]
        #print(enames)
        ycenterlist = np.arange(21,21+5*self.enumber,5)
        xcenterlist = np.arange(1,1+self.enumber)
        nlist = np.ones([self.enumber]) * self.num
        iii=0
        for ens in zip(enames,xcenterlist,ycenterlist,nlist):
            if iii < self.enumber/2:
               self.num = int(ens[3]) + 5
            else:
               self.num=int(ens[3]) - 5
            print(self.num)
            xlist.append(self.make_normal_obs(ens[1],ens[2],ens[0]))
            iii+=1
        dset = xr.concat(xlist,dim='ens')
        return dset


    def non_overlapping_ensembleB(self):
        # non-overlapping with more observations in one ensemble member
        self.num=15
        xlist = []
        obs = self.make_normal_obs(1,15,'obs') 
        xlist.append(obs)
        enames = ['e{}'.format(x) for x in np.arange(1,self.enumber)]
        #print(enames)
        ycenterlist = np.arange(21,21+5*self.enumber,5)
        xcenterlist = np.arange(1,1+self.enumber)
        nlist = np.ones([self.enumber]) * self.num
        nlist[0] = nlist[0]+20
        for ens in zip(enames,xcenterlist,ycenterlist,nlist):
            self.num=int(ens[3])
            xlist.append(self.make_normal_obs(ens[1],ens[2],ens[0]))
        dset = xr.concat(xlist,dim='ens')
        return dset


    def non_overlapping_ensemble(self):
        # non-overlapping and equal in size
        self.num=10
        xlist = []
        obs = self.make_normal_obs(1,15,'obs') 
        xlist.append(obs)
        enames = ['e{}'.format(x) for x in np.arange(1,self.enumber)]
        #print(enames)
        ycenterlist = np.arange(21,21+5*self.enumber,5)
        xcenterlist = np.arange(1,1+self.enumber)
        for ens in zip(enames,xcenterlist,ycenterlist):
            xlist.append(self.make_normal_obs(ens[1],ens[2],ens[0]))
        dset = xr.concat(xlist,dim='ens')
        return dset


    def testgeneric(self, func):
        dset = func
        #dset.max(dim='ens').plot.pcolormesh()
        tal = Talagrand(thresh=0.001,nbins=self.enumber+1)
        df = tal.add_data_xraB(dset)
        return tal

    def otestA(self):
        return self.testgeneric(self.overlapping_ensembleA())

    def otestB(self):
        return self.testgeneric(self.overlapping_ensembleB())

    def test1(self):
        return self.testgeneric(self.non_overlapping_ensemble())
        
    def test2(self):
        return self.testgeneric(self.non_overlapping_ensembleA())

    def test3(self):
        return self.testgeneric(self.non_overlapping_ensembleB())

    def test4(self):
        return self.testgeneric(self.non_overlapping_ensembleC())

def make_talagrand(dflist, thresh, bnum,
                   resample_time=None,
                   resample_type='max',
                   rname='talagrand',
                   background=0,
                   verbose=False):
    tal = Talagrand(thresh, bnum, background=background, verbose=verbose)
    for df in dflist:
        if resample_time:
            df = process_data(df, resample_time, resample_type)
        tal.add_data(df)
    # tal.plotrank(nbins=bnum)
    return tal

class Talagrand:
    # get list of forecasts and order them.
    # find where in the list the observation would go.
    # p372 Wilks
    # if obs smaller than all forecasts then rank is 1
    # if obs is larger than all forecasts then rank is n_ens+1
    # calculate rank for each observation
    # create histogram.

    # simple way when there are not duplicat values of forecasts
    # is to just sort the list and find index of observation.

    # However when there are a lot of duplicate values then must
    # create the bins first and fill them as you go along.

    # do we only look at observations above threshold?
    # What if multiple forecasts are the same? What rank is obs given?
    # This would occur in case of 0's usually. What if 10 forecasts are 0 and
    # observation is 0 and 5 forecasts are above 0?
    # ANSWER: create the bins first - one for each forecast.
    #

    def __init__(self, thresh, nbins, background=0, verbose=False):
        self.verbose = verbose
        self.thresh = float(thresh)
        self.background = background
        self.ranklist = []
        self.obsvals = []
        # create bins for histogram.
        self.binra = np.zeros(nbins)
        # when obs are 0, count how many forecasts are 0.
        self.zeronum = []
        # when rank is 27, count how many forecasts
        # are non-zero.
        # number which are 0 shows how many are completely missed.
        self.nonzero = []
        self.nonzerorow = []
        # value of obs which is higher than all forecasts
        self.obsmax = []
        # consider values with difference less than this the same.
        self.tolerance = 1e-2

    def check1(self, fname=None, fs=10):
        # creates scatter plot of observations whic
        # were higher than all forecasts and the
        # largest forecast
        import matplotlib.colors
        sns.set_style('whitegrid')
        obs = []
        maxf = []
        for vals in self.nonzerorow:
            obs.append(vals[-1])
            maxf.append(np.max(vals[0:-1]))
        yline = np.arange(0, int(np.max(maxf)), 1)
        xline = yline
        norm = matplotlib.colors.LogNorm(vmin=0.1, vmax=None, clip=False)
        cb = plt.hist2d(obs, maxf, bins=(50, 50), density=False, cmap=plt.cm.BuPu, norm=norm)
        plt.plot(xline, yline, '-k')
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel('Measured Value (ppb)', fontsize=fs)
        ax.set_ylabel('Highest forecast(ppb)', fontsize=fs)
        if fname:
            plt.savefig(fname + '.png')
        # plt.plot(obs,maxf,'k.',markersize=5)
        # plt.show()
        return ax

    def plotrank(self, fname=None, fs=10, ax=None):
        sns.set_style('whitegrid')
        if not ax:
           fig = plt.figure(1)
           ax = fig.add_subplot(1,1,1)
        nbins = self.binra.shape[0] + 1
        xval = np.arange(1, nbins)
        ax.bar(xval, self.binra, alpha=0.5)
        ax.set_xlabel('Rank', fontsize=fs)
        ax.set_ylabel('Counts', fontsize=fs)
        #plt.tight_layout()
        if fname:
            plt.savefig('{}.png'.format(fname))

    def sort_wind_direction(self, row):
        # NOT WORKING YET.
        # check for when distance between first and last
        # value is greater than 360-R-1 + R0
        # 0--------------------------------------------360
        #     R0                             R-1
        #
        if (row[-1] - row[0]) > (360-row[-1] + row[0]):
            return -1

    def add_data_xraB(self, dset, dims='ens'):
        dset = dset.fillna(0)
        df = dset.to_dataframe()
        df = df.reset_index()
        # create unique index for each measurement point.
        df['iii'] = df.apply(lambda row: '{}_{}'.format(row['x'], row['y']), axis=1)
        df = df.pivot(columns=dims, values='val', index='iii')
        self.add_data(df)
        return df

    def add_data_xra(self, obs, forecast, dims='ens'):
        """
        obs : xra
        forecast : xra
        dims: dimension for ensemble 'ens' or 'source'
        must be on same grid with dimensions of x,y,ens
        """

        obs = obs.expand_dims({dims: ['obs']})
        dra = xr.concat([obs, forecast], dim=dims)
        dra = dra.fillna(0)

        dra = dra.drop('latitude')
        dra = dra.drop('longitude')

        df = dra.to_dataframe()
        df = df.reset_index()
        # create unique index for each measurement point.
        df['iii'] = df.apply(lambda row: '{}_{}'.format(row['x'], row['y']), axis=1)

        # creates a dataframe that can be input into 'add_data' function.
        # each row represents a point with the ensemble data and observation
        # in a column labeled 'obs'
        df = df.pivot(columns=dims, values='ash_mass_loading', index='iii')
        self.add_data(df)
        return df

    def add_data(self, df, wind_direction=False, verbose=False):
        obs_col = [x for x in df.columns if 'obs' in x]
        print(obs_col)
        obs_col = obs_col[0]
        # this keeps rows which have at least one value above threshold.
        df2 = df[(df > self.thresh).any(1)]

        for iii in np.arange(0, len(df2)):
            # selects row
            row = df2.iloc[iii]
            rowcheck = row.copy()
            # if below background then set to 0.
            # otherwise was running into problems with
            # obs never having lowest rank.
            if row[obs_col] < self.background:
                row[obs_col] = 0
            for col in row.index.values:
                if row[col] < self.background:
                    row[col] = 0
            # print('ROW',row)
            # sorts values
            rowcheck = rowcheck.sort_values()
            row = row.sort_values()
            if verbose: print('ROW', row)
            # creates dataframe with columns of index, name, value
            temp = pd.DataFrame(row).reset_index().reset_index()
            temp.columns = ['iii', 'name', 'val']
            # print('TEMP',temp)
            # gets index of the observation
            # add one since index starts at 0.

            # do not add +1 because now using it as index of self.binra.
            # which starts at 0.
            rank = int(float(temp[temp.name == obs_col].iii))
            obsval = float(temp[temp.name == obs_col].val)
            if verbose: print('Rank', rank, obsval)

            temp['zval'] = np.abs(obsval - temp['val'])
            # should be one zero value where obs matches itself.
            # if multiple forecasts are within the tolerance then
            # point is split among them evenly.
            if len(temp[temp.zval <= self.tolerance]) > 1:
                val2add = 1.0 / len(temp[temp.zval <= self.tolerance])
                # indices of where forecasts match obs.
                rankvals = np.where(temp.zval <= self.tolerance)
                # add value to binra.
                #print(rankvals[0], type(rankvals))
                for rrr in rankvals[0]:
                    rrr = int(rrr)
                    #print('Adding to binra', rank, self.binra)
                    self.binra[rrr] += val2add

            # else just add one to position of obs value.
            else:
                #print('Adding to binra', rank)
                self.binra[rank] += 1

            # This is incorrect way. Making rank middle
            # of multiple 0 values.
            # if obsval < self.tolerance:
            #   rowz = row[row<self.tolerance]
            #   rank = int(len(rowz)/2.0)
            #   self.zeronum.append(len(rowz)-1)

            # this is used in check1 method.
            if rank == len(self.binra)-1:
                if self.verbose:
                    print('high val', row)
                rowz = rowcheck[rowcheck > self.tolerance]
                self.nonzero.append(len(rowz)-1)
                self.nonzerorow.append(rowcheck)
                self.obsmax.append(obsval)

            # print(rank)
            self.ranklist.append(rank)
            self.obsvals.append(obsval)


