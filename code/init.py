
# coding: utf-8

# # NETSET initializer

# In[1]:

import matplotlib.pyplot as plt, pandas as pd, numpy as np, json, copy, zipfile
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# In[2]:

#suppres warnings
import warnings
warnings.simplefilter(action = "ignore")


# In[3]:

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# ## Country and region name converters

# In[4]:

#country name converters

#EIA->pop
clist1={'North America':'Northern America',
'United States':'United States of America',
'Central & South America':'Latin America and the Caribbean',
'Bahamas, The':'Bahamas',
'Saint Vincent/Grenadines':'Saint Vincent and the Grenadines',
'Venezuela':'Venezuela (Bolivarian Republic of)',
'Macedonia':'The former Yugoslav Republic of Macedonia',
'Moldova':'Republic of Moldova',
'Russia':'Russian Federation',
'Iran':'Iran (Islamic Republic of)',
'Palestinian Territories':'State of Palestine',
'Syria':'Syrian Arab Republic',
'Yemen':'Yemen ',
'Congo (Brazzaville)':'Congo',
'Congo (Kinshasa)':'Democratic Republic of the Congo',
'Cote dIvoire (IvoryCoast)':"C\xc3\xb4te d'Ivoire",
'Gambia, The':'Gambia',
'Libya':'Libyan Arab Jamahiriya',
'Reunion':'R\xc3\xa9union',
'Somalia':'Somalia ',
'Sudan and South Sudan':'Sudan',
'Tanzania':'United Republic of Tanzania',
'Brunei':'Brunei Darussalam',
'Burma (Myanmar)':'Myanmar',
'Hong Kong':'China, Hong Kong Special Administrative Region',
'Korea, North':"Democratic People's Republic of Korea",
'Korea, South':'Republic of Korea',
'Laos':"Lao People's Democratic Republic",
'Macau':'China, Macao Special Administrative Region',
'Timor-Leste (East Timor)':'Timor-Leste',
'Virgin Islands,  U.S.':'United States Virgin Islands',
'Vietnam':'Viet Nam'}

#BP->pop
clist2={u'                 European Union #':u'Europe',
u'Rep. of Congo (Brazzaville)':u'Congo (Brazzaville)',
'Republic of Ireland':'Ireland',
'China Hong Kong SAR':'China, Hong Kong Special Administrative Region',
u'Total Africa':u'Africa',
u'Total North America':u'Northern America',
u'Total S. & Cent. America':'Latin America and the Caribbean',
u'Total World':u'World',
u'Total World ':u'World',
'South Korea':'Republic of Korea',
u'Trinidad & Tobago':u'Trinidad and Tobago',
u'US':u'United States of America'}

#WD->pop
clist3={u"Cote d'Ivoire":"C\xc3\xb4te d'Ivoire",
u'Congo, Rep.':u'Congo (Brazzaville)',
u'Caribbean small states':'Carribean',
u'East Asia & Pacific (all income levels)':'Eastern Asia',
u'Egypt, Arab Rep.':'Egypt',
u'European Union':u'Europe',
u'Hong Kong SAR, China':u'China, Hong Kong Special Administrative Region',
u'Iran, Islamic Rep.':u'Iran (Islamic Republic of)',
u'Kyrgyz Republic':u'Kyrgyzstan',
u'Korea, Rep.':u'Republic of Korea',
u'Latin America & Caribbean (all income levels)':'Latin America and the Caribbean',
u'Macedonia, FYR':u'The former Yugoslav Republic of Macedonia',
u'Korea, Dem. Rep.':u"Democratic People's Republic of Korea",
u'South Asia':u'Southern Asia',
u'Sub-Saharan Africa (all income levels)':u'Sub-Saharan Africa',
u'Slovak Republic':u'Slovakia',
u'Venezuela, RB':u'Venezuela (Bolivarian Republic of)',
u'Yemen, Rep.':u'Yemen ',
u'Congo, Dem. Rep.':u'Democratic Republic of the Congo'}

#COMTRADE->pop
clist4={u"Bosnia Herzegovina":"Bosnia and Herzegovina",
u'Central African Rep.':u'Central African Republic',
u'China, Hong Kong SAR':u'China, Hong Kong Special Administrative Region',
u'China, Macao SAR':u'China, Macao Special Administrative Region',
u'Czech Rep.':u'Czech Republic',
u"Dem. People's Rep. of Korea":"Democratic People's Republic of Korea",
u'Dem. Rep. of the Congo':"Democratic Republic of the Congo",
u'Dominican Rep.':u'Dominican Republic',
u'Fmr Arab Rep. of Yemen':u'Yemen ',
u'Fmr Ethiopia':u'Ethiopia',
u'Fmr Fed. Rep. of Germany':u'Germany',
u'Fmr Panama, excl.Canal Zone':u'Panama',
u'Fmr Rep. of Vietnam':u'Viet Nam',
u"Lao People's Dem. Rep.":u"Lao People's Democratic Republic",
u'Occ. Palestinian Terr.':u'State of Palestine',
u'Rep. of Korea':u'Republic of Korea',
u'Rep. of Moldova':u'Republic of Moldova',
u'Serbia and Montenegro':u'Serbia',
u'US Virgin Isds':u'United States Virgin Islands',
u'Solomon Isds':u'Solomon Islands',
u'United Rep. of Tanzania':u'United Republic of Tanzania',
u'TFYR of Macedonia':u'The former Yugoslav Republic of Macedonia',
u'USA':u'United States of America',
u'USA (before 1981)':u'United States of America',
}

#Jacobson->pop
clist5={u"Korea, Democratic People's Republic of":"Democratic People's Republic of Korea",
u'All countries':u'World',
u"Cote d'Ivoire":"C\xc3\xb4te d'Ivoire",
u'Iran, Islamic Republic of':u'Iran (Islamic Republic of)',
u'Macedonia, Former Yugoslav Republic of':u'The former Yugoslav Republic of Macedonia',
u'Congo, Democratic Republic of':u"Democratic Republic of the Congo",
u'Korea, Republic of':u'Republic of Korea',
u'Tanzania, United Republic of':u'United Republic of Tanzania',
u'Moldova, Republic of':u'Republic of Moldova',
u'Hong Kong, China':u'China, Hong Kong Special Administrative Region',
u'All countries.1':"World"
}

#NREL solar->pop
clist6={u"Antigua & Barbuda":u'Antigua and Barbuda',
u"Bosnia & Herzegovina":u"Bosnia and Herzegovina",
u"Brunei":u'Brunei Darussalam',
u"Cote d'Ivoire":"C\xc3\xb4te d'Ivoire",
u"Iran":u'Iran (Islamic Republic of)',
u"Laos":u"Lao People's Democratic Republic",
u"Libya":'Libyan Arab Jamahiriya',
u"Moldova":u'Republic of Moldova',
u"North Korea":"Democratic People's Republic of Korea",
u"Reunion":'R\xc3\xa9union',
u'Sao Tome & Principe':u'Sao Tome and Principe',
u'Solomon Is.':u'Solomon Islands',
u'St. Lucia':u'Saint Lucia',
u'St. Vincent & the Grenadines':u'Saint Vincent and the Grenadines',
u'The Bahamas':u'Bahamas',
u'The Gambia':u'Gambia',
u'Virgin Is.':u'United States Virgin Islands',
u'West Bank':u'State of Palestine'
}

#NREL wind->pop
clist7={u"Antigua & Barbuda":u'Antigua and Barbuda',
u"Bosnia & Herzegovina":u"Bosnia and Herzegovina",
u'Occupied Palestinian Territory':u'State of Palestine',
u'China Macao SAR':u'China, Macao Special Administrative Region',
u'East Timor':u'Timor-Leste',
u'TFYR Macedonia':u'The former Yugoslav Republic of Macedonia',
u'IAM-country Total':u'World'
}

#country entroids->pop
clist8={u'Burma':'Myanmar',
u"Cote d'Ivoire":"C\xc3\xb4te d'Ivoire",
u'Republic of the Congo':u'Congo (Brazzaville)',
u'Reunion':'R\xc3\xa9union'
}

def cnc(country): #country name converter
    if country in clist1: return clist1[country]
    elif country in clist2: return clist2[country]
    elif country in clist3: return clist3[country]
    elif country in clist4: return clist4[country]
    elif country in clist5: return clist5[country]
    elif country in clist6: return clist6[country]
    elif country in clist7: return clist7[country]
    elif country in clist8: return clist8[country]
    else: return country


# Set path to data repository

# In[5]:

dbpath='E:/Dropbox/Public/datarepo/netset/'
savepath='../'


# # Population

# Consult the notebook entitled *pop.ipynb* for the details of mining the data from the UN statistics division online database.
# 
# Due to being the reference database for country names, the cell below needs to be run first, before any other databases.

# In[12]:

#population data
pop=pd.read_json(dbpath+'db/pop.json')


# In[7]:

#population data
#deprecated
#pop=pd.read_csv(dbpath+'db/pop.csv.save').set_index(['Country','Year']).unstack(level=1)


# # Units

# In[18]:

#initialize data and constants
data={}
countries={i for i in pop.index}
years={i for i in pop.columns}
dbs={'bp','eia'}
datatypes={'prod','cons','emi','res'}
allfuels=['oil','coal','gas','nuclear','biofuels','hydro','geo_other','solar','wind']
fossils=['oil','coal','gas']+['nrg','nrg_sum']
transp=1 #transparency
#colorlist=np.array([[166,86,40,transp*255],[153,153,153,transp*255],[152,78,163,transp*255],
#                    [228,26,28,transp*255],[247,129,191,transp*255],[55,126,184,transp*255],
#                    [82,56,65,transp*255],[255,255,51,transp*255],[77,175,74,transp*255]])/255.0
colorlist=np.array([[131,13,9,transp*255],[85,20,52,transp*255],[217,20,14,transp*255],
                    [213,9,98,transp*255],[64,185,85,transp*255],[202,200,46,transp*255],
                    [106,23,9,transp*255],[251,212,31,transp*255],[112,133,16,transp*255]])/255.0
gcolors={allfuels[i]:colorlist[i] for i in range(len(allfuels))}

def reset(what='all',datatype='all'):
    global data
    if what=='all':
        #reset all values of database
        fuels=allfuels+['nrg','nrg_sum']
        data={i:{k:{'energy':{j:{m:{l:np.NaN for l in dbs} for m in datatypes}                              for j in fuels},'population':long(pop.loc[i][k])*1000,                                              'consumer_efficiency':0.5,                                              'cumulative_emissions':0}                              for k in pop.columns}                              #we use population as the default database for country names
                              for i in pop.index} 
    else:
        countries=data.keys()
        for i in countries:
            for j in years:
                if datatype=='all':
                    data[i][j]['energy'][what]={k:{l:np.NaN for l in dbs} for k in datatypes}
                else:
                    data[i][j]['energy'][what][datatype]={l:np.NaN for l in dbs}

reset()

kbpd_to_TWh=365.25*0.001628200 #unit conversion from thousand barrels of oil per day to TWh per year
Gboe_to_TWh=1628.2 #unit conversion from thousand million barrels of oil to TWh
EJ_to_TWh=277.77 #unit conversion from exa Joule to TWh
bcf_to_TWh=0.2931 #unit conversion from billion cubic feet of natural gas to TWh
tcf_to_TWh=bcf_to_TWh*1000.0 #unit conversion from trillion cubic feet of natural gas to TWh
qbtu_to_TWh=293.297222 #unit conversion from quadrillion British thermal units to TWh
mtoe_to_TWh=11.63 #unit conversion million metric tons of oil equivalent to TWh
Mtoe_to_TWh=11.63 #unit conversion million metric tons of oil equivalent to TWh
Gtoe_to_TWh=11.63*1000 #unit conversion million metric tons of oil equivalent to TWh
kgge_to_gm3=1.49 #unit conversion from kilogram of natural gas to cubic meter, based on CH4
mtlnge_to_TWh=14.45 #unit conversion million metric tons of gas (LNG) equivalent to TWh
cm_to_cf=35.3 #unit conversion from million cubic meters to million cubic feet
tcm_to_TWh=tcf_to_TWh*cm_to_cf #unit conversion from trillion cubic meters of natural gas to TWh
kgge_to_TWh=kgge_to_gm3*tcf_to_TWh*cm_to_cf*1e-18 #unit conversion from kilogram of natural gas to TWh
#mtge_to_TWh=kgge_to_gm3*tcf_to_TWh*cm_to_cf*1e-9 #unit conversion from kilogram of natural gas to TWh
mtge_to_GJ=53.6
mtge_to_TWh=mtge_to_GJ*1e-9*EJ_to_TWh
t_to_st=1.10231 #unit conversion from metric ton to short ton
tcoe_to_toe=0.7 #unit conversion from metric tons of coal equivalent to metric tons of oil equivalent
mtcoe_to_TWh=tcoe_to_toe*mtoe_to_TWh #unit conversion million metric tons of coal equivalent to TWh
#mtcoe_to_TWh=8.141
mstcoe_to_TWh=mtcoe_to_TWh*t_to_st #unit conversion million metric short tons of coal equivalent to TWh
c_to_co2=44.0/12 #unit conversion from C to CO2 mass

carbon_budget=840*c_to_co2 #840 GtC as per http://www.ipcc.ch/report/ar5/wg1/


# ## Plotters

# In[19]:

from matplotlib.patches import Rectangle
def stackplotter(country,db='navg',datatype='all',fuels='all',limits=[1965,2015]):
    x=np.sort(list(years))
    if datatype!='all':
        fig, ax = plt.subplots(1,1,subplot_kw=dict(axisbg='#EEEEEE'),figsize=(10,6))
        ax.grid(color='white', linestyle='solid')

        if type(fuels)==str: 
            if fuels=='all':fuels=allfuels
            else: fuels=[fuels]
        ind=np.argsort([np.isnan(np.array([data[country][year]['energy'][fuel][datatype][db] for year in x]).T).sum() for fuel in fuels])
        fuels=np.array(fuels)[ind]
        colors=[gcolors[fuel] for fuel in fuels]
        y=np.array([[data[country][year]['energy'][i][datatype][db] for i in fuels] for year in x]).T

        stack_coll=ax.stackplot(x,y,colors=colors)
        proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_coll][::-1]

        ax.legend(proxy_rects, fuels[::-1],loc=2,framealpha=0)
        if datatype=='emi': ax.set_ylabel('MtCO2',labelpad=10)
        else: ax.set_ylabel('TWh',labelpad=10)
        ax.set_title(db+' '+datatype)
        ax.set_xlim(limits)
    else:
        fig, ax = plt.subplots(1,2,subplot_kw=dict(axisbg='#EEEEEE'),figsize=(17,5))
        datatype=['prod','cons']
        ymax=0
        for k in range(2): 
            ax[k].grid(color='white', linestyle='solid')
        
            if type(fuels)==str: 
                if fuels=='all':fuels=allfuels
                else: fuels=[fuels]
            ind=np.argsort([np.isnan(np.array([data[country][year]['energy'][fuel][datatype[k]][db] for year in x]).T).sum() for fuel in fuels])
            fuels=np.array(fuels)[ind]
            colors=[gcolors[fuel] for fuel in fuels]
            y=np.array([[data[country][year]['energy'][i][datatype[k]][db] for i in fuels] for year in x]).T
            
            stack_coll=ax[k].stackplot(x,y,colors=colors)
            proxy_rects = np.array([Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_coll][::-1])

            ax[k].legend(proxy_rects, fuels[::-1],loc=2,framealpha=0.7)
            ax[k].set_ylabel('TWh',labelpad=10)
            ax[k].set_title(db+' '+datatype[k])
            ax[k].set_xlim(limits)
            ymax=max(ymax,ax[k].get_ylim()[1])
        for k in range(2): 
            ax[k].set_ylim([0,ymax])
    plt.suptitle(country,fontsize=14,color='green')
    plt.show()


# In[20]:

def keyplotter(d,c="royalBlue",o=1,lw=1):
    
    x=[]
    y=[]
    for key in sorted(d.keys()):
        x.append(key)
        y.append(d[key])
    plt.plot(x,y,color=c,alpha=o,lw=lw)


# In[21]:

def subplotter(country,fuel,db,datatype,ax):
    try:
        ax.plot(subgetter(country,fuel,db,datatype)['x'],subgetter(country,fuel,db,datatype)['y'],label=db+' '+datatype,linewidth=2)
    except: print 'ERROR plotting', country, fuel, db, datatype

def fracplotter(country,fuel,db,top,down,ax):
    try:
        ax.plot(fracgetter(country,fuel,db,top,down)['x'],            fracgetter(country,fuel,db,top,down)['y'],label=db+' '+top+'/'+down,linewidth=2,linestyle='--')
    except: print 'ERROR plotting', country, fuel, db, top+'/'+down
        
def plotter(country,fuel,db='avg',limits=[1965,2015]):
    fig, ax = plt.subplots(1,3,subplot_kw=dict(axisbg='#EEEEEE'),figsize=(17,4))
    for i in range(3): ax[i].grid(color='white', linestyle='solid')
    
    if fuel=='population': 
        subplotter(country,fuel,'un','population',ax[0])
        ax[0].set_ylabel('million',labelpad=14)
        ax[0].set_xlim(limits)
    elif fuel=='cumulative_emissions': 
        subplotter(country,fuel,'cumulative','emissions',ax[0])
        ax[0].plot(limits,np.ones(len(limits))*carbon_budget,'r--',label='carbon budget')
        ax[0].set_ylabel('GtCO2',labelpad=14)
        ax[0].set_xlim(limits)
    else: 
        ax[0].set_ylabel('TWh',labelpad=-60)
        if fuel in fossils:
            ax[1].set_ylabel('TWh',labelpad=-70)
            ax[2].set_ylabel('MtCO2',labelpad=-60)
            ax1=ax[1].twinx()
            ax2=ax[2].twinx()
        for i in range(3): ax[i].set_xlim(limits)
        if db=='all': db=dbs | {'avg'}
        if type(db)==str: db={db}
        for i in db:
            subplotter(country,fuel,i,'cons',ax[0])
            subplotter(country,fuel,i,'prod',ax[0])
            if fuel in fossils:
                subplotter(country,fuel,i,'res',ax[1])
                subplotter(country,fuel,i,'emi',ax[2])
                #plot extraction ratio
                fracplotter(country,fuel,i,'prod','res',ax1)
                #plot carbon intensity
                fracplotter(country,fuel,i,'emi','cons',ax2)
                
    for i in range(3): ax[i].legend(loc=2,framealpha=0.8)
    ax[0].set_title(fuel)
    if fuel in fossils:
        ax1.legend(loc=4,framealpha=0.8)
        ax2.legend(loc=4,framealpha=0.8)
        ax1.set_ylabel('fraction',labelpad=-50)
        ax2.set_ylabel('kgCO2/kWh primary',labelpad=-45)
    plt.suptitle(country,fontsize=14,color='green')
    plt.show()
    
def subgetter(country,fuel,db,datatype):
    try:
        if fuel=='population':
            x=np.sort(list(years))
            y=[data[country][i]['population']/1000000.0 for i in x]
        elif fuel=='cumulative_emissions':
            x=np.sort(list(years))
            y=[data[country][i]['cumulative_emissions']/1000.0 for i in x]
        else:
            x=[i for i in np.sort(list(years)) if not np.isnan(data[country][i]['energy'][fuel][datatype][db])]
            y=[data[country][i]['energy'][fuel][datatype][db] for i in x]
        return {'x':x,'y':y}
    except: print 'ERROR getting', country, fuel, db, datatype

def fracgetter(country,fuel,db,top,down):
    try:
        a=subgetter(country,fuel,db,top)['x']
        b=subgetter(country,fuel,db,down)['x']
        c=np.intersect1d(a,b)
        d=np.searchsorted(a,c)
        e=np.searchsorted(b,c)
        x=a[d[0]:d[::-1][0]+1]
        y=np.array(subgetter(country,fuel,db,top)['y'][d[0]:d[::-1][0]+1])/          np.array(subgetter(country,fuel,db,down)['y'][e[0]:e[::-1][0]+1])
        return {'x':x,'y':y}
    except: print 'ERROR getting', country, fuel, db, top+'/'+down
        
def getter(country,fuel,db='avg'):
    if fuel=='population': 
        return subgetter(country,fuel,'un','population')
    elif fuel=='cumulative_emissions': 
        return subgetter(country,fuel,'cumulative','emissions')
    else: 
        if db=='all': db=dbs | {'avg'}
        if type(db)==str: db={db}
        aux={}
        for i in db:
            aux[i]={}
            for datatype in {'cons','prod','emi','res'}:
                aux[i][datatype]=subgetter(country,fuel,i,datatype)
        return aux


# ## Interpolators

# In[22]:

def interpolate(d,years,gfit=2,depth=1,polyorder=1,override=False,ends=False):
#d=helper
#years=[2015]
#gfit=1
#depth=extrapolatedict[fuel]
#polyorder=1
#override=True
#ends=True
#if True:
    #depth * length of interpolation substrings will be taken to the left and right
    #for example for {1971:5,1972:6,1973:7,1974:5} interpolating it over 1969-1990
    #for the section 1960-1970 (2 elements) the values from 1972,1973,1974 (3 elements) will be taken with depth 1.5
    #for the section 1974-1990 (15 elements) all values  (4 elements) will be taken to extrapolate
    #override to extend interpolation to edges, i.e. extrapolate
    if (gfit>2): 
        print 'interpolate takes only 1 (polynomial) or 2 (exponential) as 3rd argument [default=2]'
        return
    mydict={}
    if d!={}:
        missing_points=[[]]
        onbeginning=False
        onend=False
        for year in years:
            if year not in d.keys():
                missing_points[-1].append(year)
            else:
                missing_points.append([])
        for m in missing_points:
            if m:
                fit=gfit

                #if only one point, set min extrapolation depth to 2
                if (len(m)==1): depth=max(depth,2)
                
                #check if it is ends of the interval, 
                if ((m[-1]<np.sort(d.keys())[0])|(m[0]>np.sort(d.keys())[-1])): 
                    #if not set to override then extrapolate mean only
                    if not override: 
                        fit=0                    

                if fit==0: #take average
                    y = {k: d[k] for k in set(d.keys()).intersection(range(int(max(min(years),min(m)-int(3))),                                                                           int(min(max(years),max(m)+int(3))+1)))}
                    #returned empty, on beginning
                    if y=={}:
                        if m[-1]<np.sort(d.keys())[0]:y={np.sort(d.keys())[0]:d[np.sort(d.keys())[0]]}
                        elif m[0]>np.sort(d.keys())[-1]:y={np.sort(d.keys())[-1]:d[np.sort(d.keys())[-1]]}
                    for i in range(len(m)):
                        mydict[m[i]]=np.mean(y.values())
                elif fit==1:
                    #intersector
                    y = {k: d[k] for k in set(d.keys()).intersection(range(int(max(min(years),                                min(m)-int(depth*len(m)))),int(min(max(years),max(m)+int(depth*len(m)))+1)))}
                    #returned empty
                    if y=={}:
                        if m[-1]<np.sort(d.keys())[0]:y={np.sort(d.keys())[0]:d[np.sort(d.keys())[0]]}
                        elif m[0]>np.sort(d.keys())[-1]:y={np.sort(d.keys())[-1]:d[np.sort(d.keys())[-1]]}
                            
                    w = np.polyfit(y.keys(),y.values(),polyorder) # obtaining regression parameters
                    if (polyorder==1):
                        intersector=w[0]*np.array(m)+w[1]
                    else:
                        intersector=w[0]*np.array(m)*np.array(m)+w[1]*np.array(m)+w[2]
                    for i in range(len(m)):
                        mydict[m[i]]=max(0,intersector[i])
                else:
                    #exponential intersector
                    y = {k: d[k] for k in set(d.keys()).intersection(range(int(max(min(years),                                min(m)-int(depth*len(m)))),int(min(max(years),max(m)+int(depth*len(m)))+1)))}
                    #returned empty
                    if y=={}:
                        if m[-1]<np.sort(d.keys())[0]:y={np.sort(d.keys())[0]:d[np.sort(d.keys())[0]]}
                        elif m[0]>np.sort(d.keys())[-1]:y={np.sort(d.keys())[-1]:d[np.sort(d.keys())[-1]]}
                    
                    w = np.polyfit(y.keys(),np.log(y.values()),1) # obtaining log regression parameters (exp fitting)
                    intersector=np.exp(w[1])*np.exp(w[0]*np.array(m))
                    for i in range(len(m)):
                        mydict[m[i]]=max(0,intersector[i])
                    
                #record ends adjustment beginning and end
                if ends:
                    if (m[-1]<np.sort(d.keys())[0]):
                        onbeginning=True
                        beginning=m[-1]
                    if (m[0]>np.sort(d.keys())[-1]): 
                        onend=True
                        end=m[0]
        #finish ends adjustment
        if ends:
            if onbeginning:
                #calculate adjustment scaler
                if (mydict[beginning]==0): scaler=0
                elif (beginning+1 in d): scaler=d[beginning+1]*1.0/mydict[beginning]
                else: scaler=d[np.sort(d.keys())[0]]*1.0/mydict[beginning]
                #readjust data
                for year in mydict:
                    if (year<=beginning):
                        mydict[year]*=scaler
            if onend:
                #calculate adjustment scaler
                if (mydict[end]==0): scaler=0
                elif (end-1 in d): scaler=d[end-1]*1.0/mydict[end]
                else: scaler=d[np.sort(d.keys())[-1]]*1.0/mydict[end]
                #readjust data
                for year in mydict:
                    if (year>=end):
                        mydict[year]*=scaler

    #return interpolated points
    return mydict


# In[23]:

def scurve(x):
    lamda=8 #curve steepness control
    mu=0.5
    return 1/(1+np.exp(-lamda*(x-mu)))
x=np.arange(100)/100.0
#plt.plot(x,[scurve(i) for i in x],c='#dd1c77',lw=2)
#plt.xlabel('input (normalized time)')
#plt.ylabel('output (effect strength)')
#plt.text(0.04, 0.95, u'$f(t)=1/[1+e^{-λ(t-μ)}]$\n$λ=12$\n$μ=0.5$',
#    horizontalalignment='left',
#    verticalalignment='top',size=14,alpha=0.6)
#plt.show()


# ## Country ISO codes

# In[24]:

cc=pd.read_excel(dbpath+'db/Country Code and Name ISO2 ISO3.xls')
#http://unstats.un.org/unsd/tradekb/Attachment321.aspx?AttachmentType=1


# In[25]:

ccs=cc['Country Code'].values


# ## Country neighbor list

# In[26]:

neighbors=pd.read_csv(dbpath+'db/contry-geotime.csv')
#https://raw.githubusercontent.com/ppKrauss/country-geotime/master/data/contry-geotime.csv


# In[27]:

#country name converter from iso to comtrade and back
iso2c={}
isoc2={}
for i in cc.T.iteritems():
    iso2c[i[1][0]]=i[1][1]
    isoc2[i[1][1]]=i[1][0]


# In[28]:

#country name converter from pop to iso
pop2iso={}
for i in cc.T.iteritems():
    pop2iso[cnc(i[1][1])]=int(i[1][0])


# In[29]:

#country name converter from alpha 2 to iso
c2iso={}
for i in neighbors.T.iteritems():
    c2iso[str(i[1][0])]=i[1][1]
c2iso['NA']=c2iso['nan'] #adjust for namibia
c2iso.pop('nan');


# In[30]:

#create country neighbor adjacency list based on iso country number codes
c2neighbors={}
for i in neighbors.T.iteritems():
    z=str(i[1][4]).split(' ')
    if (str(i[1][1])!='nan'): c2neighbors[int(i[1][1])]=[c2iso[k] for k in z if k!='nan']


# In[31]:

#extend iso codes not yet encountered
iso2c[729]="Sudan"
iso2c[531]="Curacao"
iso2c[535]="Bonaire, Sint Eustatius and Saba"
iso2c[728]="South Sudan"
iso2c[534]="Sint Maarten (Dutch part)"
iso2c[652]="Saint Barthélemy"


# ## GDP

# In[32]:

#deprecated
#gdp=pd.read_excel(path+'db/GDP.xls',skiprows=2)
gdp=pd.read_excel(dbpath+'db/Download-GDPcurrent-USD-countries.xls',skiprows=2)
#http://unstats.un.org/unsd/snaama/downloads/Download-GDPcurrent-USD-countries.xls
#imports are in current terms, so we need also current terms here


# In[33]:

gdpc={}
firstyear=gdp.columns[2]
lastyear=gdp.columns[len(gdp.columns)-1]
for i in gdp.T.iteritems():
    if (i[1][1]=='Total Value Added'):
        country=cnc(i[1][0])
        if country not in gdpc:gdpc[country]={}
        for j in range(firstyear,lastyear):
            gdpc[country][j]=i[1][j+2-firstyear]


# ## Coords

# In[34]:

import requests, StringIO


# In[35]:

#r = requests.get('http://gothos.info/resource_files/country_centroids.zip')
# use StringIO.StringIO(r.content) below if requests is used

z = zipfile.ZipFile(dbpath+'db/country_centroids.zip')
coord=pd.read_csv(z.open('country_centroids_all.csv'),sep='\t')    .drop(['DMS_LAT','DMS_LONG','MGRS','JOG','DSG','FULL_NAME',       'ISO3136','AFFIL','FIPS10','MOD_DATE'],axis=1)
coord.columns=['LAT','LONG','Country']
coord=coord.set_index('Country',drop=True)


# In[36]:

#create normalized distance matrix of countries
names=[]
for i in coord.index:
    names.append(cnc(i))
coord['NAME']=names
coord=coord.set_index('NAME',drop=True)


# In[37]:

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
def distance(i,j):
    if i in coord.index and j in coord.index:
        return haversine(coord.loc[i]['LONG'],coord.loc[i]['LAT'],
                    coord.loc[j]['LONG'],coord.loc[j]['LAT'])
    else: return 5000


# # Save function

# Formats `data` dictionary into `json` format for epxloration in visualization.

# In[38]:

tradealpha={}
goodcountries=sorted(data.keys())


# In[39]:

#save with zeros
def save0(sd,countrylist=[],db='navg3'):
    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED
    print 'saving... ',sd,
    popsave={}
    countries=[]
    if countrylist==[]:
        c=sorted(goodcountries)
    
    else: c=countrylist
    for country in c:
        popdummy={}
        tosave=[]
        for year in data[country]:
            popdummy[year]=data[country][year]['population']
            for fuel in data[country][year]['energy']:
            #for fuel in allfuels:
                if fuel not in {'nrg','nrg_sum'}:
                    tosave.append({"t":year,"u":fuel,"g":"f","q1":"pp","q2":999,
                               "s":round(0 if ((db in data[country][year]['energy'][fuel]['prod']) \
                                          and (np.isnan(data[country][year]['energy'][fuel]['prod'][db]))) else \
                               data[country][year]['energy'][fuel]['prod'][db] if \
                                   db in data[country][year]['energy'][fuel]['prod'] else 0,3)
                               })
                    tosave.append({"t":year,"u":fuel,"g":"m","q1":"cc","q2":999,
                               "s":round(0 if ((db in data[country][year]['energy'][fuel]['cons']) \
                                          and (np.isnan(data[country][year]['energy'][fuel]['cons'][db]))) else \
                               data[country][year]['energy'][fuel]['cons'][db] if \
                                   db in data[country][year]['energy'][fuel]['cons'] else 0,3)
                              })
                    
        #no import export flows on global
        if country not in {"World"}:
            flowg={"Import":"f","Export":"m","Re-Export":"m","Re-Import":"f"}
            if country in tradealpha:
                for year in tradealpha[country]:
                    for fuel in tradealpha[country][year]:
                        for flow in tradealpha[country][year][fuel]:
                            for partner in tradealpha[country][year][fuel][flow]:
                                tosave.append({"t":int(float(year)),"u":fuel,"g":flowg[flow],"q1":flow,"q2":partner,
                                           "s":round(tradealpha[country][year][fuel][flow][partner],3)
                                           })
        
        if tosave!=[]:
            countries.append(country)
            popsave[country]=popdummy
        
        file(savepath+'json/'+str(sd)+'/data.json','w').write(json.dumps(tosave)) 
        zf = zipfile.ZipFile(path+'json/'+str(sd)+'/'+str(country.encode('utf-8').replace('/','&&'))+'.zip', mode='w')
        zf.write(savepath+'json/'+str(sd)+'/data.json','data.json',compress_type=compression)
        zf.close()
        
    #save all countries list
    file(savepath+'json/'+str(sd)+'/'+'countries.json','w').write(json.dumps(countries)) 
    
    #save countries populations
    file(savepath+'json/'+str(sd)+'/'+'pop.json','w').write(json.dumps(popsave))     
    
    print ' done'


# In[40]:

#save without zeroes
def save(sd,countrylist=[],db='navg3'):
    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED
    print 'saving... ',sd,
    popsave={}
    countries=[]
    if countrylist==[]:
        c=sorted(goodcountries)
    
    else: c=countrylist
    for country in c:
        popdummy={}
        tosave=[]
        for year in data[country]:
            popdummy[year]=data[country][year]['population']
            for fuel in data[country][year]['energy']:
            #for fuel in allfuels:
                if fuel not in {'nrg','nrg_sum'}:
                    tosave.append({"t":year,"u":fuel,"g":"f","q1":"pp","q2":999,
                               "s":round(0 if ((db in data[country][year]['energy'][fuel]['prod']) \
                                          and (np.isnan(data[country][year]['energy'][fuel]['prod'][db]))) else \
                               data[country][year]['energy'][fuel]['prod'][db] if \
                                   db in data[country][year]['energy'][fuel]['prod'] else 0,3)
                               })
                    tosave.append({"t":year,"u":fuel,"g":"m","q1":"cc","q2":999,
                               "s":round(0 if ((db in data[country][year]['energy'][fuel]['cons']) \
                                          and (np.isnan(data[country][year]['energy'][fuel]['cons'][db]))) else \
                               data[country][year]['energy'][fuel]['cons'][db] if \
                                   db in data[country][year]['energy'][fuel]['cons'] else 0,3)
                              })
                    
        #no import export flows on global
        if country not in {"World"}:
            flowg={"Import":"f","Export":"m","Re-Export":"m","Re-Import":"f"}
            if country in tradealpha:
                for year in tradealpha[country]:
                    for fuel in tradealpha[country][year]:
                        for flow in tradealpha[country][year][fuel]:
                            for partner in tradealpha[country][year][fuel][flow]:
                                tosave.append({"t":int(float(year)),"u":fuel,"g":flowg[flow],"q1":flow,"q2":partner,
                                           "s":round(tradealpha[country][year][fuel][flow][partner],3)
                                           })
        
        tosave2=[] #eliminate zeroes
        for item in tosave:
            if item["s"]!=0: tosave2.append(item)
        
        if tosave2!=[]:
            countries.append(country)
            popsave[country]=popdummy
        
        file(savepath+'json/'+str(sd)+'/data.json','w').write(json.dumps(tosave2)) 
        zf = zipfile.ZipFile(savepath+'json/'+str(sd)+'/'+str(country.encode('utf-8').replace('/','&&'))+'.zip', mode='w')
        zf.write(savepath+'json/'+str(sd)+'/data.json','data.json',compress_type=compression)
        zf.close()
        
    #save all countries list
    file(savepath+'json/'+str(sd)+'/'+'countries.json','w').write(json.dumps(countries)) 
    
    #save countries populations
    file(savepath+'json/'+str(sd)+'/'+'pop.json','w').write(json.dumps(popsave))     
    
    print ' done'


# ## Save

# This is to automatically save this notebook as a Python script to be loaded in other notebooks. In order to prevent an infitine loop, there is a trigger variable `initlive`. PLease go to the cell on the bottom to set it to `True` and then run the cell with the `system` magic in it. On external call, `initilive` will default to `False`, thus preventing the infinite loop.

# In[41]:

initlive=False


# In[42]:

if initlive:
    get_ipython().magic(u'system jupyter nbconvert --to script init.ipynb')


# In[43]:

initlive=True

