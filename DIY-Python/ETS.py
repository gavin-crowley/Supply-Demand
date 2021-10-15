import numpy as np
import pandas as pd


def kpi(df):
    dem_ave = df.loc[df['Error'].notnull(),'Demand'].mean()
    bias_abs = df['Error'].mean()
    bias_rel = bias_abs / dem_ave
    print('Bias: {:0.2f}, {:.2%}'.format(bias_abs,bias_rel))
    MAPE = (df['Error'].abs()/df['Demand']).mean()
    print('MAPE: {:.2%}'.format(MAPE))
    MAE_abs = df['Error'].abs().mean()
    MAE_rel = MAE_abs / dem_ave
    print('MAE: {:0.2f}, {:.2%}'.format(MAE_abs,MAE_rel)) 
    RMSE_abs = np.sqrt((df['Error']**2).mean())
    RMSE_rel = RMSE_abs / dem_ave
    print('RMSE: {:0.2f}, {:.2%}'.format(RMSE_abs,RMSE_rel)) 
    
    
def moving_average(d, extra_periods=1, n=3):
    
    # Historical period length
    cols = len(d) 
    # Append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods) 
    # Define the forecast array
    f = np.full(cols+extra_periods,np.nan)
    
    # Create all the t+1 forecast until end of historical period
    for t in range(n,cols):
        f[t] = np.mean(d[t-n:t])
     
    # Forecast for all extra periods
    f[t+1:] = np.mean(d[t-n+1:t+1])
               
    # Return a dataframe with the demand, forecast & error
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Error':d-f})
  
    return df


def simple_exp_smooth(d, extra_periods=1, alpha=0.4):
   
    # Historical period length
    cols = len(d) 
    # Append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods) 
    
    # Forecast array
    f = np.full(cols+extra_periods,np.nan) 
    # Initilization of first forecast
    f[1] = d[0]
    
    # Create all the t+1 forecast until end of historical period
    for t in range(2,cols+1):
        f[t] = alpha*d[t-1]+(1-alpha)*f[t-1]
     
    # Forecast for all extra periods
    for t in range(cols+1,cols+extra_periods):
        # Update the forecast as the previous forecast
        f[t] = f[t-1]
           
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Error':d-f})
  
    return df

           
def double_exp_smooth(d, extra_periods=1, alpha=0.4, beta=0.4):
    
    # Historical period length
    cols = len(d) 
    # Append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods) 
    
    # Creation of the level, trend and forecast arrays
    f,a,b = np.full((3,cols+extra_periods),np.nan)
    
    # Level & Trend initialization
    a[0] = d[0]
    b[0] = d[1] - d[0]
 
    # Create all the t+1 forecast
    for t in range(1,cols):
        f[t] = a[t-1] + b[t-1]       
        a[t] = alpha*d[t] + (1-alpha)*(a[t-1]+b[t-1])       
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*b[t-1]
        
    # Forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t] = a[t-1] + b[t-1]
        a[t] = f[t]
        b[t] = b[t-1]
                          
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Error':d-f})
  
    return df

         
def double_exp_smooth_damped(d, extra_periods=1, alpha=0.4, beta=0.4, phi=0.9):    

    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
    
    # Creation of the level, trend and forecast arrays
    f,a,b = np.full((3,cols+extra_periods),np.nan)
    
    # Level & Trend initialization
    a[0] = d[0]
    b[0] = d[1] - d[0]

    # Create all the t+1 forecast        
    for t in range(1,cols):
        f[t] = a[t-1] + phi*b[t-1]        
        a[t] = alpha*d[t] + (1-alpha)*(a[t-1]+phi*b[t-1])       
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]
    
    # Forecast for all extra periods    
    for t in range(cols,cols+extra_periods):
        f[t] = a[t-1] + phi*b[t-1]
        a[t] = f[t]
        b[t] = phi*b[t-1]  
                           
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Error':d-f})
  
    return df


def seasonal_factors_mul(s,d,slen,cols):	   
    for i in range(slen):  
        idx = [x for x in range(cols) if x%slen==i] # Indices that correspond to this season
        s[i] = np.mean(d[idx])   # Season average
    s /= np.mean(s[:slen]) # Scale all season factors (sum of factors = slen)
    return s


def triple_exp_smooth_mul(d, slen=12, extra_periods=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3):

    cols = len(d) # Historical pteriod length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods

    # components initialization     
    f,a,b,s = np.full((4,cols+extra_periods),np.nan)
    s = seasonal_factors_mul(s,d,slen,cols)
       
    # Level & Trend initialization
    a[0] = d[0]/s[0]
    b[0] = d[1]/s[1] - d[0]/s[0]
    
    # Create the forecast for the first season
    for t in range(1,slen):
        f[t] = (a[t-1] + phi*b[t-1])*s[t]        
        a[t] = alpha*d[t]/s[t] + (1-alpha)*(a[t-1]+phi*b[t-1])        
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]
        
    # Create all the t+1 forecast
    for t in range(slen,cols):
        f[t] = (a[t-1] + phi*b[t-1])*s[t-slen]       
        a[t] = alpha*d[t]/s[t-slen] + (1-alpha)*(a[t-1]+phi*b[t-1])        
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]
        s[t] = gamma*d[t]/a[t] + (1-gamma)*s[t-slen]
        
    # Forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t] = (a[t-1] + phi*b[t-1])*s[t-slen]
        a[t] = f[t]/s[t-slen]
        b[t] = phi*b[t-1] 
        s[t] = s[t-slen]        
                       
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Season':s,'Error':d-f})

    return df


def seasonal_factors_add(s,d,slen,cols):	   
    for i in range(slen):  
        idx = [x for x in range(cols) if x%slen==i] # Indices that correspond to this season
        s[i] = np.mean(d[idx]) # Calculate season average        
    s -= np.mean(s[:slen]) # Scale all season factors (sum of factors = 0)  
    return s

    
def triple_exp_smooth_add(d, slen=12, extra_periods=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3):

    cols = len(d) # Historical pteriod length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
    
    # components initialization     
    f,a,b,s = np.full((4,cols+extra_periods),np.nan)
    s = seasonal_factors_add(s,d,slen,cols)
        
    # Level & Trend initialization
    a[0] = d[0]-s[0]
    b[0] = (d[1]-s[1]) - (d[0]-s[0])
           
    # Create the forecast for the first season
    for t in range(1,slen):
        f[t] = a[t-1] + phi*b[t-1] + s[t]       
        a[t] = alpha*(d[t]-s[t]) + (1-alpha)*(a[t-1]+phi*b[t-1])       
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]
                
    # Create all the t+1 forecast
    for t in range(slen,cols):
        f[t] = a[t-1] + phi*b[t-1] + s[t-slen]       
        a[t] = alpha*(d[t]-s[t-slen]) + (1-alpha)*(a[t-1]+phi*b[t-1])       
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]         
        s[t] = gamma*(d[t]-a[t]) + (1-gamma)*s[t-slen] 
        
    # Forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t] = a[t-1] + phi*b[t-1] + s[t-slen]
        a[t] = f[t]-s[t-slen]
        b[t] = phi*b[t-1] 
        s[t] = s[t-slen]
                      
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Season':s,'Error':d-f})

    return df


def exp_smooth_opti(d, extra_periods=6):  
  
    params = []  # contains all the different parameter sets  
    KPIs = []   # contains the results of each model  
    dfs = []  # contains all the dataframes returned by the different models  
      
    for alpha in [0.05,0.1,0.2,0.3,0.4,0.5,0.6]:  
          
        df = simple_exp_smooth(d,extra_periods=extra_periods,alpha=alpha)  
        params.append(f'Simple Smoothing, alpha: {alpha}')  
        dfs.append(df)  
        MAE = df['Error'].abs().mean()  
        KPIs.append(MAE)  
              
        for beta in [0.05,0.1,0.2,0.3,0.4]:  
              
            df = double_exp_smooth(d,extra_periods=extra_periods,alpha=alpha,beta=beta)  
            params.append(f'Double Smoothing, alpha: {alpha}, beta: {beta}')   
            dfs.append(df)  
            MAE = df['Error'].abs().mean()  
            KPIs.append(MAE)                 
     
    mini = np.argmin(KPIs)   
    print(f'Best solution found for {params[mini]} MAE of',round(KPIs[mini],2))  

    return dfs[mini]   
