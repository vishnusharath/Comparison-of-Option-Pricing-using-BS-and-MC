import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import time
#import resource
from matplotlib import style
style.use('bmh')
time_start = time.clock()

class OptionPricing:
    #Normal Black Scholes Model where...
    #S0 = Underlying Price
    #E = Strike Price
    #T = Expiration
    #rf = Risk Free
    #srigma = volatility
    #iterations = number of prices to simulate

    #This is a constructor which will assign initial variables for the option calculation
    def __init__(self, S0, E, T, rf, sigma, iterations, num_simulations, Otype):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations
        self.num_simulations = num_simulations
        self.Otype = Otype

    def option_sim_gb(self):
        #Columns For DataFrame
        if self.Otype == 'C':
            cols = ["Call Price (MC)"]
        else:
            cols = ["Put Price (MC)"]
        
        #rows for data frame appended to this list
        data_row = []
        
        #Enter loop monte carlo simulation 
        for x in range(1,self.num_simulations+1):

            #Array of options data
            data= np.zeros([self.iterations, 2])
            rand = np.random.normal(0,1,[1, self.iterations])
    
            #Calculate Stock Price applying geometric brownian motion to bs model
            stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*self.sigma**2)+ \
                                         self.sigma*np.sqrt(self.T)*rand)
                
            if self.Otype == 'C': 
                #Calculate max
                data[:,1] = stock_price - self.E
    
                #Average for MC Simulation
                average = np.sum(np.amax(data, axis=1))/float(self.iterations)
                
                #Calculate call price
                call_price = np.exp(-1.0*self.rf*self.T)*average
                                   
                #Append price as row to df
                data_row.append([call_price])
            else:

                #Note the calculation for Put Options differs here with the max
                data[:,1] = self.E - stock_price
                    
                #Average for MC Simulation
                average = np.sum(np.amax(data, axis=1))/float(self.iterations)
                
                #Calculate put price
                put_price = np.exp(-1.0*self.rf*self.T)*average
                                 
    
                                
                #Append price as row to df
                data_row.append([put_price])

        #DataFrame for simulation 
        df = pd.DataFrame(data_row, columns=cols)
        return df

    #Functions Reserved For Black Scholes Model
    def d1(self, S0, E, rf, sigma, T):
        return (np.log(self.S0/E) + (self.rf + self.sigma**2 / 2) * \
                self.T)/(self.sigma * np.sqrt(self.T))
     
    def d2(self, S0, E, r, sigma, T):
        return (np.log(self.S0 / self.E) + (self.rf - self.sigma**2 / 2) * \
                self.T) / (self.sigma * np.sqrt(self.T))
    
    #Black Scholes Model
    def BlackScholes(self):
        data_row = []       

        #We enter a loop here to graph our static call/put price along with MC results
        for x in range(1,self.num_simulations+1):
            if self.Otype=="C":
                
                call_price =  self.S0 * ss.norm.cdf(self.d1(self.S0, self.E, self.rf, \
                            sigma, self.T)) - self.E * np.exp(-self.rf * self.T) * \
                            ss.norm.cdf(self.d2(self.S0, self.E, self.rf, self.sigma, self.T))
                data_row.append([call_price])
                cols = ["Call Price (BS)"]

            else:
                 put_price =  self.E * np.exp(-self.rf * self.T) * ss.norm.cdf(-self.d2(self.S0, \
                            self.E, self.rf, self.sigma, self.T)) - self.S0 * ss.norm.cdf(-self.d1(self.S0, \
                            self.E, self.rf, self.sigma, self.T))
                 
                 cols = ["Put Price (BS)"]

                 data_row.append([put_price])
           
        df = pd.DataFrame(data_row, columns=cols)
        return df
    
    #Antithetic Model
    def antithetic(self):        
        if self.Otype=="C":
            #Set df
            cols = ["Call Price (MC)", "Call Price (BS)"]
            df = pd.DataFrame(columns=cols)
            
            #Call gb sim to get greater sample size
            op1 = np.round(m.option_sim_gb(), 2)
            op2 = np.round(m.option_sim_gb(), 2)
            
        else: 
            #Set df
            cols = ["Put Price (MC)", "Put Price (BS)"]
            df = pd.DataFrame(columns=cols)

            #Call gb sim to get greater sample size

            op1 = np.round(m.option_sim_gb(), 2)
            op2 = np.round(m.option_sim_gb(), 2)   

        #Combine df and get mean of rows  
        df = pd.concat([op1, op2], axis=1)
        df['Antithetic'] = df.mean(axis=1)
    
        return df

    #Print Option Specs
    def specs(self):
        print("S0\tstock price at time 0:", self.S0)
        print("E\tstrike price:", self.E)
        print("rf\tcontinuously compounded risk-free rate:", self.rf)
        print("sigma\tvolatility of the stock price per year:", self.sigma)
        print("T\ttime to maturity in trading years:", self.T)
        print("Sim\tnumber of simulations:", self.num_simulations)
    
    #Plot Options DF
    def plotdf(self, title, df):
        plot_df = df      
        plot_df.plot(style='.-')
        plt.suptitle(title, fontweight="bold")  
        plt.xlabel('Number of sumulations')
        plt.ylabel('Option Price')
        plt.show()
        
if __name__=="__main__":
    #Here we define the different option parameters
    #**********MODIFY PARAMS HERE***************#
    S0 = 95
    E = 95
    T = 1
    rf = 0.05
    sigma = 0.2
    iterations = 100
    num_simulations = 10000
    Otype = 'P'
    #**********MODIFY PARAMS HERE***************#
    
    #Initialize constructor
    m = OptionPricing(S0, E, T, rf, sigma, iterations, num_simulations, Otype)
    
    #Option Specs
    m.specs()
    
    #Here we plot the results of our simulations compared with BS formula and other MC approaches
    
    #Get df for regular mc sim
    gb_df = m.option_sim_gb()
        
    #Get Black Scholes output 
    bs = m.BlackScholes()
    
    
    #Get Antithetic df
    anti_df = m.antithetic()
    anti_df = anti_df['Antithetic']
    
    #Combine and plot
    df = pd.concat([gb_df, anti_df], axis=1)
    #df = pd.concat([bs], axis=1)
    if Otype == 'C':    
        m.plotdf("Call Option comparison between Antithetic and MC", df)
        #m.plotdf("Call Option Pricing Comparison", df)
    else:
        m.plotdf("Put Option comparison between Antithetic and MC", df)
        #m.plotdf("Put Option Pricing Comparison", df)
    time_elapsed = (time.clock() - time_start)
    print('Time taken to compute the code in seconds:\n',time_elapsed)
    
    #Calculate Standard Error of Each Method
    df.sem()  
    #Calculate for GB and BS
    std_err = df.iloc[:,0:2].std().std()/np.sqrt(len(df))
    print('Standard Error Between MC Sim and Antithetic Model:\n ', std_err)
        
    #resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ####################################################
    #This is for the dataframe you wanted:
    #Loop through expiration times and strike prices
    
    #********You would set the strike prices and expiration dates here********
    expiration_times = [5, 10, 15, 20, 25]
    strike_prices = [100, 110, 120, 130, 140]
    #********You would set the strike prices and expiration dates here********
  
    #Create DataFrames
    bs_df = pd.DataFrame()
    gb_df = pd.DataFrame()
    
    #Create columns of DFs
    for x in expiration_times:
        #Lists for values
        bs_list = []
        gb_list = []    
        
        #Loop through srikes
        for y in strike_prices:
            
            #Run functions to obtain values and append to list(s)
            S0 = y
            T = x
            m = OptionPricing(S0, E, T, rf, sigma, iterations, num_simulations, Otype)
            bs = m.BlackScholes()
            
            if Otype == 'C':                
                bs_list.append(bs['Call Price (BS)'].iloc[0])
                gb = m.option_sim_gb()
                gb_list.append(gb['Call Price (MC)'].iloc[0])
            else:
                bs_list.append(bs['Put Price (BS)'].iloc[0])
                gb = m.option_sim_gb()
                gb_list.append(gb['Put Price (MC)'].iloc[0])                
 
        #Concat column to df
        bs_df[str(x)+' (BS)'] = bs_list
        gb_df[str(x)+' (MC)'] = bs_list
    
    #Index Col
    bs_df['Strike Price'] = strike_prices
    gb_df['Strike Price'] = strike_prices

#Assign index col
bs_df = bs_df.set_index('Strike Price')
gb_df = gb_df.set_index('Strike Price')

#Combine DFs
#result = pd.concat([bs_df, gb_df], axis=1)
result = pd.concat([bs_df, gb_df], axis=1)
print(result)
        
    
    
