""" Utils """
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from string import ascii_letters


def info(dataframe):
    """Prints dataframe parameters
    
    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        Prints parameters: number of columns, number of rows, rate of missing values
    """
    print(str(len(dataframe.columns.values)) + " columns" )
    print(str(len(dataframe)) + " rows")
    print("Rate of missing values in df : " + str(dataframe.isnull().mean().mean()*100) + " %")
    
    
def indicator_search_regex(dataframe, search):
    """ Returns columns matching given string(regex)  
    
    Args:
        dataframe (pd.Dataframe): data source
        search (str): string(regex) to match
    Returns:
        list: columns matching given string(regex)     
    """ 
    found = (dataframe.filter(regex = search).columns.tolist())
    return found

def eta_squared(x,y):
    """Computes ANOVA η2
    
    Args:
        x (pd.Series): data source
        y (pd.Series): data source
    Returns:
        Prints η2: SCE/SCT
    """
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    #print("SCE: \n" + str(SCE))
    #print("SCT: \n" + str(SCT))
    print("η2: \n" + str(SCE/SCT))
    #return SCE/SCT


def linear_regression_function(dataframe, variable_x, variable_y, arrange) :
    """ Plots linear regression
    
    Args:
        dataframe (pd.Dataframe): data source
        variable_x (str): x axis variale
        variable_y (str): y axis variale
        arrange (int):
    Returns:
        Plotted linear regression
    """
    # Fill missing values with 0
    x_sample = dataframe[variable_x]
    y_sample = dataframe[variable_y]

    # Define variables for regression
    Y = dataframe[variable_y]
    X = dataframe[[variable_x]]

    # We will modify x_sample so we create a copy of it
    X = X.copy() 
    X['intercept'] = 1.

    # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
    result = sm.OLS(Y, X).fit() 
    print(result.summary())
    
    a, b = result.params[variable_x], result.params['intercept']
    print("a: " + str(a), "b: " + str(b))

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(dataframe[variable_x], 
             dataframe[variable_y],             
             "o", 
             alpha = 0.5
             )
    
    plt.plot(np.arange(arrange),[a*x+b for x in np.arange(arrange)], linewidth=2.5)
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    plt.show()
    
    
def linear_regression_function_grid(dataframe, variable_x, variable_y, arrange, nrows_, ncols_) :
    """ Plots multiple linear regressions in a grid 
    
    Args:
        dataframe (pd.Dataframe): data source
        variable_x (str): x axis variable
        variable_y (str): y axis variable
        arrange (int):
        nrows_ (int): number of rows
        ncols_ (int): number of columns
    Returns:
        Multiple linear regressions plotted in a grid 
    """    
    # Fill missing values with 0
    x_sample = dataframe[variable_x].fillna(value = 0, inplace = True)
    y_sample = dataframe[variable_y].fillna(value = 0, inplace = True)

    #
    #print("Pearson : " + str(st.pearsonr(x_sample, y_sample)[0]))
    #print("Cov : " + str(np.cov(x_sample, y_sample, ddof = 0)[1,0]))

    # Define variables for regression
    Y = dataframe[variable_y]
    X = dataframe[[variable_x]]

    # We will modify x_sample so we create a copy of it
    X = X.copy() 
    X['intercept'] = 1.

    # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
    result = sm.OLS(Y, X).fit() 
    print(result.summary())
    
    a, b = result.params[variable_x], result.params['intercept']
    print("\n\n" + "a: " + str(a), "b: " + str(b))

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(dataframe[variable_x], 
             dataframe[variable_y],             
             "o", 
             alpha = 0.5
             )
    
    plt.plot(np.arange(arrange),[a*x+b for x in np.arange(arrange)], linewidth=2.5)
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    plt.show()
    
        
def knn_function(variable, dataframe, n_neighbors_, cv_value) :
    """
    Returns classifier 
    
    Args:
         variable (str): column name
         dataframe (pd.Dataframe): data source
         n_neighbors_ (int): number of neighbors to consider
         cv_value (int): cross value number
    Returns:
         :knn classifier
    """
    # Create a training dataframe fully filled for input columns as well as for the target column
    df_products_outliers_missing_values_knn_training = dataframe.dropna(axis = 0, how = 'any')
    
    # Separate input and target values
    X = df_products_outliers_missing_values_knn_training.drop(columns=[variable])
    y = df_products_outliers_missing_values_knn_training[variable].values

    # Split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Create KNN classifier
    knn = KNeighborsRegressor(n_neighbors = n_neighbors_)

    # Fit the classifier to the data
    knn.fit(X_train, y_train)

    ### k-Fold Cross-Validation ###
    # Train model with cv of 5 
    cv_scores = cross_val_score(knn, X, y, cv = int(cv_value))

    # Print each cv score (accuracy) and average them
    print('Cross validation accuracy: {} %' .format(np.mean(cv_scores)*100))
    
    # Root mean square error
    pred_y = knn.predict(X_test)
    rmse = mean_squared_error(y_test, pred_y, squared = False)

    #print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)

    return knn


def plot_grid(rows, cols, dataframe, kind) : 
    """ Plots multiple given kind of plots in a grid 
    
    Args:
        rows (int): number of rows
        cols (int): number of columns
        dataframe (pd.Dataframe): data source
        kind (str): kind of plot                
    Returns:
         Multiple given kind of plots plotted in a grid 
    """
    # Sublot
    nrows = rows
    ncols = cols
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,35))
                    
    # Plot
    nrows_ = 0
    ncols_ = 0
    i = 0

    for nrows_ in range(nrows) :
        for ncols_ in range(ncols) :
            if i < len(dataframe.columns) :
                (
                dataframe[dataframe.columns[i]]
                    .plot(kind = kind, 
                          title = dataframe.columns[i],
                          xticks = [],
                          ax = axes[nrows_][ncols_]),
                     
                )   
            i = i + 1 
            ncols_ = ncols_ + 1
        nrows_ = nrows_ + 1
    plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
    plt.show()
    
    
def plot_grid_hist_kde(rows, cols, dataframe) : 
    """ Plots multiple hist+kde plots in a grid 
    
    Args:
        rows (int): number of rows
        cols (int): number of columns
        dataframe (pd.Dataframe): data source                
    Returns:
        Multiple hist+kde plots plotted in a grid 
    """
    # Sublot
    nrows = rows
    ncols = cols
    fig, axes = plt.subplots( nrows = nrows, ncols = ncols, figsize = (25, 13))

    # Plot
    nrows_ = 0
    ncols_ = 0
    i = 0

    for nrows_ in range(nrows) :
        for ncols_ in range(ncols) :
            if i < len(dataframe.columns) :
                ax = dataframe[dataframe.columns[i]].plot(kind = 'hist', 
                                                          xlabel = '', 
                                                          ylabel = '',  
                                                          fontsize = 18,
                                                          ax = axes[nrows_][ncols_],
                                                          )
                dataframe[dataframe.columns[i]].plot(kind = 'kde', 
                                                     ax = ax, 
                                                     secondary_y = True)
                plt.title(dataframe.columns[i], {'size':'20'})  
            i = i + 1 
            ncols_ = ncols_ + 1
        nrows_ = nrows_ + 1
    plt.show()
    

def plot_grid_hist_cumulative(rows, cols, dataframe) : 
    """ Plots multiple hist cumulative plots in a grid 
    
    Args:
        rows (int): number of rows
        cols (int): number of columns
        dataframe (pd.Dataframe): data source                
    Returns:
        Multiple hist cumulative plots plotted in a grid 
    """
    # Sublot
    nrows = rows
    ncols = cols
    fig, axes = plt.subplots( nrows = nrows, ncols = ncols, figsize = (20, 30))

    # Plot
    nrows_ = 0
    ncols_ = 0
    i = 0

    for nrows_ in range(nrows) :
        for ncols_ in range(ncols) :
            if i < len(dataframe.columns) :
                (
                dataframe[dataframe.columns[i]]
                    .plot(kind = 'hist',
                          cumulative=True,
                          title = dataframe.columns[i],  
                          ax = axes[nrows_][ncols_])
                )                                                                                                                  
            i = i + 1 
            ncols_ = ncols_ + 1
        nrows_ = nrows_ + 1       
    plt.show()
    
    
def dataframe_correlation_graph(dataframe):
    """ Plots seaborn graph of correlations 
    
    Args:
        dataframe (pd.Dataframe): data source
                     
    Returns:
        Graph of correlations 
    """    

    # Compute the correlation matrix
    corr = dataframe.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 14))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    #
    sns.set(font_scale=1)
        
    # Draw the heatmap with the mask and correct aspect ratio
    res = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)
          
    #
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 13)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 13)        
    plt.show()
    
    
def pairs_correlation_ranking(dataframe):
    """ Ranking of correlation pairs 
    
    Args:
        dataframe (pd.Dataframe): data source
                     
    Returns:
        nd.Series: Ranking list of correlation pairs 
    """
    # Correlation matrix dataframe
    df_corr = dataframe.corr() 

    # Filter out identical pairs    
    df_corr_filter = df_corr[df_corr != 1.000]

    # Create list and sort values
    series = (df_corr_filter.abs()
         .unstack()
         .drop_duplicates()
         .sort_values(kind="quicksort", ascending = False)
        )

    # Show
    print("Pairs correlation ranking: \n\n" + str(series))
    return series
    

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """ Plots PCA circle of correlation 
    
    Args:
        pcs (numpy.ndarray):
        n_comp (int):
        pca (sklearn.decomposition._pca.PCA):
        axis_ranks (list):       
                     
    Returns:
        Plot 
    """    
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,9))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            #plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    """ Plots PCA factorial planes
    
    Args:
        X_projected (numpy.ndarray): 
        n_comp (int): number of components to compute
        pca (sklearn.decomposition._pca.PCA):  
        axis_ranks (list):  
                               
    Returns:
        Plot 
    """    
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure  
            plt.rc('axes', unicode_minus=False)
            fig = plt.figure(figsize=(8,8))
           
            
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i], fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    """ Plots PCA scree plot
    
    Args:
        pca (sklearn.decomposition._pca.PCA):    
                     
    Returns:
        Plot 
    """    
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(8,7))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)