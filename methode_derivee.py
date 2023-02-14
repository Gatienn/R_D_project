from scipy.stats import linregress

'''from rdp import rdp

points_rdp=rdp(noise_data,0.5)

x=np.arange(len(points_rdp))
plt.plot(x,points_rdp)
plt.show()'''

def derivee_to_classe(X):
    classe_dict= {'000': [], '001': [], '010': [], '011': [], '100': [], '101': [], '110': [], '111': []}
    for i in range(len(X)-1):

        classe_dict[state].append(X[i])

    dist, sab_derivee = 1,0
    for k in range(4):
        for i in range(len(classe_dict['0'+format(k, '#04b')[-2:]])):
            for j in range(len(classe_dict['1'+format(k, '#04b')[-2:]])):

                a0=classe_dict['0'+format(k, '#04b')[-2:]][i][0]
                a1=classe_dict['1'+format(k, '#04b')[-2:]][j][0]

                if np.abs(a0-a1) < dist and (0<a0<1) and (0<a1<1):
                    dist = np.abs(a0-a1)
                    sab_derivee= (np.abs(a0)+np.abs(a1))/2
                    '''print('dist ', dist)
                    print('sab ',sab_derivee)'''
    return sab_derivee

def derivee_dataset(X,Y):
    L=np.array([np.abs(derivee_to_classe(X[i]) - Y[i][1]) for i in range(len(X))])
    return np.mean(L)

def grid2d(min = 0, max = 1, step = 0.1): #crée une grille avec les valeurs de seuil à parcourir
    x=np.arange(min,max+step,step)
    y=np.arange(min,max+step,step)
    return np.array(np.meshgrid(x,y))

def grid3d(min = 0, max = 1, step = 0.1): #crée une grille avec les valeurs de seuil à parcourir
    x=np.arange(min,max+step,step)
    y=np.arange(min,max+step,step)
    z=np.arange(min,max+step,step)
    return np.array(np.meshgrid(x,y,z))

def seuil_grid_search2d(X, seuils, step=0.1, min=0, max=1): #fonction qui parcoure la grille pour trouver les seuils avec le meilleur score
    xx,yy = grid2d(min,max,step)
    scores = np.zeros(xx.shape)
    sc = seuils[2]
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            seuils= np.array([xx[i,j],yy[i,j], sc])
            X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
            scores[i,j] = correlation_score3d(X, X_classes) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), np.shape(xx))
    return scores, xx[s], yy[s]

def seuil_grid_search3d(X, seuils, step=0.1, min=0, max=1, nb_points=5): #fonction qui parcoure la grille pour trouver les seuils avec le meilleur score
    xx,yy,zz=grid3d(min,max,step)
    scores=np.zeros(xx.shape)
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            for k in range(len(zz[0,0])):
                seuils= np.array([xx[i,j,k],yy[i,j,k],zz[i,j,k]])
                X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
                scores[i,j,k] = correlation_score3d(X, X_classes, nb_points) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), np.shape(xx))
    return scores, xx[s], yy[s], zz[s]

def seuil_grid_search_opt3d(X, seuils, step=0.1, min=0, max=1, nb_points=5): #parcourt les points de la simulation plutôt qu'une grille afin de réduire le temps de calcul
    n=len(X)
    scores=np.zeros(shape = (n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                print (i,j,k)
                seuils= np.array([X[i,0],X[j,1],X[k,2]])
                X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
                scores[i,j,k] = correlation_score3d(X, X_classes, nb_points) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), (n,n,n))
    return scores, X[s[0],0], X[s[1],1], X[s[2],2]

def sc_to_class1(X,seuils): #renvoie le vecteur X1 des états discrets dans lesquels se trouvent les points de X
    X1=[]
    state=''
    for i in range(len(X)):
        state=str(int(X[i][0]>seuils[0]))+str(int(X[i][1]>seuils[1]))+str(int(X[i][2]>seuils[2]))
        X1.append(int(state,2)) #on passe du binaire au décimal
    return to_categorical(X1)

def correlation_score3d(X, X_classes, nb_points=5):
    score = 0
    for i in range(nb_points,len(X_classes)-nb_points):
        a = np.argmax(X_classes[i]) #état supposé
        for k in range(3):
            score += point_score_value(X,a,i,k, nb_points) #ajout d'un score traduisant la corrélation état discret/valeur de la dérivée
    return score/(3*(len(X)-1))

def point_score(X,a,i,k, nb_points=5): #score traduisant la corrélation état discret/valeur de la dérivée
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    regression_range = nb_points
    Y=X[i-regression_range:i+regression_range+1, k]
    linregress_model = linregress(x=np.arange(0,len(Y)/10,0.1), y=Y) #ajustement d'une droite sur les points considérés
    slope, rvalue = linregress_model.slope, linregress_model.rvalue
    while(regression_range>1 and np.abs(rvalue)<0.90):
        regression_range-=1
        Y=X[i-regression_range:i+regression_range+1, k]
        linregress_model = linregress(x=np.arange(0,len(Y)/10,0.1), y=Y) #ajustement d'une droite sur les points considérés
        slope, rvalue = linregress_model.slope, linregress_model.rvalue
    #if regression_range>1:
        #print('yes!')
    #print(Y,'slope',slope, 'rvalue', rvalue, regression_range)
    return np.abs(slope)*regression_range*(int(slope>0)==int(state_derivative[a][k]))

def point_score_sign(X,a,i,k): #score suivant le signe de la dérivée
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    return int(int((X[i+1][k]>X[i][k])) == int(state_derivative[a][k])) #comparaison avec l'état suivant

def point_score_sign1(X,a,i,k, nb_points=5):
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    x=X[i,k]
    s=np.sum(x-X[i-nb_points:i,k])+np.sum(X[i+1:i+1+nb_points,k]-x)
    return int(int(s>0) == int(state_derivative[a][k]))

def point_score_value(X,a,i,k, nb_points=5):
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    s=0
    for j in range(1,nb_points+1): #nb de points à considérer de part et d'autre
        s+= (X[i+j,k]-X[i-j,k])/(2*j)
    return np.abs(s)*int(int(s>0) == int(state_derivative[a][k]))
    # =valeur de la pente si le signe correspond, 0 sinon


def evaluate_grid_search(Ns, step= 0.1, nb_points=5): #Utilise la méthode un grand nombre de fois et évalue son efficacité
    score_tot = 0
    for i in range(Ns):
        print(i)
        G=create_dataset_sc(1,10,50)
        X, seuils= np.array(G[0][0]), np.array(G[1][0])
        scores, sa, sb, sc = seuil_grid_search3d(X, seuils, step=step)
        score_tot = score_tot + np.abs(sa-seuils[0]) + np.abs(sb-seuils[1]) + np.abs(sb-seuils[2])
    return score_tot/(3*Ns)

def score_map(min = 0, max = 1, step = 0.1, nb_points=5):
    G = create_dataset_sc(1,10,50)
    X, seuils = np.array(G[0][0]), np.array(G[1][0])
    grid = grid2d(min, max, step)
    xx, yy = grid[0], grid[1]
    scores, sa, sb = seuil_grid_search2d(X, seuils)
    plt.pcolor(xx, yy, scores)
    #plt.imshow(scores)
    plt.colorbar()
    plt.xlabel('sab estimé : ' + str(round(sa,2))+' sab réel ' + str(round(seuils[0],2)))
    plt.ylabel(('sbc estimé : ' + str(round(sb,2))+' sbc réel ' + str(round(seuils[1],2))))
    print(scores)
    print('sa estimé : ', sa, 'sa réel', seuils[0])
    print('sb estimé : ', sb, 'sb réel', seuils[1])
    print('sc réel', seuils[2])
    plt.show()

G=create_dataset_sc(1,10,50)
X, seuils= np.array(G[0][0]), np.array(G[1][0])
classes=sc_to_class1(X,seuils)

#scipy.stats.linregress