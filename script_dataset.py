import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical

def create_dataset_v_sc(nb_data,num,nb):

    X,Y=[],[]
    initial_state  = (0.5,0.7,0.8)

    noise1 = 0.2
    noise2 = 0.2
    noise3 = 0.2

    for k in range(nb_data):
        list_entite = ['a','b','c']
        max_level = dict()
        max_level['a'] = 1
        max_level['b'] = 1
        max_level['c'] = 1

        list_colums = []
        list_colums = list_colums + list_entite
        for one_ele in list_entite:
            list_colums = list_colums + ['c_'+one_ele]
        celerities = pd.DataFrame(columns=list_colums)

        sca = np.random.random_sample(1)[0]
        scmax = 1
        sab = np.random.random_sample(1)[0]
        samax = 1
        sbc = np.random.random_sample(1)[0]
        sbmax = 1

        vac0a0=10*np.random.random_sample(1)[0]
        vac0a1=10*np.random.random_sample(1)[0]
        vac1a0=10*np.random.random_sample(1)[0]
        vac1a1=10*np.random.random_sample(1)[0]
        vba0b0=10*np.random.random_sample(1)[0]
        vba0b1=10*np.random.random_sample(1)[0]
        vba1b0=10*np.random.random_sample(1)[0]
        vba1b1=10*np.random.random_sample(1)[0]
        vcb0c0=10*np.random.random_sample(1)[0]
        vcb0c1=10*np.random.random_sample(1)[0]
        vcb1c0=10*np.random.random_sample(1)[0]
        vcb1c1=10*np.random.random_sample(1)[0]

        Y.append([sca,sab,sbc,vac0a0,vac0a1,vac1a0,vac1a1,vba0b0,vba0b1,vba1b0,vba1b1,vcb0c0,vcb0c1,vcb1c0,vcb1c1])

        cac0a0=vac0a0/(sab - 0)
        cac0a1=vac0a1/(samax - sab)
        cac1a0=vac1a0/(sab - 0)
        cac1a1=vac1a1/(samax - sab)
        cba0b0=vba0b0/(sbc - 0)
        cba0b1=vba0b1/(sbmax - sbc)
        cba1b0=vba1b0/(sbc - 0)
        cba1b1=vba1b1/(sbmax - sbc)
        ccb0c0=vcb0c0/(sca - 0)
        ccb0c1=vcb0c1/(scmax -sca)
        ccb1c0=vcb1c0/(sca - 0)
        ccb1c1=vcb1c1/(scmax -sca)

        df1 = pd.DataFrame([[0,0,0,cac0a0,cba0b0,ccb0c0]],columns=list_colums)
        df2 = pd.DataFrame([[0,0,1,-cac1a0,cba0b0,ccb0c1]],columns=list_colums)
        df3 = pd.DataFrame([[0,1,0,cac0a0,cba0b1,-ccb1c0]],columns=list_colums)
        df4 = pd.DataFrame([[0,1,1,-cac1a0,cba0b1,-ccb1c1]],columns=list_colums)
        df5 = pd.DataFrame([[1,0,0,cac0a1,-cba1b0,ccb0c0]],columns=list_colums)
        df6 = pd.DataFrame([[1,0,1,-cac1a1,-cba1b0,ccb0c1]],columns=list_colums)
        df7 = pd.DataFrame([[1,1,0,cac0a1,-cba1b1,-ccb1c0]],columns=list_colums)
        df8 = pd.DataFrame([[1,1,1,-cac1a1,-cba1b1,-ccb1c1]],columns=list_colums)

        celerities=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8])
        celerities['signature'] = celerities.apply(get_signature,axis=1)

        ini_discrete = ''
        ini_fractional = []
        if initial_state[0]>=sab:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[0]-sab)/(samax - sab)]
        elif initial_state[0]<sab:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[0]/sab]

        if initial_state[1]>=sbc:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[1]-sbc)/(sbmax - sbc)]
        elif initial_state[1]<sbc:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[1]/sbc]

        if initial_state[2]>=sca:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[2]-sca)/(scmax - sca)]
        elif initial_state[2]<sca:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[2]/sca]

        data,t = simulation(ini_discrete,ini_fractional,num)
        real_data = dc(data)
        for i in range(data.shape[0]):
            if data[i][0] < 1:
                real_data[i][0] = data[i][0]*sab
            elif data[i][0] >= 1:
                real_data[i][0] = (data[i][0] - 1)*(samax - sab) + sab
            if data[i][1] < 1:
                real_data[i][1] = data[i][1]*sbc
            elif data[i][1] >= 1:
                real_data[i][1] = (data[i][1] - 1)*(sbmax - sbc) + sbc
            if data[i][2] < 1:
                real_data[i][2] = data[i][2]*sca
            elif data[i][2] >= 1:
                real_data[i][2] = (data[i][2] - 1)*(scmax - sca) + sca
        noise_data = np.zeros((nb+1,3))
        delta_t = t[-1][0]/nb
        new_t = np.zeros((nb+1,1))
        for i in range(nb+1):
            new_t[i][0] = i*delta_t
            for j in range(t.shape[0]-1):
                if t[j][0] <= new_t[i][0] and t[j+1][0] >= new_t[i][0]:
                    noise_data[i][0] = random.gauss(0,noise1) + real_data[j][0] + (real_data[j+1][0] - real_data[j][0])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    noise_data[i][1] = random.gauss(0,noise2) + real_data[j][1] + (real_data[j+1][1] - real_data[j][1])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    noise_data[i][2] = random.gauss(0,noise3) + real_data[j][2] + (real_data[j+1][2] - real_data[j][2])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    break

        X.append(noise_data)
    return np.array(X),np.array(Y)


def create_dataset_sc(nb_data,num,nb):

    X,Y=[],[]
    initial_state  = (0.5,0.7,0.8)

    noise1 = 0.02
    noise2 = 0.02
    noise3 = 0.02

    for k in range(nb_data):
        list_entite = ['a','b','c']
        max_level = dict()
        max_level['a'] = 1
        max_level['b'] = 1
        max_level['c'] = 1

        list_colums = []
        list_colums = list_colums + list_entite
        for one_ele in list_entite:
            list_colums = list_colums + ['c_'+one_ele]
        celerities = pd.DataFrame(columns=list_colums)

        sca = np.random.random_sample(1)[0]
        scmax = 1
        sab = np.random.random_sample(1)[0]
        samax = 1
        sbc = np.random.random_sample(1)[0]
        sbmax = 1

        vac0a0=10*np.random.random_sample(1)[0]
        vac0a1=10*np.random.random_sample(1)[0]
        vac1a0=10*np.random.random_sample(1)[0]
        vac1a1=10*np.random.random_sample(1)[0]
        vba0b0=10*np.random.random_sample(1)[0]
        vba0b1=10*np.random.random_sample(1)[0]
        vba1b0=10*np.random.random_sample(1)[0]
        vba1b1=10*np.random.random_sample(1)[0]
        vcb0c0=10*np.random.random_sample(1)[0]
        vcb0c1=10*np.random.random_sample(1)[0]
        vcb1c0=10*np.random.random_sample(1)[0]
        vcb1c1=10*np.random.random_sample(1)[0]

        Y.append([sca,sab,sbc])

        cac0a0=vac0a0/(sab - 0)
        cac0a1=vac0a1/(samax - sab)
        cac1a0=vac1a0/(sab - 0)
        cac1a1=vac1a1/(samax - sab)
        cba0b0=vba0b0/(sbc - 0)
        cba0b1=vba0b1/(sbmax - sbc)
        cba1b0=vba1b0/(sbc - 0)
        cba1b1=vba1b1/(sbmax - sbc)
        ccb0c0=vcb0c0/(sca - 0)
        ccb0c1=vcb0c1/(scmax -sca)
        ccb1c0=vcb1c0/(sca - 0)
        ccb1c1=vcb1c1/(scmax -sca)

        df1 = pd.DataFrame([[0,0,0,cac0a0,cba0b0,ccb0c0]],columns=list_colums)
        df2 = pd.DataFrame([[0,0,1,-cac1a0,cba0b0,ccb0c1]],columns=list_colums)
        df3 = pd.DataFrame([[0,1,0,cac0a0,cba0b1,-ccb1c0]],columns=list_colums)
        df4 = pd.DataFrame([[0,1,1,-cac1a0,cba0b1,-ccb1c1]],columns=list_colums)
        df5 = pd.DataFrame([[1,0,0,cac0a1,-cba1b0,ccb0c0]],columns=list_colums)
        df6 = pd.DataFrame([[1,0,1,-cac1a1,-cba1b0,ccb0c1]],columns=list_colums)
        df7 = pd.DataFrame([[1,1,0,cac0a1,-cba1b1,-ccb1c0]],columns=list_colums)
        df8 = pd.DataFrame([[1,1,1,-cac1a1,-cba1b1,-ccb1c1]],columns=list_colums)

        celerities=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8])
        celerities['signature'] = celerities.apply(get_signature,axis=1)

        ini_discrete = ''
        ini_fractional = []
        if initial_state[0]>=sab:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[0]-sab)/(samax - sab)]
        elif initial_state[0]<sab:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[0]/sab]

        if initial_state[1]>=sbc:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[1]-sbc)/(sbmax - sbc)]
        elif initial_state[1]<sbc:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[1]/sbc]

        if initial_state[2]>=sca:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[2]-sca)/(scmax - sca)]
        elif initial_state[2]<sca:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[2]/sca]

        data,t = simulation(ini_discrete,ini_fractional,num)
        real_data = dc(data)
        for i in range(data.shape[0]):
            if data[i][0] < 1:
                real_data[i][0] = data[i][0]*sab
            elif data[i][0] >= 1:
                real_data[i][0] = (data[i][0] - 1)*(samax - sab) + sab
            if data[i][1] < 1:
                real_data[i][1] = data[i][1]*sbc
            elif data[i][1] >= 1:
                real_data[i][1] = (data[i][1] - 1)*(sbmax - sbc) + sbc
            if data[i][2] < 1:
                real_data[i][2] = data[i][2]*sca
            elif data[i][2] >= 1:
                real_data[i][2] = (data[i][2] - 1)*(scmax - sca) + sca
        noise_data = np.zeros((nb+1,3))
        delta_t = t[-1][0]/nb
        new_t = np.zeros((nb+1,1))
        for i in range(nb+1):
            new_t[i][0] = i*delta_t
            for j in range(t.shape[0]-1):
                if t[j][0] <= new_t[i][0] and t[j+1][0] >= new_t[i][0]:
                    noise_data[i][0] = random.gauss(0,noise1) + real_data[j][0] + (real_data[j+1][0] - real_data[j][0])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    noise_data[i][1] = random.gauss(0,noise2) + real_data[j][1] + (real_data[j+1][1] - real_data[j][1])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    noise_data[i][2] = random.gauss(0,noise3) + real_data[j][2] + (real_data[j+1][2] - real_data[j][2])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    break

        X.append(noise_data)
    return np.array(X),np.array(Y)

def sc_to_class(X,Y): #renvoie le vecteur X1 des états discrets dans lesquels se trouvent les points de X
    X1,temp=[],[]
    state=''
    for i in range(len(X)):
        for j in range(len(X[0])):
            state=str(int(X[i,j,0]>Y[i,0]))+str(int(X[i,j,1]>Y[i,1]))+str(int(X[i,j,2]>Y[i,2]))
            temp.append(int(state,2)) #on passe du binaire au décimal
        X1.append(temp)
        temp=[]
    return to_categorical(X1)

def affichage2d1d(noise_data, seuils):
    sab, sbc, sca = seuils[0], seuils[1], seuils[2]
    xdata,ydata,zdata = noise_data[0:len(noise_data),0],noise_data[0:len(noise_data),1],noise_data[0:len(noise_data),2]
    n=np.arange(len(xdata))

    #Coloration des points en fonction de leur état discret
    D_id_color = {'000': 'blue', '001': 'orange', '010': 'green', '011': 'red', '100': 'purple', '101': 'brown', '110': 'pink', '111': 'grey'}

    state_list=[] #récupération de l'état discret
    for i in range(len(noise_data)):
        state=str(int(noise_data[i,0]>sab))+str(int(noise_data[i,1]>sbc))+str(int(noise_data[i,2]>sca))
        state_list.append(state) #on passe du binaire au décimal

    color_map=[D_id_color[x] for x in state_list]

    fig=plt.figure(figsize=(9,7))

    sub1 = fig.add_subplot(2,2,1) # two rows, two columns, fist cell
    sub1.scatter(xdata, ydata, c=color_map, label=set(color_map))

    sub2 = fig.add_subplot(2,2,2) # two rows, two columns, second cell
    sub2.scatter(xdata, zdata, c=color_map, label=set(color_map))

    sub3 = fig.add_subplot(2,2,(3,4)) # two rows, two colums, combined third and fourth cell
    sub3.scatter(n, xdata, c='b')
    sub3.scatter(n, ydata, c='r')
    sub3.scatter(n, zdata, c='g')

    sub1.set_xlabel('sa, sab ='+str(round(sab,2)), fontweight='bold')
    sub1.set_ylabel('sb, sbc ='+str(round(sbc,2)), fontweight='bold')
    sub2.set_xlabel('sa, sab ='+str(round(sab,2)), fontweight='bold')
    sub2.set_ylabel('sc, sca ='+str(round(sca,2)), fontweight='bold')
    sub3.set_title('bleu : sa, rouge : sb, vert : sc', y=-0.3)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in D_id_color.values()]
    plt.legend(markers, D_id_color.keys(), numpoints=1, bbox_to_anchor=(1.13,2)) #légende pour chaque couleur

    print("nombre d'états discrets visités : ",len(set(color_map)))
    print('sab =', sab, 'sbc =', sbc, 'sca =', sca)

    plt.show()